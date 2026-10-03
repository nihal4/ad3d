#!/usr/bin/env python
"""Aggregate tagged multi-seed runs: mean +/- std over seeds, and paired deltas.

Tags must look like <config>-s<seed>  (see run_seeds.sh).
  python scripts/aggregate_seeds.py results/ [--ref base] [--metric o_auroc]
"""
import argparse, glob, os, re
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("results_dir")
ap.add_argument("--ref", default="base", help="config used as the paired reference")
ap.add_argument("--metric", default="o_auroc")
a = ap.parse_args()

runs = {}  # config -> {seed: DataFrame}
for f in sorted(glob.glob(os.path.join(a.results_dir, "*_real3d_*.csv"))):
    m = re.match(r"(.+)-s(\d+)_real3d_", os.path.basename(f))
    if not m:
        continue
    df = pd.read_csv(f, index_col=0)
    runs.setdefault(m.group(1), {})[int(m.group(2))] = df  # latest file per seed wins

cols = ["o_auroc", "p_auroc", "p_aupr", "p_aupr_pooled"]
rows = []
for cfg, by_seed in runs.items():
    means = pd.DataFrame({s: d.loc["__mean__", cols] for s, d in by_seed.items()}).T.astype(float)
    r = {"config": cfg, "n_seeds": len(by_seed)}
    for c in cols:
        r[c] = f"{means[c].mean():.4f} +/- {means[c].std(ddof=1) if len(means) > 1 else float('nan'):.4f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

if a.ref in runs:
    print(f"\nPaired delta vs '{a.ref}' on {a.metric} (same seeds only; mean over 12 categories)")
    ref = runs[a.ref]
    for cfg, by_seed in runs.items():
        if cfg == a.ref:
            continue
        common = sorted(set(ref) & set(by_seed))
        if not common:
            continue
        d = np.array([by_seed[s].loc["__mean__", a.metric] - ref[s].loc["__mean__", a.metric]
                      for s in common], dtype=float)
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        flag = "" if len(d) < 3 else ("  <-- |mean| < 2*std: not distinguishable from noise"
                                      if abs(d.mean()) < 2 * sd else "")
        print(f"  {cfg:>12s}: {100*d.mean():+.2f} pts  (std {100*sd:.2f}, seeds {common}){flag}")
