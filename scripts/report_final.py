#!/usr/bin/env python
"""Plain-text report of the frozen final runs (step 17): object AUROC of the headline and comparison channels per
dataset / seed / category, mean +- std over seeds, point-level metrics from the merged CSVs, timing and log errors.
Printed AND written to --out, so the numbers survive in the notebook log even if the session ends. Never raises.

Usage: python scripts/report_final.py [--base /kaggle/working/ad3d] [--out /kaggle/working/report_17.txt] [--tag final]
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
from collections import defaultdict

import numpy as np

CH = ["hyb_max", "h1_hyb_max", "hyb_top1pct", "hyb_p99", "loc10_max", "loc10_p99", "cal10_max",
      "glob_full_max", "glob_full_p99", "glob_cs_max", "glob_cs_p99", "knn256_max"]


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or np.isnan(s).any():
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(base, tag):
    tl = f"{base}/logs/timing.log"
    print("=== TIMING ===\n" + (open(tl).read() if os.path.exists(tl) else "(none)"))
    files = sorted(glob.glob(f"{base}/results/parts/{tag}-s*-part*_scores.json"))
    print(f"=== SCORE FILES ({len(files)}) ===")
    per = defaultdict(lambda: defaultdict(dict))          # per[ds][seed][cls] = [records]
    for f in files:
        m = re.search(r"-s(\d+)-part\d+_([a-z0-9]+)_", os.path.basename(f))
        if not m:
            continue
        try:
            for cls, recs in json.load(open(f)).items():
                per[m.group(2)][int(m.group(1))][cls] = recs
        except Exception as e:
            print(f"[!] {os.path.basename(f)}: {type(e).__name__} {e}")
    for ds in sorted(per):
        print(f"\n######## {ds}: seeds {sorted(per[ds])}")
        seed_means = defaultdict(list)
        for s in sorted(per[ds]):
            cats = sorted(per[ds][s])
            print(f"\n--- {ds} seed {s}: {len(cats)} categories ---")
            print("CAT|" + "|".join(CH))
            vals = defaultdict(list)
            for c in cats:
                recs = per[ds][s][c]
                y = [r["label"] for r in recs]
                row = [auc(y, [r.get(k, np.nan) for r in recs]) for k in CH]
                for k, v in zip(CH, row):
                    vals[k].append(v)
                print(f"CAT|{ds}|{s}|{c}|n={len(recs)}|" + "|".join(f"{v:.3f}" for v in row))
            means = {k: 100 * np.nanmean(vals[k]) for k in CH}
            print(f"MEAN|{ds}|{s}|" + "|".join(f"{k}={means[k]:.2f}" for k in CH))
            for k in CH:
                seed_means[k].append(means[k])
        print(f"\n=== {ds}: object AUROC, mean +- std over {len(per[ds])} seeds ===")
        for k in CH:
            v = np.array(seed_means[k])
            print(f"  {k:>15s}  {np.nanmean(v):6.2f} +- {np.nanstd(v):4.2f}   seeds: " + " ".join(f"{x:.2f}" for x in v))
    print("\n=== MERGED CSV MEANS (point-level metrics of the key channels) ===")
    for f in sorted(glob.glob(f"{base}/results/{tag}-s*_merged-*.csv")):
        try:
            import pandas as pd
            df = pd.read_csv(f, index_col=0)
            row = df.loc["__mean__"]
            cols = [c for c in df.columns if re.fullmatch(
                r"((hyb|h1_hyb|loc10|glob_full|glob_cs|cal10|knn256|soft1)_)?(o_auroc|p_auroc|p_aupr|aupro)", c)]
            print(os.path.basename(f) + " | " + " ".join(f"{c}={row[c]:.4f}" for c in cols))
            tcols = [c for c in df.columns if c.endswith("_s") or c.startswith("peak_")]
            if tcols:
                print("   time/memory: " + " ".join(f"{c}={row[c]:.3g}" for c in tcols))
        except Exception as e:
            print(f"[!] {os.path.basename(f)}: {type(e).__name__} {e}")
    for lg in sorted(glob.glob(f"{base}/logs/{tag}*.log")):
        L = open(lg, errors="ignore").read().splitlines()
        bad = [x for x in L if "Error" in x or "Traceback" in x]
        print(f"\n=== LOG {os.path.basename(lg)}: {len(L)} lines, {len(bad)} error lines ===")
        for x in bad[:20] + L[-3:]:
            print(x)
    print("\n=== END OF REPORT ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/kaggle/working/ad3d")
    ap.add_argument("--out", default="/kaggle/working/report_17.txt")
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            report(a.base, a.tag)
    except Exception as e:
        buf.write(f"\n[!] report failed: {type(e).__name__} {e}\n")
    text = buf.getvalue()
    try:
        open(a.out, "w").write(text)
    except Exception as e:
        text += f"\n[!] could not save {a.out}: {e}\n"
    print(text)
