#!/usr/bin/env python
"""Turn normal-only development runs (scripts/dev_protocol.py) into ONE frozen configuration.

Decisions are made one axis at a time, in a fixed order written down before any development result exists, each by
the highest development object-level AUROC (mean over categories and folds, then mean over datasets, every dataset
weighted equally). Ties within 0.5 points are broken towards the simpler option (listed first).

  1. descriptor base scale k          compared with the rule loc10 / p99 (the rule the scale was originally tuned with)
  2. matching rule                    at the chosen k, pooling p99
  3. training views                   memory subsets of the chosen rule family (loc10 / glob_full only)
  4. object-score pooling             for the chosen rule
  5. multi-hypothesis registration    chosen rule with vs without pose selection (h1_ = single pose)

Usage: python scripts/select_config.py results/dev/dev_*json --out results/dev/selection.md
Every dev file name must contain the scale tag, e.g. dev_dev-nn10_real3d.json -> k = 10.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

RULES = ["glob_cs", "glob_full", "loc10", "loc05", "loc15", "knn64", "knn256", "soft0p5", "soft1", "soft2",
         "cal10_k0", "cal10", "hyb"]                       # simpler options first (tie-break)
VIEWS = ["V0", "V01", "V02", "M2", "V012"]
POOL = ["p99", "max", "p95", "top1pct", "mean"]
TIE = 0.5


def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(paths):
    """{k: {dataset: {unit: [records]}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        d = json.load(open(p))
        k = d["config"]["max_nn"]
        ds = d["config"]["dataset"]
        for unit, recs in d["records"].items():
            alias_missing_views(recs)
            out[k][ds][unit] = recs
    return out


def alias_missing_views(recs):
    """channels.memory_masks drops a view subset that equals the full memory (e.g. V01 on a benchmark without real
    single-view scans: complete + simulated IS the whole memory). Its channel is then identical to the full-memory
    channel, so it is filled in from there instead of being treated as missing."""
    for r in recs:
        for fam in ("loc10", "glob_full"):
            for v in VIEWS[:-1]:
                for p in POOL:
                    if f"{fam}_{v}_{p}" not in r and f"{fam}_{p}" in r:
                        r[f"{fam}_{v}_{p}"] = r[f"{fam}_{p}"]


def unit_values(data_k, key):
    """{dataset: {unit: AUROC}} for one channel key."""
    out = {}
    for ds, units in data_k.items():
        out[ds] = {u: auroc([r["label"] for r in recs], [r[key] for r in recs]) for u, recs in units.items()
                   if all(key in r for r in recs)}
    return out


def boot_diff(data_k, key_a, key_b, B=2000, seed=0):
    """95 % CI of score(key_a) - score(key_b), resampling category-fold units within each dataset."""
    ua, ub = unit_values(data_k, key_a), unit_values(data_k, key_b)
    rng = np.random.default_rng(seed)
    ds_list = [ds for ds in ua if ds in ub]
    diffs = np.empty(B)
    for b in range(B):
        per = []
        for ds in ds_list:
            units = [u for u in ua[ds] if u in ub[ds] and ua[ds][u] == ua[ds][u] and ub[ds][u] == ub[ds][u]]
            pick_ = rng.choice(len(units), len(units))
            per.append(np.mean([ua[ds][units[i]] - ub[ds][units[i]] for i in pick_]) * 100)
        diffs[b] = np.mean(per)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def score(data_k, key):
    """mean over datasets of (mean over category-folds of the AUROC of `key`)."""
    per_ds = {}
    for ds, units in data_k.items():
        vals = [auroc([r["label"] for r in recs], [r[key] for r in recs]) for recs in units.values()
                if all(key in r for r in recs)]
        vals = [v for v in vals if v == v]
        if vals:
            per_ds[ds] = 100 * float(np.mean(vals))
    return (float(np.mean(list(per_ds.values()))) if per_ds else float("nan")), per_ds


def pick(cands):
    """cands: list of (name, value) in simplicity order -> the simplest within TIE of the best."""
    cands = [(n, v) for n, v in cands if v == v]
    best = max(v for _, v in cands)
    for n, v in cands:
        if v >= best - TIE:
            return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load(a.files)
    L = ["# Normal-only configuration selection (no test labels used)\n",
         f"Development files: {len(a.files)}; scales: {sorted(data)}; datasets: "
         f"{sorted({ds for k in data for ds in data[k]})}. Tie margin {TIE} AUROC points -> simpler option.\n"]

    def table(title, rows, keys=None, dk=None):
        """keys: channel key per row (same order) -> adds the bootstrap CI of (row - best row)."""
        dss = sorted({ds for _, _, pd in rows for ds in pd})
        L.append(f"\n## {title}\n")
        ci_col = keys is not None and dk is not None
        L.append("| option | mean | " + " | ".join(dss) + (" | diff to best, 95 % CI |" if ci_col else " |"))
        L.append("|---|---|" + "---|" * len(dss) + ("---|" if ci_col else ""))
        ib = int(np.nanargmax([v for _, v, _ in rows]))
        for i, (n, v, pd) in enumerate(rows):
            line = f"| {n} | {v:.2f} | " + " | ".join(f"{pd.get(d, float('nan')):.2f}" for d in dss) + " |"
            if ci_col:
                if i == ib:
                    line += " best |"
                else:
                    lo, hi = boot_diff(dk, keys[i], keys[ib])
                    line += f" {v - rows[ib][1]:+.2f} [{lo:+.2f}, {hi:+.2f}] |"
            L.append(line)

    # 1. scale
    rows = [(f"k={k}", *score(data[k], "loc10_p99")) for k in sorted(data)]
    table("1. Descriptor base scale (rule loc10, pooling p99)", rows)
    k_sel = int(pick([(n, v) for n, v, _ in rows])[2:])
    dk = data[k_sel]
    # 2. rule
    rows = [(r, *score(dk, f"{r}_p99")) for r in RULES]
    table(f"2. Matching rule (k={k_sel}, pooling p99)", rows, [f"{r}_p99" for r in RULES], dk)
    rule = pick([(n, v) for n, v, _ in rows])
    # 3. views (only for loc10 / glob_full families)
    fam = "glob_full" if rule.startswith("glob") else "loc10"
    vkeys = [f"{fam}_{v}_p99" if v != "V012" else f"{fam}_p99" for v in VIEWS]
    rows = [(v, *score(dk, k_)) for v, k_ in zip(VIEWS, vkeys)]
    table(f"3. Training views (rule family {fam}, pooling p99)", rows, vkeys, dk)
    views = pick([(n, v) for n, v, _ in rows])
    # 4. pooling
    rows = [(p, *score(dk, f"{rule}_{p}")) for p in POOL]
    table(f"4. Object-score pooling (rule {rule})", rows, [f"{rule}_{p}" for p in POOL], dk)
    pool = pick([(n, v) for n, v, _ in rows])
    # 5. MHR
    if rule in ("glob_full", "loc05", "loc10", "loc15", "knn64", "knn256", "cal10"):
        rows = [("single pose (h1)", *score(dk, f"h1_{rule}_{pool}")), ("multi-hypothesis", *score(dk, f"{rule}_{pool}"))]
        table(f"5. Pose hypotheses (rule {rule}, pooling {pool})", rows, [f"h1_{rule}_{pool}", f"{rule}_{pool}"], dk)
        mhr = pick([(n, v) for n, v, _ in rows]) == "multi-hypothesis"
    else:
        mhr = True
        L.append(f"\n## 5. Pose hypotheses\n\nnot separable for rule {rule} (kept at H = 6).\n")
    sel = {"max_nn": k_sel, "rule": rule, "views": views, "pooling": pool, "mhr": mhr}
    L.append("\n## Selected (frozen) configuration\n\n```json\n" + json.dumps(sel, indent=2) + "\n```\n")
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(sel, open(re.sub(r"\.md$", "", a.out) + ".json", "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
