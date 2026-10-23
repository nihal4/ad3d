#!/usr/bin/env python
"""Offline statistics from saved per-sample scores (no GPU, no re-run).

For one dataset / config it loads every `<tag>-s<seed>-part*_<dataset>_*_scores.json` and reports
  * object-level AUROC per channel (mean over categories), per seed;
  * paired, stratified bootstrap over TEST OBJECTS within each category (normals and anomalies resampled
    separately, so every replicate keeps both classes; the same resample is used for both channels and
    for all seeds, which share the same test files) -> 95 % CI of each channel and of the paired
    difference (A - B), aggregated exactly like the headline number (mean over categories, mean over seeds);
  * per-category paired differences with their own bootstrap CI;
  * operating points: FPR on normal objects at 95 % TPR, and TPR at 5 % FPR (per category, then mean);
  * defect-size strata: AUROC of all normals vs the anomalies in each tertile of anomalous-point fraction.

Usage
  python scripts/stats_offline.py --glob '<dir>/**/mhr6-nn10-pluscut-p99-loc10-s*_scores.json' \
      --dataset real3d --a loc10_p99 --b glob_p99 --out results_and_summary/results/stats/real3d
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_curve


def load(pattern: str) -> dict[int, dict[str, list[dict]]]:
    """{seed: {category: [records]}} merged over parts."""
    out: dict[int, dict[str, list[dict]]] = defaultdict(dict)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    for f in files:
        m = re.search(r"-s(\d+)(?:-part\d+)?_", os.path.basename(f))
        seed = int(m.group(1)) if m else 0
        for cls, recs in json.load(open(f)).items():
            out[seed][cls] = recs
    return dict(out)


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties (identical to sklearn's roc_auc_score)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def keyed(recs):
    """{sample key: record}. Older MVTec 3D-AD runs stored bare file names that repeat across defect folders
    (000.tiff in every folder); then the key is the loading position, which is the same for every seed."""
    names = [r["file"] for r in recs]
    if len(set(names)) == len(names):
        return {n: r for n, r in zip(names, recs)}
    return {f"{i:05d}|{n}": r for i, (n, r) in enumerate(zip(names, recs))}


def op_points(y, s, tpr_target=0.95, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    fpr_at = float(np.min(fpr[tpr >= tpr_target])) if np.any(tpr >= tpr_target) else 1.0
    tpr_at = float(np.max(tpr[fpr <= fpr_target])) if np.any(fpr <= fpr_target) else 0.0
    return fpr_at, tpr_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--a", default="loc10_p99")
    ap.add_argument("--b", default="glob_p99")
    ap.add_argument("--extra", default="loc10_max", help="comma-separated further channels to summarise")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load(args.glob)
    seeds = sorted(data)
    cats = sorted(set.intersection(*[set(data[s]) for s in seeds]))
    chans = [args.a, args.b] + [c for c in args.extra.split(",") if c]
    rng = np.random.default_rng(0)

    # arrays per (seed, cat, channel), test-file order made identical across seeds
    Y, S = {}, {}
    for cat in cats:
        files = sorted(keyed(data[seeds[0]][cat]))
        for sd in seeds:
            by = keyed(data[sd][cat])
            recs = [by[f] for f in files]
            Y[cat] = np.array([r["label"] for r in recs])
            for ch in chans:
                S[(sd, cat, ch)] = np.array([r[ch] for r in recs], dtype=float)
        frac = np.array([r["n_anom_points"] / max(r["n_points"], 1) for r in recs])
        S[("frac", cat)] = frac

    lines = [f"# Offline statistics: {args.dataset}  ({len(seeds)} seeds: {seeds}, {len(cats)} categories, "
             f"bootstrap B={args.B}, resampling unit = test object within category)\n"]

    # ---- point estimates per seed
    point = {ch: [np.mean([_auroc(Y[c], S[(sd, c, ch)]) for c in cats]) * 100 for sd in seeds] for ch in chans}
    lines.append("## Object-level AUROC (%), mean over categories\n")
    lines.append("| channel | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | std (seeds) |")
    lines.append("|---|" + "---|" * (len(seeds) + 2))
    for ch in chans:
        v = np.array(point[ch])
        lines.append(f"| {ch} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {v.mean():.2f} | {v.std(ddof=1) if len(v) > 1 else 0:.2f} |")

    # ---- paired stratified bootstrap
    idx_n = {c: np.where(Y[c] == 0)[0] for c in cats}
    idx_a = {c: np.where(Y[c] == 1)[0] for c in cats}
    boot = {ch: np.empty(args.B) for ch in chans}
    boot_cat = {(c, ch): np.empty(args.B) for c in cats for ch in (args.a, args.b)}
    for b in range(args.B):
        per_ch = {ch: [] for ch in chans}
        for c in cats:
            ii = np.concatenate([rng.choice(idx_n[c], len(idx_n[c])), rng.choice(idx_a[c], len(idx_a[c]))])
            y = Y[c][ii]
            for ch in chans:
                a = np.mean([_auroc(y, S[(sd, c, ch)][ii]) for sd in seeds])
                per_ch[ch].append(a)
                if ch in (args.a, args.b):
                    boot_cat[(c, ch)][b] = a
        for ch in chans:
            boot[ch][b] = np.mean(per_ch[ch]) * 100
    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)
    lines.append("\n## Bootstrap 95 % CI (mean over seeds and categories)\n")
    lines.append("| quantity | estimate | 95 % CI |")
    lines.append("|---|---|---|")
    for ch in chans:
        lo, hi = ci(boot[ch])
        lines.append(f"| {ch} | {np.mean(point[ch]):.2f} | [{lo:.2f}, {hi:.2f}] |")
    d = boot[args.a] - boot[args.b]
    lo, hi = ci(d)
    dpt = np.mean(point[args.a]) - np.mean(point[args.b])
    p_le0 = float(np.mean(d <= 0)) if dpt > 0 else float(np.mean(d >= 0))
    lines.append(f"| {args.a} - {args.b} | {dpt:+.2f} | [{lo:+.2f}, {hi:+.2f}] (one-sided bootstrap p = {p_le0:.4f}) |")

    # ---- per-category paired differences
    lines.append(f"\n## Per-category paired difference {args.a} - {args.b} (AUROC %, mean over seeds)\n")
    lines.append("| category | n normal | n anomalous | " + args.a + " | " + args.b + " | diff | 95 % CI |")
    lines.append("|---|---|---|---|---|---|---|")
    rows = []
    for c in cats:
        a = np.mean([_auroc(Y[c], S[(sd, c, args.a)]) for sd in seeds]) * 100
        bb = np.mean([_auroc(Y[c], S[(sd, c, args.b)]) for sd in seeds]) * 100
        dd = (boot_cat[(c, args.a)] - boot_cat[(c, args.b)]) * 100
        lo, hi = ci(dd)
        rows.append((c, len(idx_n[c]), len(idx_a[c]), a, bb, a - bb, lo, hi))
    for r in sorted(rows, key=lambda r: -r[5]):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:+.1f} | [{r[6]:+.1f}, {r[7]:+.1f}] |")
    sig_pos = sum(1 for r in rows if r[6] > 0)
    sig_neg = sum(1 for r in rows if r[7] < 0)
    lines.append(f"\nCategories with CI entirely above 0: {sig_pos}; entirely below 0: {sig_neg}; "
                 f"overlapping 0: {len(rows) - sig_pos - sig_neg}.")
    import csv
    with open(os.path.join(args.out, "per_category_diff.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_normal", "n_anomalous", args.a, args.b, "diff", "ci_lo", "ci_hi"])
        w.writerows(rows)

    # ---- operating points
    lines.append("\n## Operating points (per category, then mean over categories and seeds)\n")
    lines.append("| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |")
    lines.append("|---|---|---|")
    for ch in chans:
        f_, t_ = [], []
        for sd in seeds:
            for c in cats:
                a, b2 = op_points(Y[c], S[(sd, c, ch)])
                f_.append(a); t_.append(b2)
        lines.append(f"| {ch} | {np.mean(f_) * 100:.1f} | {np.mean(t_) * 100:.1f} |")

    # ---- defect-size strata (pooled tertiles of the anomalous-point fraction)
    allf = np.concatenate([S[("frac", c)][Y[c] == 1] for c in cats])
    q = np.percentile(allf, [100 / 3, 200 / 3])
    lines.append("\n## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: "
                 f"small < {q[0] * 100:.2f} % <= medium < {q[1] * 100:.2f} % <= large)\n")
    lines.append("| channel | small | medium | large |")
    lines.append("|---|---|---|---|")
    for ch in chans:
        vals = []
        for lo_, hi_ in ((0, q[0]), (q[0], q[1]), (q[1], 1.01)):
            per = []
            for sd in seeds:
                for c in cats:
                    fr = S[("frac", c)]
                    keep = (Y[c] == 0) | ((fr >= lo_) & (fr < hi_))
                    if Y[c][keep].sum() == 0:
                        continue
                    per.append(_auroc(Y[c][keep], S[(sd, c, ch)][keep]))
            vals.append(np.nanmean(per) * 100)
        lines.append(f"| {ch} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

    txt = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
