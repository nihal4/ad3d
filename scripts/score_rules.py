#!/usr/bin/env python
"""Compare object-score rules OFFLINE from the per-sample score files (<tag>_<dataset>_<stamp>_scores.json).

No feature extraction is re-run: every run saves, per test sample, max / top-k / percentile / mean of its
point-score map. This prints the object-level AUROC of each rule per category and the 12-category mean.

    python scripts/score_rules.py results/mhr6-nn10-s0_real3d_*_scores.json [more files...]

NOTE: choosing the rule by looking at this table is test-set selection; report it as such (or pick the
rule on Anomaly-ShapeNet and only apply it to Real3D-AD).
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def table(path):
    d = json.load(open(path))
    rules = None
    rows = {}
    for cls, recs in d.items():
        if not recs:
            continue
        rules = rules or [k for k in recs[0] if k not in ("file", "label", "n_points", "n_anom_points", "fitness")]
        y = np.array([r["label"] for r in recs])
        if len(set(y)) < 2:
            continue
        rows[cls] = {k: roc_auc_score(y, [r[k] for r in recs]) for k in rules}
    return rules, rows


def main(paths):
    files = [f for p in paths for f in sorted(glob.glob(p))]
    if not files:
        sys.exit("no *_scores.json files given")
    for f in files:
        rules, rows = table(f)
        print(f"\n## {os.path.basename(f)}   (O-AUROC x100 per rule)")
        head = f"{'category':>10s} " + " ".join(f"{r:>7s}" for r in rules)
        print(head)
        for cls, r in rows.items():
            print(f"{cls:>10s} " + " ".join(f"{100 * r[k]:7.1f}" for k in rules))
        means = {k: 100 * np.mean([r[k] for r in rows.values()]) for k in rules}
        print(f"{'MEAN':>10s} " + " ".join(f"{means[k]:7.1f}" for k in rules))


if __name__ == "__main__":
    main(sys.argv[1:])
