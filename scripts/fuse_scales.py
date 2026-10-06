#!/usr/bin/env python
"""Multi-scale score fusion, computed OFFLINE from runs of the SAME config at different descriptor scales.

For each category, every run gives (a) per-sample object scores (from <run>_scores.json) and (b) a TRAINING-only
reference of normal feature distances (train_ref_* columns in <run>.json, from the memory-bank coreset step).
Each run's object score is normalised with its own training reference, then the normalised scores are averaged
over scales with EQUAL weights. No test labels, no per-category choices.

    python scripts/fuse_scales.py --rule p99 --norm z  RUN_nn10.json RUN_nn40.json RUN_nn100.json

--norm: z     (s - train_ref_mean) / train_ref_std          [pre-registered primary]
        ratio s / train_ref_p99
        none  raw scores (shows why normalisation is needed)
Prints O-AUROC per category for every single scale, every subset of >= 2 scales, and the fused result.
"""
import argparse
import itertools
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def load(run_json):
    base = run_json[:-5]
    res = json.load(open(run_json))
    rows = res["results"] if "results" in res else res
    sc = json.load(open(base + "_scores.json"))
    tag = os.path.basename(base).split("_")[0]
    return tag, rows, sc


def norm(scores, row, how):
    s = np.asarray(scores, dtype=np.float64)
    if how == "none":
        return s
    if how == "z":
        return (s - row["train_ref_mean"]) / max(row["train_ref_std"], 1e-12)
    if how == "ratio":
        return s / max(row["train_ref_p99"], 1e-12)
    raise ValueError(how)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run .json files (the matching _scores.json must sit next to them)")
    ap.add_argument("--rule", default="p99")
    ap.add_argument("--norm", default="z", choices=["z", "ratio", "none"])
    a = ap.parse_args()
    runs = [load(r) for r in a.runs]
    tags = [t for t, _, _ in runs]
    cats = [c for c in runs[0][2] if all(c in r[2] for r in runs)]
    combos = [c for k in range(1, len(runs) + 1) for c in itertools.combinations(range(len(runs)), k)]
    table = {}
    for cat in cats:
        recs0 = runs[0][2][cat]
        files = [r["file"] for r in recs0]
        y = np.array([r["label"] for r in recs0])
        per = []
        for t, rows, sc in runs:
            recs = {r["file"]: r for r in sc[cat]}
            assert set(recs) == set(files), f"sample mismatch in {cat} for {t}"
            if a.norm != "none" and "train_ref_mean" not in rows[cat]:
                raise SystemExit(f"{t}: no train_ref_* stats (run made before they were saved) - use --norm none")
            per.append(norm([recs[f][a.rule] for f in files], rows[cat], a.norm))
        table[cat] = {c: roc_auc_score(y, np.mean([per[i] for i in c], axis=0)) for c in combos}
    names = {c: "+".join(tags[i].replace("-pluscut-p99", "").replace("mhr6-", "").rsplit("-s", 1)[0] for i in c) for c in combos}
    w = max(10, max(len(n) for n in names.values()))
    print(f"rule={a.rule}  norm={a.norm}  (O-AUROC x100)")
    print(f"{'category':>10s} " + " ".join(f"{names[c]:>{w}s}" for c in combos))
    for cat in cats:
        print(f"{cat:>10s} " + " ".join(f"{100 * table[cat][c]:>{w}.1f}" for c in combos))
    print(f"{'MEAN':>10s} " + " ".join(f"{100 * np.mean([table[k][c] for k in cats]):>{w}.1f}" for c in combos))


if __name__ == "__main__":
    main()
