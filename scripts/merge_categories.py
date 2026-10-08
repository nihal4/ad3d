#!/usr/bin/env python
"""Merge per-category CSVs from split runs into one full-benchmark CSV.

  python scripts/merge_categories.py <out_tag> <a.csv> <b.csv> [...] [--out results]

Drops each file's __mean__ row, concatenates the categories, recomputes the mean
row exactly like run_benchmark (mean of numeric columns), and writes
<out>/<out_tag>_<dataset>_merged-<stamp>.csv  (so aggregate_seeds.py picks it up
when out_tag looks like <config>-s<seed>). Refuses duplicates / incomplete sets.
"""
import argparse, datetime as dt, os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ad3d.datasets import get_classes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="real3d", choices=["real3d", "shapenet"])
ap.add_argument("out_tag"); ap.add_argument("files", nargs="+")
ap.add_argument("--out", default="results")
ap.add_argument("--allow-partial", action="store_true")
a = ap.parse_args()
ALL = get_classes(a.dataset)

parts = [pd.read_csv(f, index_col=0) for f in a.files]
df = pd.concat([p[p.index != "__mean__"] for p in parts])
dup = df.index[df.index.duplicated()].tolist()
if dup:
    raise SystemExit(f"duplicate categories across files: {dup}")
missing = [c for c in ALL if c not in df.index]
if missing and not a.allow_partial:
    raise SystemExit(f"missing categories: {missing} (use --allow-partial to override)")
df = df.loc[[c for c in ALL if c in df.index]]
df.loc["__mean__"] = df.mean(numeric_only=True)
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"{a.out_tag}_{a.dataset}_merged-{dt.datetime.now():%Y%m%d-%H%M%S}.csv")
df.to_csv(path)
print(f"merged {len(df)-1} categories -> {path}")
print(df.loc["__mean__", ["o_auroc", "p_auroc", "p_aupr", "p_aupr_pooled"]].round(4).to_dict())
