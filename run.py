#!/usr/bin/env python
"""CLI runner for the ad3d benchmark.

Examples
--------
# quick smoke test (2 categories, small settings - runs in minutes)
python run.py --dataset shapenet --data-root data/anomaly-shapenet/pcd \
    --classes ashtray0,bowl0 --num-group 512 --group-size 64 --tag smoke

# full Anomaly-ShapeNet benchmark (Simple3D-like settings)
python run.py --dataset shapenet --data-root data/anomaly-shapenet/pcd --tag baseline

# full Real3D-AD benchmark
python run.py --dataset real3d --data-root data/real3d-ad --tag baseline
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ad3d import get_classes  # noqa: E402
from ad3d.method import Config, run_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ad3d benchmark runner")
    p.add_argument("--dataset", required=True, choices=["shapenet", "real3d"])
    p.add_argument("--data-root", required=True,
                   help="shapenet: path to the `pcd` dir | real3d: dir with 12 category folders")
    p.add_argument("--classes", default=None,
                   help="comma-separated subset of categories (default: all)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--tag", default="run", help="experiment tag (file prefix)")

    # --- method hyperparameters (Simple3D defaults) ---
    p.add_argument("--max-nn", type=int, default=100)
    p.add_argument("--n-scales", type=int, default=3)
    p.add_argument("--num-group", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--smooth-k", type=int, default=12)
    p.add_argument("--coreset", type=float, default=0.1)
    p.add_argument("--cuts", type=int, default=0,
                   help="augment memory with N simulated single-view cuts per prototype")
    p.add_argument("--align", default="none", choices=["none", "icp"],
                   help="RANSAC+ICP registration of test clouds to the prototypes")
    p.add_argument("--align-voxel", type=float, default=0.05,
                   help="registration voxel; 0 = auto-select per category from "
                        "{0.05, 0.03} via train-side cut registration")
    p.add_argument("--align-poses", type=int, default=1,
                   help="pose hypotheses per test cloud (>1 = multi-hypothesis "
                        "registration for symmetric objects)")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--points-budget", type=int, default=100_000)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_root=args.data_root,
        dataset=args.dataset,
        max_nn=args.max_nn,
        n_scales=args.n_scales,
        num_group=args.num_group,
        group_size=args.group_size,
        smooth_k=args.smooth_k,
        coreset_ratio=args.coreset,
        cuts=args.cuts,
        align=args.align,
        align_voxel=args.align_voxel,
        align_poses=args.align_poses,
        voxel=args.voxel,
        points_budget=args.points_budget,
        topk=args.topk,
        device=args.device,
        seed=args.seed,
    )
    classes = args.classes.split(",") if args.classes else get_classes(args.dataset)

    # resolve 'auto' now so the log shows the real device being used
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device
    print(f"[ad3d] dataset={cfg.dataset}  device={dev}"
          f"{'' if dev != 'cuda' else ' (' + torch.cuda.get_device_name(0) + ')'}"
          f"  classes={len(classes)}")
    print(f"[ad3d] config: {cfg}")

    rows = run_benchmark(cfg, classes=classes)

    os.makedirs(args.out, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.out, f"{args.tag}_{cfg.dataset}_{stamp}")

    import pandas as pd
    df = pd.DataFrame(rows).T
    df.to_csv(base + ".csv")
    with open(base + ".json", "w") as f:
        json.dump({"config": cfg.__dict__, "results": rows}, f, indent=2)
    print(f"\n[ad3d] saved: {base}.csv / .json")


if __name__ == "__main__":
    main()
