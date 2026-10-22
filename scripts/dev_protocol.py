#!/usr/bin/env python
"""Normal-only development protocol (revision item 3.1): select settings WITHOUT any test label.

For every category, a normal object the method has never seen is scored against a memory built from the remaining
normal objects, once as it is and once with a synthetic defect (synth.py). The resulting development AUROC measures
how well a setting separates normal from defective surfaces using training data only.

  real3d   : leave-one-prototype-out (4 folds). Pseudo-test normals = the real single-view scans of the held-out
             prototype (GLFM training scans; assigned to their prototype by geometry); if fewer than --n-pseudo exist,
             star-shaped views of the held-out prototype fill up the set. Memory = the other 3 prototypes, their
             simulated cuts and their real scans; the frame is built from the other prototypes only.
  shapenet : leave-one-prototype-out. Pseudo-test normals = the complete held-out prototype under random rigid poses
             (with a random 90 % point subsample each), matching the complete test clouds of that benchmark.
  mvtec    : memory = the 4 prototypes; pseudo-test normals = further training goods (#5 onwards), never prototypes.
Pseudo-test anomalies = the same pseudo-normals with one synthetic bulge or dent each. Every pseudo-test is
registered exactly like a real test scan (multi-hypothesis registration from a random pose).

All rules and memory subsets are evaluated in the same pass (--channels rev); the descriptor scale k needs one run
per value. Output: <out>/dev_<tag>_<dataset>.json with per-category/per-fold metrics and per-sample records.

  python scripts/dev_protocol.py --dataset real3d --data-root ... --train-cut-root ... --align icp --align-poses 6 \
      --cuts 4 --max-nn 10 --local-mem 0.10 --train-cut-with-protos --channels rev --tag dev-nn10
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from run import build_parser, config_from_args  # noqa: E402
from ad3d import datasets as D  # noqa: E402
from ad3d.method import Simple3DLite  # noqa: E402
from ad3d.synth import directional_view, inject_defect, random_rotation  # noqa: E402


def pseudo_tests(base_clouds: list[np.ndarray], rng: np.random.Generator, n: int, mode: str):
    """-> list of TestSample (n normals + n defective copies)."""
    out = []
    srcs = []
    for i in range(n):
        c = np.asarray(base_clouds[i % len(base_clouds)], np.float32)
        if mode == "pose":                              # complete object, random rigid pose, 90 % subsample
            keep = rng.random(len(c)) < 0.9
            c = c[keep]
            c = (c - c.mean(0)) @ random_rotation(rng).T + c.mean(0) + rng.normal(scale=0.05, size=3) * np.ptp(c, 0).max()
        elif mode == "view":                            # star-shaped single view of a complete object, random pose
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            c = c[directional_view(c, u, 0.15)]
            c = (c - c.mean(0)) @ random_rotation(rng).T + c.mean(0)
        srcs.append(c.astype(np.float32))
    for i, c in enumerate(srcs):
        out.append(D.TestSample(c, np.zeros(len(c), np.float32), 0, f"pseudo_normal_{i:02d}"))
    for i, c in enumerate(srcs):
        a, gt, info = inject_defect(c, rng)
        out.append(D.TestSample(a, gt, 1, f"pseudo_defect_{i:02d}_{info['kind']}_r{info['radius']:.3f}"))
    return out


def main():
    p = build_parser()
    p.add_argument("--n-pseudo", type=int, default=6, help="pseudo-normals per fold (same number of defective copies)")
    p.add_argument("--folds", type=int, default=4, help="held-out prototypes per category (real3d/shapenet)")
    args = p.parse_args()
    cfg = config_from_args(args)
    if cfg.channels != "rev":
        raise SystemExit("run the development protocol with --channels rev")
    classes = args.classes.split(",") if args.classes else D.get_classes(cfg.dataset)
    os.makedirs(args.out, exist_ok=True)
    result = {"config": cfg.__dict__, "n_pseudo": args.n_pseudo, "folds": args.folds, "rows": [], "records": {}}
    t_start = time.time()
    for cls in classes:
        h_cls = int(hashlib.md5(cls.encode()).hexdigest()[:8], 16)
        folds = [-1] if cfg.dataset == "mvtec" else list(range(args.folds))
        for h in folds:
            rng = np.random.default_rng((cfg.seed * 7919 + h_cls + 101 * (h + 1)) & 0xFFFFFFFF)
            cfg_f = type(cfg)(**{**cfg.__dict__, "holdout": h})
            model = Simple3DLite(cfg_f)
            model.fit(cls)
            if cfg.dataset == "mvtec":
                extra = D.load_train_extra(cfg.data_root, "mvtec", cls, start=D.N_PROTOS_MVTEC, n=args.n_pseudo)
                samples = pseudo_tests(extra, rng, min(args.n_pseudo, len(extra)), mode="as_is")
                src = "train goods"
            elif cfg.dataset == "shapenet":
                samples = pseudo_tests([model._holdout_raw], rng, args.n_pseudo, mode="pose")
                src = "held-out prototype, random poses"
            else:
                real = [c for _, c in getattr(model, "_holdout_cuts", [])]
                n_real = min(len(real), args.n_pseudo)
                samples = pseudo_tests(real[:n_real], rng, n_real, mode="as_is") if n_real else []
                if n_real < args.n_pseudo:
                    samples += pseudo_tests([model._holdout_raw], rng, args.n_pseudo - n_real, mode="view")
                src = f"{n_real} real scans + {args.n_pseudo - n_real} star-shaped views"
            m = model.evaluate(cls, samples=samples)
            m.update({"category": cls, "fold": h, "pseudo_source": src})
            result["rows"].append(m)
            result["records"][f"{cls}|{h}"] = model.sample_records
            print(f"[dev] {cls:>12s} fold {h:+d}  loc10 {m.get('loc10_o_auroc', float('nan')):.3f}  "
                  f"glob_full {m.get('glob_full_o_auroc', float('nan')):.3f}  cal10 {m.get('cal10_o_auroc', float('nan')):.3f}"
                  f"  ({src}; {time.time() - t_start:.0f}s)", flush=True)
    path = os.path.join(args.out, f"dev_{args.tag}_{cfg.dataset}.json")
    with open(path, "w") as fh:
        json.dump(result, fh)
    print(f"[dev] saved {path}")


if __name__ == "__main__":
    main()
