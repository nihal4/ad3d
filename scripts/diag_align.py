#!/usr/bin/env python
"""Diagnostic (no tuning): is location-aware memory failing on some Anomaly-ShapeNet categories because of
SYSTEMATIC POSE ERRORS (a) or because of REAL NORMAL VARIATION in where shapes sit (b)?

For each category: fit the registration exactly like the frozen method (same seed, MHR 6 poses), register every
NORMAL test cloud like evaluate() does, then measure the alignment at FULL resolution:
  res_*      nearest-neighbour distance test -> merged prototypes (median / p90 / p99), unit-normalised frame
  rev_p90    prototypes -> test (catches parts that are present in training but misplaced in the test cloud)
  ref_*      a fine point-to-plane ICP started from the chosen pose: residual afterwards and how far it moved
             (rotation deg, translation). A big drop + big move = the chosen pose was off (explanation a).
  loo_*      training-only reference: each prototype vs the other three (how well NORMAL objects align at all).
Output: one CSV row per normal test sample. Join with the loc/glob scores of step 13 by file name.

  python scripts/diag_align.py --data-root <SHAPENET_ROOT> --classes cap3,vase0 --out results/diag_align.csv
"""
import argparse, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ad3d.method import Config, Simple3DLite, _transform  # noqa: E402
from ad3d import datasets as D  # noqa: E402


def sub(p, n, seed=0):
    return p if len(p) <= n else p[np.random.default_rng(seed).choice(len(p), n, replace=False)]


def nn(a, tree):
    return tree.query(a, k=1, workers=-1)[0]


def fine_icp(src, dst, voxel=0.01, max_dist=0.03):
    s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src)).voxel_down_sample(voxel)
    t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst)).voxel_down_sample(voxel)
    t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    r = o3d.pipelines.registration.registration_icp(
        s, t, max_dist, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    T = np.asarray(r.transformation)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    return T, ang, float(np.linalg.norm(T[:3, 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="also anomalous samples (default: normal only)")
    ap.add_argument("--out", default="results/diag_align.csv")
    a = ap.parse_args()
    import pandas as pd
    rows = []
    for cls in a.classes.split(","):
        t0 = time.time()
        # frozen registration settings; memory extras off (they do not touch registration)
        cfg = Config(data_root=a.data_root, dataset="shapenet", max_nn=10, align="icp", align_poses=6,
                     cuts=0, local_mem=0.0, obj_rule="p99", seed=a.seed)
        m = Simple3DLite(cfg)
        m.fit(cls)
        P = [np.asarray(p, dtype=np.float64) for p in m._proto_frames]
        merged = np.concatenate(P)
        tree = cKDTree(merged)
        loo = []
        for i in range(len(P)):
            others = cKDTree(np.concatenate([P[j] for j in range(len(P)) if j != i]))
            d = nn(sub(P[i], 20000, i), others)
            loo.append((np.median(d), np.percentile(d, 90)))
        loo = np.array(loo)
        test = D.load_test(a.data_root, "shapenet", cls)
        o3d.utility.random.seed(cfg.seed + 7)          # same RNG path as evaluate()
        for s in test:
            p_n = (np.asarray(s.points, dtype=np.float64) - m._ref_center) / m._ref_scale
            cands = m._reg_target.align_candidates(p_n, poses=cfg.align_poses, seed=cfg.seed)
            if len(cands) > 1:
                idx, (T, fit) = m._reg_target.select_pose(p_n, cands)
            else:
                idx, (T, fit) = 0, cands[0]
            if s.label == 1 and not a.all:
                continue
            q = sub(_transform(p_n, T), 20000, 1)
            d = nn(q, tree)
            rev = nn(sub(merged, 20000, 2), cKDTree(q))
            Tr, ang, tr = fine_icp(q, merged)
            d2 = nn(_transform(q, Tr), tree)
            rows.append({"cls": cls, "file": os.path.basename(s.path), "label": int(s.label), "fitness": fit,
                         "pose_idx": idx, "res_med": np.median(d), "res_p90": np.percentile(d, 90),
                         "res_p99": np.percentile(d, 99), "rev_p90": np.percentile(rev, 90),
                         "ref_res_med": np.median(d2), "ref_res_p90": np.percentile(d2, 90),
                         "ref_rot_deg": ang, "ref_trans": tr,
                         "loo_med": loo[:, 0].mean(), "loo_p90": loo[:, 1].mean()})
        print(f"[diag] {cls}: {sum(r['cls'] == cls for r in rows)} samples, {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
