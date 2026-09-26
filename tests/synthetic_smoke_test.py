"""End-to-end smoke test on synthetic data (no dataset download needed).

Builds tiny fake datasets in BOTH official layouts (Anomaly-ShapeNet and
Real3D-AD) with an obvious 'bump' anomaly, then runs the full pipeline.

Run:  python tests/synthetic_smoke_test.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ad3d.method import Config, Simple3DLite, run_benchmark  # noqa: E402

RNG = np.random.default_rng(0)
N = 4000  # points per cloud


def make_sphere(n: int, noise: float = 0.005) -> np.ndarray:
    d = RNG.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d + RNG.normal(scale=noise, size=(n, 3))


def make_bump_sample(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sphere with a local outward bump; gt marks the displaced points."""
    pts = make_sphere(n)
    direction = np.array([1.0, 0.3, 0.2])
    direction /= np.linalg.norm(direction)
    cos = pts @ direction
    mask = cos > 0.93  # small spherical cap
    pts[mask] += 0.12 * direction  # displace the cap outward
    gt = np.zeros(n, dtype=np.float32)
    gt[mask] = 1.0
    return pts.astype(np.float32), gt


def write_pcd(path: str, pts: np.ndarray):
    import open3d as o3d
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    o3d.io.write_point_cloud(path, pcd)


def write_gt_txt(path: str, pts: np.ndarray, gt: np.ndarray, comma: bool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.concatenate([pts, gt[:, None]], axis=1)
    delim = "," if comma else " "
    np.savetxt(path, arr, fmt="%.6f", delimiter=delim)


def build_shapenet_layout(root: str):
    for cls in ["ashtray0", "bowl0"]:
        for i in range(4):  # train prototypes
            write_pcd(f"{root}/{cls}/train/{i}_template.pcd", make_sphere(N))
        for i in range(5):  # normal test files carry 'positive'
            write_pcd(f"{root}/{cls}/test/{i}_positive.pcd", make_sphere(N))
        for i in range(5):  # anomalous: only the GT txt matters
            pts, gt = make_bump_sample(N)
            write_gt_txt(f"{root}/{cls}/GT/{i}_bulge.txt", pts, gt, comma=True)


def build_real3d_layout(root: str):
    for cls in ["airplane", "duck"]:
        for i in range(4):
            write_pcd(f"{root}/{cls}/train/{i}_prototype.pcd", make_sphere(N))
        for i in range(5):
            write_pcd(f"{root}/{cls}/test/{i}_good.pcd", make_sphere(N))
        for i in range(5):
            pts, gt = make_bump_sample(N)
            write_pcd(f"{root}/{cls}/test/{i}_bulge.pcd", pts)  # unused but realistic
            write_gt_txt(f"{root}/{cls}/gt/{i}_bulge.txt", pts, gt, comma=False)


def check_alignment(model: Simple3DLite):
    """Preprocessing must keep points and gt row-aligned."""
    pts, gt = make_bump_sample(N)
    p2, g2 = model.preprocess(pts, gt, path_hash=42)
    assert len(p2) == len(g2), "points/gt misaligned after preprocess!"
    # bump points must still be labeled: fraction of anomalous points preserved
    frac_before, frac_after = gt.mean(), g2.mean()
    assert abs(frac_before - frac_after) < 0.02, (frac_before, frac_after)
    # determinism
    p3, g3 = model.preprocess(pts, gt, path_hash=42)
    assert np.allclose(p2, p3) and np.array_equal(g2, g3), "preprocess not deterministic!"
    print(f"  alignment OK  ({len(pts)} -> {len(p2)} pts, anomalous frac {frac_after:.3f})")


def main():
    tmp = os.path.abspath("tests/_synthetic")
    shutil.rmtree(tmp, ignore_errors=True)
    shapenet_root = f"{tmp}/shapenet/pcd"
    real3d_root = f"{tmp}/real3d"
    build_shapenet_layout(shapenet_root)
    build_real3d_layout(real3d_root)

    cfg = Config(dataset="shapenet", data_root=shapenet_root,
                 max_nn=30, n_scales=2, num_group=256, group_size=32,
                 smooth_k=6, coreset_ratio=0.25, voxel=0.02, points_budget=4000,
                 device="cpu", seed=0)
    model = Simple3DLite(cfg)
    print("[1/4] preprocessing alignment + determinism")
    check_alignment(model)

    print("[2/4] Anomaly-ShapeNet layout (comma GT, GT-stem matching, twins)")
    rows = run_benchmark(cfg, classes=["ashtray0", "bowl0"], verbose=False)
    m = rows["__mean__"]
    print(f"  O-AUROC {m['o_auroc']:.3f} | P-AUROC {m['p_auroc']:.3f} | "
          f"P-AUPR {m['p_aupr']:.3f} | pooled P-AUROC {m['p_auroc_pooled']:.3f}")
    assert m["o_auroc"] > 0.9, "object detection failed on obvious synthetic anomalies"
    assert m["p_auroc"] > 0.7, "localization failed on obvious synthetic anomalies"
    # loader must load 5 normal + 5 anomalous (10 test pcds incl. twins)
    from ad3d.datasets import load_test
    te = load_test(shapenet_root, "shapenet", "ashtray0")
    n_norm = sum(1 for s in te if s.label == 0)
    assert (n_norm, len(te) - n_norm) == (5, 5), (n_norm, len(te))

    print("[3/4] Real3D-AD layout (space GT, 'good' normals)")
    cfg_r = Config(dataset="real3d", data_root=real3d_root,
                   max_nn=30, n_scales=2, num_group=256, group_size=32,
                   smooth_k=6, coreset_ratio=0.25, voxel=0.02, points_budget=4000,
                   device="cpu", seed=0)
    rows_r = run_benchmark(cfg_r, classes=["airplane", "duck"], verbose=False)
    m = rows_r["__mean__"]
    print(f"  O-AUROC {m['o_auroc']:.3f} | P-AUROC {m['p_auroc']:.3f} | "
          f"P-AUPR {m['p_aupr']:.3f} | pooled P-AUROC {m['p_auroc_pooled']:.3f}")
    assert m["o_auroc"] > 0.9 and m["p_auroc"] > 0.7

    print("[4/4] memory bank sanity")
    model.fit("ashtray0")
    print(f"  memory size = {model.memory.size} (coreset of 4x256=1024 @ 0.25)")
    assert 200 < model.memory.size <= 1024

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
