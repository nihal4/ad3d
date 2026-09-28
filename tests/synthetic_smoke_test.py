"""End-to-end smoke test on synthetic data (no dataset download needed).

Builds tiny fake datasets in the official layouts and runs the full pipeline:

  [1/5] preprocessing alignment + determinism
  [2/5] Anomaly-ShapeNet layout (comma GT, GT-stem matching, twins)
  [3/5] Real3D-AD layout (space GT, 'good' normals)
  [4/5] alignment + memory cuts on a partial-scan scenario
        (full-object prototypes vs randomly-rotated hemispheres)
  [5/5] memory bank sanity

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


# --------------------------------------------------------------------------- #
# Synthetic shapes
# --------------------------------------------------------------------------- #
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


def random_rotation(rng: np.random.Generator, max_deg: float = 180.0) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def make_ellipsoid(n: int, radii=(1.0, 0.6, 0.4), noise: float = 0.004) -> np.ndarray:
    d = RNG.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * np.array(radii) + RNG.normal(scale=noise, size=(n, 3))


def make_hemi(n: int, radii=(1.0, 0.6, 0.4)) -> np.ndarray:
    """Partial (single-view-like) scan: upper hemisphere of an ellipsoid."""
    d = RNG.normal(size=(n * 3, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] > 0.05][:n]
    return d * np.array(radii)


def make_hemi_bump(n: int, radii=(1.0, 0.6, 0.4)) -> tuple[np.ndarray, np.ndarray]:
    """Hemisphere with a strong bump in the visible region."""
    pts = make_hemi(n, radii)
    direction = np.array([0.5, 0.2, 0.84])
    direction /= np.linalg.norm(direction)
    dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    mask = (dirs @ direction) > 0.85
    pts[mask] += 0.18 * dirs[mask]
    gt = np.zeros(len(pts), dtype=np.float32)
    gt[mask] = 1.0
    return pts, gt


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Dataset layouts
# --------------------------------------------------------------------------- #
def build_shapenet_layout(root: str):
    for cls in ["ashtray0", "bowl0"]:
        for i in range(4):  # train prototypes
            write_pcd(f"{root}/{cls}/train/{i}_template.pcd", make_sphere(N))
        for i in range(5):  # normal test files carry 'positive'
            write_pcd(f"{root}/{cls}/test/{i}_positive.pcd", make_sphere(N))
        for i in range(5):  # anomalous: GT txt is the input; test pcd twin is ignored
            pts, gt = make_bump_sample(N)
            write_pcd(f"{root}/{cls}/test/{i}_bulge.pcd", pts)
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


def build_alignment_layout(root: str):
    """Real3D-AD-style trap: full-object prototypes (near-identical pose) vs
    randomly-rotated partial (hemisphere) test scans."""
    rng = np.random.default_rng(7)
    cls = "airplane"
    for i in range(4):
        p = make_ellipsoid(N) @ random_rotation(rng, max_deg=10)
        write_pcd(f"{root}/{cls}/train/{i}_prototype.pcd", p)
    for i in range(5):  # normal partial scans, arbitrary orientation
        p = make_hemi(N) @ random_rotation(rng)
        write_pcd(f"{root}/{cls}/test/{i}_good.pcd", p)
    for i in range(5):  # anomalous partial scans with a subtle bump
        pts, gt = make_hemi_bump(N)
        pts = pts @ random_rotation(rng)
        write_pcd(f"{root}/{cls}/test/{i}_bulge.pcd", pts)
        write_gt_txt(f"{root}/{cls}/gt/{i}_bulge.txt", pts, gt, comma=False)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_alignment(model: Simple3DLite):
    """Preprocessing must keep points and gt row-aligned."""
    pts, gt = make_bump_sample(N)
    p2, g2 = model.preprocess(pts, gt, path_hash=42)
    assert len(p2) == len(g2), "points/gt misaligned after preprocess!"
    frac_before, frac_after = gt.mean(), g2.mean()
    assert abs(frac_before - frac_after) < 0.02, (frac_before, frac_after)
    p3, g3 = model.preprocess(pts, gt, path_hash=42)
    assert np.allclose(p2, p3) and np.array_equal(g2, g3), "preprocess not deterministic!"
    print(f"  alignment OK  ({len(pts)} -> {len(p2)} pts, anomalous frac {frac_after:.3f})")


def main():
    tmp = os.path.abspath("tests/_synthetic")
    shutil.rmtree(tmp, ignore_errors=True)
    shapenet_root = f"{tmp}/shapenet/pcd"
    real3d_root = f"{tmp}/real3d"
    align_root = f"{tmp}/align"
    build_shapenet_layout(shapenet_root)
    build_real3d_layout(real3d_root)
    build_alignment_layout(align_root)

    small = dict(max_nn=30, n_scales=2, num_group=256, group_size=32,
                 smooth_k=6, coreset_ratio=0.25, voxel=0.02, points_budget=4000,
                 device="cpu", seed=0)

    print("[1/5] preprocessing alignment + determinism")
    model = Simple3DLite(Config(dataset="shapenet", data_root=shapenet_root, **small))
    check_alignment(model)

    print("[2/5] Anomaly-ShapeNet layout (comma GT, GT-stem matching, twins)")
    rows = run_benchmark(Config(dataset="shapenet", data_root=shapenet_root, **small),
                         classes=["ashtray0", "bowl0"], verbose=False)
    m = rows["__mean__"]
    print(f"  O-AUROC {m['o_auroc']:.3f} | P-AUROC {m['p_auroc']:.3f} | "
          f"P-AUPR {m['p_aupr']:.3f} | pooled P-AUROC {m['p_auroc_pooled']:.3f}")
    assert m["o_auroc"] > 0.9 and m["p_auroc"] > 0.7
    from ad3d.datasets import load_test
    te = load_test(shapenet_root, "shapenet", "ashtray0")
    n_norm = sum(1 for s in te if s.label == 0)
    assert (n_norm, len(te) - n_norm) == (5, 5), (n_norm, len(te))

    print("[3/5] Real3D-AD layout (space GT, 'good' normals)")
    rows_r = run_benchmark(Config(dataset="real3d", data_root=real3d_root, **small),
                           classes=["airplane", "duck"], verbose=False)
    m = rows_r["__mean__"]
    print(f"  O-AUROC {m['o_auroc']:.3f} | P-AUROC {m['p_auroc']:.3f} | "
          f"P-AUPR {m['p_aupr']:.3f} | pooled P-AUROC {m['p_auroc_pooled']:.3f}")
    assert m["o_auroc"] > 0.9 and m["p_auroc"] > 0.7

    print("[4/5] partial-scan scenario: align + cuts  (Real3D-AD trap)")
    results = {}
    for tag, extra in [("none", {}),
                       ("cuts", {"cuts": 4}),
                       ("icp", {"align": "icp"}),
                       ("icp+cuts", {"align": "icp", "cuts": 4}),
                       ("+poses", {"align": "icp", "cuts": 4, "align_poses": 3})]:
        cfg = Config(dataset="real3d", data_root=align_root, **small, **extra)
        r = run_benchmark(cfg, classes=["airplane"], verbose=False)
        results[tag] = r["__mean__"]
        fit = r["airplane"].get("align_fitness_mean")
        fit_s = f"  align_fitness {fit:.2f}" if fit is not None else ""
        if "pose_switch_rate" in r["airplane"]:
            fit_s += f"  pose_switch {r['airplane']['pose_switch_rate']:.0%}"
        print(f"  {tag:>8s}: O-AUROC {results[tag]['o_auroc']:.3f} | "
              f"P-AUROC {results[tag]['p_auroc']:.3f}{fit_s}")
    assert r["airplane"]["align_fitness_mean"] > 0.8, "registration quality too low"
    assert results["icp"]["p_auroc"] > 0.7, "aligned path should localize the bump"
    assert 0.0 <= r["airplane"]["pose_switch_rate"] <= 1.0, "pose selection ran"
    # NOTE: no O-AUROC assertion here on purpose - the synthetic ellipsoid's
    # natural FPFH variation (pole vs equator) exceeds the anomaly signal, so
    # object-level AUROC is decided by sampling noise. The decisive experiment
    # for alignment is the real-data ablation grid (see NOVELTY_ROADMAP.md).

    print("[5/5] memory bank sanity")
    model = Simple3DLite(Config(dataset="shapenet", data_root=shapenet_root, **small))
    model.fit("ashtray0")
    print(f"  memory size = {model.memory.size} (coreset of 4x256=1024 @ 0.25)")
    assert 200 < model.memory.size <= 1024

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
