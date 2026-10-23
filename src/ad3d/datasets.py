"""Unified dataset loading for Anomaly-ShapeNet and Real3D-AD.

Expected on-disk layouts (after extraction):

Anomaly-ShapeNet  (root = the `pcd` directory):
    <root>/<category>/train/*.pcd              4 normal prototypes
    <root>/<category>/test/*.pcd               normal + anomalous (see below)
    <root>/<category>/GT/*.txt                 anomalous samples: "x,y,z,flag"
                                               (the txt itself is the input cloud,
                                                mirroring the official benchmark code)
    A test .pcd is anomalous iff GT/<stem>.txt exists (its twin in test/ is
    ignored); every other test .pcd is a normal sample.

Real3D-AD  (root = the directory containing the 12 category folders):
    <root>/<category>/train/*.pcd              4 normal prototypes
    <root>/<category>/test/*.pcd               'good' in name -> normal (label 0)
    <root>/<category>/gt/<stem>.txt            anomalous: "x y z flag" (space separated)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pandas as pd


# --------------------------------------------------------------------------- #
# Class lists (official benchmark order)
# --------------------------------------------------------------------------- #
SHAPENET_CLASSES = [
    "ashtray0", "bag0", "bottle0", "bottle1", "bottle3",
    "bowl0", "bowl1", "bowl2", "bowl3", "bowl4", "bowl5",
    "bucket0", "bucket1",
    "cap0", "cap3", "cap4", "cap5",
    "cup0", "cup1",
    "eraser0",
    "headset0", "headset1",
    "helmet0", "helmet1", "helmet2", "helmet3",
    "jar0",
    "microphone0",
    "shelf0",
    "tap0", "tap1",
    "vase0", "vase1", "vase2", "vase3", "vase4", "vase5", "vase7", "vase8", "vase9",
]

REAL3D_CLASSES = [
    "airplane", "candybar", "car", "chicken", "diamond", "duck",
    "fish", "gemstone", "seahorse", "shell", "starfish", "toffees",
]


MVTEC_CLASSES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
]

N_PROTOS_MVTEC = 4          # few-shot setting: the first 4 (sorted) training goods are the prototypes


def get_classes(dataset: str) -> list[str]:
    if dataset == "shapenet":
        return list(SHAPENET_CLASSES)
    if dataset == "real3d":
        return list(REAL3D_CLASSES)
    if dataset == "mvtec":
        return list(MVTEC_CLASSES)
    raise ValueError(f"Unknown dataset '{dataset}' (use 'shapenet', 'real3d' or 'mvtec')")


# --------------------------------------------------------------------------- #
# Sample container
# --------------------------------------------------------------------------- #
@dataclass
class TestSample:
    points: np.ndarray   # (n, 3) float32
    gt: np.ndarray       # (n,) float32, 1 = anomalous point
    label: int           # 0 = normal object, 1 = anomalous object
    path: str


# --------------------------------------------------------------------------- #
# Low-level readers
# --------------------------------------------------------------------------- #
def read_pcd(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts


def read_txt_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a 4-column ground-truth txt (x y z flag).

    Handles both space and comma delimiters (Anomaly-ShapeNet uses ',',
    Real3D-AD uses spaces).
    """
    with open(path, "r") as f:
        first = f.readline()
    delim = "," if "," in first else None
    arr = np.loadtxt(path, dtype=np.float32, delimiter=delim)
    if arr.ndim == 1:
        arr = arr[None, :]
    points = arr[:, :3].astype(np.float32)
    gt = (arr[:, 3] > 0.5).astype(np.float32) if arr.shape[1] > 3 else np.zeros(len(points), np.float32)
    return points, gt


# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #
def load_train(root: str, dataset: str, cls: str, n_mvtec: int = 0) -> list[np.ndarray]:
    """Return the list of raw training (normal) point clouds for one category.
    MVTec 3D-AD has hundreds of training goods; the first N_PROTOS_MVTEC (sorted) are the prototypes unless n_mvtec > 0
    asks for more (training-set-size study)."""
    if dataset == "mvtec":
        paths = mvtec_train_paths(root, cls)[:(n_mvtec if n_mvtec > 0 else N_PROTOS_MVTEC)]
        return [read_mvtec_xyz(p)[0] for p in paths]
    if dataset == "shapenet":
        paths = sorted(glob.glob(os.path.join(root, cls, "train", "*.pcd")))
    elif dataset == "real3d":
        paths = sorted(glob.glob(os.path.join(root, cls, "train", "*.pcd")))
    else:
        raise ValueError(dataset)
    if not paths:
        raise FileNotFoundError(f"No training .pcd found under {os.path.join(root, cls, 'train')}")
    return [read_pcd(p) for p in paths]


def read_any_cloud(path: str) -> np.ndarray:
    """Read xyz from .pcd/.ply (open3d) or a whitespace/comma text file (.asc/.txt/.xyz)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pcd", ".ply"):
        return read_pcd(path)
    if ext in (".tif", ".tiff"):
        # MVTec-3D style organised point cloud: (H, W, 3) xyz image, zeros = no point
        import tifffile
        a = np.asarray(tifffile.imread(path), dtype=np.float32)
        pts = a.reshape(-1, a.shape[-1])[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1) & np.any(pts != 0, axis=1)]
        if pts.shape[0] == 0:
            raise ValueError(f"Empty point cloud: {path}")
        return pts
    try:
        arr = pd.read_csv(path, sep=r"\s+", header=None, comment="#").to_numpy()
    except Exception:
        arr = pd.read_csv(path, sep=",", header=None, comment="#").to_numpy()
    if arr.ndim == 1 or arr.shape[1] < 3:
        arr = pd.read_csv(path, sep=",", header=None, comment="#").to_numpy()
    pts = np.asarray(arr[:, :3], dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pts


CUT_EXTS = (".asc", ".txt", ".xyz", ".pcd", ".ply", ".tif", ".tiff")


def find_train_cut_paths(cut_root: str, cls: str) -> list[str]:
    """Pre-cut single-view TRAINING clouds. Supported layouts (first match wins):
      <cut_root>/<cls>/train_cut/*.asc            (Simple3D code: real3d_train_cut)
      <cut_root>/<cls>/train/good/xyz/*.tiff      (GLFM 'Cut Training Data' release: Real3D-mvtec, MVTec-3D format)
    plus a recursive search for either folder below cut_root. Only TRAIN data is read from here;
    the test set always comes from the official Real3D-AD release (--data-root)."""
    rel = [os.path.join("train_cut"), os.path.join("train", "good", "xyz")]
    cands = [os.path.join(cut_root, cls, r) for r in rel]
    for r in rel:
        cands += sorted(glob.glob(os.path.join(cut_root, "**", cls, r), recursive=True))
    for d in cands:
        paths = sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(CUT_EXTS))
        if paths:
            return paths
    raise FileNotFoundError(f"No training cut clouds for '{cls}' under {cut_root} "
                            f"(expected <root>/{cls}/train_cut/*.asc or <root>/{cls}/train/good/xyz/*.tiff)")


def load_train_cut(cut_root: str, cls: str) -> list[tuple[str, np.ndarray]]:
    return [(p, read_any_cloud(p)) for p in find_train_cut_paths(cut_root, cls)]


def load_test(root: str, dataset: str, cls: str) -> list[TestSample]:
    """Return all test samples for one category (normal + anomalous)."""
    samples: list[TestSample] = []

    if dataset == "shapenet":
        test_dir = os.path.join(root, cls, "test")
        gt_dir = os.path.join(root, cls, "GT")
        # Normal test files carry 'positive' in the name (official convention).
        normal_paths = sorted(p for p in glob.glob(os.path.join(test_dir, "*.pcd"))
                              if "positive" in os.path.basename(p))
        for p in normal_paths:
            pts = read_pcd(p)
            samples.append(TestSample(pts, np.zeros(len(pts), np.float32), 0, p))
        # Anomalous samples: input cloud AND mask come from GT txt.
        for p in sorted(glob.glob(os.path.join(gt_dir, "*.txt"))):
            pts, gt = read_txt_cloud(p)
            samples.append(TestSample(pts, gt, 1, p))
        if not normal_paths:
            found = [os.path.basename(p) for p in
                     sorted(glob.glob(os.path.join(test_dir, "*.pcd")))[:5]]
            raise FileNotFoundError(
                f"No '*positive*.pcd' test files in {test_dir}. "
                f"First files found: {found} - check the dataset layout "
                f"(normal test samples must contain 'positive' in the filename).")

    elif dataset == "real3d":
        test_dir = os.path.join(root, cls, "test")
        gt_dir = os.path.join(root, cls, "gt")
        test_paths = [s for s in sorted(glob.glob(os.path.join(test_dir, "*.pcd")))
                      if "temp" not in os.path.basename(s)]
        for p in test_paths:
            if "good" in os.path.basename(p):
                pts = read_pcd(p)
                samples.append(TestSample(pts, np.zeros(len(pts), np.float32), 0, p))
            else:
                stem = os.path.splitext(os.path.basename(p))[0]
                txt = os.path.join(gt_dir, stem + ".txt")
                pts, gt = read_txt_cloud(txt)
                samples.append(TestSample(pts, gt, 1, p))
    elif dataset == "mvtec":
        for tdir in sorted(glob.glob(os.path.join(root, cls, "test", "*"))):
            kind = os.path.basename(tdir)
            for p in sorted(glob.glob(os.path.join(tdir, "xyz", "*.tif*"))):
                pts, pix, shape = read_mvtec_xyz(p, return_shape=True)
                if kind == "good":
                    gt = np.zeros(len(pts), np.float32)
                else:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    m = _read_mask(os.path.join(tdir, "gt", stem + ".png"), shape)
                    gt = (m.reshape(-1)[pix] > 0).astype(np.float32)
                samples.append(TestSample(pts, gt, int(kind != "good"), p))
    else:
        raise ValueError(dataset)

    if not samples:
        raise FileNotFoundError(f"No test samples found for {cls} under {root}")
    return samples


# --------------------------------------------------------------------------- #
# MVTec 3D-AD (point clouds only: the organised xyz.tiff, RGB is never read)
# --------------------------------------------------------------------------- #
def mvtec_train_paths(root: str, cls: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(root, cls, "train", "good", "xyz", "*.tif*")))
    if not paths:
        raise FileNotFoundError(f"No MVTec 3D-AD training xyz under {os.path.join(root, cls, 'train', 'good', 'xyz')}")
    return paths


def load_train_extra(root: str, dataset: str, cls: str, start: int, n: int) -> list[np.ndarray]:
    """Further NORMAL training samples beyond the prototypes (MVTec 3D-AD only; used by the normal-only
    development protocol as held-out normals, never as memory)."""
    if dataset != "mvtec":
        raise ValueError("extra training normals exist only for MVTec 3D-AD")
    return [read_mvtec_xyz(p)[0] for p in mvtec_train_paths(root, cls)[start:start + n]]


def _read_mask(path: str, shape) -> np.ndarray:
    from PIL import Image
    m = np.asarray(Image.open(path).convert("L"))
    if m.shape != tuple(shape):
        m = np.asarray(Image.open(path).convert("L").resize((shape[1], shape[0]), Image.NEAREST))
    return m


def read_mvtec_xyz(path: str, border: int = 10, return_shape: bool = False):
    """Organised MVTec 3D-AD point cloud -> foreground points + their flat pixel indices.

    Background removal (data-independent rule, the usual one for this benchmark): fit a plane by RANSAC to the valid
    points of the image border (`border` pixels wide), which show only the support surface, and drop every point
    closer to that plane than max(3 x the border points' median plane residual, 0.25 % of the cloud's extent);
    finally drop small connected components (DBSCAN; < 5 % of the largest) to remove stray points."""
    import tifffile
    a = np.asarray(tifffile.imread(path), dtype=np.float32)
    H, W = a.shape[:2]
    flat = a.reshape(-1, a.shape[-1])[:, :3]
    valid = np.isfinite(flat).all(axis=1) & np.any(flat != 0, axis=1)
    yy, xx = np.divmod(np.arange(H * W), W)
    on_border = (yy < border) | (yy >= H - border) | (xx < border) | (xx >= W - border)
    bpts = flat[valid & on_border]
    pix = np.where(valid)[0]
    pts = flat[pix]
    if len(bpts) >= 100:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bpts.astype(np.float64)))
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
        (a_, b_, c_, d_), _ = pc.segment_plane(distance_threshold=0.002 * extent, ransac_n=3, num_iterations=1000)
        nrm = np.linalg.norm([a_, b_, c_]) or 1.0
        res_b = np.abs(bpts @ np.array([a_, b_, c_]) + d_) / nrm
        thr = max(3.0 * float(np.median(res_b)), 0.0025 * extent)
        dist = np.abs(pts @ np.array([a_, b_, c_]) + d_) / nrm
        keep = dist > thr
        pts, pix = pts[keep], pix[keep]
    if len(pts) > 50:
        from scipy.spatial import cKDTree
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(len(pts), min(20_000, len(pts)), replace=False))
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[sub].astype(np.float64)))
        lab_s = np.asarray(pc.cluster_dbscan(eps=0.03 * extent, min_points=5))
        if (lab_s >= 0).any():
            _, nn = cKDTree(pts[sub]).query(pts, k=1)
            lab = lab_s[nn]
            cnt = np.bincount(lab[lab >= 0])
            keep_lab = np.where(cnt >= 0.05 * cnt.max())[0]      # main object + any sizeable part of it
            keep = np.isin(lab, keep_lab)
            pts, pix = pts[keep], pix[keep]
    if len(pts) == 0:
        raise ValueError(f"Empty point cloud after background removal: {path}")
    pts = pts.astype(np.float32)
    return (pts, pix, (H, W)) if return_shape else (pts, pix)
