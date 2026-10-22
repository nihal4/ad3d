"""Synthetic surface defects for the NORMAL-ONLY development protocol (no real anomaly is ever used).

A defect is a smooth local displacement of the surface along its normal (bulge outward or dent inward), the two defect
types that dominate Real3D-AD and are present in Anomaly-ShapeNet and MVTec 3D-AD. The generator is generic and is
NOT tuned to any benchmark: one centre point drawn uniformly from the cloud, radius r drawn uniformly from
[0.04, 0.12] x the object radius, amplitude drawn uniformly from [0.25, 0.6] x r, profile (1 - (d/r)^2)^2. Points
displaced by more than 1 % of r are labelled anomalous.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def outward_normals(pts: np.ndarray, k: int = 20) -> np.ndarray:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=k))
    n = np.asarray(pc.normals)
    out = pts - pts.mean(axis=0)
    flip = (n * out).sum(axis=1) < 0
    n[flip] *= -1
    return n.astype(np.float32)


def inject_defect(pts: np.ndarray, rng: np.random.Generator, scale: float | None = None,
                  kind: str | None = None, r_range=(0.04, 0.12), amp_range=(0.25, 0.6)):
    """Return (points, gt, info) with one synthetic bulge or dent. `scale` = object radius (default: from pts)."""
    pts = np.asarray(pts, dtype=np.float32)
    if scale is None:
        scale = float(np.linalg.norm(pts - pts.mean(0), axis=1).max()) or 1.0
    kind = kind or ("bulge" if rng.random() < 0.5 else "dent")
    c = pts[rng.integers(len(pts))]
    r = float(rng.uniform(*r_range)) * scale
    amp = float(rng.uniform(*amp_range)) * r * (1.0 if kind == "bulge" else -1.0)
    d = np.linalg.norm(pts - c, axis=1)
    w = np.clip(1.0 - (d / r) ** 2, 0.0, None) ** 2
    sel = w > 0
    nrm = np.zeros_like(pts)
    if sel.sum() >= 3:
        nrm_all = outward_normals(pts)
        nrm[sel] = nrm_all[sel]
    disp = (amp * w)[:, None] * nrm
    out = pts + disp
    gt = (np.linalg.norm(disp, axis=1) > 0.01 * r).astype(np.float32)
    return out.astype(np.float32), gt, {"kind": kind, "radius": r / scale, "amplitude": abs(amp) / scale,
                                        "frac": float(gt.mean())}


def directional_view(pts: np.ndarray, u: np.ndarray, tau: float = 0.15) -> np.ndarray:
    """Points of a (centred) cloud whose direction from the centroid faces u - star-shaped single-view model."""
    c = pts - pts.mean(axis=0)
    dirs = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
    return dirs @ u > tau
