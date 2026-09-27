"""Feature extraction: multi-scale FPFH (MSND) + Local Feature Spatial Aggregation (LFSA).

This is the Simple3D recipe, reimplemented cleanly and dependency-light:

  1. MSND : concatenate FPFH descriptors computed at several neighborhood
            scales (max_nn, 2*max_nn, 3*max_nn) -> (n, 33 * n_scales)
  2. FPS  : farthest-point-sample `num_group` center points
  3. LFSA : each center's feature = mean FPFH of its `group_size` neighbors

Extension point #1 for novelty: add new descriptors / learned features here.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import torch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))


def _resolve_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# MSND: multi-scale FPFH
# --------------------------------------------------------------------------- #
def compute_fpfh_ms(
    points: np.ndarray,
    max_nn: int = 100,
    n_scales: int = 3,
    radius: float = 1e6,
    normal_max_nn: int = 10,
) -> np.ndarray:
    """Multi-scale FPFH features, shape (n_points, 33 * n_scales).

    A huge search radius makes the descriptor effectively scale-invariant;
    the neighborhood size is then controlled purely by `max_nn` (as in the
    official Simple3D implementation).
    """
    pcd = to_o3d(points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=normal_max_nn))
    feats = []
    for s in range(1, n_scales + 1):
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn * s))
        feats.append(np.asarray(fpfh.data.T, dtype=np.float32))
    return np.concatenate(feats, axis=1)


# --------------------------------------------------------------------------- #
# FPS (pure torch - no compiled extensions needed)
# --------------------------------------------------------------------------- #
def fps_torch(coords: np.ndarray, n_center: int, device: str | None = None) -> np.ndarray:
    """Farthest point sampling. Returns center coordinates (n_center, 3)."""
    dev = _resolve_device(device)
    t = torch.from_numpy(coords.astype(np.float32)).to(dev)
    n = t.shape[0]
    n_center = min(n_center, n)
    idx = torch.empty(n_center, dtype=torch.long, device=dev)
    min_dist = torch.full((n,), float("inf"), device=dev)
    far = 0
    for i in range(n_center):
        idx[i] = far
        d = torch.cdist(t[far:far + 1], t).squeeze(0)
        min_dist = torch.minimum(min_dist, d)
        far = int(torch.argmax(min_dist).item())
    return coords[idx.cpu().numpy()]


# --------------------------------------------------------------------------- #
# Chunked k-NN (memory-safe on CPU and GPU)
# --------------------------------------------------------------------------- #
def knn_indices(query: np.ndarray, ref: np.ndarray, k: int,
                device: str | None = None, work_size: int = 2 ** 24) -> np.ndarray:
    """Indices of the k nearest ref points for each query point -> (n_query, k)."""
    dev = _resolve_device(device)
    q = torch.from_numpy(query.astype(np.float32)).to(dev)
    r = torch.from_numpy(ref.astype(np.float32)).to(dev)
    k = min(k, r.shape[0])
    chunk = max(1, work_size // max(1, r.shape[0]))
    out = []
    for i in range(0, q.shape[0], chunk):
        d = torch.cdist(q[i:i + chunk], r)
        _, idx = torch.topk(d, k, largest=False)
        out.append(idx)
    return torch.cat(out).cpu().numpy()


# --------------------------------------------------------------------------- #
# LFSA: group + aggregate
# --------------------------------------------------------------------------- #
def lfsa(
    coords: np.ndarray,
    feats: np.ndarray,
    num_group: int = 2048,
    group_size: int = 128,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Local Feature Spatial Aggregation.

    Returns:
        centers     (num_group, 3)  FPS center coordinates
        center_feats(num_group, F)  mean of the group members' features
    """
    centers = fps_torch(coords, num_group, device=device)
    idx = knn_indices(centers, coords, group_size, device=device)
    center_feats = feats[idx].mean(axis=1).astype(np.float32)
    return centers.astype(np.float32), center_feats


# --------------------------------------------------------------------------- #
# Test-time registration (Direction E of NOVELTY_ROADMAP.md)
# --------------------------------------------------------------------------- #
class RegistrationTarget:
    """Precomputed reference for RANSAC(FPFH)+ICP alignment of incoming clouds.

    Built once per category from the (merged, mutually aligned) training
    prototypes. ``align()`` returns the 4x4 transform that puts a source
    cloud into the prototype reference frame, plus the ICP fitness.
    """

    def __init__(self, target_pts: np.ndarray, voxel: float = 0.05):
        self.voxel = float(voxel)
        reg = o3d.pipelines.registration

        self.pcd = to_o3d(target_pts).voxel_down_sample(self.voxel)
        self.pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 2, max_nn=30))
        self.fpfh = reg.compute_fpfh_feature(
            self.pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 5, max_nn=100))

        self.pcd_fine = to_o3d(target_pts).voxel_down_sample(self.voxel * 0.5)
        self.pcd_fine.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel, max_nn=30))

    def align(self, source_pts: np.ndarray) -> tuple[np.ndarray, float]:
        """Coarse global registration (FPFH + RANSAC) + point-to-plane ICP refine."""
        reg = o3d.pipelines.registration
        v = self.voxel

        src = to_o3d(source_pts).voxel_down_sample(v)
        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
        f_src = reg.compute_fpfh_feature(
            src, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 5, max_nn=100))

        ransac = reg.registration_ransac_based_on_feature_matching(
            src, self.pcd, f_src, self.fpfh,
            mutual_filter=False,
            max_correspondence_distance=v * 1.5,
            estimation_method=reg.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            criteria=reg.RANSACConvergenceCriteria(50000, 0.999))

        icp = reg.registration_icp(
            to_o3d(source_pts), self.pcd_fine, v * 0.6, ransac.transformation,
            reg.TransformationEstimationPointToPlane())

        return np.asarray(icp.transformation), float(icp.fitness)
