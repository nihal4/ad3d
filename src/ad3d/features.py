"""Feature extraction: multi-scale FPFH (MSND) + Local Feature Spatial Aggregation (LFSA).

This is the Simple3D recipe, reimplemented cleanly and dependency-light:

  1. MSND : concatenate FPFH descriptors computed at several neighborhood
            scales (max_nn, 2*max_nn, 3*max_nn) -> (n, 33 * n_scales)
  2. FPS  : farthest-point-sample `num_group` center points
  3. LFSA : each center's feature = mean FPFH of its `group_size` neighbors

Extension point #1 for novelty: add new descriptors / learned features here.
"""
from __future__ import annotations

import zlib

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
def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit `axis` and `angle` (radians)."""
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _pose_distance(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Distance between two poses: rotation angle (deg) + weighted shift."""
    R = T_a[:3, :3].T @ T_b[:3, :3]
    tr = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(tr)))
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    return ang + 200.0 * dt          # 0.05 units of shift == 10 "degrees"


class RegistrationTarget:
    """Precomputed reference for RANSAC(FPFH)+ICP alignment of incoming clouds.

    Built once per category from the (merged, mutually aligned) training
    prototypes. ``align()`` returns the 4x4 transform that puts a source
    cloud into the prototype reference frame, plus the ICP fitness.
    ``align_candidates()`` additionally returns symmetry-equivalent poses
    (multi-hypothesis registration): symmetric objects admit several valid
    alignments, and picking the wrong one inflates normal clouds' scores.

    Coarse-to-fine pipeline (fast): RANSAC at a coarse voxel with mutual
    filtering, then two point-to-plane ICP passes at increasingly fine
    resolution - never on the full-resolution cloud.
    """

    def __init__(self, target_pts: np.ndarray, voxel: float = 0.05):
        self.voxel = float(voxel)
        v = self.voxel
        reg = o3d.pipelines.registration

        # coarse level (global registration)
        self.pcd = to_o3d(target_pts).voxel_down_sample(v * 2)
        self.pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v * 4, max_nn=30))
        self.fpfh = reg.compute_fpfh_feature(
            self.pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 10, max_nn=100))

        # mid level (ICP pass 1 + hypothesis refinement)
        self.pcd_mid = to_o3d(target_pts).voxel_down_sample(v)
        self.pcd_mid.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
        # capped torch copy for residual computation (memory-safe cdist)
        mid = np.asarray(self.pcd_mid.points, dtype=np.float32)
        if mid.shape[0] > 30_000:
            mid = mid[np.linspace(0, mid.shape[0] - 1, 30_000).astype(int)]
        self._ref_mid = torch.from_numpy(mid)

        # fine level (ICP pass 2)
        self.pcd_fine = to_o3d(target_pts).voxel_down_sample(v * 0.5)
        self.pcd_fine.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v, max_nn=30))

    # ------------------------------------------------------------------ #
    def _coarse_icp(self, source_pts: np.ndarray, T_init, iters: int = 40):
        reg = o3d.pipelines.registration
        return reg.registration_icp(
            to_o3d(source_pts).voxel_down_sample(self.voxel), self.pcd_mid,
            self.voxel, T_init, reg.TransformationEstimationPointToPlane(),
            reg.ICPConvergenceCriteria(relative_fitness=1e-4, max_iteration=iters))

    # ------------------------------------------------------------------ #
    def align_candidates(self, source_pts: np.ndarray, poses: int = 1,
                         seed: int = 0) -> list[tuple[np.ndarray, float]]:
        """RANSAC+ICP pose plus, if `poses > 1`, symmetry-equivalent ones.

        Perturbation restarts: large deterministic rotations around the best
        pose, each refined by a short ICP, converge into whichever
        symmetry-equivalent basin is nearest. Returns up to `poses` distinct
        (transform, fitness) candidates, best fitness first. Deterministic
        for a given (points, seed).
        """
        reg = o3d.pipelines.registration
        v = self.voxel

        src = to_o3d(source_pts).voxel_down_sample(v * 2)
        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v * 4, max_nn=30))
        f_src = reg.compute_fpfh_feature(
            src, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 10, max_nn=100))

        def ransac(mutual: bool, iters: int):
            return reg.registration_ransac_based_on_feature_matching(
                src, self.pcd, f_src, self.fpfh,
                mutual_filter=mutual,
                max_correspondence_distance=v * 2,
                estimation_method=reg.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                criteria=reg.RANSACConvergenceCriteria(iters, 0.999))

        r = ransac(True, 20_000)
        if r.fitness < 0.2:                     # ambiguous features -> broader search
            r = ransac(False, 50_000)

        icp1 = self._coarse_icp(source_pts, r.transformation, iters=50)
        if icp1.fitness < 0.1:                  # registration failed; keep RANSAC pose
            return [(np.asarray(r.transformation), float(r.fitness))]

        icp2 = reg.registration_icp(
            to_o3d(source_pts).voxel_down_sample(v * 0.5), self.pcd_fine, v * 0.5,
            icp1.transformation, reg.TransformationEstimationPointToPlane(),
            reg.ICPConvergenceCriteria(relative_fitness=1e-4, max_iteration=30))
        best = icp2 if icp2.fitness >= icp1.fitness else icp1
        candidates: list[tuple[np.ndarray, float]] = [
            (np.asarray(best.transformation), float(best.fitness))]

        if poses <= 1:
            return candidates

        # --- multi-hypothesis: deterministic perturbation restarts --------- #
        pts64 = np.ascontiguousarray(source_pts, dtype=np.float64)
        rng = np.random.default_rng(
            (zlib.crc32(pts64.tobytes()) ^ int(seed)) & 0xFFFFFFFF)
        raw = []
        for _ in range(poses - 1):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-12
            ang = rng.uniform(np.deg2rad(40.0), np.deg2rad(180.0))
            T_rot = np.eye(4)
            T_rot[:3, :3] = _rotation(axis, ang)
            T_init = T_rot @ candidates[0][0]
            ic = self._coarse_icp(source_pts, T_init, iters=40)
            if ic.fitness >= 0.8 * candidates[0][1]:
                raw.append((np.asarray(ic.transformation), float(ic.fitness)))
        for T, f in raw:                        # deduplicate near-identical poses
            if all(_pose_distance(T, T0) > 10.0 for T0, _ in candidates):
                candidates.append((T, f))
        candidates.sort(key=lambda c: -c[1])
        return candidates[:poses]

    # ------------------------------------------------------------------ #
    def _res_src(self, source_pts: np.ndarray) -> np.ndarray:
        """Deterministic capped downsample for residual evaluation."""
        src = np.asarray(to_o3d(source_pts).voxel_down_sample(
            self.voxel).points, dtype=np.float32)
        if src.shape[0] > 20_000:
            rng = np.random.default_rng(
                zlib.crc32(np.ascontiguousarray(src, dtype=np.float64).tobytes())
                & 0xFFFFFFFF)
            src = src[np.sort(rng.choice(src.shape[0], 20_000, replace=False))]
        return src

    def _trimmed_residual(self, src: np.ndarray, T: np.ndarray) -> float:
        """90th-percentile point-to-surface residual of a posed cloud."""
        s = torch.from_numpy(src) @ torch.from_numpy(T[:3, :3].T.astype(np.float32)) \
            + torch.from_numpy(T[:3, 3].astype(np.float32))
        mins = []
        step = max(1, 2 ** 24 // max(1, self._ref_mid.shape[0]))
        for i in range(0, s.shape[0], step):
            mins.append(torch.cdist(s[i:i + step], self._ref_mid).min(dim=1).values)
        return float(torch.quantile(torch.cat(mins), 0.9))

    def residual(self, source_pts: np.ndarray, T: np.ndarray) -> float:
        """Trimmed point-to-surface residual of `source_pts` under pose `T`."""
        return self._trimmed_residual(self._res_src(source_pts), T)

    # ------------------------------------------------------------------ #
    def select_pose(self, source_pts: np.ndarray,
                    candidates: list[tuple[np.ndarray, float]]):
        """Pick the pose with the lowest trimmed point-to-surface residual.

        A registration decision based on geometry ONLY - never on detection
        scores, which would adversarially hide anomalies (choosing the pose
        where a defect is least visible). The 90th-percentile residual is
        robust to the defect itself (outliers are trimmed) but sensitive to
        a systematic misfit of a whole wrongly-posed region.
        Returns (index, (transform, fitness)).
        """
        src = self._res_src(source_pts)
        best_i, best_res = 0, None
        for i, (T, _f) in enumerate(candidates):
            res = self._trimmed_residual(src, T)
            if best_res is None or res < best_res:
                best_i, best_res = i, res
        return best_i, candidates[best_i]

    # ------------------------------------------------------------------ #
    def align(self, source_pts: np.ndarray) -> tuple[np.ndarray, float]:
        """Single-pose alignment (backward compatible)."""
        T, f = self.align_candidates(source_pts)[0]
        return T, f
