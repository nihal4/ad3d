"""Prototype Tolerance Field (PTF): a geometric anomaly channel built from the few normal prototypes.

Idea
----
After registration, every test point can be compared with the *surface* of the training prototypes: a bulge or a
sink displaces points away from it. A single fixed threshold is a poor judge of that displacement, because normal
objects differ from each other by different amounts at different places (a shell's ridges vary a lot between
instances, a car's flat panels hardly at all). With several aligned normal prototypes we can MEASURE that normal
variation per location:

  1. Align the prototypes into one frame (done by the registration stage) and refine them with point-to-plane ICP.
  2. Leave-one-out residuals: for every point of prototype i, its distance to the surface formed by the OTHER
     prototypes. This is how far a *normal* instance sits from the normal surface, location by location.
  3. Tolerance field sigma(x): those leave-one-out residuals averaged over the k nearest surface points (spatially
     smooth), plus a global floor eps (median sigma).
  4. Test point p (already registered): residual r(p) to the merged prototype surface, normalised by the tolerance
     at its nearest surface location: u(p) = r(p) / (sigma(q) + eps), then averaged over its k nearest test points.

No test data and no labels are used to build the field: it comes from the training prototypes only.

The residual r is point-to-plane, plus the tangential offset beyond the sampling spacing (so points that project
off the sampled surface, e.g. beyond an edge, are not hidden by the plane approximation).
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def _down(pts: np.ndarray, voxel: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64)))
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points)


def _normals(pts: np.ndarray, voxel: float, max_nn: int = 20) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel, 1e-6) * 4, max_nn=max_nn))
    return np.asarray(pcd.normals)


def _residual(query: np.ndarray, surf: np.ndarray, normals: np.ndarray, tree: cKDTree, spacing: float):
    """Point-to-plane residual + tangential offset beyond the sampling spacing. Returns (r, nearest index)."""
    d, idx = tree.query(query, k=1, workers=-1)
    diff = query - surf[idx]
    normal_off = np.abs(np.einsum("ij,ij->i", diff, normals[idx]))
    tang = np.sqrt(np.maximum(d ** 2 - normal_off ** 2, 0.0))
    return normal_off + np.maximum(tang - spacing, 0.0), idx


def _icp_refine(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray, max_dist: float) -> np.ndarray:
    s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst))
    t.normals = o3d.utility.Vector3dVector(dst_normals)
    res = o3d.pipelines.registration.registration_icp(
        s, t, max_dist, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    return np.asarray(res.transformation)


def _spacing(p: np.ndarray) -> float:
    dd, _ = cKDTree(p).query(p[:: max(1, len(p) // 20000)], k=2, workers=-1)
    return float(np.median(dd[:, 1]))


def _apply(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


class ToleranceField:
    def __init__(self, frames: list[np.ndarray], voxel: float = 0.005, k_smooth: int = 16,
                 refine: bool = True):
        self.voxel = float(voxel)
        self.k = int(k_smooth)
        P = [_down(f, self.voxel) for f in frames]
        N = [_normals(p, self.voxel) for p in P]
        if refine and len(P) > 1:                      # tighten the mutual alignment (point-to-plane ICP)
            for i in range(1, len(P)):
                T = _icp_refine(P[i], P[0], N[0], max_dist=self.voxel * 6)
                P[i] = _apply(P[i], T)
                N[i] = _normals(P[i], self.voxel)
        trees = [cKDTree(p) for p in P]

        # leave-one-out residual of every prototype point w.r.t. the union of the other prototypes
        loo = []
        for i in range(len(P)):
            others = [j for j in range(len(P)) if j != i]
            if not others:
                loo.append(np.zeros(len(P[i])))
                continue
            r = np.min(np.stack([_residual(P[i], P[j], N[j], trees[j], max(self.voxel, _spacing(P[j])))[0] for j in others]), axis=0)
            loo.append(r)

        self.surf = np.concatenate(P)
        self.surf_normals = np.concatenate(N)
        self.tree = cKDTree(self.surf)
        dd, _ = self.tree.query(self.surf[:: max(1, len(self.surf) // 50000)], k=2, workers=-1)
        self.spacing = max(self.voxel, float(np.median(dd[:, 1])))   # actual sampling spacing of the surface
        raw = np.concatenate(loo)
        # spatially smooth tolerance (mean of LOO residuals over the k nearest surface points)
        _, nn = self.tree.query(self.surf, k=min(self.k, len(self.surf)), workers=-1)
        self.sigma = raw[nn].mean(axis=1) if nn.ndim == 2 else raw
        self.eps = float(np.median(self.sigma)) if len(P) > 1 else self.voxel
        if self.eps <= 0:
            self.eps = self.voxel

        # training reference of the normalised residual (normal-to-normal), smoothed like test maps
        # uniform tolerance (ablation = one fixed threshold everywhere, Template3D-AD-like): global mean LOO residual
        self.uniform = float(raw.mean()) if len(P) > 1 and raw.mean() > 0 else self.spacing
        u_ref = raw / (self.sigma + self.eps)
        self.ref = {"u_mean": float(u_ref.mean()), "u_std": float(u_ref.std()),
                    "u_p99": float(np.percentile(u_ref, 99)), "sigma_median": float(np.median(self.sigma)),
                    "sigma_p90": float(np.percentile(self.sigma, 90)), "eps": self.eps,
                    "uniform": self.uniform, "spacing": self.spacing, "n_surface": int(len(self.surf))}

    def refine_pose(self, pts: np.ndarray) -> np.ndarray:
        """Small point-to-plane ICP of an already registered test cloud onto the prototype surface."""
        sub = pts if len(pts) <= 20000 else pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
        T = _icp_refine(np.asarray(sub, dtype=np.float64), self.surf, self.surf_normals, max_dist=self.voxel * 6)
        return T

    def score(self, pts: np.ndarray, smooth_k: int = 12, refine: bool = True):
        """For every point of a registered test cloud (same frame) returns
        (u, u_uniform): residual normalised by the tolerance field, and by one global threshold (ablation)."""
        q = np.asarray(pts, dtype=np.float64)
        if refine:
            q = _apply(q, self.refine_pose(q))
        r, idx = _residual(q, self.surf, self.surf_normals, self.tree, self.spacing)
        u = r / (self.sigma[idx] + self.eps)
        uu = r / self.uniform
        if smooth_k and smooth_k > 1 and len(q) > smooth_k:
            _, nn = cKDTree(q).query(q, k=smooth_k, workers=-1)
            u, uu = u[nn].mean(axis=1), uu[nn].mean(axis=1)
        return u.astype(np.float32), uu.astype(np.float32)
