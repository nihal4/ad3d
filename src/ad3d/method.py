"""Baseline method: Simple3D-lite (MSND-FPFH + LFSA + prototype memory bank).

This ties the modules together and is the class you subclass / modify when
prototyping a novel method.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from . import datasets as D
from .features import compute_fpfh_ms, lfsa
from .memory import MemoryBank
from .metrics import compute_metrics
from .scoring import object_score, point_scores


@dataclass
class Config:
    data_root: str = "data"
    dataset: str = "shapenet"          # 'shapenet' | 'real3d'
    # features
    max_nn: int = 100                  # FPFH base neighborhood (Simple3D default)
    n_scales: int = 3                  # MSND scales: max_nn * [1, 2, 3]
    # grouping
    num_group: int = 2048              # FPS centers
    group_size: int = 128              # LFSA neighbors per group
    smooth_k: int = 12                 # centers averaged per point when scoring
    # memory
    coreset_ratio: float = 0.1
    # preprocessing
    voxel: float = 0.01                # relative voxel size (cloud unit-normalized)
    points_budget: int = 100_000       # hard cap on points after voxel down
    # scoring
    topk: int = 1                      # object score = mean of top-k point scores
    # misc
    device: str = "auto"
    seed: int = 0
    extra: dict = field(default_factory=dict)


class Simple3DLite:
    """Clean reimplementation of the Simple3D-style prototype pipeline."""

    name = "Simple3D-lite"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.memory = MemoryBank(device=cfg.device)

    # ------------------------------------------------------------------ #
    # Preprocessing (kept identical for train and test!)
    # ------------------------------------------------------------------ #
    def preprocess(self, points: np.ndarray, gt: np.ndarray | None = None,
                   path_hash: int = 0):
        """Center + unit-scale, deterministic voxel downsample, budget cap.

        Returns (points, gt) with rows kept aligned. `path_hash` makes the
        random budget-cap reproducible per sample.
        """
        pts = points.astype(np.float32)
        pts = pts - pts.mean(axis=0, keepdims=True)
        scale = float(np.linalg.norm(pts, axis=1).max())
        if scale > 0:
            pts = pts / scale

        # deterministic voxel subsample (keeps point <-> gt alignment)
        if self.cfg.voxel and self.cfg.voxel > 0:
            grid = np.floor(pts / self.cfg.voxel).astype(np.int64)
            _, idx = np.unique(grid, axis=0, return_index=True)
            idx = np.sort(idx)
            pts = pts[idx]
            if gt is not None:
                gt = gt[idx]

        # random budget cap (seeded by path so re-runs are identical)
        if pts.shape[0] > self.cfg.points_budget:
            rng = np.random.default_rng(self.cfg.seed + path_hash)
            keep = rng.choice(pts.shape[0], self.cfg.points_budget, replace=False)
            keep.sort()
            pts = pts[keep]
            if gt is not None:
                gt = gt[keep]

        if gt is None:
            gt = np.zeros(pts.shape[0], dtype=np.float32)
        return pts.astype(np.float32), gt.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Per-sample feature extraction (override this for new methods)
    # ------------------------------------------------------------------ #
    def sample_features(self, points: np.ndarray):
        feats = compute_fpfh_ms(points, max_nn=self.cfg.max_nn,
                                n_scales=self.cfg.n_scales)
        centers, center_feats = lfsa(points, feats,
                                     num_group=self.cfg.num_group,
                                     group_size=self.cfg.group_size,
                                     device=self.cfg.device)
        return centers, center_feats

    # ------------------------------------------------------------------ #
    # Fit / evaluate
    # ------------------------------------------------------------------ #
    def fit(self, cls: str):
        self.memory = MemoryBank(device=self.cfg.device)
        for pts in D.load_train(self.cfg.data_root, self.cfg.dataset, cls):
            pts, _ = self.preprocess(pts)
            _, center_feats = self.sample_features(pts)
            self.memory.add(center_feats)
        self.memory.build(coreset_ratio=self.cfg.coreset_ratio, seed=self.cfg.seed)

    def evaluate(self, cls: str) -> dict:
        test = D.load_test(self.cfg.data_root, self.cfg.dataset, cls)
        obj_labels, obj_scores, point_gts, point_scores_all = [], [], [], []
        for s in test:
            h = int(hashlib.md5(s.path.encode()).hexdigest()[:8], 16)
            pts, gt = self.preprocess(s.points, s.gt, path_hash=h)
            centers, center_feats = self.sample_features(pts)
            center_dists = self.memory.min_dists(center_feats)
            p_scores = point_scores(pts, centers, center_dists,
                                    smooth_k=self.cfg.smooth_k,
                                    device=self.cfg.device)
            obj_labels.append(s.label)
            obj_scores.append(object_score(p_scores, topk=self.cfg.topk))
            point_gts.append(gt)
            point_scores_all.append(p_scores)
        return compute_metrics(obj_labels, obj_scores, point_gts, point_scores_all)


def run_benchmark(cfg: Config, classes: list[str] | None = None,
                  verbose: bool = True):
    """Run every category, return {category: metrics} plus the mean row."""
    import pandas as pd

    classes = classes or D.get_classes(cfg.dataset)
    rows = {}
    for cls in classes:
        model = Simple3DLite(cfg)
        model.fit(cls)
        m = model.evaluate(cls)
        rows[cls] = m
        if verbose:
            print(f"[{cfg.dataset}] {cls:>14s}  "
                  f"O-AUROC {m['o_auroc']:.3f}  "
                  f"P-AUROC {m['p_auroc']:.3f}  "
                  f"P-AUPR {m['p_aupr']:.3f}  "
                  f"({m['n_anom']}/{m['n_test']} anomalous)", flush=True)

    df = pd.DataFrame(rows).T
    mean_row = df.mean(numeric_only=True)
    rows["__mean__"] = {k: (None if v != v else float(v)) for k, v in mean_row.items()}
    if verbose:
        cols = ["o_auroc", "p_auroc", "p_aupr", "p_auroc_pooled", "p_aupr_pooled"]
        print("\n=== Mean over", len(classes), "categories ===")
        for c in cols:
            print(f"  {c:>14s}: {mean_row[c]:.4f}")
    return rows
