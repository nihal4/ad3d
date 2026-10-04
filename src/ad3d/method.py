"""Baseline method: Simple3D-lite (MSND-FPFH + LFSA + prototype memory bank).

This ties the modules together and is the class you subclass / modify when
prototyping a novel method.

Two optional helpers target the Real3D-AD 360°-train vs single-view-test gap
(Direction E of NOVELTY_ROADMAP.md):

  * ``align='icp'`` : RANSAC(FPFH)+ICP-register every test cloud to the merged
    training prototypes before feature matching, all in one shared reference
    frame (prototypes are mutually aligned at fit time).
  * ``cuts=N``      : augment the memory bank with N simulated single-view
    cuts of each prototype, so scan-boundary artifacts become "known normal"
    (mimics the train_cut trick used by the Simple3D paper).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from . import datasets as D
from .features import RegistrationTarget, compute_fpfh_ms, lfsa
from .memory import MemoryBank
from .metrics import compute_metrics
from .scoring import object_score, point_scores


@dataclass
class Config:
    data_root: str = "data"
    dataset: str = "shapenet"          # 'shapenet' | 'real3d'
    # features
    max_nn: int = 100                  # FPFH base neighborhood. NOTE: Simple3D paper (arXiv 2507.07435) states 40/80/120
                                       # (i.e. max_nn=40, n_scales=3); 100 is NOT the paper default - test --max-nn 40
    n_scales: int = 3                  # MSND scales: max_nn * [1, 2, 3]
    # grouping
    num_group: int = 2048              # FPS centers
    group_size: int = 128              # LFSA neighbors per group
    smooth_k: int = 12                 # centers averaged per point when scoring
    # memory
    coreset_ratio: float = 0.1
    cuts: int = 0                      # simulated single-view cuts per prototype
    cuts_diverse: bool = False         # True: different cut directions per prototype
                                       # (False reproduces all runs r1-r8 exactly)
    # alignment
    align: str = "none"                # 'none' | 'icp'
    align_voxel: float = 0.05          # RANSAC/ICP voxel; <=0 = auto per category
    align_poses: int = 1               # pose hypotheses per test cloud (>1 = multi-hypothesis)
    # preprocessing
    voxel: float = 0.01                # relative voxel size (cloud unit-normalized)
    points_budget: int = 100_000       # hard cap on points after voxel down
    # scoring
    topk: int = 1                      # object score = mean of top-k point scores
    # misc
    device: str = "auto"
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    return pts @ T[:3, :3].T + T[:3, 3]


class Simple3DLite:
    """Clean reimplementation of the Simple3D-style prototype pipeline."""

    name = "Simple3D-lite"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.memory = MemoryBank(device=cfg.device)
        self._ref_center: np.ndarray | None = None
        self._ref_scale: float | None = None
        self._reg_target: RegistrationTarget | None = None
        self._align_voxel_used: float | None = None

    # ------------------------------------------------------------------ #
    # Preprocessing (kept identical for train and test!)
    # ------------------------------------------------------------------ #
    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """Center + unit-scale.

        With ``align='icp'`` everything lives in the prototype reference frame
        (fixed center/scale from prototype 0) instead of per-cloud stats - a
        prerequisite for meaningful registration.
        """
        pts = np.asarray(points, dtype=np.float64)
        if self.cfg.align == "icp" and self._ref_center is not None:
            return (pts - self._ref_center) / self._ref_scale
        pts = pts - pts.mean(axis=0, keepdims=True)
        s = float(np.linalg.norm(pts, axis=1).max())
        return pts / s if s > 0 else pts

    def _finalize(self, pts: np.ndarray, gt: np.ndarray | None = None,
                  path_hash: int = 0):
        """Deterministic voxel downsample + budget cap, keeping rows aligned."""
        pts = pts.astype(np.float32)
        if self.cfg.voxel and self.cfg.voxel > 0:
            grid = np.floor(pts / self.cfg.voxel).astype(np.int64)
            _, idx = np.unique(grid, axis=0, return_index=True)
            idx = np.sort(idx)
            pts = pts[idx]
            if gt is not None:
                gt = gt[idx]
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

    def preprocess(self, points: np.ndarray, gt: np.ndarray | None = None,
                   path_hash: int = 0):
        """Center + unit-scale, voxel downsample, budget cap (rows aligned)."""
        return self._finalize(self._normalize(points), gt, path_hash)

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

    def _add_to_memory(self, pts: np.ndarray) -> None:
        _, center_feats = self.sample_features(pts)
        self.memory.add(center_feats)

    def _simulated_cuts(self, pts: np.ndarray, n_cuts: int, proto_idx: int = 0) -> int:
        """Augment the memory with simulated single-view scans of a prototype.

        Each cut keeps the directional cap visible from a random viewpoint
        (star-shape approximation) - the resulting scan-boundary artifacts
        match those of real single-view test clouds.
        """
        # Legacy behaviour re-used the SAME seed for every prototype, so all
        # prototypes got identical cut directions (n_cuts distinct views in
        # total, not n_cuts * n_prototypes). cuts_diverse fixes that.
        off = 7919 * proto_idx if self.cfg.cuts_diverse else 0
        rng = np.random.default_rng(self.cfg.seed + 1234 + off)
        dirs = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9)
        added = 0
        for _ in range(n_cuts):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            cut = pts[dirs @ u > 0.15]
            if cut.shape[0] < 200:
                continue
            cut_f, _ = self._finalize(cut)
            self._add_to_memory(cut_f)
            added += 1
        return added

    # ------------------------------------------------------------------ #
    # Adaptive registration resolution
    # ------------------------------------------------------------------ #
    def _select_voxel(self, raws: list[np.ndarray]) -> float:
        """Pick the registration voxel per category from TRAINING data only.

        r7 evidence: slender objects (airplane) need a fine voxel, bulky
        near-symmetric ones (starfish) need a coarse one. We register
        simulated single-view cuts of the other prototypes to the base
        prototype at both candidate resolutions; lower trimmed residual
        wins, with a stability penalty when mean ICP fitness drops below
        0.95 (fine voxels can destabilise RANSAC/ICP). No test data is
        touched - this is a legitimate per-category hyperparameter choice.
        """
        o3d.utility.random.seed(self.cfg.seed + 99)
        rng = np.random.default_rng(self.cfg.seed + 555)
        base_n = (raws[0] - self._ref_center) / self._ref_scale
        cuts: list[np.ndarray] = []
        for r in raws[1:]:
            r_n = (r - self._ref_center) / self._ref_scale
            d = r_n / np.maximum(np.linalg.norm(r_n, axis=1, keepdims=True), 1e-9)
            for _ in range(2):
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                cut = r_n[d @ u > 0.15]
                if cut.shape[0] >= 200:
                    cuts.append(cut)
        if not cuts:
            print("  [align-voxel auto] no usable cuts; defaulting to 0.05")
            return 0.05
        best_vox, best_score = 0.05, None
        for vox in (0.05, 0.03):
            tgt = RegistrationTarget(base_n, voxel=vox)
            res, fits = [], []
            for cut in cuts:
                T, f = tgt.align(cut)
                if f >= 0.1:
                    res.append(tgt.residual(cut, T))
                    fits.append(f)
            if not res:
                continue
            score = float(np.mean(res)) + max(0.0, 0.95 - float(np.mean(fits)))
            print(f"  [align-voxel auto] voxel {vox:.2f}: cut-reg residual "
                  f"{np.mean(res):.4f}  fitness {np.mean(fits):.2f}")
            if best_score is None or score < best_score:
                best_vox, best_score = vox, score
        print(f"  [align-voxel auto] selected {best_vox:.2f}")
        return best_vox

    # ------------------------------------------------------------------ #
    # Fit / evaluate
    # ------------------------------------------------------------------ #
    def fit(self, cls: str):
        self.memory = MemoryBank(device=self.cfg.device)
        trains = D.load_train(self.cfg.data_root, self.cfg.dataset, cls)

        if self.cfg.align == "icp":
            o3d.utility.random.seed(self.cfg.seed)
            raws = [np.asarray(t, dtype=np.float64) for t in trains]
            base = raws[0]
            self._ref_center = base.mean(axis=0)
            self._ref_scale = float(np.linalg.norm(
                base - self._ref_center, axis=1).max()) or 1.0
            base_n = (base - self._ref_center) / self._ref_scale
            vox = self.cfg.align_voxel
            if vox <= 0:                        # auto: pick from {0.05, 0.03}
                vox = self._select_voxel(raws)
            self._align_voxel_used = float(vox)
            self._reg_target = RegistrationTarget(base_n, voxel=vox)

            proto_frames = [base_n]
            for r in raws[1:]:                       # mutually align prototypes
                r_n = (r - self._ref_center) / self._ref_scale
                T, _ = self._reg_target.align(r_n)
                proto_frames.append(_transform(r_n, T))
            # denser registration target: the merged aligned prototypes
            self._reg_target = RegistrationTarget(
                np.concatenate(proto_frames), voxel=vox)
        else:
            self._ref_center, self._reg_target = None, None
            proto_frames = [self._normalize(t) for t in trains]

        for pi, pf in enumerate(proto_frames):
            pts, _ = self._finalize(pf)
            self._add_to_memory(pts)
            if self.cfg.cuts > 0:
                self._simulated_cuts(pts, self.cfg.cuts, proto_idx=pi)

        self.memory.build(coreset_ratio=self.cfg.coreset_ratio, seed=self.cfg.seed)

    def evaluate(self, cls: str) -> dict:
        test = D.load_test(self.cfg.data_root, self.cfg.dataset, cls)
        if self.cfg.align == "icp":
            o3d.utility.random.seed(self.cfg.seed + 7)

        obj_labels, obj_scores, point_gts, point_scores_all = [], [], [], []
        fitnesses: list[float] = []
        pose_switched = 0
        for s in test:
            h = int(hashlib.md5(s.path.encode()).hexdigest()[:8], 16)
            if self.cfg.align == "icp" and self._reg_target is not None:
                p_n = (np.asarray(s.points, dtype=np.float64)
                       - self._ref_center) / self._ref_scale
                cands = self._reg_target.align_candidates(
                    p_n, poses=self.cfg.align_poses, seed=self.cfg.seed)
                if len(cands) > 1:
                    i, (T, fitness) = self._reg_target.select_pose(p_n, cands)
                    pose_switched += int(i != 0)
                else:
                    T, fitness = cands[0]
                fitnesses.append(fitness)
                pts, gt = self._finalize(_transform(p_n, T), s.gt, path_hash=h)
            else:
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

        m = compute_metrics(obj_labels, obj_scores, point_gts, point_scores_all)
        if fitnesses:
            m["align_fitness_mean"] = float(np.mean(fitnesses))
        if self._align_voxel_used is not None:
            m["align_voxel"] = self._align_voxel_used
        if self.cfg.align_poses > 1 and test:
            m["pose_switch_rate"] = pose_switched / len(test)
        return m


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
            extra = (f"  align_fitness {m['align_fitness_mean']:.2f}"
                     if "align_fitness_mean" in m else "")
            if "pose_switch_rate" in m:
                extra += f"  pose_switch {m['pose_switch_rate']:.0%}"
            if "align_voxel" in m:
                extra += f"  voxel {m['align_voxel']:.2f}"
            print(f"[{cfg.dataset}] {cls:>14s}  "
                  f"O-AUROC {m['o_auroc']:.3f}  "
                  f"P-AUROC {m['p_auroc']:.3f}  "
                  f"P-AUPR {m['p_aupr']:.3f}  "
                  f"({m['n_anom']}/{m['n_test']} anomalous){extra}", flush=True)

    df = pd.DataFrame(rows).T
    mean_row = df.mean(numeric_only=True)
    rows["__mean__"] = {k: (None if v != v else float(v)) for k, v in mean_row.items()}
    if verbose:
        cols = ["o_auroc", "p_auroc", "p_aupr", "p_auroc_pooled", "p_aupr_pooled"]
        print("\n=== Mean over", len(classes), "categories ===")
        for c in cols:
            print(f"  {c:>14s}: {mean_row[c]:.4f}")
    return rows
