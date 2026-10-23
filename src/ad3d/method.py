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
import os
import resource
import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from . import datasets as D
from .features import RegistrationTarget, compute_fpfh_ms, lfsa
from .memory import MemoryBank
from .metrics import compute_metrics
from .scoring import object_score, object_score_rule, object_score_stats, point_scores  # noqa: F401


# rules also scored under the single RANSAC/ICP pose T0 (pose-hypothesis ablation for free); hyb added 2026-10-23
H1_CHANNELS = ("glob_full", "loc05", "loc10", "loc15", "knn64", "knn256", "cal10", "hyb")


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
    topk: int = 1                      # object score: 1 = max, k>1 = mean of top-k, <=0 = mean of all points
    obj_rule: str = ""                 # if set, overrides topk: 'max' | 'mean' | 'p99' | 'top200' | 'top1pct' ...
    # training data
    train_cut_root: str = ""           # if set: train on pre-cut single-view clouds <root>/<cls>/train_cut/*
                                       # (GLFM/Simple3D 'Cut Training Data'); official prototypes then only
                                       # serve as the registration target (align=icp)
    train_cut_with_protos: bool = False  # also keep the full prototypes in the memory bank
    # geometric channel: Prototype Tolerance Field (needs align='icp'); see geometry.py
    geo: str = "none"                  # 'none' | 'fuse' (feature x (1 + u)) | 'only' (u alone, ablation)
    geo_voxel: float = 0.005           # surface sampling of the merged prototypes (unit-normalised frame)
    geo_k: int = 12                    # test-point smoothing of the normalised residual
    # location-aware memory (needs align='icp'); see localmem.py. 0 = off (global memory only)
    local_mem: float = 0.0             # PRIMARY radius rho (unit-normalised frame); e.g. 0.10
    local_radii: tuple = (0.05, 0.10, 0.15)   # also reported from the same run (sensitivity)
    # revision experiments (2026-10): many decision rules from one feature extraction (channels.py)
    channels: str = ""                 # "" = original behaviour; "rev" = all revision channels + diagnostics
    save_maps: str = ""                # directory: save point clouds + score maps of a fixed subset of test samples
    n_protos: int = 0                  # use only the first n prototypes (0 = all) - robustness to training-set size
    holdout: int = -1                  # normal-only dev protocol: prototype held out of memory and frame (-1 = none)
    test_noise: float = 0.0            # robustness: Gaussian noise (std, unit-normalised frame) added to test scans
    test_density: float = 1.0          # robustness: fraction of test points kept (random subsample)
    test_crop: float = 1.0             # robustness: fraction of the test scan kept by a directional crop
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

    def _add_to_memory(self, pts: np.ndarray, vtype: int = 0, obj: int = 0, sub: int = 0) -> None:
        centers, center_feats = self.sample_features(pts)
        self.memory.add(center_feats)
        if getattr(self, "_local", None) is not None:
            self._local.add(centers, center_feats, vtype=vtype, obj=obj, sub=sub)

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
        for l in range(n_cuts):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            cut = pts[dirs @ u > 0.15]
            if cut.shape[0] < 200:
                continue
            cut_f, _ = self._finalize(cut)
            self._add_to_memory(cut_f, vtype=1, obj=proto_idx, sub=l)
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
        self._local = None
        if self.cfg.local_mem > 0:
            if self.cfg.align != "icp":
                raise ValueError("--local-mem needs --align icp (positions must live in one registered frame)")
            from .localmem import LocalMemory
            self._local = LocalMemory(device=self.cfg.device)
        self.timing = {}
        t_fit0 = time.perf_counter()
        trains = getattr(self, "_train_override", None) or D.load_train(self.cfg.data_root, self.cfg.dataset, cls)
        if self.cfg.n_protos > 0:
            trains = trains[:self.cfg.n_protos]
        obj_ids = list(range(len(trains)))
        self._holdout_raw = None
        if 0 <= self.cfg.holdout < len(trains):
            self._holdout_raw = np.asarray(trains[self.cfg.holdout], dtype=np.float64)
            obj_ids = [i for i in obj_ids if i != self.cfg.holdout]
            trains = [trains[i] for i in obj_ids]
        self._obj_ids = obj_ids

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
            self._proto_frames = proto_frames      # kept for diagnostics (scripts/diag_align.py)
            if self._holdout_raw is not None:      # dev protocol: held-out object registered ONLY to assign real scans
                h_n = (self._holdout_raw - self._ref_center) / self._ref_scale
                T_h, _ = self._reg_target.align(h_n)
                self._holdout_frame = _transform(h_n, T_h)
            # denser registration target: the merged aligned prototypes
            self._reg_target = RegistrationTarget(
                np.concatenate(proto_frames), voxel=vox)
            self._tol = None
            if self.cfg.geo != "none":
                from .geometry import ToleranceField
                self._tol = ToleranceField(proto_frames, voxel=self.cfg.geo_voxel)
                print(f"  [geo] {cls}: tolerance field on {self._tol.ref['n_surface']} surface pts, "
                      f"median sigma {self._tol.ref['sigma_median']:.4f}", flush=True)
        else:
            if self.cfg.geo != "none":
                raise ValueError("--geo needs --align icp (the tolerance field lives in the registered frame)")
            self._ref_center, self._reg_target, self._tol = None, None, None
            proto_frames = [self._normalize(t) for t in trains]

        self.timing["fit_register_s"] = time.perf_counter() - t_fit0
        t_mem0 = time.perf_counter()
        use_protos = (not self.cfg.train_cut_root) or self.cfg.train_cut_with_protos
        if use_protos:
            for pi, pf in zip(obj_ids, proto_frames):
                pts, _ = self._finalize(pf)
                self._add_to_memory(pts, vtype=0, obj=pi)
                if self.cfg.cuts > 0:
                    self._simulated_cuts(pts, self.cfg.cuts, proto_idx=pi)

        self.train_cut_info = {}
        self._holdout_cuts = []                  # dev protocol: raw real scans of the held-out object (pseudo-test normals)
        if self.cfg.train_cut_root:
            cut_fits = []
            cut_list = D.load_train_cut(self.cfg.train_cut_root, cls)
            frames = [np.asarray(f, np.float32) for f in proto_frames]
            frame_ids = list(obj_ids)
            if getattr(self, "_holdout_frame", None) is not None and self._holdout_raw is not None:
                frames.append(np.asarray(self._holdout_frame, np.float32))
                frame_ids.append(self.cfg.holdout)
            from .features import to_o3d
            reg_ = o3d.pipelines.registration
            fpcs = [to_o3d(f).voxel_down_sample(0.01) for f in frames]
            ftargets = [RegistrationTarget(f, voxel=self._align_voxel_used or 0.05) for f in frames] \
                if self.cfg.align == "icp" else []
            assigned = []
            self._assign_log = []
            n_real_mem = 0
            for path, c in cut_list:
                h = int(hashlib.md5(path.encode()).hexdigest()[:8], 16)
                if self.cfg.align == "icp" and self._reg_target is not None:
                    # register the cut to the merged prototypes exactly like a test cloud
                    c_n = (np.asarray(c, dtype=np.float64) - self._ref_center) / self._ref_scale
                    cands = self._reg_target.align_candidates(
                        c_n, poses=self.cfg.align_poses, seed=self.cfg.seed)
                    if len(cands) > 1:
                        _, (T, fit) = self._reg_target.select_pose(c_n, cands)
                    else:
                        T, fit = cands[0]
                    cut_fits.append(fit)
                    pts, _ = self._finalize(_transform(c_n, T), path_hash=h)
                else:
                    pts, _ = self.preprocess(c, path_hash=h)
                # which physical object is this scan of? Refine the registered scan against each prototype separately
                # (point-to-point ICP from the current pose) and take the one it fits best (geometry only).
                src_pc = to_o3d(pts).voxel_down_sample(0.01)
                fit_rmse = []
                for fi, fpc in enumerate(fpcs):
                    T_ = np.eye(4)
                    if ftargets:                             # re-register the scan to THIS prototype alone first
                        T_, _ = ftargets[fi].align(np.asarray(pts, np.float64))
                    for thr_ in (0.10, 0.05, 0.02):          # coarse-to-fine, from the current pose
                        r_ = reg_.registration_icp(src_pc, fpc, thr_, T_,
                                                   reg_.TransformationEstimationPointToPoint(),
                                                   reg_.ICPConvergenceCriteria(max_iteration=30))
                        T_ = r_.transformation
                    fit_rmse.append(r_.inlier_rmse if r_.fitness > 0.5 else np.inf)
                owner = frame_ids[int(np.argmin(fit_rmse))] if np.isfinite(np.min(fit_rmse)) \
                    else -100 - len(assigned)              # source unknown: own object id (never pooled with others)
                if os.environ.get("AD3D_DEBUG_ASSIGN"):
                    print("    [assign]", os.path.basename(path), np.round(fit_rmse, 4).tolist(), "->", owner)
                assigned.append(owner)
                rm = np.asarray(fit_rmse, float)
                fin = np.isfinite(rm)
                srt = np.sort(rm[fin]) if fin.any() else np.array([np.inf])
                ratio = float(srt[1] / srt[0]) if len(srt) > 1 and srt[0] > 0 else float("inf")
                self._assign_log.append({"scan": os.path.basename(path), "rmse": [float(x) for x in rm],
                                         "frames": list(frame_ids), "owner": int(owner), "ratio_2nd_best": ratio})
                if self._holdout_raw is not None:
                    # dev protocol, conservative against leakage: a scan is a pseudo-test of the held-out object only if
                    # the held-out object fits it clearly best (2nd best >= 1.5 x worse); it enters the memory only if
                    # the held-out object is NOT a plausible source (its fit >= 1.5 x worse than the best). Everything
                    # in between, and every scan of unknown source, is left out of both.
                    hi = frame_ids.index(self.cfg.holdout) if self.cfg.holdout in frame_ids else None
                    r_h = rm[hi] if hi is not None else np.inf
                    best = srt[0]
                    if owner == self.cfg.holdout and ratio >= 1.5:
                        self._holdout_cuts.append((path, np.asarray(c, np.float32)))
                        continue
                    if owner <= -100 or not (r_h >= 1.5 * best):
                        continue
                if pts.shape[0] >= 200:
                    self._add_to_memory(pts, vtype=2, obj=owner)
                    n_real_mem += 1
            self.train_cut_info = {"train_cut_n": len(cut_list),
                                   "train_cut_owners": ",".join(str(a) for a in assigned),
                                   "train_cut_unknown": int(sum(a <= -100 for a in assigned)),
                                   "train_cut_assign": self._assign_log,
                                   "train_cut_in_memory": n_real_mem}
            if cut_fits:
                self.train_cut_info["train_cut_fit_mean"] = float(np.mean(cut_fits))
            print(f"  [train-cut] {cls}: {len(cut_list)} cut clouds"
                  + (f", registration fitness {np.mean(cut_fits):.2f}" if cut_fits else ""), flush=True)

        self.memory.build(coreset_ratio=self.cfg.coreset_ratio, seed=self.cfg.seed)
        if self._local is not None:
            self._local.build()
            print(f"  [local-mem] {cls}: {self._local.size} positioned descriptors", flush=True)
            if self.cfg.channels == "rev":
                from .channels import memory_masks
                info = self._local.build_reference(rho=self.cfg.local_mem)
                self._masks = memory_masks(self._local, seed=self.cfg.seed)
                self.train_cut_info.update({k: v for k, v in info.items()})
                self.train_cut_info.update({f"mem_{k}": int(v.sum()) for k, v in self._masks.items()})
        self.timing["fit_memory_s"] = time.perf_counter() - t_mem0

    def evaluate(self, cls: str, samples: list | None = None) -> dict:
        test = samples if samples is not None else D.load_test(self.cfg.data_root, self.cfg.dataset, cls)
        if self.cfg.channels == "rev":
            return self._evaluate_rev(cls, test)
        if self.cfg.align == "icp":
            o3d.utility.random.seed(self.cfg.seed + 7)

        obj_labels, obj_scores, point_gts, point_scores_all = [], [], [], []
        tol = getattr(self, "_tol", None)
        lm = getattr(self, "_local", None)
        if lm is not None and self.cfg.local_mem not in tuple(self.cfg.local_radii):
            self.cfg.local_radii = tuple(sorted(set(tuple(self.cfg.local_radii) + (self.cfg.local_mem,))))
        LCH = (["glob"] + [f"loc{int(round(r * 100)):02d}" for r in self.cfg.local_radii]) if lm is not None else []
        lch_obj = {k: [] for k in LCH}
        lch_pts = {k: [] for k in LCH}
        loc_cand = {r: [] for r in self.cfg.local_radii}
        CH = ("feat", "geo", "geou", "fuseu")        # ablation channels computed in the SAME run
        ch_obj = {k: [] for k in CH}                 # per-channel object scores (same object rule)
        ch_pts = {k: [] for k in CH}
        self.sample_records: list[dict] = []
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
                fitness = None
                pts, gt = self.preprocess(s.points, s.gt, path_hash=h)

            centers, center_feats = self.sample_features(pts)
            center_dists = self.memory.min_dists(center_feats)
            p_scores = point_scores(pts, centers, center_dists,
                                    smooth_k=self.cfg.smooth_k,
                                    device=self.cfg.device)
            loc_maps = {}
            if self._local is not None:
                ld, lc = self._local.dists(centers, center_feats, tuple(self.cfg.local_radii), fallback=center_dists)
                for r_, dr in ld.items():
                    loc_maps[r_] = point_scores(pts, centers, dr, smooth_k=self.cfg.smooth_k, device=self.cfg.device)
                    loc_cand[r_].append(lc[r_])
                glob_map = p_scores
                p_scores = loc_maps[self.cfg.local_mem]          # PRIMARY: location-aware score
            def _obj(v):
                return (object_score_rule(v, self.cfg.obj_rule) if self.cfg.obj_rule
                        else object_score(v, topk=self.cfg.topk))
            ch = {}
            if tol is not None:
                u, uu = tol.score(pts, smooth_k=self.cfg.geo_k)
                # feat = features only; geo = tolerance-normalised residual only; geou = residual with ONE global
                # threshold (Template3D-AD-like); fuseu = features x (1 + geou)  -> isolates the tolerance field
                ch = {"feat": p_scores, "geo": u, "geou": uu,
                      "fuseu": (p_scores * (1.0 + uu)).astype(np.float32)}
                p_scores = (p_scores * (1.0 + u) if self.cfg.geo == "fuse" else u).astype(np.float32)
                for k_, v_ in ch.items():
                    ch_obj[k_].append(_obj(v_))
                    ch_pts[k_].append(v_)
            if lm is not None:
                chs = {"glob": glob_map}
                chs.update({f"loc{int(round(r_ * 100)):02d}": v_ for r_, v_ in loc_maps.items()})
                for k_, v_ in chs.items():
                    lch_obj[k_].append(_obj(v_))
                    lch_pts[k_].append(v_)
                    st = object_score_stats(v_)
                    ch[k_] = v_
            obj_labels.append(s.label)
            obj_scores.append(_obj(p_scores))
            rec = {"file": os.path.basename(s.path), "label": int(s.label),
                   "n_points": int(pts.shape[0]), "n_anom_points": int(gt.sum())}
            rec.update(object_score_stats(p_scores))
            for k_, v_ in ch.items():
                st = object_score_stats(v_)
                rec.update({f"{k_}_{r_}": st[r_] for r_ in ("max", "p99", "mean")})
            if fitness is not None:
                rec["fitness"] = float(fitness)
            self.sample_records.append(rec)
            point_gts.append(gt)
            point_scores_all.append(p_scores)

        m = compute_metrics(obj_labels, obj_scores, point_gts, point_scores_all)
        if lm is not None:                           # global vs location-aware channels from the SAME run
            for k_ in LCH:
                mc = compute_metrics(obj_labels, lch_obj[k_], point_gts, lch_pts[k_])
                for key in ("o_auroc", "p_auroc", "p_aupr", "p_auroc_pooled", "p_aupr_pooled"):
                    m[f"{k_}_{key}"] = mc[key]
            for r_, v_ in loc_cand.items():
                m[f"loc{int(round(r_ * 100)):02d}_candidates"] = float(np.mean(v_)) if v_ else 0.0
            m["local_mem_size"] = lm.size
        if tol is not None:                          # channel ablation from the SAME run
            for k_ in CH:
                mc = compute_metrics(obj_labels, ch_obj[k_], point_gts, ch_pts[k_])
                for key in ("o_auroc", "p_auroc", "p_aupr", "p_auroc_pooled", "p_aupr_pooled"):
                    m[f"{k_}_{key}"] = mc[key]
            for k_, v_ in tol.ref.items():
                m[f"geo_{k_}"] = v_
        if fitnesses:
            m["align_fitness_mean"] = float(np.mean(fitnesses))
        if self._align_voxel_used is not None:
            m["align_voxel"] = self._align_voxel_used
        if self.cfg.align_poses > 1 and test:
            m["pose_switch_rate"] = pose_switched / len(test)
        m.update(getattr(self, "train_cut_info", {}) or {})
        for k, v in (self.memory.ref or {}).items():
            m[f"train_ref_{k}"] = v
        return m


    # ------------------------------------------------------------------ #
    # Revision path: many decision rules from one feature extraction
    # ------------------------------------------------------------------ #
    def _perturb(self, p_n: np.ndarray, gt: np.ndarray, h: int):
        """Robustness perturbations of a test scan in the unit-normalised frame (all off by default).
        Crop keeps a directional cap holding `test_crop` of the points; for an anomalous scan the cap is
        re-drawn (up to 20 times) until at least half of its defect points stay visible, so that the
        crop tests partial views of visible defects rather than deleting the defect."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed * 1000003 + h)
        if cfg.test_crop < 1.0:
            c = p_n - p_n.mean(0)
            dirs = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
            best = None
            for _ in range(20):
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                proj = dirs @ u
                thr = np.quantile(proj, 1.0 - cfg.test_crop)
                keep = proj >= thr
                if gt.sum() == 0 or gt[keep].sum() >= 0.5 * gt.sum():
                    best = keep
                    break
                if best is None or gt[keep].sum() > gt[best].sum():
                    best = keep
            p_n, gt = p_n[best], gt[best]
        if cfg.test_density < 1.0:
            keep = rng.random(len(p_n)) < cfg.test_density
            p_n, gt = p_n[keep], gt[keep]
        if cfg.test_noise > 0:
            p_n = p_n + rng.normal(scale=cfg.test_noise, size=p_n.shape)
        return p_n, gt

    def _channels_for(self, pts: np.ndarray):
        from .channels import score_channels
        centers, feats = self.sample_features(pts)
        glob_cs = self.memory.min_dists(feats)
        ch = score_channels(self._local, glob_cs, centers, feats, self._masks, rho=self.cfg.local_mem,
                            radii=tuple(self.cfg.local_radii))
        return centers, ch

    def _evaluate_rev(self, cls: str, test: list) -> dict:
        from .channels import KEY_CHANNELS
        from .features import knn_indices
        cfg = self.cfg
        if self._local is None or cfg.align != "icp":
            raise ValueError("channels='rev' needs --align icp and --local-mem > 0")
        o3d.utility.random.seed(cfg.seed + 7)
        if torch_cuda():
            import torch
            torch.cuda.reset_peak_memory_stats()
        tsum = {"test_register_s": 0.0, "test_features_match_s": 0.0, "test_score_s": 0.0}
        labels, gts = [], []
        obj = {}            # channel -> list of object scores (main rule = cfg.obj_rule or p99)
        pmaps = {k: [] for k in KEY_CHANNELS}
        self.sample_records = []
        rule = cfg.obj_rule or "p99"
        save_left = self._maps_plan(test) if cfg.save_maps else set()
        switched, fits = 0, []
        for s in test:
            h = int(hashlib.md5(s.path.encode()).hexdigest()[:8], 16)
            t0 = time.perf_counter()
            p_n = (np.asarray(s.points, dtype=np.float64) - self._ref_center) / self._ref_scale
            gt0 = np.asarray(s.gt, np.float32)
            if cfg.test_crop < 1 or cfg.test_density < 1 or cfg.test_noise > 0:
                p_n, gt0 = self._perturb(p_n, gt0, h)
            cands = self._reg_target.align_candidates(p_n, poses=cfg.align_poses, seed=cfg.seed)
            res = []
            if len(cands) > 1:
                i_sel, (T, fit) = self._reg_target.select_pose(p_n, cands)
                res = list(getattr(self._reg_target, "last_residuals", []))
            else:
                i_sel, (T, fit) = 0, cands[0]
                res = [self._reg_target.residual(p_n, T)]
            switched += int(i_sel != 0)
            fits.append(fit)
            pts, gt = self._finalize(_transform(p_n, T), gt0, path_hash=h)
            tsum["test_register_s"] += time.perf_counter() - t0
            t1 = time.perf_counter()
            centers, ch = self._channels_for(pts)
            if i_sel != 0:          # single-pose ablation for free: the same rules under the RANSAC/ICP pose T0
                pts0, _ = self._finalize(_transform(p_n, cands[0][0]), gt0, path_hash=h)
                c0, ch0 = self._channels_for(pts0)
                idx0 = knn_indices(pts0, c0, cfg.smooth_k, device=cfg.device)
                h1 = {f"h1_{k}": ch0[k][idx0].mean(axis=1) for k in H1_CHANNELS if k in ch0}
            tsum["test_features_match_s"] += time.perf_counter() - t1
            t2 = time.perf_counter()
            idx = knn_indices(pts, centers, cfg.smooth_k, device=cfg.device)
            maps = {k: v[idx].mean(axis=1).astype(np.float32) for k, v in ch.items()}
            if i_sel == 0:
                h1 = {f"h1_{k}": maps[k] for k in H1_CHANNELS if k in maps}
            maps.update(h1)
            rec = {"file": os.path.basename(s.path), "label": int(s.label), "n_points": int(len(pts)),
                   "n_anom_points": int(gt.sum()), "fitness": float(fit), "pose_idx": int(i_sel),
                   "n_hyp": len(cands), "reg_res": float(res[i_sel] if res else np.nan),
                   "reg_res_T0": float(res[0] if res else np.nan)}
            for k, v in maps.items():
                st = object_score_stats(v)
                rec.update({f"{k}_{r}": st[r] for r in ("max", "p99", "p95", "mean", "top1pct")})
                obj.setdefault(k, []).append(object_score_rule(v, rule))
            for k in KEY_CHANNELS:
                if k in maps:
                    pmaps[k].append(maps[k])
            if s.path in save_left:
                self._save_map(cls, s, pts, gt, maps, rec)
            self.sample_records.append(rec)
            labels.append(s.label)
            gts.append(gt)
            tsum["test_score_s"] += time.perf_counter() - t2

        main = f"loc{int(round(cfg.local_mem * 100)):02d}"
        m = compute_metrics(labels, obj[main], gts, pmaps[main] if pmaps.get(main) else pmaps["loc10"])
        for k, v in obj.items():
            m[f"{k}_o_auroc"] = compute_metrics(labels, v, [np.zeros(1)], [np.zeros(1)])["o_auroc"]
        for k in KEY_CHANNELS:
            if pmaps.get(k):
                mc = compute_metrics(labels, obj[k], gts, pmaps[k])
                for key in ("p_auroc", "p_aupr", "p_auroc_pooled", "p_aupr_pooled"):
                    m[f"{k}_{key}"] = mc[key]
        m["align_fitness_mean"] = float(np.mean(fits)) if fits else float("nan")
        m["pose_switch_rate"] = switched / max(1, len(test))
        m["local_mem_size"] = self._local.size
        m.update(getattr(self, "train_cut_info", {}) or {})
        n = max(1, len(test))
        m.update({k: v for k, v in self.timing.items()})
        m.update({k.replace("_s", "_per_scan_s"): v / n for k, v in tsum.items()})
        m["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        if torch_cuda():
            import torch
            m["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return m

    def _maps_plan(self, test: list) -> set:
        """Documented, score-independent selection of samples whose maps are saved: per category the first two
        anomalous and the first normal test scans in file-name order."""
        an = sorted(s.path for s in test if s.label == 1)[:2]
        no = sorted(s.path for s in test if s.label == 0)[:1]
        return set(an + no)

    def _save_map(self, cls, s, pts, gt, maps, rec):
        os.makedirs(self.cfg.save_maps, exist_ok=True)
        keep = np.arange(len(pts))
        if len(pts) > 30000:
            keep = np.sort(np.random.default_rng(0).choice(len(pts), 30000, replace=False))
        stem = os.path.splitext(os.path.basename(s.path))[0]
        arrs = {"points": pts[keep].astype(np.float16), "gt": gt[keep].astype(np.uint8)}
        for k in ("glob_cs", "glob_full", "loc10", "cal10", "hyb", "soft1"):
            if k in maps:
                arrs[k] = maps[k][keep].astype(np.float16)
        np.savez_compressed(os.path.join(self.cfg.save_maps, f"{cls}__{stem}__s{self.cfg.seed}.npz"),
                            meta=np.array(json_dumps(rec)), **arrs)


def torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def json_dumps(d: dict) -> str:
    import json
    return json.dumps(d)


def run_benchmark(cfg: Config, classes: list[str] | None = None,
                  verbose: bool = True, scores_out: dict | None = None):
    """Run every category, return {category: metrics} plus the mean row.
    If `scores_out` is a dict, it is filled with per-sample score records per category."""
    import pandas as pd

    classes = classes or D.get_classes(cfg.dataset)
    rows = {}
    for cls in classes:
        model = Simple3DLite(cfg)
        model.fit(cls)
        m = model.evaluate(cls)
        rows[cls] = m
        if scores_out is not None:
            scores_out[cls] = model.sample_records
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
