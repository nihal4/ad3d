"""Evaluation metrics for 3D anomaly detection.

Two point-level conventions are reported because papers differ:
  * per-sample  : AUROC / AUPR computed per anomalous test sample, then averaged
                  (M3DM / MVTec-3D / Real3D-AD benchmark convention)
  * pooled      : AUROC / AUPR over every point of every test sample in the
                  category, concatenated

Always report which convention you use in a paper.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_aupr(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def compute_metrics(obj_labels, obj_scores, point_gts, point_scores) -> dict:
    """obj_labels: list[int]; obj_scores: list[float];
    point_gts / point_scores: list[np.ndarray] (one per test sample)."""
    obj_labels = np.asarray(obj_labels)
    obj_scores = np.asarray(obj_scores, dtype=np.float64)

    # ---- object level -----------------------------------------------------
    o_auroc = _safe_auroc(obj_labels, obj_scores)
    o_aupr = (float(average_precision_score(obj_labels, obj_scores))
              if len(np.unique(obj_labels)) > 1 else float("nan"))

    # ---- point level (per anomalous sample, then averaged) ----------------
    per_sample_auroc, per_sample_aupr = [], []
    for gt, s in zip(point_gts, point_scores):
        gt = np.asarray(gt)
        if (gt > 0).any() and (gt == 0).any():
            per_sample_auroc.append(_safe_auroc(gt, s))
            per_sample_aupr.append(_safe_aupr(gt, s))
    p_auroc = float(np.nanmean(per_sample_auroc)) if per_sample_auroc else float("nan")
    p_aupr = float(np.nanmean(per_sample_aupr)) if per_sample_aupr else float("nan")

    # ---- point level (pooled over the whole category) ---------------------
    gt_all = np.concatenate([np.asarray(g).ravel() for g in point_gts])
    s_all = np.concatenate([np.asarray(s).ravel() for s in point_scores])
    p_auroc_pooled = _safe_auroc(gt_all, s_all)
    p_aupr_pooled = _safe_aupr(gt_all, s_all)

    return {
        "o_auroc": o_auroc,
        "o_aupr": o_aupr,
        "p_auroc": p_auroc,
        "p_aupr": p_aupr,
        "p_auroc_pooled": p_auroc_pooled,
        "p_aupr_pooled": p_aupr_pooled,
        "n_test": int(len(obj_labels)),
        "n_anom": int((obj_labels == 1).sum()),
    }
