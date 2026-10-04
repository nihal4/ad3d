"""Score assignment: group-level distances -> per-point map -> object score.

Extension point #3 for novelty: alternative scoring rules live here
(top-k pooling, density calibration, per-sample z-normalisation, ...).
"""
from __future__ import annotations

import numpy as np

from .features import knn_indices


def point_scores(
    coords: np.ndarray,          # (n, 3)  all points of the test cloud
    centers: np.ndarray,         # (G, 3)  group centers
    center_dists: np.ndarray,    # (G,)    distance of each group to the memory bank
    smooth_k: int = 12,
    device: str | None = None,
) -> np.ndarray:
    """Per-point anomaly score = mean score of the `smooth_k` nearest centers.

    (Equivalent in spirit to Simple3D's interpolate -> re-group -> average ->
    interpolate smoothing, in a single step.)
    """
    idx = knn_indices(coords, centers, smooth_k, device=device)
    return center_dists[idx].mean(axis=1).astype(np.float32)


def object_score(point_scores: np.ndarray, topk: int = 1) -> float:
    """Object-level score: max point score (topk=1), mean of top-k (topk>1),
    or the mean over ALL points (topk<=0; Simple3D's rule on Real3D-AD)."""
    if topk <= 0:
        return float(point_scores.mean())
    k = min(topk, point_scores.shape[0])
    if k <= 1:
        return float(point_scores.max())
    part = np.partition(point_scores, -k)[-k:]
    return float(part.mean())


SCORE_RULES = ("max", "top10", "top32", "top80", "top200", "top1pct", "p99", "p95", "p90", "mean")


def object_score_stats(point_scores: np.ndarray) -> dict:
    """Every candidate object-score rule for one sample (saved so that scoring rules
    can be compared offline from a single run, without re-running feature extraction)."""
    s = np.asarray(point_scores, dtype=np.float64)
    out = {"max": float(s.max()), "mean": float(s.mean())}
    for k in (10, 32, 80, 200):
        out[f"top{k}"] = float(np.partition(s, -min(k, s.size))[-min(k, s.size):].mean())
    k = max(1, int(round(0.01 * s.size)))
    out["top1pct"] = float(np.partition(s, -k)[-k:].mean())
    for q in (99, 95, 90):
        out[f"p{q}"] = float(np.percentile(s, q))
    return out
