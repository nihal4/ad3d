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
    """Object-level score: max point score (topk=1) or mean of top-k."""
    k = min(topk, point_scores.shape[0])
    if k <= 1:
        return float(point_scores.max())
    part = np.partition(point_scores, -k)[-k:]
    return float(part.mean())
