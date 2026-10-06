"""Prototype memory bank with greedy k-center coreset subsampling.

Extension point #2 for novelty: the memory construction / distance metric /
coreset strategy are all swappable (e.g. per-prototype banks, Mahalanobis
distance, PatchCore-style reweighting).
"""
from __future__ import annotations

import numpy as np
import torch

from .features import _resolve_device


class MemoryBank:
    def __init__(self, device: str | None = None):
        self.device = _resolve_device(device)
        self.feats: torch.Tensor | None = None      # coreset (m, F)
        self._raw: list[torch.Tensor] = []
        self.ref: dict = {}                           # training-normal distance stats (for score normalisation)

    # ------------------------------------------------------------------ #
    def add(self, feats: np.ndarray) -> None:
        self._raw.append(torch.from_numpy(feats.astype(np.float32)).to(self.device))

    def build(self, coreset_ratio: float = 0.1, seed: int = 0) -> None:
        """Concatenate stored features and subsample a greedy k-center coreset."""
        if not self._raw:
            raise RuntimeError("Memory bank is empty - call add() first.")
        all_feats = torch.cat(self._raw, dim=0)
        n = all_feats.shape[0]
        m = max(1, int(n * coreset_ratio))
        if m >= n:
            self.feats = all_feats
            self.ref = {}
            return
        rng = np.random.default_rng(seed)
        selected = torch.tensor([rng.integers(n)], dtype=torch.long, device=self.device)
        min_dist = torch.cdist(all_feats[selected], all_feats).squeeze(0)
        for _ in range(m - 1):
            far = int(torch.argmax(min_dist).item())
            selected = torch.cat([selected,
                                  torch.tensor([far], device=self.device)])
            d = torch.cdist(all_feats[far:far + 1], all_feats).squeeze(0)
            min_dist = torch.minimum(min_dist, d)
        self.feats = all_feats[selected]
        # min_dist = distance of every TRAINING feature to the final coreset (0 for the selected ones):
        # a training-only reference of how far normal features sit from the bank. Used to put scores
        # from different descriptor scales on a common footing (no test data involved).
        nz = min_dist[min_dist > 0].float()
        if nz.numel() > 10:
            q = torch.quantile(nz[torch.randperm(nz.numel(), device=nz.device)[:200000]],
                               torch.tensor([0.5, 0.9, 0.99], device=nz.device))
            self.ref = {"mean": float(nz.mean()), "std": float(nz.std()),
                        "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2])}

    # ------------------------------------------------------------------ #
    def min_dists(self, query: np.ndarray, work_size: int = 2 ** 24) -> np.ndarray:
        """Distance from every query row to its nearest memory entry -> (n_query,)."""
        if self.feats is None:
            raise RuntimeError("Memory bank not built - call build() first.")
        q = torch.from_numpy(query.astype(np.float32)).to(self.device)
        chunk = max(1, work_size // max(1, self.feats.shape[0]))
        out = []
        for i in range(0, q.shape[0], chunk):
            d = torch.cdist(q[i:i + chunk], self.feats)
            out.append(d.min(dim=1).values)
        return torch.cat(out).cpu().numpy()

    @property
    def size(self) -> int:
        return 0 if self.feats is None else int(self.feats.shape[0])
