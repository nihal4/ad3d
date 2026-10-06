"""Location-aware memory: compare a test region only with training regions from the SAME place on the object.

The global memory bank (PatchCore / Simple3D style) asks "does this local shape occur ANYWHERE on a normal object?".
A defect that happens to look like normal geometry elsewhere (a bump on a shell resembling a ridge, a dent resembling a
crease) is then matched and missed. Because every training cloud (aligned prototypes, simulated cuts, registered real
cuts) and every test cloud live in ONE registered frame, each memory descriptor can keep the 3D position of its group
centre, and a test descriptor is compared only with descriptors whose centres lie within radius rho of its own centre
("does this shape occur HERE on a normal object?").

All descriptors are kept (no coreset), so every location is covered by all training clouds that see it. If no training
centre lies within rho (a region never seen in training), the global nearest-neighbour distance is used instead.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .features import _resolve_device


class LocalMemory:
    def __init__(self, device: str | None = None):
        self.device = _resolve_device(device)
        self._c: list[np.ndarray] = []
        self._f: list[np.ndarray] = []
        self.centers = None
        self.feats = None
        self.tree = None

    def add(self, centers: np.ndarray, feats: np.ndarray) -> None:
        self._c.append(np.asarray(centers, dtype=np.float32))
        self._f.append(np.asarray(feats, dtype=np.float32))

    def build(self) -> None:
        self.centers = np.concatenate(self._c)
        self.feats = torch.from_numpy(np.concatenate(self._f)).to(self.device)
        self.tree = cKDTree(self.centers)
        self._c, self._f = [], []

    @property
    def size(self) -> int:
        return 0 if self.centers is None else int(len(self.centers))

    def dists(self, centers: np.ndarray, feats: np.ndarray, radii: tuple[float, ...], fallback: np.ndarray,
              k: int = 1024, chunk: int = 256) -> tuple[dict, dict]:
        """For every test group centre: min feature distance over memory entries within each radius.
        Returns ({radius: (G,) distances}, {radius: mean number of candidates})."""
        k = min(k, self.size)
        dpos, idx = self.tree.query(np.asarray(centers, dtype=np.float32), k=k, workers=-1)
        if k == 1:
            dpos, idx = dpos[:, None], idx[:, None]
        q = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(self.device)
        dpos_t = torch.from_numpy(dpos.astype(np.float32)).to(self.device)
        idx_t = torch.from_numpy(idx.astype(np.int64)).to(self.device)
        out = {r: np.empty(len(q), dtype=np.float32) for r in radii}
        cnt = {r: 0.0 for r in radii}
        fb = torch.from_numpy(np.asarray(fallback, dtype=np.float32)).to(self.device)
        for s in range(0, len(q), chunk):
            e = min(s + chunk, len(q))
            m = self.feats[idx_t[s:e]]                                  # (c, k, F)
            d = torch.linalg.norm(m - q[s:e, None, :], dim=2)           # (c, k)
            for r in radii:
                valid = dpos_t[s:e] <= r
                dm = torch.where(valid, d, torch.full_like(d, float("inf"))).min(dim=1).values
                has = valid.any(dim=1)
                out[r][s:e] = torch.where(has, dm, fb[s:e]).cpu().numpy()
                cnt[r] += float(valid.sum().item())
        return out, {r: cnt[r] / max(1, len(q)) for r in radii}
