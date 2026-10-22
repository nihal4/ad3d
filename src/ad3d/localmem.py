"""Location-aware memory: compare a test region only with training regions from the SAME place on the object.

The global memory bank (PatchCore / Simple3D style) asks "does this local shape occur ANYWHERE on a normal object?".
A defect that happens to look like normal geometry elsewhere (a bump on a shell resembling a ridge, a dent resembling a
crease) is then matched and missed. Because every training cloud (aligned prototypes, simulated cuts, registered real
cuts) and every test cloud live in ONE registered frame, each memory descriptor can keep the 3D position of its group
centre, and a test descriptor is compared only with descriptors whose centres lie within radius rho of its own centre
("does this shape occur HERE on a normal object?").

All descriptors are kept (no coreset), so every location is covered by all training clouds that see it. If no training
centre lies within rho (a region never seen in training), the global nearest-neighbour distance is used instead.

Revision additions (2026-10): every entry carries metadata - the view type (0 = complete prototype, 1 = simulated
single-view cut, 2 = real single-view scan), the physical object it comes from (prototype index) and a view id - so
that one test-time pass can score the same features under many memory subsets and decision rules (see channels.py),
and so that a NORMAL-ONLY reference distance can be computed per entry (leave-one-object-out), used for calibration.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .features import _resolve_device

VIEW_COMPLETE, VIEW_SIM, VIEW_REAL = 0, 1, 2


class LocalMemory:
    def __init__(self, device: str | None = None):
        self.device = _resolve_device(device)
        self._c: list[np.ndarray] = []
        self._f: list[np.ndarray] = []
        self._meta: list[np.ndarray] = []
        self.centers = None
        self.feats = None
        self.tree = None
        self.vtype = self.obj = self.view = self.sub = None   # (N,) metadata
        self.ref = None                                          # (N,) leave-one-object-out reference distance
        self._n_views = 0

    def add(self, centers: np.ndarray, feats: np.ndarray, vtype: int = VIEW_COMPLETE, obj: int = 0,
            sub: int = 0) -> None:
        """Add one training view. `sub` = index of a simulated cut within its prototype (for cut-count masks)."""
        n = len(centers)
        self._c.append(np.asarray(centers, dtype=np.float32))
        self._f.append(np.asarray(feats, dtype=np.float32))
        self._meta.append(np.stack([np.full(n, vtype), np.full(n, obj), np.full(n, self._n_views),
                                    np.full(n, sub)], axis=1).astype(np.int64))
        self._n_views += 1

    def build(self) -> None:
        self.centers = np.concatenate(self._c)
        self.feats = torch.from_numpy(np.concatenate(self._f)).to(self.device)
        meta = np.concatenate(self._meta)
        self.vtype, self.obj, self.view, self.sub = (meta[:, i] for i in range(4))
        self.tree = cKDTree(self.centers)
        self._c, self._f, self._meta = [], [], []

    @property
    def size(self) -> int:
        return 0 if self.centers is None else int(len(self.centers))

    # ------------------------------------------------------------------ #
    def candidates(self, centers: np.ndarray, k: int = 1024):
        """k nearest memory positions of every query centre -> (dpos (G,k), idx (G,k)) as torch tensors."""
        k = min(k, self.size)
        dpos, idx = self.tree.query(np.asarray(centers, dtype=np.float32), k=k, workers=-1)
        if k == 1:
            dpos, idx = dpos[:, None], idx[:, None]
        return (torch.from_numpy(dpos.astype(np.float32)).to(self.device),
                torch.from_numpy(idx.astype(np.int64)).to(self.device))

    def dists(self, centers: np.ndarray, feats: np.ndarray, radii: tuple[float, ...], fallback: np.ndarray,
              k: int = 1024, chunk: int = 256) -> tuple[dict, dict]:
        """For every test group centre: min feature distance over memory entries within each radius.
        Returns ({radius: (G,) distances}, {radius: mean number of candidates}). (Original rule, unchanged.)"""
        dpos_t, idx_t = self.candidates(centers, k)
        q = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(self.device)
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

    # ------------------------------------------------------------------ #
    def global_full_dists(self, feats: np.ndarray, mask: np.ndarray | None = None,
                          work_size: int = 2 ** 25) -> np.ndarray:
        """Global nearest-neighbour distance over ALL (or the masked) memory entries, no coreset.
        This is the memory-size control: same entries as the location-aware rule, no spatial restriction."""
        bank = self.feats if mask is None else self.feats[torch.from_numpy(mask).to(self.device)]
        q = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(self.device)
        chunk = max(1, work_size // max(1, bank.shape[0]))
        out = [torch.cdist(q[i:i + chunk], bank).min(dim=1).values for i in range(0, len(q), chunk)]
        return torch.cat(out).cpu().numpy()

    # ------------------------------------------------------------------ #
    def build_reference(self, rho: float = 0.10, k: int = 1024, chunk: int = 512) -> dict:
        """NORMAL-ONLY reference: for every memory entry, the distance to its nearest descriptor within rho that comes
        from a DIFFERENT physical object (leave-one-object-out). This measures how much normal descriptors vary at
        that location across instances; no test data is involved. Also returns the same quantity without the
        spatial restriction (global leave-one-object-out), used to put global distances on the same scale."""
        dpos_t, idx_t = self.candidates(self.centers, k)
        obj_t = torch.from_numpy(self.obj).to(self.device)
        ref = torch.full((self.size,), float("nan"), device=self.device)
        for s in range(0, self.size, chunk):
            e = min(s + chunk, self.size)
            m = self.feats[idx_t[s:e]]
            d = torch.linalg.norm(m - self.feats[s:e, None, :], dim=2)
            other = obj_t[idx_t[s:e]] != obj_t[s:e, None]
            valid = (dpos_t[s:e] <= rho) & other
            dm = torch.where(valid, d, torch.full_like(d, float("inf"))).min(dim=1).values
            ref[s:e] = torch.where(valid.any(dim=1), dm, torch.full_like(dm, float("nan")))
        self.ref = ref
        finite = ref[torch.isfinite(ref)]
        cat_med = float(finite.median()) if finite.numel() else float("nan")
        # global leave-one-object-out on a subsample (scale of global distances among normal instances)
        g = []
        rng = np.random.default_rng(0)
        sub = rng.choice(self.size, min(4000, self.size), replace=False)
        for i0 in range(0, len(sub), 512):
            ii = torch.from_numpy(sub[i0:i0 + 512]).to(self.device)
            d = torch.cdist(self.feats[ii], self.feats)
            d[obj_t[ii][:, None] == obj_t[None, :]] = float("inf")
            g.append(d.min(dim=1).values)
        g = torch.cat(g)
        g = g[torch.isfinite(g)]
        self.ref_info = {"ref_cat_median": cat_med,
                         "ref_glob_median": float(g.median()) if g.numel() else float("nan"),
                         "ref_coverage": float(torch.isfinite(ref).float().mean())}
        return self.ref_info
