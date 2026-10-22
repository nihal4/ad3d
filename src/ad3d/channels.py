"""Many decision rules from ONE feature extraction (revision experiments).

Given the descriptors of one registered test scan, `score_channels` returns, for every rule ("channel"), one distance
per test group centre. All channels share the same registration, training views and descriptors, so their differences
isolate the decision rule (or the memory subset) alone. Channels:

  glob_cs              global nearest neighbour over the greedy coreset (original global baseline)
  glob_full            global nearest neighbour over ALL memory entries (memory-size control for the local rule)
  loc05/loc10/loc15    original location-aware rule, fixed radius, fallback = glob_cs (main method = loc10)
  knn{M}               density-adaptive: nearest feature among the M nearest memory POSITIONS (no fixed radius)
  soft{a}              soft spatial penalty: min_j ||f - f_j|| + a * s * (||c - c_j|| / rho)^2, s = normal-only
                       feature scale (median leave-one-object-out distance), candidates = J nearest positions
  cal10, cal10_k0      normal-only calibration: loc10 / shrunk median of the leave-one-object-out reference of the
                       entries in the ball (kappa = 64 pseudo-entries towards the category median; k0 = no shrinkage)
  hyb                  geometric mean of the calibrated local and the calibrated global distance
  loc10_{V}, glob_full_{V}       memory subsets: V0 complete only, V01 complete + simulated, V02 complete + real,
                       M2 = all but only the first 2 simulated cuts per prototype; `_eq` = same subset randomly
                       reduced to the size of V0 (memory-budget control)
"""
from __future__ import annotations

import numpy as np
import torch

from .localmem import VIEW_COMPLETE, VIEW_REAL, VIEW_SIM, LocalMemory

KEY_CHANNELS = ("glob_cs", "glob_full", "loc10", "knn256", "soft1", "cal10", "hyb")


def memory_masks(lm: LocalMemory, seed: int = 0) -> dict[str, np.ndarray]:
    vt, sub = lm.vtype, lm.sub
    masks = {
        "V0": vt == VIEW_COMPLETE,
        "V01": (vt == VIEW_COMPLETE) | (vt == VIEW_SIM),
        "V02": (vt == VIEW_COMPLETE) | (vt == VIEW_REAL),
        "V012": np.ones(lm.size, bool),
        "M2": (vt != VIEW_SIM) | (sub < 2),
    }
    n0 = int(masks["V0"].sum())
    rng = np.random.default_rng(seed + 31)
    for name in ("V01", "V02", "V012"):
        idx = np.where(masks[name])[0]
        if len(idx) > n0 > 0:
            keep = np.zeros(lm.size, bool)
            keep[rng.choice(idx, n0, replace=False)] = True
            masks[name + "_eq"] = keep
    # drop subsets that are empty or identical to the full memory (e.g. no real scans on a dataset)
    return {k: v for k, v in masks.items() if v.any() and (k == "V012" or not v.all())}


def score_channels(lm: LocalMemory, glob_cs: np.ndarray, centers: np.ndarray, feats: np.ndarray,
                   masks: dict[str, np.ndarray], rho: float = 0.10, radii=(0.05, 0.10, 0.15),
                   knn=(64, 256), soft=(0.5, 1.0, 2.0), kappa: float = 64.0, k: int = 1024,
                   chunk: int = 256) -> dict[str, np.ndarray]:
    dev = lm.device
    G = len(feats)
    out: dict[str, np.ndarray] = {"glob_cs": np.asarray(glob_cs, np.float32)}
    gfull = {name: lm.global_full_dists(feats, None if name == "V012" else m) for name, m in masks.items()}
    out["glob_full"] = gfull["V012"]
    for name in masks:
        if name != "V012":
            out[f"glob_full_{name}"] = gfull[name]

    dpos_t, idx_t = lm.candidates(centers, k)
    q = torch.from_numpy(np.asarray(feats, np.float32)).to(dev)
    mask_t = {name: torch.from_numpy(m).to(dev) for name, m in masks.items() if name != "V012"}
    ref = lm.ref if lm.ref is not None else None
    info = getattr(lm, "ref_info", {}) or {}
    r_cat = info.get("ref_cat_median", float("nan"))
    r_glob = info.get("ref_glob_median", float("nan"))
    s_feat = r_cat if np.isfinite(r_cat) and r_cat > 0 else 1.0
    fb_cs = torch.from_numpy(np.asarray(glob_cs, np.float32)).to(dev)
    fb_sub = {name: torch.from_numpy(gfull[name]).to(dev) for name in mask_t}

    acc: dict[str, list[torch.Tensor]] = {}

    def put(name, t):
        acc.setdefault(name, []).append(t)

    INF = float("inf")
    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        idx = idx_t[s:e]
        dp = dpos_t[s:e]
        d = torch.linalg.norm(lm.feats[idx] - q[s:e, None, :], dim=2)          # (c, J)
        for r in radii:
            valid = dp <= r
            dm = torch.where(valid, d, torch.full_like(d, INF)).min(dim=1).values
            put(f"loc{int(round(r * 100)):02d}", torch.where(valid.any(1), dm, fb_cs[s:e]))
        ball = dp <= rho
        for name, mt in mask_t.items():
            valid = ball & mt[idx]
            dm = torch.where(valid, d, torch.full_like(d, INF)).min(dim=1).values
            put(f"loc{int(round(rho * 100)):02d}_{name}", torch.where(valid.any(1), dm, fb_sub[name][s:e]))
        for M in knn:
            put(f"knn{M}", d[:, :min(M, d.shape[1])].min(dim=1).values)
        for a in soft:
            pen = d + a * s_feat * (dp / rho) ** 2
            put(f"soft{a:g}".replace(".", "p"), pen.min(dim=1).values)
        if ref is not None:
            rr = ref[idx]                                                       # (c, J), nan where undefined
            vr = ball & torch.isfinite(rr)
            n = vr.sum(dim=1).float()
            med = torch.nanmedian(torch.where(vr, rr, torch.full_like(rr, float("nan"))), dim=1).values
            med = torch.where(n > 0, med, torch.full_like(med, r_cat))
            dloc = torch.where(ball, d, torch.full_like(d, INF)).min(dim=1).values
            dloc = torch.where(ball.any(1), dloc, fb_cs[s:e])
            for nm, kap in (("cal10", kappa), ("cal10_k0", 0.0)):
                den = (n * med + kap * r_cat) / (n + kap).clamp_min(1e-9) if kap > 0 else med
                put(nm, dloc / den.clamp_min(1e-9))
    for name, parts in acc.items():
        out[name] = torch.cat(parts).cpu().numpy().astype(np.float32)
    if "cal10" in out and np.isfinite(r_glob) and r_glob > 0:
        out["hyb"] = np.sqrt(np.maximum(out["cal10"], 0) * (out["glob_full"] / r_glob)).astype(np.float32)
    return out
