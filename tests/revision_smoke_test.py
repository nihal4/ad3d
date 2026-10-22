"""Smoke test of the revision code on synthetic data (no dataset download).

[1] channels='rev' on a Real3D-AD-style layout with real single-view training scans: every channel present,
    metadata/masks consistent, reference built, maps saved, timing recorded, legacy loc10 unchanged
[2] robustness perturbations (crop / density / noise / n_protos)
[3] normal-only development protocol (scripts/dev_protocol.py) on real3d and shapenet layouts
[4] MVTec 3D-AD loader: organised tiff, plane removal, mask -> point labels
Run:  python tests/revision_smoke_test.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from synthetic_smoke_test import random_rotation, write_gt_txt, write_pcd  # noqa: E402
from ad3d.method import Config, Simple3DLite  # noqa: E402

RNG = np.random.default_rng(3)
N = 3000


def ridged(n):
    """Ellipsoid with ridges on one half: local shapes repeat across the object (the location-aware case)."""
    d = RNG.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p = d * np.array([1.0, 0.6, 0.45])
    ridge = (p[:, 0] > 0) * 0.04 * np.sin(18 * p[:, 1])
    p = p + ridge[:, None] * d
    return (p + RNG.normal(scale=0.003, size=p.shape)).astype(np.float32)


def view(p, u, tau=0.1):
    c = p - p.mean(0)
    dirs = c / np.linalg.norm(c, axis=1, keepdims=True)
    return p[dirs @ u > tau]


def build_real3d(root, cut_root):
    for cls in ["airplane", "duck"]:
        protos = [ridged(N) @ random_rotation(RNG, 8) for _ in range(4)]
        for i, p in enumerate(protos):
            write_pcd(f"{root}/{cls}/train/{i}_prototype.pcd", p)
            for j in range(2):                                   # 2 real single-view scans per prototype
                u = RNG.normal(size=3); u /= np.linalg.norm(u)
                np.savetxt(f"{cut_root}/{cls}/train_cut/{i}_{j}.asc" if os.makedirs(
                    f"{cut_root}/{cls}/train_cut", exist_ok=True) is None else "", view(p, u) @ random_rotation(RNG))
        for i in range(4):
            u = RNG.normal(size=3); u /= np.linalg.norm(u)
            write_pcd(f"{root}/{cls}/test/{i}_good.pcd", view(ridged(N), u) @ random_rotation(RNG))
        for i in range(4):
            p = ridged(N)
            u = np.array([-0.3, 0.2, 0.9]); u /= np.linalg.norm(u)
            v = view(p, u)
            c = v[np.argmax(v @ u)]
            m = np.linalg.norm(v - c, axis=1) < 0.15
            v[m] += 0.06 * (v[m] - v.mean(0)) / np.linalg.norm(v[m] - v.mean(0), axis=1, keepdims=True)
            gt = m.astype(np.float32)
            v = v @ random_rotation(RNG)
            write_pcd(f"{root}/{cls}/test/{i}_bulge.pcd", v)
            write_gt_txt(f"{root}/{cls}/gt/{i}_bulge.txt", v, gt, comma=False)


def build_shapenet(root):
    for cls in ["ashtray0"]:
        for i in range(4):
            write_pcd(f"{root}/{cls}/train/{i}_template.pcd", ridged(N))
        for i in range(3):
            write_pcd(f"{root}/{cls}/test/{i}_positive.pcd", ridged(N) @ random_rotation(RNG))
        for i in range(3):
            p = ridged(N)
            m = p[:, 0] < -0.8
            p[m, 0] -= 0.08
            write_gt_txt(f"{root}/{cls}/GT/{i}_bulge.txt", p, m.astype(np.float32), comma=True)


def build_mvtec(root):
    import tifffile
    from PIL import Image
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / H
    for split, kind, k in (("train", "good", 6), ("test", "good", 2), ("test", "hole", 2)):
        for i in range(k):
            z = np.zeros((H, W), np.float32) + 0.001 * RNG.normal(size=(H, W))
            blob = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) < 0.08
            z[blob] += 0.2 * np.sqrt(np.clip(0.08 - ((xx - 0.5) ** 2 + (yy - 0.5) ** 2), 0, None))[blob] * 5
            mask = np.zeros((H, W), np.uint8)
            if kind != "good":
                r2 = (xx - 0.55) ** 2 + (yy - 0.45) ** 2
                d = r2 < 0.004
                z[d] -= 0.03 * (1 - r2[d] / 0.004) ** 2            # smooth dent
                mask[d] = 255
            xyz = np.stack([xx, yy, z], -1)
            d_ = f"{root}/bagel/{split}/{kind}"
            os.makedirs(f"{d_}/xyz", exist_ok=True)
            tifffile.imwrite(f"{d_}/xyz/{i:03d}.tiff", xyz)
            if kind != "good":
                os.makedirs(f"{d_}/gt", exist_ok=True)
                Image.fromarray(mask).save(f"{d_}/gt/{i:03d}.png")


def main():
    tmp = os.path.join(ROOT, "tests", "_synthetic_rev")
    shutil.rmtree(tmp, ignore_errors=True)
    r_root, c_root, s_root, m_root = (f"{tmp}/real3d", f"{tmp}/cuts", f"{tmp}/shapenet", f"{tmp}/mvtec")
    build_real3d(r_root, c_root)
    build_shapenet(s_root)
    small = dict(max_nn=10, n_scales=3, num_group=256, group_size=32, smooth_k=6, coreset_ratio=0.25,
                 voxel=0.02, points_budget=4000, device="cpu", seed=0, align="icp", align_poses=3, cuts=2,
                 local_mem=0.10, obj_rule="p99")
    small["cuts"] = 3

    print("[1] channels='rev' (real3d layout + real single-view training scans)")
    cfg = Config(dataset="real3d", data_root=r_root, train_cut_root=c_root, train_cut_with_protos=True,
                 channels="rev", save_maps=f"{tmp}/maps", **small)
    model = Simple3DLite(cfg)
    model.fit("airplane")
    lm = model._local
    vt = np.bincount(lm.vtype, minlength=3)
    print(f"  memory {lm.size} entries; per view type {vt.tolist()}; objects {sorted(set(lm.obj.tolist()))}; "
          f"masks {sorted(model._masks)}; ref coverage {lm.ref_info['ref_coverage']:.2f}")
    assert vt[0] > 0 and vt[1] > 0 and vt[2] > 0, "all three view types must be present"
    assert set(model._masks) >= {"V0", "V01", "V02", "V012", "M2", "V01_eq", "V02_eq", "V012_eq"}
    assert model._masks["V0_eq"].sum() if "V0_eq" in model._masks else True
    owners = model.train_cut_info["train_cut_owners"].split(",")
    print(f"  real scans assigned to objects {owners}")
    known = [o for o in owners if int(o) >= 0]
    exp = [str(i) for i in range(4) for _ in range(2)]
    # every scan whose source is identified must be identified correctly (unknown sources are allowed)
    assert all(o == e for o, e in zip(owners, exp) if int(o) >= 0), (owners, exp)
    print(f"  {len(known)}/8 scans identified, all correctly")
    m = model.evaluate("airplane")
    need = ["glob_cs", "glob_full", "loc05", "loc10", "loc15", "knn64", "knn256", "soft0p5", "soft1", "soft2",
            "cal10", "cal10_k0", "hyb", "loc10_V0", "loc10_V01", "loc10_V02", "loc10_M2", "loc10_V012_eq",
            "glob_full_V0", "h1_loc10", "h1_cal10", "h1_glob_full"]
    missing = [c for c in need if f"{c}_o_auroc" not in m]
    assert not missing, missing
    for c in need:
        print(f"    {c:>15s} O-AUROC {m[c + '_o_auroc']:.3f}")
    rec = model.sample_records[0]
    assert {"reg_res", "reg_res_T0", "n_hyp", "pose_idx", "loc10_p99", "cal10_max"} <= set(rec)
    assert all(k in m for k in ("fit_register_s", "fit_memory_s", "test_register_per_scan_s", "peak_rss_gb"))
    maps = sorted(glob.glob(f"{tmp}/maps/*.npz"))
    assert len(maps) == 3, maps
    z = np.load(maps[0])
    assert {"points", "gt", "loc10", "glob_full", "cal10"} <= set(z.files)
    # loc10 under 'rev' must equal the legacy loc10 of the same run (same features, same rule)
    cfg_leg = Config(**{**cfg.__dict__, "channels": "", "save_maps": ""})
    leg = Simple3DLite(cfg_leg)
    leg.fit("airplane")
    ml = leg.evaluate("airplane")
    print(f"  legacy loc10 {ml['loc10_o_auroc']:.4f} vs rev loc10 {m['loc10_o_auroc']:.4f}")
    assert abs(ml["loc10_o_auroc"] - m["loc10_o_auroc"]) < 1e-9

    print("[2] robustness perturbations")
    for kw in ({"test_crop": 0.6}, {"test_density": 0.5}, {"test_noise": 0.005}, {"n_protos": 2}):
        mm = Simple3DLite(Config(**{**cfg.__dict__, "save_maps": "", **kw}))
        mm.fit("airplane")
        r = mm.evaluate("airplane")
        print(f"    {kw}: loc10 {r['loc10_o_auroc']:.3f}  mean points {np.mean([x['n_points'] for x in mm.sample_records]):.0f}")
        if "test_crop" in kw:
            assert all(x["n_anom_points"] > 0 for x in mm.sample_records if x["label"] == 1)

    print("[3] normal-only development protocol")
    common = ["--max-nn", "10", "--num-group", "256", "--group-size", "32", "--smooth-k", "6", "--coreset", "0.25",
              "--voxel", "0.02", "--points-budget", "4000", "--device", "cpu", "--align", "icp", "--align-poses", "3",
              "--cuts", "3", "--local-mem", "0.10", "--obj-rule", "p99", "--channels", "rev", "--out", f"{tmp}/dev",
              "--n-pseudo", "2", "--folds", "2"]
    for ds, root, extra in (("real3d", r_root, ["--train-cut-root", c_root, "--train-cut-with-protos",
                                                 "--classes", "airplane"]),
                            ("shapenet", s_root, ["--classes", "ashtray0"])):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dev_protocol.py"), "--dataset", ds,
                              "--data-root", root, "--tag", "t"] + common + extra, capture_output=True, text=True)
        print("   ", "\n    ".join(l for l in out.stdout.splitlines() if l.startswith("[dev]")))
        assert out.returncode == 0, out.stderr[-3000:]
        d = json.load(open(f"{tmp}/dev/dev_t_{ds}.json"))
        assert len(d["rows"]) == 2 and all(r["n_test"] == 4 for r in d["rows"])

    print("[4] MVTec 3D-AD loader")
    build_mvtec(m_root)
    from ad3d import datasets as D
    tr = D.load_train(m_root, "mvtec", "bagel")
    te = D.load_test(m_root, "mvtec", "bagel")
    ex = D.load_train_extra(m_root, "mvtec", "bagel", start=4, n=4)
    n_bg = 64 * 64
    print(f"    train {len(tr)} clouds of {[len(t) for t in tr]} pts (from {n_bg} pixels); extra {len(ex)}; "
          f"test {len(te)} ({sum(s.label for s in te)} anomalous, gt points {[int(s.gt.sum()) for s in te]})")
    assert len(tr) == 4 and len(ex) == 2 and len(te) == 4
    assert all(len(t) < 0.5 * n_bg for t in tr), "background plane not removed"
    assert all(s.gt.sum() > 0 for s in te if s.label == 1) and all(s.gt.sum() == 0 for s in te if s.label == 0)
    print("ALL REVISION CHECKS PASSED")


if __name__ == "__main__":
    main()
