#!/usr/bin/env python
"""Plain-text report of the normal-only development runs (step 16): timing, per-fold provenance of the pseudo-tests,
real-scan assignment, AUROC of every channel per category-fold, the selection table and log errors.
Printed AND written to --out, so the content survives in the notebook log even if the session ends.
Usage: python scripts/report_dev.py [--base /kaggle/working/ad3d] [--out /kaggle/working/report_16.txt]
Never raises."""
import argparse
import contextlib
import glob
import io
import json
import os

import numpy as np

def report_16(BASE="/kaggle/working/ad3d", OUT="/kaggle/working/report_16.txt"):
    from scipy.stats import rankdata
    def auc(y, s):
        y = np.asarray(y); s = np.asarray(s, float); n1 = int(y.sum()); n0 = len(y) - n1
        if n1 == 0 or n0 == 0: return float("nan")
        r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tl = f"{BASE}/logs/timing.log"
        print("=== TIMING ===\n" + (open(tl).read() if os.path.exists(tl) else "(none)"))
        files = sorted(glob.glob(f"{BASE}/results/parts/dev_*.json"))
        print(f"=== DEV FILES ({len(files)}) ===")
        POOL = ("max", "p99", "p95", "mean", "top1pct")
        for f in files:
            try:
                d = json.load(open(f)); ds, k = d["config"]["dataset"], d["config"]["max_nn"]
                print(f"\n### FILE {os.path.basename(f)} dataset={ds} k={k} rows={len(d['rows'])} units={len(d['records'])}")
                for r in d["rows"]:
                    t = r.get("fit_register_s", 0) + r.get("fit_memory_s", 0)
                    print(f"ROW|{ds}|{k}|{r['category']}|{r['fold']}|src={r.get('pseudo_source')}|n_test={r.get('n_test')}"
                          f"|in_mem={r.get('train_cut_in_memory')}|unknown={r.get('train_cut_unknown')}"
                          f"|owners={r.get('train_cut_owners')}|t_fit={t:.0f}s")
                    for a in (r.get("train_cut_assign") or []):
                        rm = ",".join(f"{x:.4g}" for x in a.get("rmse", []))
                        print(f"ASSIGN|{ds}|{k}|{r['category']}|{r['fold']}|{a.get('scan')}|owner={a.get('owner')}"
                              f"|ratio={a.get('ratio_2nd_best', float('nan')):.3f}|rmse={rm}")
                keys = None
                for unit, recs in d["records"].items():
                    if keys is None:
                        keys = sorted(c for c in recs[0] if c.endswith(POOL) and not c.startswith(("n_", "reg_")))
                        print("KEYS|" + "|".join(keys))
                    y = [x["label"] for x in recs]
                    print(f"AUC|{ds}|{k}|{unit}|" + "|".join(
                        f"{auc(y, [x.get(c, np.nan) for x in recs]):.3f}" for c in keys))
            except Exception as e:
                print(f"[!] could not read {f}: {type(e).__name__} {e}")
        sel = f"{BASE}/results/selection.md"
        print("\n=== SELECTION ===\n" + (open(sel).read() if os.path.exists(sel) else "(16e has not run yet)"))
        for lg in sorted(glob.glob(f"{BASE}/logs/*.log")):
            if lg.endswith("timing.log"): continue
            L = open(lg, errors="ignore").read().splitlines()
            bad = [l for l in L if "Error" in l or "Traceback" in l]
            print(f"\n=== LOG {os.path.basename(lg)}: {len(L)} lines, {len(bad)} error lines ===")
            for l in [l for l in L if "MVTec" in l or "loader" in l][:5] + bad[:20] + L[-4:]: print(l)
        print("\n=== END OF REPORT ===")
    text = buf.getvalue()
    try: open(OUT, "w").write(text)
    except Exception as e: text += f"\n[!] could not save {OUT}: {e}"
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/kaggle/working/ad3d")
    ap.add_argument("--out", default="/kaggle/working/report_16.txt")
    a = ap.parse_args()
    try:
        report_16(a.base, a.out)
    except Exception as e:
        print("[!] report failed:", type(e).__name__, e)
