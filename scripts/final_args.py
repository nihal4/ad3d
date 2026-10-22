#!/usr/bin/env python
"""Selection json (scripts/select_config.py) -> run.py flags of the frozen final configuration.

The final runs always use --channels rev, so every rule is still reported; the selected rule is the headline channel
(written to the run's csv as <rule>_o_auroc). Views decide which training views enter the memory:
  V0 complete prototypes only | V01 + simulated cuts | V02 + real single-view scans | V012 both | M2 both, 2 cuts each
Real scans are only used where the benchmark provides them (Real3D-AD via GLFM)."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("--dataset", default="real3d")
    ap.add_argument("--cut-root", default="")
    a = ap.parse_args()
    s = json.load(open(a.selection))
    views = s["views"]
    cuts = {"V0": 0, "V02": 0, "M2": 2}.get(views, 4)
    real = views in ("V02", "V012", "M2") and a.dataset == "real3d" and a.cut_root
    args = ["--align icp", f"--align-poses {6 if s.get('mhr', True) else 1}", f"--cuts {cuts}",
            f"--max-nn {s['max_nn']}", f"--obj-rule {s['pooling']}", "--local-mem 0.10", "--channels rev"]
    if real:
        args += [f"--train-cut-root {a.cut_root}", "--train-cut-with-protos"]
    print(" ".join(args))


if __name__ == "__main__":
    main()
