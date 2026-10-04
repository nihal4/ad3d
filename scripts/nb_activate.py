#!/usr/bin/env python
"""Notebook convention: ONLY the cells of the current step are active, every other code cell is fully commented.
So the user can simply press "Run All".

    python scripts/nb_activate.py kaggle_ad3d.ipynb "STEP 5" "%pip install" "# ---- get the ad3d" "# ---- dataset roots" "# 4a" "# 5a" "# 5b" "# 5c"

Arguments after the notebook: first the step label (shown in the banner cell), then the prefixes of the code cells
to activate (matched against the start of the cell source). In active cells, lines of the form "# !..." and
"# CUT_ROOT = ..." are uncommented; all other code cells get every non-comment line prefixed with "# ".
"""
import json
import sys

BANNER = "<!-- RUN-BANNER -->"


def comment_all(src: str) -> str:
    out = []
    for line in src.splitlines():
        out.append(line if (not line.strip() or line.lstrip().startswith("#")) else "# " + line)
    return "\n".join(out) + ("\n" if src.endswith("\n") else "")


def uncomment_runs(src: str) -> str:
    out = []
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith("# !") or s.startswith("# CUT_ROOT =") or s.startswith("# %"):
            line = line.replace("# ", "", 1)
        elif s.startswith("# get_ipython()"):
            line = line.replace("# ", "", 1)
        out.append(line)
    return "\n".join(out) + ("\n" if src.endswith("\n") else "")


def main(path, label, prefixes):
    nb = json.load(open(path))
    nb["cells"] = [c for c in nb["cells"] if BANNER not in "".join(c["source"])]
    active = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # match on the original (possibly commented) first line
        first = src.lstrip()
        first_unc = first[2:] if first.startswith("# %") or first.startswith("# get_ipython") else first
        if any(first.startswith(p) or first_unc.startswith(p) for p in prefixes):
            if first.startswith("# %"):
                src = src.replace("# %", "%", 1)
            src = uncomment_runs(src)
            active.append(src.splitlines()[0][:70])
        else:
            src = comment_all(src)
        c["source"] = src.splitlines(True)
        c["outputs"], c["execution_count"] = [], None
    banner = {"cell_type": "markdown", "metadata": {}, "source": [
        f"{BANNER}\n", f"# ▶ CURRENT RUN: {label} — just press **Run All**\n",
        "Only the cells of this step are active; every other code cell is commented out with `#`.\n",
        "Active cells, in order:\n"] + [f"- `{a}`\n" for a in active]}
    nb["cells"].insert(1, banner)
    json.dump(nb, open(path, "w"), indent=1)
    print("active cells:"); [print("  ", a) for a in active]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
