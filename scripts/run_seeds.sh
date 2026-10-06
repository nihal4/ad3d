#!/usr/bin/env bash
# Multi-seed driver for the Real3D-AD ablation (Direction F: >=3 seeds).
# Usage: bash scripts/run_seeds.sh <real3d-root> [seeds...]   (default seeds: 0 1 2)
# Writes results/<config>-s<seed>_real3d_*.csv ; never overwrites (timestamped).
set -euo pipefail
ROOT="$1"; shift
SEEDS="${@:-0 1 2}"
CUT_ROOT="${CUT_ROOT:-}"   # folder with <cls>/train_cut/* (GLFM/Simple3D "Cut Training Data"); needed by cut-* configs
declare -A CFG=(
  [base]=""
  [cuts4]="--cuts 4"
  [icp]="--align icp"
  [icp-cuts4]="--align icp --cuts 4"
  [mhr6]="--align icp --cuts 4 --align-poses 6"
  [mhr6-div]="--align icp --cuts 4 --cuts-diverse --align-poses 6"
  [icp-cuts4-div]="--align icp --cuts 4 --cuts-diverse"

  [base-nn40]="--max-nn 40"
  [base-nn60]="--max-nn 60"
  [base-nn30]="--max-nn 30"
  [base-nn20]="--max-nn 20"
  [base-nn10]="--max-nn 10"
  [base-nn6]="--max-nn 6"
  [icp-cuts4-nn10]="--align icp --cuts 4 --max-nn 10"
  [mhr6-nn10]="--align icp --cuts 4 --align-poses 6 --max-nn 10"
  [base-nn10-mean]="--max-nn 10 --topk 0"
  [cut-nn10]="--max-nn 10 --train-cut-root ${CUT_ROOT}"
  [cut-icp-nn10]="--align icp --max-nn 10 --train-cut-root ${CUT_ROOT}"
  [cut-mhr6-nn10]="--align icp --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT}"
  [mhr6-nn10-pluscut]="--align icp --cuts 4 --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT} --train-cut-with-protos"
  [base-nn10-p99]="--max-nn 10 --obj-rule p99"
  [mhr6-nn10-p99]="--align icp --cuts 4 --align-poses 6 --max-nn 10 --obj-rule p99"
  [mhr6-nn10-pluscut-p99]="--align icp --cuts 4 --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT} --train-cut-with-protos --obj-rule p99"
  [mhr6-nn40-pluscut-p99]="--align icp --cuts 4 --align-poses 6 --max-nn 40 --train-cut-root ${CUT_ROOT} --train-cut-with-protos --obj-rule p99"
  [mhr6-nn100-pluscut-p99]="--align icp --cuts 4 --align-poses 6 --max-nn 100 --train-cut-root ${CUT_ROOT} --train-cut-with-protos --obj-rule p99"
  [mhr6-nn10-pluscut-p99-g4096]="--align icp --cuts 4 --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT} --train-cut-with-protos --obj-rule p99 --num-group 4096 --coreset 0.05"
  [mhr6-nn20-pluscut-p99-vx007]="--align icp --cuts 4 --align-poses 6 --max-nn 20 --voxel 0.007 --train-cut-root ${CUT_ROOT} --train-cut-with-protos --obj-rule p99"
  [cut-mhr6-nn10-protos]="--align icp --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT} --train-cut-with-protos"
  [icp-cuts4-nn20]="--align icp --cuts 4 --max-nn 20"
  [mhr6-nn20]="--align icp --cuts 4 --align-poses 6 --max-nn 20"
  [icp-cuts4-nn40]="--align icp --cuts 4 --max-nn 40"
  [mhr6-nn40]="--align icp --cuts 4 --align-poses 6 --max-nn 40"
)
# CONFIGS env var picks a subset, e.g. CONFIGS="icp-cuts4-div mhr6-div" bash scripts/run_seeds.sh ROOT 0 1 2
# order matters: cheapest first so a disconnect still leaves usable rows
for name in ${CONFIGS:-base cuts4 icp icp-cuts4 mhr6}; do
  for s in $SEEDS; do
    python run.py --dataset real3d --data-root "$ROOT" ${CFG[$name]} \
        --seed "$s" --tag "${name}-s${s}"
  done
done
