#!/usr/bin/env bash
# Multi-seed driver for the Real3D-AD ablation (Direction F: >=3 seeds).
# Usage: bash scripts/run_seeds.sh <real3d-root> [seeds...]   (default seeds: 0 1 2)
# Writes results/<config>-s<seed>_real3d_*.csv ; never overwrites (timestamped).
set -euo pipefail
ROOT="$1"; shift
SEEDS="${@:-0 1 2}"
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
