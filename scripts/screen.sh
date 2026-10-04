#!/usr/bin/env bash
# Hyper-parameter SCREEN on a few categories (default: the three worst ones: car, seahorse, chicken).
# Base setting for every preset: --max-nn 10 (= base-nn10, no registration). The reference numbers for these
# categories already exist in results/base-nn10-s0_real3d_*.csv (deterministic), so the reference is NOT re-run.
# One config per GPU at a time, round-robin. Tags: scr-<preset>-s<seed>.
#   PRESETS="g64 g256 ng1024 ng4096 sm6 sm24" bash scripts/screen.sh <REAL_ROOT>
# Env: PRESETS  CLASSES (comma list)  SEED (default 0)  NGPU
# NOTE: screening on 3 categories can overfit them. A winner MUST be validated on all 12 categories before it is adopted.
set -uo pipefail
ROOT="$1"
SEED="${SEED:-0}"
CLASSES="${CLASSES:-car,seahorse,chicken}"
NGPU="${NGPU:-$(python -c 'import torch;print(max(1,torch.cuda.device_count()))' 2>/dev/null || echo 2)}"
THREADS=$(( $(nproc) / NGPU )); [ "$THREADS" -lt 1 ] && THREADS=1

declare -A P=(
  [g64]="--group-size 64"
  [g256]="--group-size 256"
  [ng1024]="--num-group 1024"
  [ng4096]="--num-group 4096"
  [sm6]="--smooth-k 6"
  [sm24]="--smooth-k 24"
  [cs20]="--coreset 0.2"
  [vx005]="--voxel 0.005"
  [vx02]="--voxel 0.02"
  [top32]="--topk 32"
)
PRESETS="${PRESETS:-g64 g256 ng1024 ng4096 sm6 sm24}"
mkdir -p logs results
echo "[screen] gpus=$NGPU threads/proc=$THREADS classes=$CLASSES seed=$SEED presets: $PRESETS"
i=0; declare -a QUEUE
for n in $PRESETS; do
  if [ -z "${P[$n]+x}" ]; then echo "[screen] unknown preset $n (skipped)"; continue; fi
  g=$(( i % NGPU )); QUEUE[$g]="${QUEUE[$g]:-} $n"; i=$((i+1))
done
for g in $(seq 0 $((NGPU-1))); do
  (
    for n in ${QUEUE[$g]:-}; do
      echo "[gpu$g] start $n"
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$THREADS python run.py --dataset real3d \
        --data-root "$ROOT" --max-nn 10 ${P[$n]} --seed "$SEED" --classes "$CLASSES" \
        --out results --tag "scr-${n}-s${SEED}" > "logs/scr-${n}-s${SEED}.log" 2>&1
      echo "[gpu$g] done  $n (exit $?)"
    done
  ) &
done
wait
echo "[screen] all done"
