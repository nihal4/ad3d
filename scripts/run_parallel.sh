#!/usr/bin/env bash
# Run the Real3D-AD queue on several GPUs in parallel (Kaggle: 2x T4).
# The expensive part (open3d FPFH + RANSAC/ICP) is CPU-bound, so we use one
# process per GPU with OMP_NUM_THREADS = cores / n_gpus. Do NOT launch more
# processes than GPUs - they just fight for the ~4 CPU cores.
#
# MODE 1 (default): jobs = CONFIGS x SEEDS, dealt round-robin to the GPUs.
#   CONFIGS="base icp-cuts4 mhr6" bash scripts/run_parallel.sh <REAL_ROOT> 1 2
# MODE 2: split ONE config+seed by category across the GPUs, then merge.
#   SPLIT=1 CONFIGS="mhr6" bash scripts/run_parallel.sh <REAL_ROOT> 0
#
# MEASURED (Kaggle T4x2, 4 vCPU, base config): 2 concurrent jobs ~2h15 each vs ~1h25 alone,
# i.e. only ~1.25x faster than sequential - the CPU, not the GPU, is the bottleneck.
#
# Env: NGPU (default: detected, else 2)  CONFIGS  SPLIT=1  DATASET=real3d|shapenet (default real3d;
#      shapenet supports MODE 1 only). e.g. DATASET=shapenet CONFIGS="base" bash scripts/run_parallel.sh <SHAPENET_ROOT> 1 2
set -uo pipefail
ROOT="$1"; shift
SEEDS="${@:-0 1 2}"
NGPU="${NGPU:-$(python -c 'import torch;print(max(1,torch.cuda.device_count()))' 2>/dev/null || echo 2)}"
NCPU="$(nproc)"; THREADS=$(( NCPU / NGPU )); [ "$THREADS" -lt 1 ] && THREADS=1

declare -A CFG=(
  [base]=""
  [cuts4]="--cuts 4"
  [icp]="--align icp"
  [icp-cuts4]="--align icp --cuts 4"
  [mhr6]="--align icp --cuts 4 --align-poses 6"
  [icp-cuts4-div]="--align icp --cuts 4 --cuts-diverse"
  [mhr6-div]="--align icp --cuts 4 --cuts-diverse --align-poses 6"

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
CONFIGS="${CONFIGS:-base icp-cuts4 mhr6}"
DATASET="${DATASET:-real3d}"
if [ "$DATASET" != "real3d" ] && [ "${SPLIT:-0}" = "1" ]; then echo "SPLIT=1 supports real3d only"; exit 1; fi
mkdir -p logs results/parts
echo "[parallel] gpus=$NGPU cpus=$NCPU threads/proc=$THREADS  configs: $CONFIGS  seeds: $SEEDS"

if [ "${SPLIT:-0}" = "1" ]; then
  # interleave categories so heavy ones are spread across GPUs
  CATS=(airplane candybar car chicken diamond duck fish gemstone seahorse shell starfish toffees)
  for name in $CONFIGS; do for s in $SEEDS; do
    pids=()
    for g in $(seq 0 $((NGPU-1))); do
      list=""; for i in "${!CATS[@]}"; do [ $((i % NGPU)) -eq "$g" ] && list="${list:+$list,}${CATS[$i]}"; done
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$THREADS python run.py --dataset "$DATASET" \
        --data-root "$ROOT" ${CFG[$name]} --seed "$s" --classes "$list" \
        --out results/parts --tag "${name}-s${s}-part${g}" > "logs/${name}-s${s}-part${g}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    python scripts/merge_categories.py "${name}-s${s}" results/parts/${name}-s${s}-part*_real3d_*.csv
  done; done
  exit 0
fi

# MODE 1: build the job list, deal round-robin
jobs=(); for name in $CONFIGS; do for s in $SEEDS; do jobs+=("$name:$s"); done; done
for g in $(seq 0 $((NGPU-1))); do
  (
    for i in "${!jobs[@]}"; do
      [ $((i % NGPU)) -eq "$g" ] || continue
      name="${jobs[$i]%%:*}"; s="${jobs[$i]##*:}"
      echo "[gpu$g] start $name seed $s"
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$THREADS python run.py --dataset "$DATASET" \
        --data-root "$ROOT" ${CFG[$name]} --seed "$s" --tag "${name}-s${s}" \
        > "logs/${name}-s${s}.log" 2>&1 && echo "[gpu$g] done  $name seed $s" || echo "[gpu$g] FAILED $name seed $s (see logs/)"
    done
  ) &
done
wait
echo "[parallel] all done -> python scripts/aggregate_seeds.py results/"
