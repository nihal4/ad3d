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
# Env: NGPU (default: detected, else 2)  CONFIGS  SPLIT=1  CLASSES=a,b,c (SPLIT mode: only these categories)  DATASET=real3d|shapenet (default real3d;
#      both modes). e.g. DATASET=shapenet CONFIGS="base" bash scripts/run_parallel.sh <SHAPENET_ROOT> 1 2
set -uo pipefail
ROOT="$1"; shift
SEEDS="${@:-0 1 2}"
NGPU="${NGPU:-$(python -c 'import torch;print(max(1,torch.cuda.device_count()))' 2>/dev/null || echo 2)}"
NCPU="$(nproc)"; THREADS=$(( NCPU / NGPU )); [ "$THREADS" -lt 1 ] && THREADS=1

CUT_ROOT="${CUT_ROOT:-}"   # folder with <cls>/train_cut/* (GLFM/Simple3D "Cut Training Data"); needed by cut-* configs
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
  [mhr6-nn10-pluscut-p99-geo]="--align icp --cuts 4 --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT} --train-cut-with-protos --obj-rule p99 --geo fuse"
  [mhr6-nn10-p99-loc10]="--align icp --cuts 4 --align-poses 6 --max-nn 10 --obj-rule p99 --local-mem 0.10"
  [mhr6-nn10-pluscut-p99-loc10]="--align icp --cuts 4 --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT} --train-cut-with-protos --obj-rule p99 --local-mem 0.10"
  [cut-mhr6-nn10-protos]="--align icp --align-poses 6 --max-nn 10 --train-cut-root ${CUT_ROOT} --train-cut-with-protos"
  [icp-cuts4-nn20]="--align icp --cuts 4 --max-nn 20"
  [mhr6-nn20]="--align icp --cuts 4 --align-poses 6 --max-nn 20"
  [icp-cuts4-nn40]="--align icp --cuts 4 --max-nn 40"
  [mhr6-nn40]="--align icp --cuts 4 --align-poses 6 --max-nn 40"
)
for _c in ${CONFIGS:-}; do case "$_c" in cut-*|*pluscut*) [ -z "$CUT_ROOT" ] && { echo "[!] config $_c needs CUT_ROOT=<folder with <cls>/train_cut>"; exit 1; };; esac; done
CONFIGS="${CONFIGS:-base icp-cuts4 mhr6}"
DATASET="${DATASET:-real3d}"
mkdir -p logs results/parts
echo "[parallel] gpus=$NGPU cpus=$NCPU threads/proc=$THREADS  configs: $CONFIGS  seeds: $SEEDS"

if [ "${SPLIT:-0}" = "1" ]; then
  # interleave categories so heavy ones are spread across GPUs
  CATS=($(PYTHONPATH=src python -c "from ad3d.datasets import get_classes; print(' '.join(get_classes('$DATASET')))"))
  [ "${#CATS[@]}" -gt 0 ] || { echo "[!] could not list categories for $DATASET"; exit 1; }
  PARTIAL=""
  if [ -n "${CLASSES:-}" ]; then IFS=',' read -r -a CATS <<< "$CLASSES"; PARTIAL="--allow-partial"; fi   # category subset (screening)
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
    python scripts/merge_categories.py --dataset "$DATASET" $PARTIAL "${name}-s${s}" results/parts/${name}-s${s}-part*_${DATASET}_*[0-9].csv
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
