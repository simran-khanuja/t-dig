#!/bin/bash
# Score the i.i.d. renders of chosen T2I models with HPSv3, one shard per GPU, then merge.
#
#   T_DIG_ROOT=<repository root> QUALITY_OUT=<new dir> T2I_MODELS=<name+name> \
#     bash tdig/iid_quality.sh
#
# Resources: 1 node with 8 GPUs, about 4 h. Run with bash on an allocated node, or submit with
# your scheduler passing these variables in the job environment.
#
#   QUALITY_OUT  output directory
#   T2I_MODELS    '+'-joined T2I model names (sbatch --export splits on commas)
#   INDEX        image index (default: $T_DIG_RUNS/tdig/iid-vqa-v1/index.jsonl)
#
# Resubmitting continues: scored images are skipped.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
OUT=${QUALITY_OUT:?output directory}
PY=$PY_REWARD
INDEX=${INDEX:-$T_DIG_RUNS/tdig/iid-vqa-v1/index.jsonl}
NAMES=$(echo "${T2I_MODELS:?T2I model names}" | tr '+' ' ')
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir -p "$OUT"
GPUS=($(t_dig_gpus)); NG=${#GPUS[@]}
PIDS=()
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" -u -m tdig.iid_quality score --index "$INDEX" --out "$OUT" --shard "$i" \
    --num-shards "$NG" --t2i-models $NAMES --cache "$HF_HUB_CACHE" > "$OUT/shard-$(printf %02d "$i").log" 2>&1 &
  PIDS+=($!)
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
[ "$FAILED" -eq 0 ] || { echo "a shard failed; see $OUT/shard-*.log. Resubmit to continue."; exit 1; }
"$PY" -m tdig.iid_quality merge --index "$INDEX" --out "$OUT" --num-shards "$NG" --t2i-models $NAMES
