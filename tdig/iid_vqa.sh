#!/bin/bash
# Score every i.i.d. render in an existing index with VQAScore, one shard per GPU, then merge.
#
#   T_DIG_ROOT=<repository root> VQA_OUT=<dir holding index.jsonl> bash tdig/iid_vqa.sh
#
# Resources: 1 node with 8 GPUs, about 8 h. Run with bash on an allocated node, or submit with
# your scheduler passing these variables in the job environment.
#
#   VQA_OUT      directory holding index.jsonl
#   VQA_PYTHON   interpreter with torch and t2v_metrics (default: $PY_VQA)
#   HF_CACHE     Hugging Face hub cache holding zhiqiulin/clip-flant5-xxl (default: $HF_HUB_CACHE)
#   NUM_SHARDS   shards in total; must equal the usable GPUs (default: their number)
#
# Rerunning with the same VQA_OUT continues unfinished shards without rescoring. A shard is a
# stride over the index, so NUM_SHARDS must stay the same across reruns.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
OUT=${VQA_OUT:?directory holding index.jsonl}
PY=${VQA_PYTHON:-$PY_VQA}
CACHE=${HF_CACHE:-$HF_HUB_CACHE}
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PATH="$(dirname "$PY"):$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

GPUS=($(t_dig_gpus)); NG=${#GPUS[@]}
NUM_SHARDS=${NUM_SHARDS:-$NG}
[ "$NUM_SHARDS" -eq "$NG" ] || { echo "NUM_SHARDS=$NUM_SHARDS but $NG GPUs are usable"; exit 1; }
[ -f "$OUT/index.jsonl" ] || { echo "no index at $OUT/index.jsonl; run: python -m tdig.iid_vqa index"; exit 1; }

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 \
  > "$OUT/gpu-utilization-${T_DIG_JOB_ID:-${SLURM_JOB_ID:-local}}.csv" &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" -m tdig.iid_vqa score --out "$OUT" --shard "$i" \
    --num-shards "$NUM_SHARDS" --cache "$CACHE" > "$OUT/shard-$(printf %02d "$i").log" 2>&1 &
  PIDS+=($!)
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
[ "$FAILED" -eq 0 ] || { echo "a shard failed; see $OUT/shard-*.log. Resubmit to continue."; exit 1; }
"$PY" -m tdig.iid_vqa merge --out "$OUT" --num-shards "$NUM_SHARDS"
