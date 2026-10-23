#!/bin/bash
# Score indexed baseline image sets with VQAScore and TPIPS diversity, then merge. Each GPU runs
# one VQAScore shard and one TPIPS shard at once.
#
#   T_DIG_ROOT=<repository root> SCORE_OUT=<dir holding index.jsonl and sets.jsonl> \
#     bash tdig/score_baselines.sh
#
# Resources: 1 node with 8 GPUs, about 6 h. Run with bash on an allocated node, or submit with
# your scheduler passing these variables in the job environment.
#
#   SCORE_OUT         directory written by `python -m tdig.score_baselines index`
#   TPIPS_CHECKPOINT  TPIPS snapshot or repo@revision
#                     (default: sywang/TPIPS-Embed-Qwen3VL-8B@86a1493c60121e656a1c075bc7aed88f25fcd4d4)
#
# Resubmitting with the same SCORE_OUT continues unfinished shards.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
OUT=${SCORE_OUT:?directory holding index.jsonl and sets.jsonl}
PY_TPIPS=$PY_REWARD
CACHE=$HF_HUB_CACHE
TPIPS=${TPIPS_CHECKPOINT:-sywang/TPIPS-Embed-Qwen3VL-8B@86a1493c60121e656a1c075bc7aed88f25fcd4d4}
cd "$ROOT"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
GPUS=($(t_dig_gpus)); NG=${#GPUS[@]}
[ -f "$OUT/index.jsonl" ] && [ -f "$OUT/sets.jsonl" ] || { echo "run: python -m tdig.score_baselines index"; exit 1; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 \
  > "$OUT/gpu-utilization-${T_DIG_JOB_ID:-${SLURM_JOB_ID:-local}}.csv" &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT
PIDS=()
for i in $(seq 0 $((NG - 1))); do
  n=$(printf %02d "$i")
  PATH="$(dirname "$PY_VQA"):$PATH" CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY_VQA" -m tdig.iid_vqa score --out "$OUT" \
    --shard "$i" --num-shards "$NG" --cache "$CACHE" > "$OUT/vqa-shard-$n.log" 2>&1 &
  PIDS+=($!)
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY_TPIPS" -m tdig.score_baselines tpips --out "$OUT" \
    --shard "$i" --num-shards "$NG" --checkpoint "$TPIPS" --cache "$CACHE" > "$OUT/tpips-shard-$n.log" 2>&1 &
  PIDS+=($!)
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
[ "$FAILED" -eq 0 ] || { echo "a shard failed; see $OUT/*-shard-*.log. Resubmit to continue."; exit 1; }
"$PY_VQA" -m tdig.iid_vqa merge --out "$OUT" --num-shards "$NG" > /dev/null
"$PY_VQA" -m tdig.score_baselines merge --out "$OUT" --num-shards "$NG"
