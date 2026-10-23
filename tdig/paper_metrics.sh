#!/bin/bash
# Compute the paper's metrics for an indexed directory (sets.jsonl and index.jsonl), then merge.
# Each GPU runs one VQAScore shard (PY_VQA) and one shard of every other scorer (PY_REWARD) at once.
#
#   T_DIG_ROOT=<repository root> METRICS_OUT=<indexed directory> bash tdig/paper_metrics.sh
#
# Resources: 1 node with 8 GPUs, about 12 h. Run with bash on an allocated node, or submit with
# your scheduler passing these variables in the job environment.
#
# Weights: DreamSim, DINOv2 and the LAION aesthetic head from $T_DIG_MODELS; TPIPS and the rest
# from $HF_HUB_CACHE. Resubmitting with the same METRICS_OUT continues unfinished shards. After
# adding sets, rerun the index step, delete done-*.json and metrics-done-*.json, and resubmit;
# scored sets are kept.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
OUT=${METRICS_OUT:?indexed directory}
CACHE=$HF_HUB_CACHE
cd "$ROOT"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
GPUS=($(t_dig_gpus)); NG=${#GPUS[@]}
[ -f "$OUT/index.jsonl" ] && [ -f "$OUT/sets.jsonl" ] || { echo "run an index subcommand of tdig.paper_metrics first"; exit 1; }
PIDS=()
for i in $(seq 0 $((NG - 1))); do
  n=$(printf %02d "$i")
  PATH="$(dirname "$PY_VQA"):$PATH" CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY_VQA" -m tdig.iid_vqa score --out "$OUT" \
    --shard "$i" --num-shards "$NG" --cache "$CACHE" > "$OUT/vqa-shard-$n.log" 2>&1 &
  PIDS+=($!)
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY_REWARD" -m tdig.paper_metrics score --out "$OUT" \
    --shard "$i" --num-shards "$NG" --cache "$CACHE" \
    --dreamsim-cache "$T_DIG_MODELS/dreamsim_cache" --dino-path "$T_DIG_MODELS/dino" \
    --aesthetic-head "$T_DIG_MODELS/aesthetic/sac_logos_ava1_l14_linearMSE.pth" \
    --tpips-checkpoint "sywang/TPIPS-Embed-Qwen3VL-8B@86a1493c60121e656a1c075bc7aed88f25fcd4d4" \
    > "$OUT/metrics-shard-$n.log" 2>&1 &
  PIDS+=($!)
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
[ "$FAILED" -eq 0 ] || { echo "a shard failed; see $OUT/*-shard-*.log. Resubmit to continue."; exit 1; }
"$PY_VQA" -m tdig.iid_vqa merge --out "$OUT" --num-shards "$NG" > /dev/null
"$PY_VQA" -m tdig.paper_metrics merge --out "$OUT" --num-shards "$NG"
