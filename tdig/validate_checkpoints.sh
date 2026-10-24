#!/bin/bash
# Score checkpoints of one training run on held-out prompt sets. Of the usable GPUs
# (CUDA_VISIBLE_DEVICES, else all; at least three):
#   first        vLLM policy (tdig.validate_checkpoints)
#   second       VQAScore service
#   the others   reward services: T2I model, TPIPS and the tracked scorers (six on eight GPUs)
#
#   T_DIG_ROOT=<repository root> TRAIN_RUN=<training run dir> VAL_OUT=<new dir> STEPS=<steps> \
#     SETS='<name=parquet;name=parquet>' bash tdig/validate_checkpoints.sh
#
# Resources: 1 node with 8 GPUs, about 24 h. Run with bash on an allocated node, or submit with
# your scheduler passing these variables in the job environment.
#
#   TRAIN_RUN      training run directory
#   VAL_OUT        output directory
#   STEPS          'all', 'latest', 'every:<N>', or '+'-joined integers such as 0+300+600+1200;
#                  0 is the untrained policy
#   SETS           ';'-separated name=parquet entries (sbatch --export splits on commas, so neither
#                  STEPS nor SETS may contain one). Name a set heldout* or probe* when its rows
#                  carry their own align_thresh: paper_metrics.py keeps the per-row threshold
#                  (fidelity_penalty, reward) only for sets named so.
#   LIMIT          score only the first N rows of each set
#   CONCURRENCY    completions scored at once (default: 48)
#   VAL_STEPS, VAL_RES  override the validation render steps and resolution
#   PAPER_METRICS  0 skips the paper metrics (default: 1)
#
# LIMIT, VAL_STEPS, VAL_RES and PAPER_METRICS=0 together give a quick selection pass. The services
# run the training run's reward-config.json with save_images=true and the render batch sizes the
# training job selected. Images land in VAL_OUT/reward-<i>/images/<request id>/. When validation
# finishes, the job runs tdig/paper_metrics.py on the saved images into VAL_OUT/paper-metrics/.
# Resubmitting continues: scored steps are skipped and the reward services reuse stored results.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
RUN=${TRAIN_RUN:?training run directory}
OUT=${VAL_OUT:?output directory}
: "${STEPS:?all, latest, every:N or +-joined steps}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT/staging"
GPUS=($(t_dig_gpus))
[ "${#GPUS[@]}" -ge 3 ] || { echo "need at least 3 GPUs (policy, VQAScore, one reward service)"; exit 1; }
NR=$(( ${#GPUS[@]} - 2 ))
VQA_PORT=18740
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT

CACHE=$("$PY_REWARD" -c "import json;print(json.load(open('$RUN/reward-config.json'))['cache'])")
VQA_MODEL=$("$PY_REWARD" -c "import json;print(json.load(open('$RUN/reward-config.json'))['vqa_model'])")
PATH="$(dirname "$PY_VQA"):$PATH" CUDA_VISIBLE_DEVICES=${GPUS[1]} "$PY_VQA" -m axis_grpo.training.vqa_service \
  --cache "$CACHE" --staging "$OUT/staging" --port $VQA_PORT --model "$VQA_MODEL" --batch-size 16 --execute \
  > "$OUT/vqa.log" 2>&1 &
PIDS+=($!)

wait_healthy() {  # url, seconds
  for _ in $(seq 1 $(( $2 / 10 ))); do
    curl -sf --max-time 5 "$1/health" | grep -q '"loaded": true' && return 0
    for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "a service exited while waiting for $1"; return 1; }; done
    sleep 10
  done
  echo "$1 never became healthy"; return 1
}
# Log GPU use every 30 s.
( while true; do echo "T $(date +%s)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 30; done ) \
  > "$OUT/gpu-utilization.log" 2>&1 &
PIDS+=($!)

# The T2I model services start while the VQAScore model is still loading; both take minutes.
ENDPOINTS=()
for i in $(seq 0 $((NR - 1))); do
  "$PY_REWARD" - <<PYEOF
import json
from pathlib import Path
c = json.load(open("$RUN/reward-config.json"))
selected = Path("$RUN/node-0/selected-batching.json")
if selected.exists():
    s = json.load(open(selected)); c["render_batch"], c["val_render_batch"] = s["render_batch"], s["val_render_batch"]
c.update(staging="$OUT/staging", vqa_url="http://127.0.0.1:$VQA_PORT", save_images=True)
# VAL_STEPS / VAL_RES override the run's validation render settings for a quick selection pass;
# the paper's numbers use the run's own settings. The override is recorded in reward-<i>.json.
if "${VAL_STEPS:-}": c["render"]["val"]["steps"] = int("${VAL_STEPS:-0}")
if "${VAL_RES:-}": c["render"]["val"]["resolution"] = int("${VAL_RES:-0}")
json.dump(c, open("$OUT/reward-$i.json", "w"), indent=2)
PYEOF
  PORT=$((18750 + i))
  CUDA_VISIBLE_DEVICES=${GPUS[$((i + 2))]} "$PY_REWARD" -m axis_grpo.training.service --config "$OUT/reward-$i.json" \
    --out "$OUT/reward-$i" --host 0.0.0.0 --port $PORT --execute > "$OUT/reward-$i.log" 2>&1 &
  PIDS+=($!)
  ENDPOINTS+=(--endpoint "http://127.0.0.1:$PORT")
done
wait_healthy "http://127.0.0.1:$VQA_PORT" 3600
for i in $(seq 0 $((NR - 1))); do wait_healthy "http://127.0.0.1:$((18750 + i))" 3600; done

SET_ARGS=()
IFS=';' read -ra ITEMS <<< "${SETS:?name=parquet;name=parquet}"
for item in "${ITEMS[@]}"; do SET_ARGS+=(--set "$item"); done
STEP_ARGS=(${STEPS//+/ })
PATH="$(dirname "$PY_TRAIN"):$PATH" CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY_TRAIN" -u -m tdig.validate_checkpoints \
  --run "$RUN" "${SET_ARGS[@]}" --steps "${STEP_ARGS[@]}" "${ENDPOINTS[@]}" --out "$OUT" \
  ${LIMIT:+--limit "$LIMIT"} ${CONCURRENCY:+--concurrency "$CONCURRENCY"} --execute

# Paper metrics on the saved images, after stopping the services to free their GPUs.
# PAPER_METRICS=0 skips this; tdig/paper_metrics.sh can run it later.
if [ "${PAPER_METRICS:-1}" = "1" ]; then
  cleanup; PIDS=(); sleep 30
  "$PY_TRAIN" -m tdig.paper_metrics index-validation --val-out "$OUT" --out "$OUT/paper-metrics" > "$OUT/paper-metrics-index.log"
  rm -f "$OUT"/paper-metrics/done-*.json "$OUT"/paper-metrics/metrics-done-*.json
  T_DIG_ROOT="$ROOT" METRICS_OUT="$OUT/paper-metrics" bash "$ROOT/tdig/paper_metrics.sh"
fi
