#!/bin/bash
# Render i.i.d. samples of a prompt list, then measure both row thresholds: VQAScore
# (align_thresh) and HPSv3 (quality_thresh). Each stage runs one shard per GPU.
#
#   T_DIG_ROOT=<repository root> PROMPTS=<prompts.jsonl> REWARD_CONFIG=<reward-config.json> \
#     IID_OUT=<new dir> bash tdig/render_iid.sh
#
# Resources: 1 node with 8 GPUs, about 6 h. Run with bash on an allocated node, or submit with
# your scheduler passing these variables in the job environment.
#
#   PROMPTS        JSON lines with prompt_id and base_prompt
#   REWARD_CONFIG  reward-config.json of a run; sets the T2I model and the render settings
#   IID_OUT        output directory
#
# Outputs in IID_OUT: images/, index.jsonl, thresholds.jsonl (iid_vqa merge) and quality.jsonl
# (iid_quality merge). Resubmitting continues each stage.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT; OUT=${IID_OUT:?output directory}
PY=$PY_REWARD
cd "$ROOT"; export PYTHONPATH="$ROOT"
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"
GPUS=($(t_dig_gpus)); NG=${#GPUS[@]}
T2I_MODEL=$("$PY" -c "import json;print(json.load(open('${REWARD_CONFIG:?reward config}'))['renderer'])")
stage() { # name, command prefix run once per GPU with --shard/--num-shards appended
  local name=$1; shift; local pids=() failed=0
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$@" --shard "$i" --num-shards "$NG" > "$OUT/$name-$(printf %02d "$i").log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [ "$failed" -eq 0 ] || { echo "stage $name failed; see $OUT/$name-*.log"; exit 1; }
}
stage render "$PY" -u -m tdig.render_iid --prompts "${PROMPTS:?prompt list}" --reward-config "$REWARD_CONFIG" --out "$OUT"
cat "$OUT"/index-*.jsonl > "$OUT/index.jsonl"
PATH="$(dirname "$PY_VQA"):$PATH" stage vqa "$PY_VQA" -m tdig.iid_vqa score --out "$OUT" --cache "$HF_HUB_CACHE"
"$PY_VQA" -m tdig.iid_vqa merge --out "$OUT" --num-shards "$NG"
stage quality "$PY" -u -m tdig.iid_quality score --index "$OUT/index.jsonl" --out "$OUT" --t2i-models "$T2I_MODEL" --cache "$HF_HUB_CACHE"
"$PY" -m tdig.iid_quality merge --index "$OUT/index.jsonl" --out "$OUT" --num-shards "$NG" --t2i-models "$T2I_MODEL"
