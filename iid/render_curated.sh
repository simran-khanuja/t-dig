#!/bin/bash
# Generate one T2I model's images of a prepared curated i.i.d. run (python -m iid.curated prepare):
# verify the weights once, run the eight shards over the visible GPUs, then merge their summaries.
#
#   CURATED_RUN=<prepared run> T2I_MODEL=<name> bash iid/render_curated.sh
#
# Resources: 1 node with 8 GPUs per T2I model (fewer GPUs run the shards in turns). Rerunning
# continues where an interrupted run stopped. MAX_NEW_IMAGES caps the new images of each shard.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
OUT=${CURATED_RUN:?prepared run directory}; MODEL=${T2I_MODEL:?T2I model}
cd "$T_DIG_ROOT"; export PYTHONPATH="$T_DIG_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY_REWARD" -m iid.curated verify --out "$OUT" --t2i-model "$MODEL"
GPUS=($(t_dig_gpus)); NG=${#GPUS[@]}
pids=()
for g in $(seq 0 $((NG - 1))); do
  ( # GPU g runs shards g, g + NG, ... in turn.
    for shard in $(seq "$g" "$NG" 7); do
      CUDA_VISIBLE_DEVICES=${GPUS[$g]} "$PY_REWARD" -u -m iid.curated worker --out "$OUT" \
        --t2i-model "$MODEL" --shard "$shard" ${MAX_NEW_IMAGES:+--max-new-images "$MAX_NEW_IMAGES"} \
        >> "$OUT/$MODEL/shard-$(printf %02d "$shard").log" 2>&1
    done
  ) &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "a shard failed; see $OUT/$MODEL/shard-*.log"; exit 1; }
"$PY_REWARD" -m iid.curated status --out "$OUT"
[ -n "${MAX_NEW_IMAGES:-}" ] || "$PY_REWARD" -m iid.curated aggregate --out "$OUT" --t2i-model "$MODEL"
