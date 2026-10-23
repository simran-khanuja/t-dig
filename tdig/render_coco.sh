#!/bin/bash
# Render the COCO-250 captions with one T2I model's baseline methods, split over the usable GPUs.
#
#   T_DIG_ROOT=<repository root> CONFIGS=<method.json>:<method.json> ASSETS=<asset lock> \
#     RENDER_OUT=<dir> bash tdig/render_coco.sh
#
# Resources: 1 node with 8 GPUs, about 10 h. Run with bash on an allocated node, or submit with
# your scheduler passing these variables in the job environment.
#
#   CONFIGS     ':'-separated method configurations from `baselines config`
#   ASSETS      asset lock of the T2I model
#   RENDER_OUT  output directory
#   SUBSET      caption subset (default: $T_DIG_DATA/tdig/coco250-v1/subset.json)
#   BUNDLE      COCO bundle directory (default: $T_DIG_DATA/benchmarks/coco2017-val-5000)
#
# Resubmitting continues: `baselines generate` reuses saved images of an interrupted run.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
cd "$ROOT"
export PYTHONPATH="$ROOT"
ARGS=()
IFS=':' read -ra CFGS <<< "${CONFIGS:?colon-separated method configurations}"
for c in "${CFGS[@]}"; do ARGS+=(--config "$c"); done
exec "$PY_REWARD" -u -m tdig.render_coco "${ARGS[@]}" \
  --subset "${SUBSET:-$T_DIG_DATA/tdig/coco250-v1/subset.json}" --assets "${ASSETS:?asset lock}" \
  --bundle "${BUNDLE:-$T_DIG_DATA/benchmarks/coco2017-val-5000}" --out "${RENDER_OUT:?output directory}" --execute
