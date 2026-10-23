#!/bin/bash
# Compute FID and KID of every complete group in FID_OUT/sets.jsonl against COCO val2017.
#
#   T_DIG_ROOT=<repository root> FID_OUT=<dir holding sets.jsonl> bash tdig/coco_fid.sh
#
# Resources: 1 GPU (the first usable one), about 3 h. Run with bash on an allocated node, or submit
# with your scheduler passing these variables in the job environment.
#
#   FID_OUT        directory holding sets.jsonl; results and feature caches are written there
#   REAL_IMAGES    val2017 photographs (default: $T_DIG_DATA/downloads/coco-images/val2017)
#   INCEPTION_DIR  Clean-FID Inception weights (default: $T_DIG_DATA/downloads/fid-inception)
#
# Photograph and per-group features are cached in FID_OUT, so a rerun after adding groups to
# sets.jsonl computes only the new ones.
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
cd "$ROOT"
# clean-fid 0.1.35 needs the SciPy pinned in the overlay; it must precede the environment's own.
export PYTHONPATH="$ROOT/.cache/metric-python:$ROOT"
read -r GPU _ <<< "$(t_dig_gpus)"
CUDA_VISIBLE_DEVICES=$GPU "$PY_REWARD" -u -m tdig.coco_fid score \
  --out "${FID_OUT:?directory holding sets.jsonl}" \
  --real-images "${REAL_IMAGES:-$T_DIG_DATA/downloads/coco-images/val2017}" \
  --inception-dir "${INCEPTION_DIR:-$T_DIG_DATA/downloads/fid-inception}"
