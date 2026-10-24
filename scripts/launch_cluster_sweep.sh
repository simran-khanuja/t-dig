#!/bin/bash
# Run a prepared HPDv2 clustering sweep (scripts/run_cluster_sweep.py prepare) on one GPU.
#
#   T_DIG_CLUSTER_RUN=<prepared sweep directory> bash scripts/launch_cluster_sweep.sh
#
# Resources: 1 node, 1 GPU, 64 GiB host memory for the distance matrix. The prepared directory
# holds a frozen copy of the code and of this file; .cache/cluster-python holds the clustering
# dependencies (scripts/run_cluster_sweep.py --dependencies).
set -euo pipefail
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
: "${T_DIG_CLUSTER_RUN:?Frozen clustering sweep directory required}"
export PYTHONPATH="$T_DIG_ROOT/.cache/cluster-python:$T_DIG_CLUSTER_RUN/source"
read -r GPU _ <<< "$(t_dig_gpus)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$T_DIG_CLUSTER_RUN/source"
exec "$PY_REWARD" -X faulthandler -u scripts/run_cluster_sweep.py run --out "$T_DIG_CLUSTER_RUN" --execute
