#!/bin/bash
# Train one T-DIG policy on this machine: scripts/train_policy.py in the training environment.
#   bash scripts/train_policy.sh --t2i-model sd35-medium --out $T_DIG_RUNS/my-policy --execute
# Resources: one machine with one or more GPUs (CUDA_VISIBLE_DEVICES if set, else all of them).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$PY_TRAIN" "$T_DIG_ROOT/scripts/train_policy.py" "$@"
