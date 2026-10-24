#!/bin/bash
# One node of a training job: training_fleet_node.sh <frozen source> <training python> <fleet config>
# <node index> [<job id>]. Node zero is started by training_fleet.sh, the others by node zero.
set -euo pipefail
cd "${1:?frozen source}"
export T_DIG_ROOT="$1"
source "$1/scripts/env.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ -n "${5:-}" ]; then export T_DIG_JOB_ID="$5"; fi
# A further node can inherit node zero's GPU UUID list; unset it so fleet.py lists the
# node's own GPUs.
if [ "${4:?node index}" != "0" ]; then unset CUDA_VISIBLE_DEVICES; fi
exec "${2:?training Python}" -m axis_grpo.training.fleet --config "${3:?fleet config}" --node-index "${4:?node index}" --execute
