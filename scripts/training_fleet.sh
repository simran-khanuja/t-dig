#!/bin/bash
# Start node zero of one GRPO training job. <run>/job.sh, written by
# scripts/prepare_training_matrix.py, sets the three required variables and execs this file:
#   T_DIG_TRAIN_SOURCE   frozen source directory of the matrix
#   T_DIG_TRAIN_PYTHON   training interpreter
#   T_DIG_FLEET_CONFIG   the run's fleet-config.json
#
# Resources: --nodes 8-GPU nodes (2 by default), one task per node, up to 72 h. Node zero reads the
# host list from T_DIG_NODEFILE and starts the reward services of the other nodes with
# T_DIG_REMOTE_LAUNCH; under Slurm both have defaults (axis_grpo/training/fleet.py), e.g.
#   sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --time=72:00:00 <run>/job.sh
set -euo pipefail
exec /bin/bash "${T_DIG_TRAIN_SOURCE:?}/scripts/training_fleet_node.sh" \
  "$T_DIG_TRAIN_SOURCE" "${T_DIG_TRAIN_PYTHON:?}" "${T_DIG_FLEET_CONFIG:?}" 0
