#!/bin/bash
# Export a ladder from a round-one verl training run: each checkpoint of one run becomes
# one rung, axis_framework/adapters/<EXP>_step<N>/lora_adapter (export_lora.py), so the slider
# moves along one objective's training trajectory. The released policies do not need this: their
# ladders come from scripts/build_alpha_ladders.sh.
#
#   EXP=<training run> bash axis_framework/scripts/export_ladder.sh
#
# Reads $T_DIG_RUNS/axis_grpo/ckpt/$EXP/global_step_N (AXIS_FW_CKPT_ROOT overrides the ckpt
# directory). STEPS="25 200 400" exports a subset (default: every global_step_N, in order).
# PY_POLICY (default PY_TRAIN) runs export_lora.py. Existing rungs are skipped; the log goes to
# axis_framework/runs/export/ladder_<jobid>.out.
#
# Resources: CPU only, ~8 GB of host memory, ~26 s per checkpoint of a 4B policy. Run it on a
# compute node, not a shared login node.
set -x
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
cd "$ROOT" || exit 1
JOBID=${T_DIG_JOB_ID:-${SLURM_JOB_ID:-local-$$}}

mkdir -p axis_framework/runs/export
exec > >(tee -a "axis_framework/runs/export/ladder_${JOBID}.out") 2>&1

export AXIS_MODELS=${AXIS_MODELS:-$T_DIG_MODELS}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY=${PY_POLICY:-$PY_TRAIN}
EXP=${EXP:?pass EXP=<training run> -- the run whose checkpoints become a ladder}
CKPT="${AXIS_FW_CKPT_ROOT:-$T_DIG_RUNS/axis_grpo/ckpt}/$EXP"
[ -d "$CKPT" ] || { echo "no such run: $CKPT"; exit 1; }

STEPS=${STEPS:-$(ls "$CKPT" | sed -n 's/^global_step_\([0-9]*\)$/\1/p' | sort -n | tr '\n' ' ')}
echo "[ladder] $EXP: $(echo $STEPS | wc -w) checkpoints"

rc=0; done_n=0; skip_n=0
for st in $STEPS; do
  out="axis_framework/adapters/${EXP}_step${st}"
  if [ -f "$out/lora_adapter/adapter_config.json" ]; then
    skip_n=$((skip_n+1)); continue
  fi
  echo "[ladder] exporting step $st ..."
  if $PY axis_framework/export_lora.py --ckpt "$CKPT" --step "$st"; then
    done_n=$((done_n+1))
  else
    echo "[ladder] FAILED: step $st"; rc=1
  fi
done
echo "[ladder] exported $done_n, already present $skip_n"
ls -d axis_framework/adapters/${EXP}_step*/ 2>/dev/null | wc -l
exit $rc
