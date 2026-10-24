#!/bin/bash
# Run the whole loop on one node, one service per GPU. The GPUs are those of
# CUDA_VISIBLE_DEVICES when set (by the scheduler or by hand), else every GPU of the node
# (scripts/env.sh, t_dig_gpus); in that list:
#
#     gpu 0        policy      vLLM + the served runs' base model + their LoRA ladders
#     gpu 1        suggest     the axis-suggestion VLM
#     gpu 2..1+R   render      R replicas of one T2I model; the first also loads TPIPS (audit)
#
# then the orchestrator on PORT (the UI) in the foreground; stopping it (Ctrl-C, scancel) stops
# the services.
#
#   SERVE=<generation run>[:<editing run>] bash axis_framework/scripts/launch_pool.sh
#
# Resources: 1 node with 2 + T2I_GPUS GPUs (6 by default), for as long as the tool should be
# up. No scheduler directives: run it with bash on an allocated node, or submit it with
# sbatch --export=ALL, passing T_DIG_ROOT, SERVE and the variables below.
#
# SERVE (required unless FAKE=1) names served_runs/*.json descriptors, e.g.
#   SERVE=tdig_flux9b_alpha_gen:tdig_flux9b_edit_alpha
# Everything model-specific (T2I model, policy base, VLM or not, LoRA rank, context length, render
# settings) is read from those runs by axis_framework/profiles.py and printed before anything
# loads. Build their adapters first with axis_framework/scripts/build_alpha_ladders.sh.
#
# Other variables (defaults):
#   MODEL          served T2I model to pool (the first)
#   T2I_GPUS    render replicas, at most K = 4 are useful (4)
#   RUN            run name, logs in axis_framework/runs/<RUN>/ (pool_<jobid>)
#   PORT           orchestrator port (19100)
#   SUB_PORT_BASE  internal ports: policy +1, suggest +3, render replicas +10.. (19100); a
#                  second instance on the node sets it and CUDA_VISIBLE_DEVICES
#   ARMS           ours | baseline | ab (ours)
#   LORA           adapter for requests that name no rung (last rung of the generation ladder)
#   OFF_AXIS       slider start, 0..1 along the ladder (0.2)
#   GPU_UTIL       vLLM GPU memory share (0.85)
#   VQA_URL        VQAScore service for the audit (none)
#   PY_POLICY, PY_T2I, PY_SUGGEST   interpreters (PY_TRAIN, PY_REWARD, PY_REWARD)
#   NO_AUDIT=1, NO_PREFETCH=1, NO_CUDAGRAPHS=1, FAKE=1 (no GPU)
#   Study mode and relay: see the section further down.
#
# Each grid is sharded across the render replicas; every shard uses the latents it would have
# had on one server, so the noise is unchanged (see t2i_server._render for the batch-size
# caveat).
set -x
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/env.sh"
JOBID=${T_DIG_JOB_ID:-${SLURM_JOB_ID:-local-$$}}
ROOT=$T_DIG_ROOT
cd "$ROOT" || exit 1
[ -f axis_framework/orchestrator.py ] || { echo "not the repo root: $ROOT"; exit 1; }

RUN=${RUN:-pool_${JOBID}}
LOGS="axis_framework/runs/$RUN"
mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/job.out") 2>&1

# Site paths come from scripts/env.sh (T_DIG_MODELS, HF_HOME; T_DIG_SETUP for module loads).
export AXIS_MODELS=${AXIS_MODELS:-$T_DIG_MODELS}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY_POLICY=${PY_POLICY:-$PY_TRAIN}
PY_T2I=${PY_T2I:-$PY_REWARD}
PY_SUGGEST=${PY_SUGGEST:-$PY_REWARD}

[ "${FAKE:-0}" = "1" ] && export AXIS_FW_FAKE=1
export AXIS_FW_SERVE=${SERVE:-${AXIS_FW_SERVE:-}}
if [ -z "$AXIS_FW_SERVE" ] && [ "${FAKE:-0}" != "1" ]; then
  echo "SERVE is required: SERVE=<round-one run>[:<layered run>] (see the header)"; exit 1
fi
echo "---------------- served profile (derived from the runs) ----------------"
$PY_T2I -m axis_framework.profiles || { echo "the served runs do not resolve -- see above"; exit 1; }
echo "------------------------------------------------------------------------"
MODEL=${MODEL:-$($PY_T2I -c 'from axis_framework import config; print(config.default_model())')}
export AXIS_FW_MODEL=$MODEL
# Default --lora (answers requests that name no rung): rung 100 of this T2I model's generation
# ladder, i.e. the released adapter at scale 1 on an alpha ladder; else its last rung.
if [ -z "${LORA:-}" ] && [ "${FAKE:-0}" != "1" ]; then
  EXP1=$($PY_T2I -c "from axis_framework import config; print(config.off_axis_exp_for('$MODEL'))")
  LORA="$ROOT/axis_framework/adapters/${EXP1}_step100"
  [ -d "$LORA" ] || LORA=$(ls -d "$ROOT/axis_framework/adapters/${EXP1}"_step* 2>/dev/null \
         | awk -F_step '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2)
  [ -n "$LORA" ] || { echo "no adapters for $EXP1 -- run axis_framework/scripts/build_alpha_ladders.sh $EXP1"; exit 1; }
fi

ARMS=${ARMS:-ours}
PORT=${PORT:-19100}
SP=${SUB_PORT_BASE:-19100}
GPUS=($(t_dig_gpus)); [ ${#GPUS[@]} -gt 0 ] || GPUS=(0 1 2 3 4 5 6 7)  # no GPU: fake mode
export AXIS_FW_POLICY_URL="http://127.0.0.1:$((SP + 1))" AXIS_FW_SUGGEST_URL="http://127.0.0.1:$((SP + 3))"
# Slider start, 0..1 along the ladder (see config.OFF_AXIS_DEFAULT).
export AXIS_FW_OFF_AXIS=${OFF_AXIS:-0.2}
GPU_UTIL=${GPU_UTIL:-0.85}
FAKE_FLAG=""; [ "${FAKE:-0}" = "1" ] && FAKE_FLAG="--fake"

# CUDA graphs are on here because the policy has its own GPU (policy_server keeps them off
# by default for shared GPUs). They cut a 4B policy's expansion from ~14.5 s to ~2-3 s on an
# H200, at the cost of a longer startup (~49 s -> ~112 s). NO_CUDAGRAPHS=1 runs eager.
CG_FLAG="--cudagraphs"; [ "${NO_CUDAGRAPHS:-0}" = "1" ] && CG_FLAG=""
[ "${FAKE:-0}" = "1" ] && CG_FLAG=""

# A grid of K images splits into at most K shards (orchestrator._shards), so more than K = 4
# replicas are never used.
R=${T2I_GPUS:-4}
if [ "${FAKE:-0}" != "1" ] && [ "$((R + 2))" -gt "${#GPUS[@]}" ]; then
  R=$((${#GPUS[@]} - 2))
  echo "only ${#GPUS[@]} GPUs visible -> $R render replicas"
fi
[ "$R" -lt 1 ] && { echo "need at least 3 GPUs (policy, suggest, one T2I model)"; exit 1; }

PIDS=()
cleanup() {
  # Kill whole process trees: vLLM v1's engine runs in a child process that outlives the
  # server and keeps its GPU memory.
  t_dig_kill_tree "${PIDS[@]}"
}
trap cleanup EXIT INT TERM

# gpu 0: policy. Its env's bin goes on PATH (this launch only) because flashinfer, used by
# vLLM's sampler, JIT-builds a kernel with `ninja` on first use.
CUDA_VISIBLE_DEVICES=${GPUS[0]} PATH="$(dirname "$PY_POLICY"):$PATH" \
    $PY_POLICY axis_framework/servers/policy_server.py \
    --port $((SP + 1)) --gpu-util "$GPU_UTIL" ${LORA:+--lora "$LORA"} \
    $CG_FLAG $FAKE_FLAG \
    > "$LOGS/policy.log" 2>&1 &
PIDS+=($!)

# gpu 1: suggest. --preload (unless faking) so the ~50 s VLM load happens at startup.
# vendor/ provides qwen_vl_utils, which the reward environment lacks.
PRELOAD=""; [ "${FAKE:-0}" = "1" ] || PRELOAD="--preload"
CUDA_VISIBLE_DEVICES=${GPUS[1]} PYTHONPATH="$ROOT/axis_framework/vendor:${PYTHONPATH:-}" \
    $PY_SUGGEST axis_framework/servers/suggest_server.py \
    --port $((SP + 3)) $PRELOAD $FAKE_FLAG > "$LOGS/suggest.log" 2>&1 &
PIDS+=($!)

# gpu 2..: render replicas, each serving only $MODEL. Only replica 0 loads TPIPS (~16 GiB)
# and serves the audit for the pool.
POOL=""
for i in $(seq 0 $((R - 1))); do
  p=$((SP + 10 + i))
  AUDIT_FLAG="--no-audit"
  [ "$i" = "0" ] && [ "${NO_AUDIT:-0}" != "1" ] && AUDIT_FLAG=""
  CUDA_VISIBLE_DEVICES=${GPUS[2 + i]} $PY_T2I axis_framework/servers/t2i_server.py \
      --port "$p" --models "$MODEL" --model "$MODEL" ${VQA_URL:+--vqa-url "$VQA_URL"} \
      $AUDIT_FLAG $FAKE_FLAG > "$LOGS/render_$i.log" 2>&1 &
  PIDS+=($!)
  POOL="$POOL${POOL:+,}http://127.0.0.1:$p"
  echo "gpu ${GPUS[2 + i]} -> render replica $i of $MODEL on :$p"
done

any_dead() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || return 0; done; return 1; }

# Wait until every service reports ready (up to 1 h each).
WAIT_URLS="http://127.0.0.1:$((SP + 1)) http://127.0.0.1:$((SP + 3))"
for i in $(seq 0 $((R - 1))); do WAIT_URLS="$WAIT_URLS http://127.0.0.1:$((SP + 10 + i))"; done
for u in $WAIT_URLS; do
  for _ in $(seq 1 240); do
    curl -sf --max-time 5 "$u/health" | grep -q '"ready": true' && break
    any_dead && { echo "a service died before $u came up -- see $LOGS/*.log"; exit 1; }
    sleep 15
  done
  curl -sf --max-time 5 "$u/health" | grep -q '"ready": true' \
    || { echo "$u never came up -- see $LOGS/*.log"; exit 1; }
  echo "  $u OK"
done

# Pre-render the start gallery (prefetch.DEFAULT_PROMPTS) before the UI opens, dealing the
# prompts round-robin across the replicas. Already-cached prompts are skipped.
if [ "${NO_PREFETCH:-0}" != "1" ]; then
  echo "warming the start gallery ($MODEL) across $R replicas ..."
  $PY_T2I -c 'from axis_framework import prefetch
print(chr(10).join(prefetch.DEFAULT_PROMPTS))' > "$LOGS/prompts.txt"
  split -n "r/$R" -d "$LOGS/prompts.txt" "$LOGS/prompts_part"
  # Wait on the prefetch PIDs only: a bare `wait` would also wait for the servers and hang.
  PRE=()
  for i in $(seq 0 $((R - 1))); do
    part=$(printf "%s/prompts_part%02d" "$LOGS" "$i")
    [ -s "$part" ] || continue
    $PY_T2I axis_framework/prefetch.py --model "$MODEL" --prompts "$part" \
        --t2i-url "http://127.0.0.1:$((SP + 10 + i))" >> "$LOGS/prefetch.log" 2>&1 &
    PRE+=($!)
  done
  [ ${#PRE[@]} -gt 0 ] && wait "${PRE[@]}"
  echo "gallery warm -- see $LOGS/prefetch.log"
fi

# ---- Public study mode (all off by default) -----------------------------------------------
# BASE_PATH=/axis        also serve under a URL prefix (orchestrator --base-path)
# ASK_PARTICIPANT=1      ask for a name / participant id first (orchestrator --ask-participant)
# RELAY=1                once the orchestrator answers, run relay/relay_agent.py, which dials out
#                        to the Cloudflare Worker at RELAY_URL (relay/worker.js, deployed with
#                        relay/deploy_worker.sh). A newer agent takes the socket over and this
#                        one exits, so handovers need no manual step.
STUDY_FLAGS=()
[ -n "${BASE_PATH:-}" ] && STUDY_FLAGS+=(--base-path "$BASE_PATH")
[ "${ASK_PARTICIPANT:-0}" = "1" ] && STUDY_FLAGS+=(--ask-participant)
# Study mode (README, "Study mode"): STUDY_PICTURES=N turns it on; STUDY_SUGGESTIONS=on|off|split;
# COMPLETION_CODE is shown after the questionnaire (e.g. a Prolific completion code).
[ "${STUDY_PICTURES:-0}" != "0" ] && STUDY_FLAGS+=(--study-pictures "$STUDY_PICTURES" --study-suggestions "${STUDY_SUGGESTIONS:-on}")
[ -n "${COMPLETION_CODE:-}" ] && STUDY_FLAGS+=(--completion-code "$COMPLETION_CODE")

if [ "${RELAY:-0}" = "1" ]; then
  (
    for _ in $(seq 1 360); do
      curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null && break
      sleep 5
    done
    echo "$(date '+%F %T') orchestrator answering on :$PORT -> starting relay agent"
    exec "$PY_T2I" axis_framework/relay/relay_agent.py \
        --relay "${RELAY_URL:?set RELAY_URL=wss://<your domain>/axis/__relay}" \
        --upstream "http://127.0.0.1:$PORT"
  ) >> "$LOGS/relay_agent.log" 2>&1 &
  PIDS+=($!)
fi

echo "============================================================"
echo "  axis_framework (pooled) on $(hostname)  port $PORT   run=$RUN"
CG_NOTE="cudagraphs off"; [ -n "$CG_FLAG" ] && CG_NOTE="cudagraphs on"
echo "  policy gpu${GPUS[0]} ($CG_NOTE) | suggest gpu${GPUS[1]} | $R x $MODEL on gpu ${GPUS[*]:2:R}"
echo "  tunnel:  ssh -N -L ${PORT}:$(hostname):${PORT} <login node>"
echo "  then open http://localhost:${PORT}"
echo "  logs -> $LOGS ; stop job ${JOBID} (or Ctrl-C) when finished"
echo "============================================================"

# --audit-url is replica 0 (the one with TPIPS). Bind 0.0.0.0 so the login node can reach it.
# Not exec'd: the script waits here so that `cleanup` runs when the orchestrator or the job ends.
AUDIT_FLAG=""; [ "${NO_AUDIT:-0}" = "1" ] && AUDIT_FLAG="--no-audit"
$PY_T2I axis_framework/orchestrator.py --host 0.0.0.0 --port "$PORT" \
    --run "$RUN" --arms "$ARMS" --model "$MODEL" \
    --t2i-pool "$MODEL=$POOL" \
    --audit-url "http://127.0.0.1:$((SP + 10))" \
    $AUDIT_FLAG "${STUDY_FLAGS[@]}" &
PIDS+=($!)
wait $!
