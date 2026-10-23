#!/usr/bin/env bash
# Start the policy, render and suggest servers and the orchestrator on the current machine, all
# on one GPU (or CPU-only with FAKE=1). Logs and the pidfile go to axis_framework/runs/<RUN>/.
#
#   FAKE=1 bash axis_framework/scripts/launch_local.sh          # no GPU, drives the loop
#   SERVE=tdig_flux9b_alpha_gen:tdig_flux9b_edit_alpha bash axis_framework/scripts/launch_local.sh
#   RUN=<same RUN> bash axis_framework/scripts/launch_local.sh stop
#
# SERVE names served_runs/*.json descriptors (<generation run>[:<editing run>]); build their
# adapters first with scripts/build_alpha_ladders.sh.
#
# Variables (defaults): RUN (local); PORT (19100) for the orchestrator, with policy, render and
# suggest on PORT+1, +2, +3; GPU_UTIL (0.25), vLLM's share of the GPU; LORA; VQA_URL; ARMS (ours);
# NO_AUDIT=1; FAKE=1; PY_POLICY (PY_TRAIN) for the policy server, PY_T2I (PY_REWARD) for the rest.
set -u
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/env.sh"
ROOT=$T_DIG_ROOT
cd "$ROOT" || exit 1

RUN=${RUN:-local}
LOGS="axis_framework/runs/$RUN"
PIDFILE="$LOGS/pids"
mkdir -p "$LOGS"

if [ "${1:-start}" = "stop" ]; then
  # Kill whole process trees: vLLM v1's engine runs in a child process that outlives the
  # server and keeps its GPU memory.
  [ -f "$PIDFILE" ] && t_dig_kill_tree $(cat "$PIDFILE")
  rm -f "$PIDFILE"; echo "stopped"; exit 0
fi

PORT=${PORT:-19100}
AUDIT_FLAG=""; [ "${NO_AUDIT:-0}" = "1" ] && AUDIT_FLAG="--no-audit"
PY_POLICY=${PY_POLICY:-$PY_TRAIN}
PY_T2I=${PY_T2I:-$PY_REWARD}
# vLLM's zmq needs GLIBCXX_3.4.30, which the system libstdc++ may lack ("version
# `GLIBCXX_3.4.30' not found" at LLM()). The policy launch alone puts its env's lib dir first.
POLICY_LD="$(dirname "$(dirname "$PY_POLICY")")/lib"
FAKE_FLAG=""; [ "${FAKE:-0}" = "1" ] && FAKE_FLAG="--fake" && export AXIS_FW_FAKE=1
# The runs to serve; everything model-specific is read from them (axis_framework/profiles.py).
export AXIS_FW_SERVE=${SERVE:-${AXIS_FW_SERVE:-}}
: > "$PIDFILE"

# GPU_UTIL is vLLM's share of the whole GPU, which all three services share here. 0.25 of a
# 48 GiB GPU (~12 GiB) fits a 4B policy beside SD3.5-M and the suggest VLM (~16 GiB each).
GPU_UTIL=${GPU_UTIL:-0.25}
LD_LIBRARY_PATH="$POLICY_LD:${LD_LIBRARY_PATH:-}" \
  nohup $PY_POLICY  axis_framework/servers/policy_server.py  --port $((PORT + 1)) --gpu-util "$GPU_UTIL" ${LORA:+--lora "$LORA"} $FAKE_FLAG > "$LOGS/policy.log"  2>&1 & echo $! >> "$PIDFILE"
nohup $PY_T2I  axis_framework/servers/t2i_server.py  --port $((PORT + 2)) ${VQA_URL:+--vqa-url "$VQA_URL"} $AUDIT_FLAG $FAKE_FLAG > "$LOGS/render.log"  2>&1 & echo $! >> "$PIDFILE"
# --preload (unless faking) so the ~50 s VLM load happens at startup.
PRELOAD=""; [ "${FAKE:-0}" = "1" ] || PRELOAD="--preload"
PYTHONPATH="$ROOT/axis_framework/vendor:${PYTHONPATH:-}" \
  nohup $PY_T2I  axis_framework/servers/suggest_server.py --port $((PORT + 3)) $PRELOAD $FAKE_FLAG > "$LOGS/suggest.log" 2>&1 & echo $! >> "$PIDFILE"
sleep 3
nohup $PY_T2I  axis_framework/orchestrator.py --host 0.0.0.0 --port "$PORT" --run "$RUN" \
      --policy-url "http://127.0.0.1:$((PORT + 1))" --t2i-url "http://127.0.0.1:$((PORT + 2))" \
      --suggest-url "http://127.0.0.1:$((PORT + 3))" \
      --arms "${ARMS:-ours}" $AUDIT_FLAG > "$LOGS/orch.log" 2>&1 & echo $! >> "$PIDFILE"

echo "logs -> $LOGS ; UI -> http://127.0.0.1:$PORT ; stop with: RUN=$RUN bash $0 stop"
