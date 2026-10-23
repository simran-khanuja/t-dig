#!/bin/bash
# Start (or restart) relay_agent.py in the background on the node the orchestrator runs on.
#
#   bash axis_framework/relay/start_agent.sh          # start / restart
#   bash axis_framework/relay/start_agent.sh stop
#
# RELAY_URL (required): wss://<your domain>/axis/__relay. UPSTREAM: orchestrator URL
# (http://127.0.0.1:19100). The agent reconnects by itself. Log:
# axis_framework/runs/relay_agent_<host>.log
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=$ROOT/axis_framework/runs/relay_agent_$(hostname -s).log
PIDF=$ROOT/axis_framework/runs/relay_agent_$(hostname -s).pid

if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  kill "$(cat "$PIDF")"; sleep 1
  echo "stopped agent $(cat "$PIDF")"
fi
rm -f "$PIDF"
[[ "${1:-}" == "stop" ]] && exit 0

T_DIG_ROOT=${T_DIG_ROOT:-$ROOT}
source "$ROOT/scripts/env.sh"
nohup "$PY_REWARD" "$ROOT/axis_framework/relay/relay_agent.py" \
  --relay "${RELAY_URL:?set RELAY_URL=wss://<your domain>/axis/__relay}" \
  --upstream "${UPSTREAM:-http://127.0.0.1:19100}" >> "$LOG" 2>&1 &
echo $! > "$PIDF"
sleep 4
tail -2 "$LOG"
echo "agent pid $(cat "$PIDF"), log $LOG"
