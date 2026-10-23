# Shared environment of the launchers (sourced, not executed). Every variable can be set beforehand;
# the defaults follow the repository layout. See README.md, "Environment variables".
#
#   T_DIG_ROOT    repository root (default: the parent of this file's directory)
#   T_DIG_DATA    datasets                      (default: $T_DIG_ROOT/data)
#   T_DIG_RUNS    outputs                       (default: $T_DIG_ROOT/runs)
#   T_DIG_MODELS  local model and scorer weights (default: $T_DIG_ROOT/models)
#   HF_HOME       Hugging Face home; snapshots are read from $HF_HUB_CACHE (default: $HF_HOME/hub)
#   PY_TRAIN, PY_REWARD, PY_VQA   interpreters of the three environments (default: python)
#   T_DIG_SETUP   optional file sourced first, for site setup (module loads, LD_LIBRARY_PATH, CUDA)
if [ -n "${T_DIG_SETUP:-}" ]; then
  # shellcheck disable=SC1090
  source "$T_DIG_SETUP"
fi
export T_DIG_ROOT=${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export T_DIG_DATA=${T_DIG_DATA:-$T_DIG_ROOT/data}
export T_DIG_RUNS=${T_DIG_RUNS:-$T_DIG_ROOT/runs}
export T_DIG_MODELS=${T_DIG_MODELS:-$T_DIG_ROOT/models}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
# Absolute interpreter paths, so that "$(dirname "$PY_VQA")" names the environment's bin directory.
_t_dig_which() { command -v "$1" 2>/dev/null || echo "$1"; }
export PY_TRAIN=$(_t_dig_which "${PY_TRAIN:-python}")
export PY_REWARD=$(_t_dig_which "${PY_REWARD:-python}")
export PY_VQA=$(_t_dig_which "${PY_VQA:-python}")
unset -f _t_dig_which
export PYTHONPATH="$T_DIG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The GPUs a launcher may use, one per word: CUDA_VISIBLE_DEVICES when the scheduler or the user
# set it, else every GPU of the node. Launchers index this list instead of numbering GPUs from 0.
t_dig_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "${CUDA_VISIBLE_DEVICES//,/ }"
  else nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '; fi
}
# Stop processes and every descendant: vLLM runs its engine in a child that outlives its server.
# The tree is collected before anything is killed, since an orphan leaves it.
_t_dig_tree() { local p; for p in "$@"; do echo "$p"; _t_dig_tree $(pgrep -P "$p" 2>/dev/null); done; }
t_dig_kill_tree() {
  local all; all=$(_t_dig_tree "$@")
  [ -n "$all" ] || return 0
  kill $all 2>/dev/null; sleep 2; kill -9 $all 2>/dev/null; return 0
}
