#!/bin/bash
# Build the alpha-ladder adapters served by the served_runs/*_alpha_gen.json and *_edit_alpha.json
# descriptors, from the released checkpoints (https://huggingface.co/neulab/t-dig).
#
#   CHECKPOINTS=<local copy of neulab/t-dig> bash axis_framework/scripts/build_alpha_ladders.sh [exp ...]
#
# Rung N is the released LoRA adapter with its lora_B matrices scaled by N/100, for
# N = 5, 10, ..., 200 (alpha 0.05 to 2; rung 0 is the base model and needs no adapter). The slider
# covers alpha 0 to 1 in steps of 0.05; the alpha box goes up to 2. Rungs are written to
# axis_framework/adapters/<exp>_step<N>/lora_adapter/. Existing rungs are kept, so delete a
# ladder's directories to rebuild it from another checkpoint.
#
# Without arguments every ladder below is built. CPU only; needs torch and safetensors (PY_TRAIN).
# Default CHECKPOINTS: $T_DIG_MODELS/t-dig-checkpoints.
set -eu
source "${T_DIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/env.sh"
cd "$T_DIG_ROOT/axis_framework"
CHECKPOINTS=${CHECKPOINTS:-$T_DIG_MODELS/t-dig-checkpoints}

# exp -> released folder (see each descriptor's "note" for the training update it served).
declare -A SOURCE=(
  [tdig_qimg9b_alpha_gen]=qwen-image-2512/9b-generate-paper
  [tdig_qimg9b_edit_alpha]=qwen-image-2512/9b-edit
  [tdig_sdl9b_alpha_gen]=sd35-large/9b-generate
  [tdig_sdl9b_edit_alpha]=sd35-large/9b-edit
  [tdig_sdm9b_alpha_gen]=sd35-medium/9b-generate-paper
  [tdig_sdm9b_edit_alpha]=sd35-medium/9b-edit
  [tdig_flux9b_alpha_gen]=flux2-klein-9b/9b-generate
  [tdig_flux9b_edit_alpha]=flux2-klein-9b/9b-edit
  [tdig_flux35b_alpha_gen]=flux2-klein-9b/35b-a3b-generate
  [tdig_flux35b_edit_alpha]=flux2-klein-9b/35b-a3b-edit
)
EXPS=("$@")
[ ${#EXPS[@]} -gt 0 ] || EXPS=(tdig_qimg9b_alpha_gen tdig_qimg9b_edit_alpha tdig_sdl9b_alpha_gen
  tdig_sdl9b_edit_alpha tdig_sdm9b_alpha_gen tdig_sdm9b_edit_alpha tdig_flux9b_alpha_gen
  tdig_flux9b_edit_alpha tdig_flux35b_alpha_gen tdig_flux35b_edit_alpha)

for exp in "${EXPS[@]}"; do
  folder=${SOURCE[$exp]:?unknown ladder $exp}
  src=$CHECKPOINTS/$folder
  [ -f "$src/adapter_model.safetensors" ] || { echo "missing $src/adapter_model.safetensors"; exit 1; }
  for a in $(seq 5 5 200); do
    dst=adapters/${exp}_step$a/lora_adapter
    [ -f "$dst/adapter_model.safetensors" ] && continue
    mkdir -p "$dst"
    "$PY_TRAIN" - "$src" "$dst" "$a" <<'PY'
import sys, json, shutil
from safetensors.torch import load_file, save_file
src, dst, a = sys.argv[1], sys.argv[2], int(sys.argv[3]) / 100
t = load_file(f"{src}/adapter_model.safetensors")
for k in t:
    if "lora_B" in k: t[k] = (t[k].float() * a).to(t[k].dtype)
save_file(t, f"{dst}/adapter_model.safetensors", metadata={"format": "pt"})
shutil.copy(f"{src}/adapter_config.json", f"{dst}/adapter_config.json")
json.dump({"source": src, "alpha": a}, open(f"{dst}/scale.json", "w"))
PY
  done
  echo "$exp: $(ls -d adapters/${exp}_step* | wc -l) rungs from $src"
done
