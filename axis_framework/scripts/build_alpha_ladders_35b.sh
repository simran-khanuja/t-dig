#!/bin/bash
# The two FLUX.2 [klein] ladders of the Qwen3.5-35B-A3B policies (generation and editing); see
# build_alpha_ladders.sh, which this calls.
exec bash "$(dirname "${BASH_SOURCE[0]}")/build_alpha_ladders.sh" tdig_flux35b_alpha_gen tdig_flux35b_edit_alpha
