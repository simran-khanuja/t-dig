# axis_framework: interactive axis-conditioned diversity

A web tool that puts the trained T-DIG policies in a loop with a person; it is the interface of the paper's human study.

1. The user types a prompt. The served T2I model draws an *i.i.d. set* of 4 images at independent noise.
2. The user names an *axis* in their own words ("the dog's breed", "the lighting"), or takes one of the axes a VLM suggests from the images.
3. The *policy* (Qwen3.5 with the GRPO LoRA) writes K = 4 prompts that vary along the axis (generation), or K edit instructions for the i.i.d. images (editing).
4. The T2I model draws them. The user changes the axis, layers another axis on top, moves the diversity slider, or downloads the set; in [study mode](#study-mode) they also rate it and accept or reject it.

The axis text reaches the policy verbatim, and the audit metrics condition TPIPS on the same text.

## Controls

| Control | Side | Effect |
|---|---|---|
| Diversity slider | policy | Selects a rung of the run's LoRA alpha ladder: rung N is the released adapter with its `lora_B` matrices scaled by N/100. The slider covers alpha 0 (base model, no adapter) to 1 in steps of 0.05; a text box accepts up to 2. All rungs are loaded at once. |
| Seed | policy | The policy's sampling seed. It stays fixed while the slider or K moves; "Change seed" draws a new set of ideas at the same level. |
| Axis / mode | policy | `POST /session/respin` re-expands as a new trial: `mode=fresh` varies the original prompt again, `mode=layer` varies this round's K prompts so the set gains a second kind of variation. |
| Generate / Edit | policy | `POST /session/axis` with `mode=generate` writes K prompts; `mode=edit` (only when an editing run is served) writes K edit instructions applied to the i.i.d. images. |
| Set size K | policy | `k` on `/session/axis` and `/session/respin`: 2, 4, 8 or 16 (`AXIS_FW_K_CHOICES`); the UI does not offer it. Policies were trained at K = 4; other K are logged with `k_off_distribution`. |
| Initial diversity a in [0, 1] | render | Share of independent per-image noise: `eps_k = sqrt(1-a) * eps_shared + sqrt(a) * eps_k`. At 0 (the live default) all images share the latent the policy was trained under, so they differ through their wording only. The i.i.d. stage uses 1. |
| Text guidance | render | The T2I model's CFG scale; default is the served run's validation setting. |

The web UI shows the policy-side controls other than K. `POST /session/rerender {session_id, guidance_scale, init_diversity}` redraws the same prompts with other render settings. `/session/axis` and `/session/respin` accept `temperature`, `top_p` and `top_k`. Two actions use the editor without the policy: `/session/fix` applies the user's own instruction to one picture, and `/session/merge` combines several pictures.

Axis suggestions: whenever a set of images exists, the orchestrator asks the suggest server in the background what is the same across the images (or across the user's selected subset). The chip `overall` is always offered; rejected axes are not suggested again. With `--speculate N` (default 3) the policy expands the top N suggestions in one batched call before the user picks.

### Audit metrics

The T2I model server's `/audit` computes these for the grid and for the i.i.d. set under the same axis. They are logged and never sent to the browser.

| Metric | Definition |
|---|---|
| `axis_dist` | mean pairwise TPIPS distance conditioned on the axis text (the reward's distance) |
| `axis_vendi`, `axis_vendi_norm` | Vendi score of the same embeddings |
| `blind_vendi` | Vendi score with TPIPS's axis-conditioning LoRA disabled |
| `ratio` | `axis_vendi / blind_vendi` |
| `vqa_base`, `vqa_own`, `vqa_base_min` | VQAScore of each image against the base prompt and against its own prompt; only with a VQAScore service, no substitute scorer |

## Architecture

A CPU orchestrator talks JSON over localhost HTTP to three GPU servers (all stdlib `http.server`):

| Process | File | Environment | Port (`launch_local.sh`) | Holds |
|---|---|---|---|---|
| orchestrator | `orchestrator.py` | `PY_REWARD` | 19100 | state machine, session store, event log, web UI (`web/index.html`) |
| policy | `servers/policy_server.py` | `PY_TRAIN` (vLLM with Qwen3.5 LoRA) | 19101 | the base model and all LoRA rungs |
| T2I model | `servers/t2i_server.py` | `PY_REWARD` | 19102 | one T2I model from `t2i_models.json`, its editor, TPIPS for `/audit` |
| suggest | `servers/suggest_server.py` | `PY_REWARD` | 19103 | the suggestion VLM (`Qwen/Qwen3-VL-8B-Instruct`); imports `vendor/qwen_vl_utils` |
| VQAScore (optional) | `axis_grpo/training/vqa_service.py` | `PY_VQA` | any | CLIP-FlanT5 VQAScore for the audit |

For the `vqa_*` metrics, start a VQAScore service and pass its URL as `VQA_URL`. The T2I model server writes its images into the staging directory the service reports on `/health`:

```bash
$PY_VQA axis_grpo/training/vqa_service.py --execute --port 19096 \
    --cache "$HF_HUB_CACHE" --staging "${TMPDIR:-/tmp}/axis_grpo_vqa"
VQA_URL=http://127.0.0.1:19096 SERVE=... bash axis_framework/scripts/launch_local.sh
```

Sessions go `START -> GATE -> CHOOSE_AXIS -> REVIEW -> DONE` (`session.py`). In study mode GATE asks "do these need more variety?" and REVIEW ends with accept or reject, where rejecting returns to CHOOSE_AXIS; the open demo answers the gate itself and stays in REVIEW, making further sets. A respin is a new trial linked by `respin_of`, so every logged render points at the prompts it drew.

### Seeds

The K images of a grid share one latent (initial diversity 0). Its seed is:

- for the i.i.d. set, the training seed of the base prompt, `seed_for(base_prompt)` (`backend/rollout.py`);
- for a policy grid, the policy draw's seed, a hash of session, base prompt, axis and draw number. The slider keeps it, so two grids differ by adapter scale only; a re-roll draws a new one.

Completions are parsed with `parse_prompts` (`backend/rollout.py`), as in training.

### Logs

- `axis_framework/runs/<run>/events.jsonl`: one line per transition with its full payload.
- `axis_framework/runs/<run>/sessions.sqlite`: the session store (`--store memory` keeps it in memory).
- A JSON sidecar next to each grid's PNGs.

Use a separate `--run` for each data collection.

## Served runs

Everything model-specific (base model, LoRA rank, context length, T2I model, render settings) comes from run descriptors in `served_runs/`, read by `profiles.py`:

```
SERVE=<generation run>[:<editing run>][+<generation run>[:<editing run>]...]
```

`:` pairs a generation run with the editing run of the same T2I model; `+` adds another T2I model. All served runs must share one policy base model; startup names the runs that disagree.

| T2I model | Generation | Editing |
|---|---|---|
| Qwen-Image-2512 | `tdig_qimg9b_alpha_gen` | `tdig_qimg9b_edit_alpha` |
| SD3.5-Large | `tdig_sdl9b_alpha_gen` | `tdig_sdl9b_edit_alpha` |
| SD3.5-Medium | `tdig_sdm9b_alpha_gen` | `tdig_sdm9b_edit_alpha` |
| FLUX.2 [klein] 9B | `tdig_flux9b_alpha_gen` | `tdig_flux9b_edit_alpha` |
| FLUX.2 [klein] 9B, Qwen3.5-35B-A3B policy | `tdig_flux35b_alpha_gen` | `tdig_flux35b_edit_alpha` |

The other runs use Qwen3.5-9B. SD3.5 edits with img2img SDEdit, FLUX.2 [klein] with its own pipeline, Qwen-Image with `Qwen/Qwen-Image-Edit-2509`.

| Descriptor field | Meaning |
|---|---|
| `format: "tdig"` | instruction from `tdig.instruction` (`instruction` or `edit_instruction`), as in training |
| `policy_base`, `lora_rank`, `max_model_len`, `max_prompt_length`, `limit_images` | the policy as trained |
| `renderer` | a key of `t2i_models.json` (weights under `$T_DIG_MODELS/<path>`) |
| `tdig_renderer` | the T2I model's name in `tdig/` |
| `task` | `expand` (generation) or `layered` (editing) |
| `vlm` | whether the policy sees images (true for the editing runs) |
| `editor_kind` | `edit` (instruction editor) or `sdedit` (img2img) |
| `relative_budget` | word cap = base prompt's word count + this value |
| `train`, `val`, `edit` | `{resolution, steps, guidance}` |
| `ladder`, `alpha_ladder: true` | rungs in slider order (`0` = base model); rung N is labelled `alpha N/100` |
| `source`, `note` | where the adapter comes from |

Grids render at `val`, the edit stage at the editing run's `edit`, the i.i.d. stage at the grid settings, and an accepted set re-renders at the larger of the grid and i.i.d. resolutions. `AXIS_FW_<STAGE>_{RESOLUTION,STEPS,GUIDANCE}` (STAGE = GRID, IID, EDIT, ACCEPT) overrides a value for every T2I model; the run's `run_start` log record lists the overridden fields. To see what a `SERVE` resolves to without loading anything:

```bash
source scripts/env.sh
AXIS_FW_SERVE=tdig_flux9b_alpha_gen:tdig_flux9b_edit_alpha $PY_REWARD -m axis_framework.profiles
```

## Building the ladders

Download the released checkpoints (`https://huggingface.co/neulab/t-dig`; `build_alpha_ladders.sh` maps each ladder to its folder, e.g. `flux2-klein-9b/9b-generate` or `qwen-image-2512/9b-generate-paper`), then build the rungs on CPU with `PY_TRAIN`:

```bash
bash axis_framework/scripts/build_alpha_ladders.sh                      # all ten ladders
bash axis_framework/scripts/build_alpha_ladders.sh tdig_flux9b_alpha_gen tdig_flux9b_edit_alpha
bash axis_framework/scripts/build_alpha_ladders_35b.sh                  # the two 35B-A3B ladders
```

- `CHECKPOINTS` defaults to `$T_DIG_MODELS/t-dig-checkpoints`.
- Rungs 5 to 200 in steps of 5 go to `axis_framework/adapters/<exp>_step<N>/lora_adapter/`, each with a `scale.json` recording source and alpha.
- Existing rungs are kept; delete a ladder's directories to rebuild it.
- The launchers serve the adapters found on disk. A missing rung is reported as unavailable, never replaced by a neighbour.

`scripts/export_ladder.sh` and `export_lora.py` build a checkpoint-trajectory ladder (every saved checkpoint of one run is a rung) for runs of the older verl trainer, from `$T_DIG_RUNS/axis_grpo/ckpt/$EXP/global_step_N` (or `AXIS_FW_CKPT_ROOT`). The released policies do not need them.

## Running on one GPU

`scripts/launch_local.sh` starts all four processes on the current machine; logs and the pid file go to `axis_framework/runs/$RUN/`. `FAKE=1` needs no GPU: placeholder images, templated prompts and canned axes exercise the loop and the UI.

```bash
FAKE=1 bash axis_framework/scripts/launch_local.sh       # UI on http://127.0.0.1:19100
RUN=local bash axis_framework/scripts/launch_local.sh stop
SERVE=tdig_flux9b_alpha_gen:tdig_flux9b_edit_alpha bash axis_framework/scripts/launch_local.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `RUN` | `local` | log directory name |
| `PORT` | 19100 | orchestrator port; policy, T2I model and suggest servers use `PORT`+1 to +3, so instances can run side by side |
| `GPU_UTIL` | 0.25 | vLLM's share of the GPU, which all three services share |
| `ARMS` | `ours` | see [Study mode](#study-mode) |
| `LORA`, `VQA_URL`, `NO_AUDIT=1`, `FAKE=1` | | default adapter, VQAScore service, skip audit, no GPU |
| `PY_POLICY`, `PY_T2I` | `PY_TRAIN`, `PY_REWARD` | interpreters |

The 35B-A3B policy's weights alone take about 70 GB: serve it with `launch_pool.sh`, which gives the policy its own GPU, or raise `GPU_UTIL`. `stop` kills the instance's whole process tree, including vLLM's `VLLM::EngineCore` child, which otherwise keeps its GPU memory; other instances are not touched.

## Running on a whole node

`scripts/launch_pool.sh` gives each service its own GPU, on 2 + `T2I_GPUS` GPUs. The GPUs are those of `CUDA_VISIBLE_DEVICES` when it is set, else every GPU of the node; in that list:

```
gpu 0          policy   (port 19101)  vLLM + all LoRA rungs, CUDA graphs on
gpu 1          suggest  (port 19103)  the suggestion VLM, preloaded
gpu 2..1+R     T2I      (19110+i)     R replicas of one T2I model; TPIPS (audit) on the first
```

```bash
SERVE=tdig_qimg9b_alpha_gen:tdig_qimg9b_edit_alpha bash axis_framework/scripts/launch_pool.sh
# or sbatch --export=ALL,T_DIG_ROOT=<repo>,SERVE=... axis_framework/scripts/launch_pool.sh
```

It prints the resolved profile (and exits if the runs do not resolve), waits for every `/health`, pre-renders the start gallery (`prefetch.py`), then runs the orchestrator on `PORT` with `--t2i-pool` in the foreground; when it ends (Ctrl-C, `scancel`) the services are stopped. A grid is split across the replicas; each shard slices the latents of the whole grid, so the images match a single-server render up to batch effects (launch the replicas with `--render-batch 1` for exact parity).

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | first served T2I model | T2I model to replicate |
| `T2I_GPUS` | 4 | replicas; more than K are not used |
| `GPU_UTIL` | 0.85 | vLLM's share of the policy GPU |
| `OFF_AXIS` | 0.2 | slider start |
| `RUN`, `PORT`, `ARMS`, `LORA` | `pool_<job id>`, 19100, `ours` | as in `launch_local.sh` |
| `SUB_PORT_BASE` | 19100 | internal port base; a second instance on the node sets it and `CUDA_VISIBLE_DEVICES` |
| `NO_AUDIT=1`, `NO_PREFETCH=1`, `NO_CUDAGRAPHS=1`, `FAKE=1` | | |
| `PY_POLICY`, `PY_T2I`, `PY_SUGGEST` | `PY_TRAIN`, `PY_REWARD`, `PY_REWARD` | interpreters |

To offer several T2I models in one session, start one T2I model server per model and pass `--t2i-urls 'name=http://host:port,...'` or one `--t2i-pool` per model to `orchestrator.py`.

## Study mode

Orchestrator flags for a study; the defaults give the open demo. `launch_pool.sh` sets them from the environment variables shown.

| Flag | Variable | Effect |
|---|---|---|
| `--arms ours\|baseline\|ab` | `ARMS` | `ab` assigns the tuned policy or the untuned base per trial, deterministically from `--arm-salt` (default: the run name). Arm and audit metrics are removed from every response, and the slider is withheld, since the base arm has no ladder. |
| `--no-off-axis` | | withhold the slider in a single-arm run |
| `--base-path /prefix` | `BASE_PATH` | also serve every route under a URL prefix |
| `--ask-participant` | `ASK_PARTICIPANT=1` | ask for a participant id first, unless the link has `?pid=` |
| `--study-pictures N` | `STUDY_PICTURES` | each participant completes N pictures, rates every finished set twice on 1 to 5 (varied as asked; matches the prompt), then a questionnaire; the example gallery is hidden |
| `--study-suggestions on\|off\|split` | `STUDY_SUGGESTIONS` | show the suggestion chips, hide them, or assign per participant |
| `--completion-code`, `--completion-url` | `COMPLETION_CODE` | code returned by `/study/questionnaire`; the URL defaults to Prolific's `?cc={code}` |
| `--no-hq`, `--no-cache`, `--no-audit`, `--speculate N`, `--store` | | no re-render on accept; render the i.i.d. stage live; no metrics; speculative expansions; session store |

Prolific's `PROLIFIC_PID`, `STUDY_ID` and `SESSION_ID` link parameters are stored with each session. Instructions and consent come from `web/study_instructions.html`, a template for your own study; `{N}` becomes the number of pictures.

## Relay for nodes without inbound connections

A Cloudflare Worker (`relay/worker.js`) on your domain holds a WebSocket that `relay/relay_agent.py` opens from the node; participant requests travel down it to the orchestrator and responses stream back.

```bash
ZONE_NAME=<your domain> CF_API_TOKEN_FILE=<token file> bash axis_framework/relay/deploy_worker.sh
RELAY_URL=wss://<your domain>/axis/__relay bash axis_framework/relay/start_agent.sh
# or RELAY=1 RELAY_URL=... with launch_pool.sh
```

`deploy_worker.sh` creates the shared secret in `~/.config/axis_relay/secret` and deploys it with the Worker; the agent reads it from `RELAY_SECRET_FILE` (same default). Serve the orchestrator with `BASE_PATH=/axis` so routes match the Worker's prefix.

## Configuration

Read by `config.py`, `profiles.py` and `orchestrator.py`. The launchers first source `scripts/env.sh` (`T_DIG_ROOT`, `T_DIG_RUNS`, `T_DIG_MODELS`, `HF_HOME`, interpreters, and the site file `T_DIG_SETUP`).

| Variable | Default | Meaning |
|---|---|---|
| `AXIS_FW_SERVE` | required unless faking | served runs (`SERVE` in the launchers) |
| `AXIS_FW_FAKE` | | `1`: no GPU, placeholder outputs |
| `AXIS_FW_MODEL` | first served T2I model | T2I model a session opens on |
| `AXIS_FW_POLICY_BASE` | from the served runs | policy base model |
| `AXIS_FW_POLICY_URL`, `AXIS_FW_T2I_URL`, `AXIS_FW_SUGGEST_URL` | `http://127.0.0.1:19101`, `:19102`, `:19103` | service URLs |
| `AXIS_FW_VQA_URL` | empty | VQAScore service |
| `AXIS_FW_PY_POLICY`, `AXIS_FW_PY_T2I`, `AXIS_FW_PY_SUGGEST` | `PY_TRAIN`, `PY_REWARD`, `PY_REWARD` | interpreters |
| `AXIS_FW_SUGGEST_VLM` | `Qwen/Qwen3-VL-8B-Instruct` | suggestion VLM |
| `AXIS_FW_T2I_MODELS` | `axis_framework/t2i_models.json` | T2I model registry |
| `AXIS_MODELS` | `$T_DIG_MODELS` | T2I model weights root |
| `AXIS_FW_RUNS`, `AXIS_FW_CACHE` | `axis_framework/runs`, `axis_framework/cache` | sessions and logs; pre-rendered i.i.d. sets |
| `AXIS_FW_LEGACY_RUNS`, `AXIS_FW_CKPT_ROOT` | `$T_DIG_RUNS/axis_grpo`, `.../ckpt` | outputs of older-trainer runs, only for serving them |
| `AXIS_FW_K`, `AXIS_FW_K_CHOICES` | `4`, `2,4,8,16` | trained K; sizes the UI offers |
| `AXIS_FW_N_IID`, `AXIS_FW_N_GALLERY` | `4`, `8` | i.i.d. set size; start-gallery prompts |
| `AXIS_FW_INIT_DIVERSITY`, `AXIS_FW_OFF_AXIS` | `0.0`, `0.2` | initial diversity of live grids; slider start |
| `AXIS_FW_SPECULATE` | `3` | suggestions expanded ahead of time |
| `AXIS_FW_TEMPERATURE`, `AXIS_FW_MAX_TOKENS`, `AXIS_FW_EXPAND_RETRIES` | `1.0`, `1024`, `3` | policy sampling |
| `AXIS_FW_BASE_PATH`, `AXIS_FW_INDEX` | empty, `web/index.html` | default `--base-path`; UI page |

## Files

| File | Role |
|---|---|
| `orchestrator.py`, `session.py` | routes, state machine, arm assignment, event log, session store |
| `servers/{policy,t2i,suggest}_server.py` | the three GPU services |
| `config.py`, `profiles.py` | paths and defaults; the profile derived from the served runs |
| `policy_prompt.py` | policy instruction per run format and task |
| `latents.py` | initial-diversity noise mix |
| `client.py`, `http_util.py` | HTTP client and server helpers |
| `cache.py`, `prefetch.py` | i.i.d.-set cache; start-gallery pre-render |
| `export_lora.py` | verl FSDP checkpoint to PEFT adapter (trajectory ladders) |
| `t2i_models.json`, `served_runs/` | T2I model registry; run descriptors |
| `backend/render.py` | T2I model and editor loading, latents, TPIPS, Vendi, VQAScore client |
| `backend/rollout.py` | `parse_prompts`, `seed_for` |
| `backend/instructions.py`, `backend/layered.py`, `backend/axes.py` | instructions of round-one and edit runs and the baseline arm; the seven study axes (released runs use `tdig/instruction.py`) |
| `backend/suggest.py` | axis-suggestion prompt and parser |
| `scripts/`, `relay/`, `web/` | launchers and ladder builders; optional relay; UI and study template |
| `vendor/qwen_vl_utils/` | qwen-vl-utils for the suggest server (Apache-2.0) |
