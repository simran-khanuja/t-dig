# tdig: datasets, thresholds, validation and paper metrics

This package builds the T-DIG policies' training datasets with their per-row fidelity and quality thresholds, scores saved checkpoints outside training, picks a run's strength scale from its checkpoints, and computes every metric the paper reports, for baselines and policies alike.

The *generation policy* rewrites one base prompt into `k = 4` image prompts that vary along a requested *axis*. The *editing policy* writes `k` edit instructions for the four i.i.d. images of the base prompt. Both are trained with GRPO; the reward is TPIPS diversity conditioned on the axis text, minus a VQAScore fidelity hinge and optional length, quality and drift terms.

Run every command from the repository root after `source scripts/env.sh`; each launcher (`*.sh`) states its resources and variables in its header, and rerunning it with the same output directory continues unfinished shards.

## Files

| Path | Purpose |
| --- | --- |
| `instruction.py` | Policy instruction per reward T2I model; edit instructions per editor kind |
| `build_dataset.py` | Filter the shared split and publish one dataset directory per policy variant |
| `build_edit_dataset.py` | Turn a generation dataset into an editing-policy dataset |
| `recap_dataset.py` | Copy variants with a shorter word cap, fixed or per row |
| `iid_vqa.py`, `iid_vqa.sh` | VQAScore of the i.i.d. images; `align_thresh` per prompt and T2I model |
| `iid_quality.py`, `iid_quality.sh` | HPSv3 of the i.i.d. images; `quality_thresh` per prompt |
| `render_iid.py`, `render_iid.sh` | Generate i.i.d. images of new prompts, then measure both thresholds |
| `heldout_bundle.py` | Held-out axis bundle: the 200 held-out base prompts with their axes |
| `eval_sets.py`, `eval_sets_quality.py` | COCO-250 as policy input rows (axis `overall`); add `quality_thresh` |
| `test_set.py` | Policy rows and baseline bundle of the interactive test set, per T2I model |
| `score_baselines.py`, `score_baselines.sh` | VQAScore and TPIPS diversity of baseline image sets |
| `render_coco.py`, `render_coco.sh`, `coco_fid.py`, `coco_fid.sh` | COCO-250 images per baseline method; FID and KID |
| `validate_checkpoints.py`, `validate_checkpoints.sh` | Score saved checkpoints on held-out prompt sets |
| `select_frontier.py` | Order checkpoints on the alignment-diversity frontier into a strength scale |
| `paper_metrics.py`, `paper_metrics.sh` | Every paper metric, for baseline sets and policy checkpoints |
| `scorers/` | PickScore v1 and HPSv3 loaders |

## Dataset pipeline

```bash
# 0. Image twin of the split, for the image-input variant (CPU; axis_grpo/training/image_data.py).
$PY_REWARD -m axis_grpo.training.image_data --text-dataset $T_DIG_DATA/generation-policy-val200-v1 \
  --records $T_DIG_DATA/generation-policy-verbatim/source-records.jsonl \
  --image-cache $T_DIG_DATA/policy-image-inputs512-v1 \
  --out $T_DIG_DATA/generation-policy-val200-images512-v1 --execute
# 1. Datasets (without --execute: print row counts only).
$PY_REWARD -m tdig.build_dataset --text-source $T_DIG_DATA/generation-policy-val200-v1 \
  --image-source $T_DIG_DATA/generation-policy-val200-images512-v1 \
  --out $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated --order rotated --execute
# 2. Index the i.i.d. images of every dataset prompt (CPU; --render-root may repeat).
$PY_REWARD -m tdig.iid_vqa index --dataset $T_DIG_DATA/generation-policy-val200-v1 \
  --render-root <directory of i.i.d. images> --out $T_DIG_RUNS/tdig/iid-vqa-v1
# 3. Score on one 8-GPU node and merge.
VQA_OUT=$T_DIG_RUNS/tdig/iid-vqa-v1 bash tdig/iid_vqa.sh
# 4. Publish a copy whose rows carry align_thresh.
$PY_REWARD -m tdig.iid_vqa attach --thresholds $T_DIG_RUNS/tdig/iid-vqa-v1/thresholds.jsonl \
  --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated \
  --out $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau
# 5. Optional quality threshold (HPSv3) from the same index.
QUALITY_OUT=$T_DIG_RUNS/tdig/iid-quality-v1 \
  T2I_MODELS='qwen-image-2512+sd35-large+sd35-medium+flux2-klein-9b' bash tdig/iid_quality.sh
$PY_REWARD -m tdig.iid_quality attach --quality $T_DIG_RUNS/tdig/iid-quality-v1/quality.jsonl \
  --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau \
  --t2i-models qwen-image-2512 sd35-large sd35-medium flux2-klein-9b --out <new dataset dir>
# Tests
$PY_REWARD -m unittest tests.test_tdig -v
```

For prompts without i.i.d. images, `render_iid.sh` (`PROMPTS`, `REWARD_CONFIG`, `IID_OUT`) generates them with a run's `reward-config.json` and writes `thresholds.jsonl` and `quality.jsonl` in one job. Published directories are written once and never edited.

### Variants

`build_dataset.py` publishes one subdirectory per *policy variant*, which fixes the reward T2I model named in the instruction and whether the policy also sees images. Variants hold the same rows in the same order; only `prompt` and the `instruction_*` fields differ.

| Subdirectory | Policy input | Word cap in rule 4 |
| --- | --- | --- |
| `qwen-image-2512`, `flux2-klein-9b` | text | 150 words, 512 tokens |
| `sd35-large`, `sd35-medium` | text | 60 words, 77 tokens |
| `flux2-klein-9b-images` | text and four 512 px images | 150 words, 512 tokens |

| File | Rows |
| --- | --- |
| `train.parquet` | Four passes over the base prompts, one axis per prompt per pass |
| `val.parquet` | Every remaining row of the 200 held-out base prompts |
| `probe-contrastive.parquet` | Hard-contrastive rows of the held-out prompts; never trained on |
| `dropped-contrastive-train.parquet` | Hard-contrastive rows removed from training |
| `instruction.txt` | The instruction with `{base_prompt}` and `{axis}` open |
| `manifest.json` | Row counts, row order, source hashes, SHA-256 of every file |

### Row order of `train.parquet`

`order` in `manifest.json` names the order:

- `source`: pass `p` holds axis slot `p - 1` of every prompt, so early training sees slot 0 only.
- `rotated`: each pass holds one axis per prompt, the slot chosen by a hash of the prompt id and advanced by one each pass, so every slot appears from the first step. A pass is a body (one row per prompt: its `overall` row when that belongs to the pass, else its named axis) followed by a tail of the axes displaced by `overall` rows (`build_dataset.rotate_slots`). Use this order with `--max-steps` below one pass.

Fields added to `extra_info`:

| Field | Type | Meaning |
| --- | --- | --- |
| `index` | int | Position in its file, from 0; the trainer checks it against the row order |
| `tdig_source_index` | int | `index` of the row in the source split |
| `tdig_source_pass` | int, 1-4 | Rotated order only: the row's pass in the source order |
| `instruction_version` | string | `tdig.instruction.VERSION` (`VERSION_SHORT` after recapping) |
| `instruction_renderer` | string | Reward T2I model named in rule 4 |
| `align_thresh` | float in [0, 1] | `-customtau` datasets only ([Fidelity thresholds](#fidelity-thresholds)) |
| `align_thresh_images` | int, 1-4 | `-customtau` datasets only: number of i.i.d. images averaged |
| `quality_thresh` | float | After `iid_quality.py attach`: mean HPSv3 of the i.i.d. images |

### Hard-contrastive filter

A *hard-contrastive* axis names something the images must not show. `build_dataset. is_hard_contrastive` flags an axis that contains `instead of`, `rather than` or `without` (except `with or without`, `with and without`, `with/without`), or starts with `Remove` (case-insensitive). The reward cannot score these: diversity is conditioned on the axis text and fidelity on the base prompt, so for "full-body shots rather than portraits" a set with one portrait scores higher than four full-body shots. Axes that only add something ("Include tennis players") stay in training.

## Instruction

`instruction.py` holds verbatim copies of `INSTRUCTION_V4`, `LENGTH_RULE`, `LENGTH_BUDGETS` and `INSTRUCTION_OVERALL` from `axis_framework/backend/instructions.py`; `tests/test_tdig.py` compares them byte for byte.

- Named axes get `INSTRUCTION_V4` (rule 3: "Vary ONLY the given axis"); `overall` rows get `INSTRUCTION_OVERALL`, whose rules 3 and 5 ask for variety across several attributes.
- The image variant inserts `IMAGE_CONTEXT` and four `<image>` placeholders before "Write exactly".
- `instruction(..., word_cap=N)` swaps rule 4 for `SHORT_LENGTH_RULE` and sets version `tdig-v4-2`. `recap_dataset.py` publishes such copies: `--word-cap N`, or `--relative-budget B` for a per-row cap of the base prompt's word count plus `B`, stored as `length_cap`.
- `edit_instruction(..., kind)` is the editing instruction. `EDITOR_KIND` maps each T2I model to `edit` (instruction editor: FLUX.2 [klein], Qwen-Image) or `sdedit` (SD3.5 image-to-image, where the policy describes the finished image).

## Fidelity thresholds

`align_thresh` is the mean VQAScore of a prompt's i.i.d. images (up to four, independent seeds) from the variant's reward T2I model. *VQAScore* is the probability in [0, 1] that CLIP-FlanT5-XXL answers "yes" when asked whether the image shows the base prompt. The fidelity term is a hinge over the `k` images of a completion, `mean_i max(0, align_thresh - a_i)`, so an image above the threshold does not offset one below it. Since the threshold is a mean, returning the base prompt unchanged still pays a small penalty.

- `iid_vqa.py merge` writes `thresholds.jsonl` (per T2I model and prompt: `align_thresh`, `score_min`, `images`, per-image `scores`) and `thresholds.json` (mean, standard deviation and percentiles per T2I model). `attach` writes `align_thresh` into `reward_model.ground_truth` and `extra_info`.
- Thresholds are measured at inference settings: 1024 px, 40 (SD3.5), 4 (FLUX.2 [klein]) or 50 (Qwen-Image) steps. Training images are 512 px with 16 steps (4 for Klein); validation images use inference settings, so only the validation penalty matches the threshold's view.
- The optional ground-truth keys are `align_thresh`, `quality_thresh`, `length_cap`, `task` and `source_images` (`axis_grpo.training.protocol.parse_spec` rejects others). The reward uses the row's `align_thresh` when present, else `--tau` (`axis_grpo.training.metrics.score`); the logged `fidelity_threshold` is the value applied. For one threshold on every row, use the dataset without `-customtau`.

## Metrics recorded during training

The trainer writes step means of the reward metrics to `<run>/policy/checkpoints/reward-metrics/<step>.json` (fields `step`, `completions`, `malformed`, `mean_valid`, `per_completion`). `malformed` completions did not parse into exactly `k` prompts and are never rendered. Per-image values are in `detail` of the reward service's `results/<request>.json`. Metrics are means over the `k` images unless stated:

| Metric | Range | Meaning |
| --- | --- | --- |
| `score` | below 1 | The reward |
| `normalized_vendi` | 0-1 | Per completion: `(vendi - 1) / (k - 1)` of the axis-conditioned TPIPS embeddings; the diversity term for `--metric vendi` |
| `pairwise` | 0-2 | Per completion: mean pairwise cosine distance of the same embeddings |
| `fidelity_mean`, `fidelity_min` | 0-1 | VQAScore against the base prompt |
| `fidelity_penalty`, `fidelity_threshold` | 0-1 | The hinge and the threshold it used |
| `drift_mean`, `drift_penalty` | 0-1 | Edit task: DreamSim distance of each edit from its source, and the drift hinge |
| `imagereward`, `hpsv3`, `hpsv2`, `clip_score`, `aesthetic` | | When listed in `--track-metrics` (`axis_grpo/training/tracked.py`); logged only, except `hpsv3` with a quality term |

## Training settings for these datasets

The prepare, preflight and submit steps are in [axis_grpo/training/README.md](../axis_grpo/training/README.md). Options used with these datasets, and their values when omitted:

| Option | Omitted | Effect |
| --- | --- | --- |
| `--tau 0.9` | required | Threshold of a row without `align_thresh` |
| `--metric vendi` | `pairwise` | Diversity term |
| `--save-freq 50` | `100` | Optimizer steps between checkpoints |
| `--max-steps 1200` | `0` (one epoch) | Step cap; not part of the resume identity, so raising it extends a finished run |
| `--length-weight 0.1` | `0` | Subtract `w * mean_i min(3, max(0, words_i - cap) / cap)` |
| `--length-cap 30` | T2I model family's cap | `cap`: 60 for SD3.5, 150 for Klein and Qwen, 30 for edit instructions; a row's `length_cap` wins. Set it to a recapped dataset's cap |
| `--quality-weight 0.05`, `--quality-margin 1.0` | `0`, `1.0` | Subtract `w * mean_i min(6, max(0, quality_thresh - margin - h_i))`, `h_i` the HPSv3 of image `i`; needs `hpsv3` tracked and `quality_thresh` |
| `--sizes 9B` | `4B 9B` | From `2B`, `4B`, `9B`, `35B-A3B` (rollout tensor parallel 2; MLP LoRA reaches the shared expert only) |
| `--lora mixer` | `all` | `all`: 12 module types (attention, linear-attention, MLP); `mixer`: the 9 without MLP |
| `--validation offline` | `inline` | Score checkpoints with `validate_checkpoints.sh` instead of pausing training |
| `--track-metrics imagereward hpsv3 clip_score aesthetic` | none | Logged scores |
| `--task edit` | `generate` | Editing policy ([Editing policy](#editing-policy)) |

```bash
$PY_TRAIN scripts/prepare_training_matrix.py \
  --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau \
  --t2i-models qwen-image-2512 --sizes 9B --out $T_DIG_RUNS/<matrix> \
  --tau 0.9 --lam 2.0 --learning-rate 2e-5 --metric vendi --length-weight 0.1 --quality-weight 0.05 --quality-margin 0 \
  --save-freq 50 --save-freq-early 25 --max-steps 2000 --validation offline \
  --track-metrics imagereward hpsv3 clip_score aesthetic --execute
```

## Held-out axis bundle for baselines

Baselines are generated and scored as in [baselines/README.md](../baselines/README.md). Their held-out prompt set is a bundle of the 200 held-out base prompts, loaded by `benchmarks/heldout.py`; each row's `metadata.axes` lists the axes to score diversity along and `metadata.align_thresh` the fidelity threshold per T2I model:

```bash
$PY_REWARD -m tdig.heldout_bundle \
  --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau --out <new bundle dir>
```

## Checkpoint validation

`validate_checkpoints.sh` scores checkpoints of one run, during or after training, on one 8-GPU node. Per checkpoint it merges the FSDP shards into a LoRA adapter, samples one completion per row with vLLM (temperature 1.0, thinking off), and scores it with the run's own `reward-config.json` at validation settings (1024 px, inference step counts). Step 0 is the untrained model.

| Set | File | Axis |
| --- | --- | --- |
| Held-out axes | `<dataset>/<variant>/val.parquet` | each row's axis, with its `align_thresh` |
| Contrastive probe | `<dataset>/<variant>/probe-contrastive.parquet` | hard-contrastive axes |
| COCO-250 | `<eval sets>/<variant>/coco250.parquet` | `overall` |

`python -m tdig.eval_sets --benchmark coco250 --out <eval sets>` writes the COCO-250 rows (`--iid-run` supplies the image variant's input images). They carry no `align_thresh`, so they use `--tau`; for a run with the quality term add `quality_thresh` with `tdig.eval_sets_quality`.

```bash
TRAIN_RUN=$T_DIG_RUNS/<matrix>/<job> VAL_OUT=$T_DIG_RUNS/tdig/<new dir> \
  STEPS=0+300+600+900+1200 SETS='heldout=<val.parquet>;coco250=<coco250.parquet>' \
  bash tdig/validate_checkpoints.sh
```

Outputs per step and set: `<out>/step_<N>/<set>.jsonl` (per row: `row_id`, `base_prompt`, `axis`, `k`, `completion`, `request_id`, `metrics`) and `<set>.summary.json` (`rows`, `malformed`, `words_per_prompt`, `mean_valid`). Images are in `<out>/reward-<i>/images/<request_id>/`. The job then writes the paper metrics to `<out>/paper-metrics/` unless `PAPER_METRICS=0`.

## Strength scale from the checkpoint frontier

`select_frontier.py` orders a run's validated checkpoints into strengths. Per step it takes the held-out means of `fidelity_mean` (VQAScore against the base prompt) and `normalized_vendi` (TPIPS Vendi on the axis). Step B beats step A when B is better on one by more than `--margin` (0.005) and no worse on the other by more than `--margin`. Unbeaten steps, from most aligned to most diverse, are strengths 1 to X; step 0 can be one.

- A step whose completions ask for text absent from the base prompt more than `--max-text-share` (0.10) of the time is excluded, since printed captions raise VQAScore. `frontier.csv` reports each step's `text_share`.
- The held-out prompts are split in two by a hash of their text: the frontier is chosen on one half and reported on the other (`--select-fraction`, default 0.5; 1 is for inspection only).
- Validation directories with the same name pool into one run; they must share parquet, seeds and reward configuration. Only completed steps are read, so reruns are safe while jobs write.

```bash
$PY_REWARD tdig/select_frontier.py $T_DIG_RUNS/tdig/<validation dir>/*/ \
  [$T_DIG_RUNS/tdig/<more steps of the same runs>/*/] --out $T_DIG_RUNS/tdig/frontier-v1
```

Outputs: `frontier.json`, `frontier.csv` and `frontier.png` (needs matplotlib; `--no-plot` skips it).

## Editing policy

The editing policy sees a base prompt, an axis and the four i.i.d. images of the prompt, and writes four edit instructions; the reward T2I model's editor applies instruction `i` to image `i`. Editors: FLUX.2 [klein]'s own image input; Qwen-Image-Edit-2509 for Qwen-Image (`--editor-path`, default the pinned hub snapshot); SD3.5 image-to-image with strength `--edit-strength` (service default 0.6). Every edit starts from the row's seed.

`build_edit_dataset.py` derives the dataset from one text variant of a `-customtau` dataset, keeping rows, order and axes. Each row gains the `edit_instruction` for the editor's kind, an `images` column of 512 px policy inputs, and ground-truth keys `task: "edit"` and `source_images` (`k` full-size paths, resized by the service to the split's resolution). `--images` picks the sources; every editing run in the paper uses `assigned`, the i.i.d. images of the T2I model each prompt was assigned at elicitation, with thresholds recomputed on those sources (the module docstring lists the other choices).

```text
reward = diversity(edited images, axis)
       - fidelity_weight * mean_i max(0, align_thresh - a_i)
       - drift_weight    * mean_i min(1, max(0, d_i - drift_threshold))
       - length_weight   * mean_i min(3, max(0, words_i - cap) / cap)
       [- quality term, as for generation]
```

`d_i` is the DreamSim distance between source `i` and its edit (`--drift-weight 1.0`, `--drift-threshold 0.35` by default); without the drift term, replacing the scene maximises diversity. `cap` is 30 words for an instruction editor and 60 for SD3.5.

```bash
$PY_REWARD -m tdig.build_edit_dataset \
  --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau/qwen-image-2512 \
  --t2i-model qwen-image-2512 --images assigned \
  --iid-index $T_DIG_RUNS/tdig/iid-vqa-v1/index.jsonl \
  --thresholds $T_DIG_RUNS/tdig/iid-vqa-v1/thresholds.jsonl \
  --quality $T_DIG_RUNS/tdig/iid-quality-v1/quality.jsonl \
  --input-cache $T_DIG_DATA/tdig/edit-inputs512-v1 \
  --out $T_DIG_DATA/tdig/edit-policy-tdig-qwen-image-2512 --execute
$PY_TRAIN scripts/prepare_training_matrix.py --task edit \
  --dataset $T_DIG_DATA/tdig/edit-policy-tdig-qwen-image-2512 --t2i-models qwen-image-2512 \
  --policy-observation elicitation_images --out $T_DIG_RUNS/<new matrix> \
  --tau 0.9 --lam 1.0 --learning-rate 2e-5 --metric vendi --length-weight 0.1 --quality-weight 0.05 --quality-margin 0 \
  --save-freq 50 --save-freq-early 25 --max-steps 800 --validation offline \
  --track-metrics imagereward hpsv3 clip_score aesthetic --execute
```

`--task edit` needs exactly one T2I model, `--policy-observation elicitation_images`, and a dataset built for that editor.

## Paper metrics

`paper_metrics.py` scores baselines and policy checkpoints with the same code; its module docstring defines every key. The paper reports keys scored against the base prompt:

| Paper section | Keys |
| --- | --- |
| Alignment | `clipscore`, `vqascore_mean`, `vqascore_min` |
| Diversity | `tpips_vendi` (per axis), `dino_vendi`, `dreamsim_vendi` |
| Preference and aesthetics | `pickscore`, `imagereward`, `hpsv3`, `aesthetic` |

Methods that rewrite the prompt (PAG and every generation policy) also get `*_own` scores against their rewritten prompts. `reward` and `fidelity_penalty` appear on held-out axis rows only.

```bash
# Baselines: COCO-250 and the held-out axis set in one directory.
$PY_REWARD -m tdig.paper_metrics index-baselines --run <baseline run dir> [--run ...] \
  --coco-subset $T_DIG_DATA/tdig/coco250-v1/subset.json --out $T_DIG_RUNS/tdig/paper-metrics-baselines
METRICS_OUT=$T_DIG_RUNS/tdig/paper-metrics-baselines bash tdig/paper_metrics.sh
# Policy checkpoints (validate_checkpoints.sh runs this itself).
$PY_REWARD -m tdig.paper_metrics index-validation --val-out <VAL_OUT> --out <VAL_OUT>/paper-metrics
METRICS_OUT=<VAL_OUT>/paper-metrics bash tdig/paper_metrics.sh
```

`paper_metrics.sh` runs one VQAScore shard and one shard of every other scorer per GPU, then merges into `rows.jsonl` (one line per set and axis), `summary.json` and `summary.csv`. A group is `<T2I model>|<method>|<benchmark>` for a baseline and `<training run>|step_<N>|<set>` for a policy.
