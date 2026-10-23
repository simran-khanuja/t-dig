# Generation-policy training data

This package builds the text-only training data of the T-DIG generation policy from the 20,000-prompt T2I model assignment and its axis elicitation results. Each row gives the policy one base prompt and one variation request (the *axis*, for example "dog breed") and asks for `k` expanded prompts (default 4). Building the data renders no images and calls no API.

GRPO training on this data is in [training/](training/README.md). `tdig/` derives the paper's final policy datasets from this export ([tdig/README.md](../tdig/README.md)).

| File | Purpose |
| --- | --- |
| `prepare.py` | Command line: `plan`, `audit`, `build`, `check-loader` |
| `dataset.py` | Schedule, row construction, audit and Parquet/JSONL export |
| `policy.py` | Generation instruction shown to the policy |
| `training/` | GRPO trainer, reward services and multi-node launcher |

## Row order

The elicitor proposes up to four candidate axes per base prompt. The export makes four passes over the prompts, one per candidate slot, and inserts an `overall` row (vary unspecified attributes) after each block of 40 planned conditioned rows:

| Pass | Conditioned rows | `overall` rows | Total |
| --- | ---: | ---: | ---: |
| Each of passes 1-4 | 20,000 | 500 | 20,500 |
| One epoch | 80,000 | 2,000 | 82,000 |

- The table gives maxima. A prompt with fewer than four axes has fewer rows (one with three appears in passes 1-3 only), and a valid `no_variation` reply gives none. The row count is the sum of available axes per prompt plus the retained `overall` rows.
- Candidate slots are local to each prompt, in the elicitor's order: "axis 1" is not one attribute shared across prompts.
- The 20,000 prompts are 5,000 per T2I model (Qwen-Image-2512, SD3.5-Large, SD3.5-Medium, FLUX.2 [klein] 9B). With seed 42, each T2I model's ids are shuffled separately and interleaved, so each block of 40 rows holds 10 prompts per T2I model. All passes use the same prompt order.
- From each block, one prompt per T2I model is drawn for an `overall` row, and the four draws go to the four passes, rotating the T2I model order between blocks. 2,000 distinct prompts (10%, 500 per T2I model) get an `overall` row, which is 2.44% of the rows.
- Provider-blocked replies and API errors drop the prompt, including its `overall` row; they are listed with their source hashes in `exclusions.json`. Missing or invalid replies block the export instead.
- `--k` sets the number of expanded prompts per row, not the number of passes. The GRPO group size is a training setting.

Split off held-out prompts by base prompt with `axis_grpo/training/split.py`, never by row, so that no prompt has axes in both splits.

## Commands

Run in the reward environment (`PY_REWARD`, which provides PyArrow). Without `--execute` every command is read-only. Outputs go to a new directory, published atomically; nothing is overwritten. `python -m elicitation prepare` writes the assignment and `python -m elicitation run` the elicitation run.

```bash
# 1. Freeze the order and the overall selections (assignment.json, plan.json, schedule.jsonl).
python -m axis_grpo.prepare plan --assignment "$T_DIG_RUNS/<assignment-run>/assignment.json" \
  --out "$T_DIG_DATA/generation-policy-plan" --seed 42 --k 4 --execute

# 2. Optional: list missing or problematic inputs (readiness.json) and preview rows.
python -m axis_grpo.prepare audit --plan "$T_DIG_DATA/generation-policy-plan" \
  --elicitation-run "$T_DIG_RUNS/<elicitation-run>" \
  --out "$T_DIG_DATA/generation-policy-audit" --execute

# 3. Build the export once every input is resolved.
python -m axis_grpo.prepare build --plan "$T_DIG_DATA/generation-policy-plan" \
  --elicitation-run "$T_DIG_RUNS/<elicitation-run>" \
  --out "$T_DIG_DATA/generation-policy-verbatim" --reward-text request --execute
```

- A plan is immutable; a different seed needs a new output path.
- The audit's `preview.txt` and `preview.jsonl` are for inspection, not training.
- An incomplete build exits with code 2 and writes nothing. It never fills a missing slot by duplicating an axis, synthesizing one, or swapping in another prompt. Every result must match its saved raw reply; source files are never edited.

## Export

| File | Purpose |
| --- | --- |
| `train.parquet` | verl-compatible rows, in the frozen order |
| `train.jsonl` | Same rows and order as text |
| `source-records.jsonl` | One record per base prompt: all axis candidates, source paths and hashes, the four image paths, seeds and receipts, the elicitor model |
| `assignment.json`, `plan.json` | Frozen T2I model assignment; row order and overall selections |
| `loader-settings.json` | Order-preserving loader settings; not a launch configuration |
| `readiness.json`, `manifest.json` | Completeness checks, source identity, export hashes |
| `builder-source/` | Snapshot of the builder and the generation instruction |

`extra_info.index` is the training index, `planned_index` the position in the maximum-size schedule, and `row_id` stays stable when earlier slots are omitted. The build checks image receipts against the T2I model manifests but does not reread the images, which the text policy never sees.

## Policy input and reward text

The policy sees only the base prompt and the axis request, through the instruction in [policy.py](policy.py): keep the base prompt, copy the request verbatim, and write exactly `k` self-contained prompts as `<prompt>...</prompt>`. `overall` rows have their own wording.

The reward reads `reward_model.ground_truth`, a JSON string with `base_prompt`, `axis`, `tpips_factor` and `k`. With `--reward-text request` (the paper's setting), `axis` and `tpips_factor` both hold the request verbatim, so TPIPS is conditioned on the words the policy sees; the elicited short name is kept in `extra_info.concise_factor`. `--reward-text tpips_factor` conditions the reward on the short name instead. For `overall` rows both fields are `overall`.

## Order in the trainer

The [GRPO trainer](training/README.md) runs one epoch in the saved row order, keeps short batches and finishes each pass before the next. `python -m axis_grpo.prepare check-loader --dataset <export> --config <resolved.json>` checks a configuration for an unmodified verl loader instead, which drops partial batches and so needs the row count to be divisible by the batch size.

## Tests

```bash
python -m unittest discover -s tests -p 'test_training_data.py' -v   # needs PyArrow
```
