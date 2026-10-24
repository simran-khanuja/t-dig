# Multimodal axis elicitation

This package produces the training axes. For each curated base prompt, a vision LLM looks at the four i.i.d. images from one T2I model, with in-context examples of human variation requests, and proposes kinds of variation a viewer might ask for. Each candidate pairs a human-style request with a short axis name. HPDv2 screening (`hpdv2/`) proposes its own candidate axes; they are not used here.

Run from the repository root with `PY_TRAIN`, the environment that has both `openai` (the API client) and Pillow. `run` calls a paid vision-LLM API and needs a key.

| File | Purpose |
|---|---|
| `__main__.py` | `prepare` and `run` subcommands |
| `selection.py` | Prompt-to-T2I-model assignment, frozen before looking at images; image verification |
| `policy.py` | System prompt, JSON response schema and strict parser |
| `examples.py` | Example bank loading and in-context example sampling |
| `curated_examples.py` | Builds and validates a merged in-context bank from source-linked choices |
| `image_inputs.py` | Optional resized API copies of images; originals are never changed |
| `api.py` | OpenAI-compatible gateway client; no SDK retries, no fallback model |
| `runner.py` | Bounded runner: pinned requests, raw replies saved before parsing, explicit resume |
| `mixture.py` | Chooses the policy-input wording (short name or natural request) per training row |
| `tpips_compat.py` | Tokenizer workaround for loading the TPIPS checkpoint; used by training and scoring |

Tests: `python -m unittest discover -s tests -p 'test_elicitation.py' -v` (also `test_curated_examples.py`, `test_image_inputs.py`).

## Assignment

`prepare` reads a finished i.i.d. run with K = 4 images per prompt from each of the four T2I models (`qwen-image-2512`, `sd35-large`, `sd35-medium`, `flux2-klein-9b`). The run directory holds `plan.json`, `plan-hash.json`, `prompts.json` and `<T2I model>/shard-NN/` (eight shards), each with `manifest.json` and `images/<prompt id>/<index>/{image.png,receipt.json}`.

```bash
python -m elicitation prepare --render-root <i.i.d. run> \
  --out $T_DIG_RUNS/axis-elicitation --per-t2i-model 5000 --seed 42
```

- Prompts are sorted by id, shuffled with the seed and assigned round-robin: exactly `--per-t2i-model` per T2I model (default 5,000, so 20,000 in total). The rest are listed as `unused_ids`. Selection never looks at image availability or quality.
- Outputs: `assignment.json`, `assignment-hash.json`, `summary.json`.
- At run time each image is checked against its shard manifest and receipt (prompt, seed, index). An incomplete image set waits; it never falls back to fewer images or another T2I model.

## Response schema

The system prompt is in `policy.py` (`VERSION` names its revision). It asks for variation that is visible in the images and left open by the prompt, never a change to what the prompt fixes ("red chair" fixes the chair's color, not the table's), and never a repair. It asks for fewer candidates, `no_variation` or `unassessable` rather than padding. Request wording should mix concise names, brief requests and some one-sentence requests (about 65/20/15 percent across the dataset).

A response has `status` (`ok` exactly when axes are present, else `no_variation` or `unassessable`), `reason`, `alignment_notes` and `axes`. Each axis has exactly:

- `request`: human-style wording; `observation`: visible evidence; `why_free`: why the prompt leaves it open; `constraints`.
- `tpips_factor`: short noun phrase for the same axis (letters, digits and spaces; no `and`, `or`, `not`).
- `mapping_status`: `faithful` or `needs_review`. This is the model's own claim; schema validation cannot show that a request respects the prompt or matches its factor, so audit mappings.

Duplicate requests and factors are rejected.

## In-context examples

Examples are human-written variation requests, kept verbatim. A bank is one of:

- `{"kind": "reviewed_axis_examples", "reviewer": ..., "examples": [...]}`: records from an inventory, each with `approved: true`, a `review_note` and source metadata (file, line, SHA-256), re-hashed on load. Interactive-session examples need `provenance: "interactive_reviewed"`; images are attached only with `image_reviewed`.
- `{"kind": "curated_merged_axis_examples", ...}`: a merged bank from `curated_examples.py`. A version-2 plan picks one exact source string per similarity group; a version-1 plan holds edited wording (`provenance: "human_derived_edited"`).

The banks used for the paper were built from an inventory of the human-study responses, which are not released; `run` refuses a bank whose examples are not reviewed.

Sampling excludes examples whose normalized prompt equals the target's. A reviewed bank is sampled by source group, then participant, without repeating a source prompt; a merged bank uniformly.

## Running

The API config has exactly these fields, plus an optional `image_max_side`:

```json
{
  "api_base": "https://YOUR-GATEWAY/v1",
  "model": "YOUR-VISION-MODEL-ID",
  "reasoning_effort": "medium",
  "timeout": 180,
  "detail": "high",
  "max_tokens": 4096,
  "max_axes": 4,
  "example_count": 4,
  "seed": 42
}
```

- `max_axes` is a cap, not a target. `detail`: `low`, `high` or `auto`. `reasoning_effort`: `null`, `none`, `low`, `medium` or `high`. `example_count`: a positive integer or `"all"`.
- `seed` controls example selection, not API sampling.
- An unsupported parameter fails the request; it is not dropped or rerouted.
- `image_max_side` sends aspect-preserving LANCZOS-resized PNG copies, never enlarged; `request.json` records what was sent.

```bash
# Dry run (default --limit 40): checks image sets and examples, makes no API calls.
python -m elicitation run --assignment $T_DIG_RUNS/axis-elicitation/assignment.json \
  --examples <reviewed or merged bank> --config <config.json> \
  --out $T_DIG_RUNS/axis-elicitation/pilot --limit 40
# Paid run: --max-calls is required, a cumulative request bound across resumes.
python -m elicitation run ... --execute --max-calls 40
```

`--limit` takes the first N items of the assignment and defaults to 40, a pilot; a full run passes the assignment's size (`--limit 20000` for the default assignment). The selected ids are part of the run identity in `manifest.json`, so a run cannot be extended with a larger `--limit` later: start a full run in its own output directory.

- The API key is read from the variable named by `--api-key-env` (default `LITELLM_API_KEY`).
- Requests are serial, one per prompt, with no automatic retries. `--retry-errors` resends provider failures; `--retry-uncertain` resends interrupted calls that may have been billed. Refusals, truncations and schema failures with saved replies are not resent.
- `manifest.json` pins code, config, policy, bank and assignment; a changed identity is refused, so use a new output directory. `status.json` counts pending, invalid, `no_variation`, `unassessable` and budget-pending prompts.
- Per prompt: `request.json`, `attempt.json`, `reply.json`, then `result.json` (with `semantic_review: pending`) or `validation.json`.

## Training rows

Training uses one row per prompt and axis, with the request verbatim. `mixture.mix(rows, short_fraction, seed, reward_wording=...)` sets each row's policy input to the short factor or the natural request (`input_style`: `tpips` or `natural`) within each T2I model and split; `reward_wording` (`short` or `natural`) fixes the reward phrase. It accepts only rows with `semantic_review: approved`, `mapping_status: faithful` and no constraints, and requires that all rows of a prompt use one T2I model and all rows of a cluster stay in one split.
