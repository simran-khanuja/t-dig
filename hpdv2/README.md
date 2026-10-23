# HPDv2 prompt curation

This package selects base prompts from HPDv2 that leave at least three meaningful visual dimensions open to variation, in four stages:

1. `prepare`: a versioned, deduplicated prompt pool from HPDv2 JSON.
2. `filter --stage pass1`: high-recall LLM screening.
3. `filter --stage pass2`: LLM eligibility check that names three axes with example values.
4. `cluster`, or the GPU sweep `scripts/run_cluster_sweep.py`: embedding-based deduplication that keeps one prompt per semantic cluster.

Run from the repository root with `PY_REWARD`. Preparation, filtering and review need no GPU. `filter` needs `litellm`, which no requirements file includes; install it in its own environment so the reward environment's pins stay intact (`python -m venv envs/api && envs/api/bin/pip install litellm`), and run `filter` with that interpreter. Clustering needs `sentence-transformers` 5.7.0, installed as an import overlay:

```bash
$PY_REWARD -m pip install --target .cache/cluster-python --no-deps sentence-transformers==5.7.0
```

`scripts/launch_cluster_sweep.sh` puts the overlay on `PYTHONPATH`; for `cluster`, prefix the command with `PYTHONPATH=.cache/cluster-python:.`. Every `filter` and `cluster` command only prints a plan unless `--execute` is given.

| File | Contents |
|---|---|
| `cli.py` | `show-policy`, `prepare`, `filter`, `review`, `cluster` |
| `policy.py` | Versioned system prompts of both passes; response validation |
| `dataset.py` | Pool preparation and input loading (schema version 2) |
| `api.py` | LiteLLM-proxy or direct Anthropic client |
| `runner.py` | Batched execution, checkpointing, resume, summaries |
| `storage.py` | Atomic JSON, journals, locks, hashing, job id |
| `review.py` | Stratified HTML review page |
| `cluster.py`, `cluster_sweep.py` | Embedding, average-linkage clustering, per-cluster selection; multi-threshold cuts and review page |

## Selection policy

`python -m hpdv2 show-policy --stage pass1` (or `pass2`) prints the exact system prompt. The user message is the batch as a JSON array of `{id, prompt}` objects.

- Both passes exclude identifiable real people, sexual or graphic content, prompts with no depictable subject, and stacks of artist/platform/render descriptors. Explicit requirements of the prompt stay fixed: "a red chair" fixes the colour but leaves material and design open.
- Pass 1 rejects only prompts clearly too constrained for three dimensions, returning `id`, `keep`, `reason`.
- Pass 2 keeps a prompt only if it names three distinct, meaningful, unconstrained axes, each with two example values. Axes can be prompt-specific (time on a wall clock) or broad (artistic medium); a generic background/lighting/style trio does not qualify.

```json
{"id": "<prompt SHA-256>", "keep": true,
 "reason": "The clock's time, face design and housing material are open.",
 "variation_axes": [{"dimension": "time displayed", "values": ["3:00", "9:00"]},
                    {"dimension": "clock-face design", "values": ["Roman", "Arabic"]},
                    {"dimension": "housing material", "values": ["wood", "metal"]}]}
```

Rejects have `variation_axes: []`. These axes only screen prompts; the training axes come from the image-based elicitation (`elicitation/`).

## 1. Prepare the pool

```bash
python -m hpdv2 prepare --source /path/to/train.json --split train \
  --output $T_DIG_DATA/hpdv2-train-v1.json
# or download ymhao/HPDv2 at a full 40-character commit (the paper's pool: bb0f2563e451e11c3f376b2ba93615db00e042e1)
python -m hpdv2 prepare --download --revision bb0f2563e451e11c3f376b2ba93615db00e042e1 --split train \
  --output $T_DIG_DATA/hpdv2-train-v1.json
```

The input is a list of objects or an object with a `data` list. Prompts fall back to captions, are stripped, and exact duplicates are removed, keeping first-occurrence order. Prepare the full pool and limit spending with `filter --limit`, so the input hash stays fixed. An existing output is never replaced.

## 2. Pass 1 and 3. Pass 2

```bash
export LITELLM_API_KEY=...   # or --backend anthropic with ANTHROPIC_API_KEY; --api-key-env NAME
python -m hpdv2 filter --stage pass1 --input $T_DIG_DATA/hpdv2-train-v1.json \
  --run-dir $T_DIG_RUNS/hpdv2-v1/pass1 --backend proxy --api-base https://<gateway> \
  --model <model alias> --batch-size 30 --limit 90 --execute --max-requests 6
python -m hpdv2 filter --stage pass2 --input $T_DIG_RUNS/hpdv2-v1/pass1/summary.json \
  --run-dir $T_DIG_RUNS/hpdv2-v1/pass2 --backend proxy --api-base https://<gateway> \
  --model <model alias> --batch-size 20 --cache-system --execute --max-requests <N>
```

- Rerun the same command to continue. `--limit` (pending prompts, in pool order, including earlier failures) and `--max-requests` (every call, including retries; required with `--execute`) bound one invocation. `--limit 0` makes no calls and rebuilds the summary.
- Pass 2 needs a complete pass 1. `--allow-partial-input` (also on `cluster`) accepts an incomplete upstream summary for a pilot.
- Defaults: batch 20, 8,192 output tokens, temperature 0, 120 s timeout, 3 attempts per batch, 1 worker (`--workers`). Changing the batch size can change labels.
- `--backend anthropic` calls the Anthropic API directly (omit `--api-base`). No key is read from `.env` or a fallback variable.
- `--reasoning-effort {none,low,medium,high,xhigh,max}` (proxy, not with `--cache-system`) omits the temperature. `--short-request-ids` sends batch-local ids for models that miscopy long hashes.
- `--cache-system` requests system-prompt caching; the recorded usage shows whether it applied.

Review a stratified sample of kept, rejected, error and pending prompts (checkboxes are not saved):

```bash
python -m hpdv2 review --input $T_DIG_RUNS/hpdv2-v1/pass2/summary.json \
  --output $T_DIG_RUNS/hpdv2-v1/pass2-review.html --per-group 30 --seed 42
```

## 4. Deduplicate

### Threshold sweep (paper)

One GPU embeds all pass-2 keeps with all-MiniLM-L6-v2, builds one exact average-linkage hierarchy on cosine distance, cuts it at 0.15 to 0.45 in steps of 0.05, keeps one random prompt per cluster at each cut, and writes a blinded review page for choosing the threshold.

```bash
python scripts/run_cluster_sweep.py prepare --execute \
  --input $T_DIG_RUNS/hpdv2-v1/pass2/summary.json --out $T_DIG_RUNS/hpdv2-cluster-sweep
T_DIG_CLUSTER_RUN=$T_DIG_RUNS/hpdv2-cluster-sweep bash scripts/launch_cluster_sweep.sh
```

- `--model` defaults to `$T_DIG_MODELS/all-MiniLM-L6-v2`, which must hold a `tdig-provenance.json` with `repository` and `revision`. `--dependencies` defaults to `.cache/cluster-python`.
- `prepare` requires a complete pass 2; prompts it rejected or failed on are listed in `unresolved.json` and excluded. It freezes the input, code, launcher, model and dependencies; `run` refuses if any changed. The distance matrix is capped at 64 GiB.
- Embeddings and the tree are cached with hash receipts, so a rerun resumes after them.
- Outputs: `clustered-distance-<t>.json` per threshold, `sweep.json` (statistics, provenance, `final_threshold: null`), `audit-pairs.json`, `review.html`, `status.json`. The paper uses the 0.35 cut, `clustered-distance-0.35.json`.

### Single threshold (`cluster`)

```bash
python -m hpdv2 cluster --input $T_DIG_RUNS/hpdv2-v1/pass2/summary.json \
  --output $T_DIG_DATA/hpdv2-v1-deduped.json --distance 0.35 --max-per-cluster 1 \
  --selection random --seed 42 --embedding-revision <commit> --execute --allow-download
```

- Runs on CPU by default (`--device`); without `--allow-download` the weights must be cached.
- The distance matrix takes 8N² bytes for N prompts; `--max-matrix-gib` (default 4) caps it.
- Each cluster gives one random prompt, chosen from the seed and cluster membership only. `--selection max-min --max-per-cluster 2` takes the centroid-nearest prompt, then greedy max-min; `--min-cluster-size` drops small clusters. The output size equals the number of clusters, so a lower threshold keeps more prompts.

## Run directory of a filter pass

| File | Contents |
|---|---|
| `manifest.json` | Input, pool, code and system-prompt hashes, the system prompt, model, backend, generation settings, package versions |
| `checkpoint.jsonl` | One fsynced event per batch: input ids, raw responses, usage, validated decisions or the error |
| `summary.json` | Kept and rejected decisions, pending prompts (errors are a subset), counts, `complete` |
| `.lock` | Exclusive-writer lock |

- The prompt id is the SHA-256 of the stripped UTF-8 prompt. A response must cover exactly the requested ids; a missing, extra or duplicate id invalidates the batch. Pass-2 axes need a nonempty `dimension` and two distinct values (compared case-insensitively). Failures never count as keeps.
- `n_kept + n_rejected + n_pending == n_input`. `full_pipeline_complete` also requires complete upstream stages.
- Resume refuses a changed input, code, policy, package versions, model, backend, batch size, token limit, timeout, cache mode or partial-input setting. Workers, attempts, `--limit` and `--max-requests` may change.
- Invalid responses and transient API errors are retried with waits of `min(2^attempt, 30)` s; authentication and configuration errors stop new calls. An interruption can lose up to `--workers` in-flight batches.
- An incomplete last checkpoint line stops resume; `--execute --repair-tail --max-requests N` backs it up to `.bak` and truncates it.
- Exit codes: 0 plan or complete, 1 configuration or integrity error, 2 filtering incomplete (including a reached `--limit` or request cap).

Tests: `tests/test_pipeline.py`, `tests/test_api_review.py`, `tests/test_cluster_sweep.py`.
