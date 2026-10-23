# Verified i.i.d. image sets

This package generates `k` i.i.d. images (default 4) for every prompt of a curated pool, from the unchanged base prompt, with no axis and no prompt expansion. Every image is verified and checksummed, and a run resumes without regenerating completed images. It has two entry points:

- `python -m iid.curated` with `iid/render_curated.sh`: the images that axis elicitation reads ([Curated runs for elicitation](#curated-runs-for-elicitation)). Each T2I model generates with its native Diffusers pipeline from `baselines/`, with the settings of `python -m baselines config --method iid`.
- `python -m iid`: images requested one at a time from the `axis_framework` T2I model server (`axis_framework/servers/t2i_server.py`) over HTTP ([Setup](#setup) onwards).

| File | Purpose |
| --- | --- |
| `curated.py` | Curated runs: plan, shard workers, merge, status |
| `render_curated.sh` | Launcher of one T2I model's curated images on one node |
| `cli.py` | Command line (`python -m iid`): planning, execution, resume |
| `runner.py` | Seeds, run manifest, image verification, receipts and summary |
| `service.py` | HTTP client for the T2I model server |

## Curated runs for elicitation

Run in the reward environment (`PY_REWARD`). One run holds all four T2I models; each needs one 8-GPU node.

```bash
# Freeze the prompts, configs and weight locks (read-only without --execute).
$PY_REWARD -m iid.curated prepare --input <clustered pool JSON> --out $T_DIG_RUNS/curated-iid --execute
# Generate one T2I model's images; repeat for qwen-image-2512, sd35-large, sd35-medium, flux2-klein-9b.
CURATED_RUN=$T_DIG_RUNS/curated-iid T2I_MODEL=flux2-klein-9b bash iid/render_curated.sh
$PY_REWARD -m iid.curated status --out $T_DIG_RUNS/curated-iid
```

- `--input` is a schema-2 `clustered_pool` from `hpdv2`. A pool whose `upstream_complete` is false (the clustering sweep excludes unresolved prompts) needs `--allow-partial-input`, which the plan records.
- Prompts are split into eight shards by sorted id. The launcher verifies the weights against `configs/t2i_models/<T2I model>.assets.lock.json` once, runs the shards over the visible GPUs (one per GPU with eight), and writes `<T2I model>/summary.json` when all are complete. `MAX_NEW_IMAGES` caps the new images per shard, for a smoke run.
- Image `i` of a prompt uses the seed `seed_for(prompt id, 42, i)` (`baselines/assets.py`), independent of shard, GPU and resume boundaries. Rerun the launcher to resume; committed images are verified and kept.
- Layout: `plan.json`, `plan-hash.json`, `prompts.json` and `<T2I model>/shard-NN/{manifest.json,summary.json,images/<prompt id>/<NNNN>/{image.png,receipt.json}}`. `python -m elicitation prepare --render-root` and `tdig.iid_vqa index --render-root` read it. A run refuses to continue when its plan, prompts, weight lock, package versions or a committed image changed.

## Setup

This section and the next ones describe `python -m iid`. Use the reward environment (`PY_REWARD`). Start a T2I model server with the chosen model; it must share a filesystem with this client and see the same absolute run directory:

```bash
$PY_REWARD -m axis_framework.servers.t2i_server --model sd35-large --port 19102
```

`--input` is a schema-2 `clustered_pool` JSON from `hpdv2`. `--allow-partial-input` accepts an incomplete pilot input and records the incomplete lineage; do not use it for a final dataset. Per-prompt `split` labels are kept; screening axes are dropped.

## Plan, execute, resume

```bash
python -m iid --input "$T_DIG_DATA/hpdv2-clustered.json" \
  --run-dir "$T_DIG_RUNS/iid-sd35-large-k4" --t2i-url http://127.0.0.1:19102 \
  --model sd35-large --model-path /absolute/path/to/pinned/checkpoint \
  --t2i-model-revision CHECKPOINT_COMMIT --resolution 1024 --steps 40 --guidance 4.5
# Then add, for example: --execute --limit 2 --max-requests 8
```

- Without `--execute` the command reports the remaining work; it makes no network calls and writes nothing.
- `--max-requests` (required with `--execute`) caps the image requests of one invocation; `--limit` caps the pending prompt sets (`--limit 0` makes no calls). There are no automatic retries.
- `--model-path` must equal the `model_path` the server returns, which is checked for every image. `--t2i-model-revision` is recorded but not attested: the server reports no revision or weight checksum, so pin its checkpoint.
- Resume with the same command and a new budget. An existing run must match the input, model settings, `k`, seeds, revision, code, Python and Pillow version; change any of them with a new run directory. Saved images are decoded and checksummed before reuse, and only missing indices are generated.
- Exit codes: 0 for a plan or a complete pool, 2 for a partial pool, 1 for an error. Before using pilot outputs, check both `complete` and `full_pipeline_complete` in `summary.json`.

## Seeds and verification

- Seeds depend on `--seed` (default 42), the prompt id and the image index, not on prompt order, `k` or invocation boundaries, so the first four seeds of a `k = 8` set equal those of `k = 4`.
- The server draws the latent from the request seed plus 10,000,019, modulo 2^31 - 1 (`init_diversity=1.0`, `index_offset=0`, `n_total=1`). Both seeds are recorded.
- A response must name exactly one local PNG and echo the requested prompt, model, seed, steps, guidance, resolution and noise settings. Fake server responses are rejected. An image is accepted after PNG structure checks, full decoding, mode and size checks, and a SHA-256 checksum. These checks establish integrity and provenance, not visual quality.

## Output

- `manifest.json`: input, configuration, code and environment identity; `input.json`: the input.
- `images/<prompt id>/<index>.png`, with a `.json` receipt: prompt id, index, seeds, checksum, byte count and the saved T2I model response.
- `staging/<request id>/response.json`: saved responses; failed outputs stay here.
- `summary.json`: complete sets (`items`), missing indices, errors and upstream lineage, rewritten every 100 new images and at the end of each invocation.

One process owns a run at a time. A changed or corrupt image, a missing receipt or an incompatible manifest stops the run. To regenerate an image, move its image and receipt out of the run directory first.

## Tests

```bash
python -m unittest tests.test_iid tests.test_iid_curated -v
```
