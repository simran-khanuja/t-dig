# GRPO policy training

This package trains a T-DIG policy with GRPO. The policy is Qwen3.5 (4B, 9B or 35B-A3B) with LoRA; it rewrites one base prompt and one axis request into `k = 4` expanded prompts (generation) or `k` edit instructions (editing). A frozen T2I model turns each expansion into an image, and the reward scores the set for diversity along the axis (TPIPS) and fidelity to the base prompt (VQAScore).

| File | Purpose |
| --- | --- |
| `run.py` | Validate or run one policy experiment (`--execute`, `--resume`) |
| `config.py` | Training and reward-service configuration, composed into the verl configuration |
| `trainer.py` | Adapter around verl's V1 synchronous trainer: ordered batches, checkpoint markers, resume checks |
| `order.py` | Pass-bounded batch sampler that consumes each row once, including short tails |
| `reward_fn.py` | verl reward client: parses `<prompt>` tags, calls the reward service |
| `protocol.py` | JSON wire contract, reward settings, seeds, malformed-output penalty |
| `metrics.py` | Diversity and reward math (CPU-testable) |
| `service.py` | Reward service: T2I model, TPIPS, batching, cached per-request results |
| `backend.py` | Diffusers T2I models, TPIPS embedding, VQAScore client |
| `vqa_service.py` | VQAScore service around the official CLIP-FlanT5 backend |
| `tracked.py`, `hpsv3_model.py` | Logged-only image scores (ImageReward, HPSv3, HPSv2, CLIP score, aesthetic) |
| `split.py` | Prompt-disjoint train/validation split of a dataset export |
| `image_data.py` | Image twin of a split: each row plus 512 px copies of the four i.i.d. images the elicitor saw (from the export's `source-records.jsonl`), the input of `tdig/build_dataset.py --image-source` |
| `fleet.py` | Multi-node launcher: places the policy, reward and VQAScore processes on GPUs |
| `gpu_probe.py` | Bounded T2I model and scorer probe; never optimizes a policy |
| `runtime.lock.json` | Pinned package versions and hashes of the verl files the adapter depends on |

## Policies and data

- One policy per reward T2I model: `qwen-image-2512`, `sd35-large`, `sd35-medium`, `flux2-klein-9b`. All policies of a task share the training export, row order and held-out set.
- The dataset's `source_renderer` field names the images the elicitor saw; it does not choose the training T2I model.
- Each training row gets `group_size` completions of `k` prompts each. `batch_size = 8`, `group_size = 8`, `k = 4` needs 256 images per optimizer update.

## Order and checkpoints

- One epoch in the row order of the export, with no reshuffling. Each axis pass finishes before the next; `overall` rows stay where the builder put them.
- Short batches at pass boundaries and at the end are kept: no padding, dropped rows or wrap-around. Each step checks the actual `index` and `row_id` values against the schedule. `group_size` must be divisible by the number of trainer GPUs.
- Resume needs model and optimizer state, `data.pt`, and a completion marker (`tdig-complete.json`) with the same curriculum hash; data, code, runtime and service identities must also match. Resume restores the data cursor, not bitwise-identical vLLM sampling.
- Sequence packing is off: the Qwen3.5 fallback recurrent and convolution kernels ignore packed example boundaries. Token-based dynamic batching stays on.

The adapter subclasses verl's V1 synchronous trainer (`trainer.use_v1=True`, `trainer_mode=sync`) and targets verl 0.9.0.dev0, vLLM 0.24.0, torch 2.11.0, transformers 5.14.1 and flash-linear-attention 0.5.2 (`requirements/train.txt`). [runtime.lock.json](runtime.lock.json) pins the verl, torch, vLLM, transformers, ray and torchdata versions and hashes the verl files the adapter relies on, including the one `scripts/patch_verl.py` patches; `run.py` refuses to start when the installed runtime differs.

## Reward

Let `D` be the mean pairwise cosine distance between the normalized TPIPS embeddings of the `k` images, conditioned on the axis text; with `metric=vendi`, `D` is `(exp(entropy(eigenvalues(Gram/k))) - 1)/(k-1)`. Let `a_i` be the VQAScore of image `i` against the base prompt.

```text
reward = D - lambda * mean(max(tau - a_i, 0))
```

`tau` is `fidelity_threshold` and `lambda` is `fidelity_weight` (defaults 0.5 and 0.3). A row whose `ground_truth` carries `align_thresh` in [0, 1] uses that value as `tau`, reported in the `fidelity_threshold` metric. Optional terms are subtracted:

- length: `length_weight` times the mean fraction by which an expansion's word count exceeds `length_cap`, each capped at 300%;
- quality: `quality_weight` times the mean HPSv3 deficit below the row's `quality_thresh` less `quality_margin`, capped at 6 points;
- drift (editing): `drift_weight` times the mean DreamSim distance of each edited image from its source above `drift_threshold`, capped at 1.

Variants and logging:

- `mode=linear` replaces the hinge with `+ fidelity_weight * mean(a_i)`.
- `floor=overall` subtracts generic diversity from named-axis rows.
- `track_metrics` adds logged-only image scores (`tracked.py`).
- `validation: "offline"` turns off validation inside training; a separate job scores the checkpoints ([tdig/README.md](../../tdig/README.md)).

The four expansions of a completion share one initial noise, seeded from the base prompt (MD5), so every rollout of a base prompt uses the same seed whatever the axis. Malformed policy output gets the fixed penalty `-3 - fidelity_weight - 3*length_weight - drift_weight - 6*quality_weight`, below every valid reward. Scorer failures, NaNs, timeouts and missing files raise errors; they are never turned into rewards.

## Services

| Process | Environment | Role |
| --- | --- | --- |
| `vqa_service` | `PY_VQA` (`requirements/vqa.txt`) | Official CLIP-FlanT5 VQAScore backend |
| `service` | `PY_REWARD` (`requirements/reward.txt`) | T2I model, TPIPS, reward math, cached per-request results |
| `run` | `PY_TRAIN` (`requirements/train.txt`) | Qwen policy, LoRA, vLLM rollout, GRPO, checkpoints |

The services bind to 127.0.0.1 by default and their JSON API is not authenticated. Rewards are cached by exact request and configuration. Weights come from `$HF_HUB_CACHE` (default `$HF_HOME/hub`) and `$T_DIG_MODELS`. The HPSv3 scorer reads `AXIS_HPSV3_BASE` (its Qwen2-VL base, default `Qwen/Qwen2-VL-7B-Instruct`) and `AXIS_HPSV3_CKPT` (a local `HPSv3.safetensors`, default: the `MizzenAI/HPSv3` file in the hub cache).

### Single run

`run.py` reads a JSON configuration with the keys checked in [config.py](config.py). The matrix preparation below writes `reward-config.json` and `train-config.json` for every run.

```bash
# Services; --staging is the "staging" directory of reward-config.json.
$PY_VQA -m axis_grpo.training.vqa_service --cache "$HF_HUB_CACHE" --staging /path/to/staging \
  --port 18740 --execute
$PY_REWARD -m axis_grpo.training.service --config reward-config.json --out /path/to/service-out \
  --port 18750 --execute

# Validate the data and installed framework; add --execute to train, --execute --resume to resume.
$PY_TRAIN -m axis_grpo.training.run --config train-config.json --out /path/to/new-run
```

The run refuses to resume from an incomplete latest checkpoint; `--resume-from
<run>/checkpoints/global_step_N` picks an earlier complete one.

## Multi-node training matrix

The default matrix is Qwen3.5-4B and 9B for each of the four T2I models: eight jobs of two 8-GPU nodes (`--sizes` adds `35B-A3B`, `--nodes` changes the node count). Node zero gives two GPUs to the policy, five to the T2I model and TPIPS and one to VQAScore; each further node gives seven to the T2I model and TPIPS and one to VQAScore.

- LoRA rank 32, alpha 64 on the attention, Gated DeltaNet and MLP projections (`--lora mixer` drops the MLP).
- Training images are 512 px (SD3.5 16 steps, CFG 4.5; Klein 4 steps, CFG 1; Qwen-Image 16 steps, CFG 4); validation images are 1024 px (SD3.5 40 steps, Klein 4, Qwen-Image 50).
- Batch size 8 rows, 8 completions each; a checkpoint every 100 updates (`--save-freq`); with inline validation, the held-out set is scored before training and every 500 updates; `--val-axes-per-prompt N` validates on the N lowest-slot named axes of each held-out prompt plus its overall rows instead of every held-out row.
- Before the epoch, every job benchmarks batch sizes on each node at both resolutions, checks shared seeds and scorer batching, and trains and resumes two steps on three real rows. A failure stops the job before training.

1. Split the export by base prompt (50 held-out prompts per source T2I model by default). The split writes `train.parquet` and `val.parquet` (every held-out axis).

   ```bash
   source scripts/env.sh
   $PY_REWARD -m axis_grpo.training.split --source "$T_DIG_DATA/generation-policy-verbatim" \
     --out "$T_DIG_DATA/generation-policy-split" --execute
   ```

2. Freeze the source, the configurations and one `<run>/job.sh` per job. `T_DIG_SUBMIT` is the submit command stored in `matrix.json` (default `bash`, which runs the job in the foreground). Without `--execute` the script prints the plan; `--help` lists the task, reward and checkpoint options.

   ```bash
   T_DIG_SUBMIT="sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --time=72:00:00" \
   $PY_TRAIN scripts/prepare_training_matrix.py \
     --dataset "$T_DIG_DATA/generation-policy-split" --out "$T_DIG_RUNS/grpo-matrix" \
     --tau 0.9 --lam 1 --learning-rate 2e-5 --execute
   ```

3. Check every frozen configuration, the data, the runtime lock and image preprocessing on CPU (writes `preflight.json`):

   ```bash
   $PY_TRAIN scripts/preflight_training_matrix.py --matrix "$T_DIG_RUNS/grpo-matrix"
   ```

4. Submit each job once; receipts are journaled and a second submission is refused. `--expected-jobs` (default 8) must match the number of prepared jobs.

   ```bash
   $PY_TRAIN scripts/submit_training_matrix.py --matrix "$T_DIG_RUNS/grpo-matrix" --execute
   ```

### Reaching the other nodes

Each `job.sh` runs `scripts/training_fleet.sh` on node zero, which reads the host list from `T_DIG_NODEFILE` and starts the other nodes with `T_DIG_REMOTE_LAUNCH`:

- Slurm: both have defaults, `scontrol show hostnames "$SLURM_JOB_NODELIST"` and `srun --nodes=1 --ntasks=1 -w {host} --`.
- Plain bash on two machines: write the host names to a file, node zero first, and run `bash <run>/job.sh` on node zero with `T_DIG_NODEFILE=<file>` and `T_DIG_REMOTE_LAUNCH='ssh {host} T_DIG_SETUP=<site setup file>'`. ssh does not pass the environment, so name the variables the other node needs after the host.

The job id that ties the nodes of one run together is `T_DIG_JOB_ID`, else `SLURM_JOB_ID`, else `local-<pid>`. `$T_DIG_RUNS/<run>` and the frozen source must be on a filesystem shared by all nodes: node zero reads each node's `ready.json` from the run directory, and the other nodes read the configurations and the stop marker from it.

A job that outlives the scheduler's time limit is resubmitted and resumes from its last complete checkpoint. With `allow_endpoint_rebind: true` in the training configuration (default false), the resumed job may use other hosts as long as every reward slot's service hash matches.

## Tests

```bash
PYTHONPATH=tests:. $PY_TRAIN -m unittest test_training_reward test_training_fleet -v
```

The tests run by bare module name because the training environment has a site-packages `tests` package that shadows this repository's. [THIRD_PARTY.md](THIRD_PARTY.md) lists the verl code this package adapts.
