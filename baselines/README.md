# Baselines

The `baselines` package generates and scores T-DIG's comparison methods on the four T2I models (Qwen-Image-2512, SD3.5-Large, SD3.5-Medium, FLUX.2 [klein] 9B):

- PAG (`pag`): Prompt Adaptation with GFlowNets (Yun et al., CVPR 2025), a learned prompt-rewriting policy; not Perturbed Attention Guidance.
- CADS (`cads`): Condition-Annealed Diffusion Sampling.
- Scaling Group Inference (`group`): selects K of M candidates during denoising.
- i.i.d. sampling (`iid`): K independent seeds, the control arm.

All methods of one T2I model share its Diffusers pipeline, scheduler, resolution, step count, guidance and K, so the results do not reproduce published numbers on other T2I models ([Upstream code and published settings](#upstream-code-and-published-settings)). Environment setup is in [SETUP.md](SETUP.md).

## Protocol

- Held-out axes: base prompts the T-DIG policy never trains on, with their axes (`benchmarks/heldout.py`). A baseline generates from the base prompt alone and is scored once per axis.
- COCO-250: 250 COCO 2017 validation captions for FID and KID ([COCO-250 FID](#coco-250-fid)).
- K = 4 by default. Group Inference needs M >= K; PAG with K = 1 decodes greedily.
- Every method is scored against the base prompt, never against PAG's rewritten prompt, which is saved separately.

## T2I model settings

| T2I model (`--t2i-model`) | Steps | Guidance | Text length | Asset lock |
|---|---:|---|---:|---|
| `qwen-image-2512` | 50 | true CFG 4.0 with the pipeline's norm correction | 512 | `configs/t2i_models/qwen-image-2512.assets.lock.json` |
| `sd35-large` | 40 | CFG 4.5 | 256 | `configs/t2i_models/sd35-large.assets.lock.json` |
| `sd35-medium` | 40 | CFG 4.5 | 256 | `configs/t2i_models/sd35-medium.assets.lock.json` |
| `flux2-klein-9b` | 4 | 1, no negative branch (distilled) | 512 | `configs/t2i_models/flux2-klein-9b.assets.lock.json` |

All use 1024 x 1024 px, bfloat16, seed 42, an empty negative prompt and the checkpoint's flow scheduler; Klein's and Qwen's settings follow their model cards. `baselines/config.py` (`t2i_model_defaults`, `validate`) holds the defaults. SD3.5 runs through `baselines/sd3.py` (TF32 off, no skip-layer guidance); Qwen and Klein through `baselines/native_pool.py`.

## Method settings

| Method | Settings |
|---|---|
| PAG | Released policy checkpoint (step 10000) in a child process with Transformers 4.46.3 (`pag_worker.py`). Input `base_prompt.rstrip() + ' Rephrase:\n'`; only the continuation is used. Released diverse-beam decoding: K beams and groups, length penalty -1, diversity penalty 1, 15 to 75 new tokens. Outputs are not filtered, so PAG can add style or artist phrases. |
| CADS | Independent noise on the token embeddings of each guidance branch, from the clean embeddings at every step. tau1 = 0.6, tau2 = 0.9, noise scale 0.25, psi = 1, per-sample rescaling. Annealing time is sampling progress, `1 - step_index / (steps - 1)`. |
| Group Inference | M = 128, K = 4, keeps `max(K, ceil(0.75 N))` candidates per step, lambda = 1. Previews `x - sigma*v` decoded by the full VAE. Unary term: CLIP-B/32 similarity to the caption; pairwise term: DINOv2-B patch distance; solved exactly with Gurobi (one thread, 30 s limit). A solve that is not proven optimal stops the run unless `accept_incumbent: true`. |

CADS conditioning (`cads.conditioning`): SD3.5 and Qwen-Image perturb the token embeddings of both branches and leave SD3.5's pooled projections clean (perturbing them collapses SD3.5 outputs); Klein has a single conditional branch, so only that is perturbed.

## Upstream code and published settings

The adaptations follow the authors' released code where it exists:

- PAG: [dbsxodud-11/PAG](https://github.com/dbsxodud-11/PAG/tree/d5f6ba2d2e896fb4be0660f8a2f1bacf3e5d5608) at `d5f6ba2`.
- Scaling Group Inference: [GaParmar/group-inference](https://github.com/GaParmar/group-inference/tree/a1fff209d3796df16d87a7876c5076721ccb1772) at `a1fff20`.
- CADS: no author release; [Algorithm 1 and Appendix G](https://arxiv.org/html/2310.17347v3#A7) of the paper.

Differences from the published methods:

- PAG: the paper gives a beam-search length penalty of +1.0; the released `eval.py`, which `baselines/` follows, uses -1.0 (`pag.length_penalty`). The paper evaluates 16 rewrites per prompt with three images each on SD1.4 at 512 px; here each of the K images comes from one rewrite on the shared T2I model.
- Group Inference: the paper uses the pruning ratio both as the retained fraction (Equation 6) and as the dropped fraction (Algorithm 1 and the CLI). `baselines/` retains 0.75 per step, following Equation 6, with M = 128 as in the paper's sweeps. The released CLI covers FLUX.1 only and decodes with the small `taef1` VAE; `baselines/` uses each T2I model's full VAE and saves PNG. The paper's first-step CFG suppression is `group.first_step_cfg`.
- CADS: the paper's setting is Stable Diffusion with 100 steps and CFG 9 on 1,000 COCO captions; tau1, tau2, the noise scale and psi are kept, and their use on the 4-step distilled Klein is an adaptation.
- Compute: Group Inference generates M = 128 candidates per prompt and PAG runs a language model, so an equal image count is not an equal compute budget.

## Generating one method

```bash
source scripts/env.sh
# A method's configuration for a T2I model (loads no models): --method iid, cads, group or pag.
$PY_REWARD -m baselines config --method iid --t2i-model sd35-large > iid-sd35-large.json
# Without --execute: a read-only plan. With it: generate up to --max-new-images new images.
$PY_REWARD -m baselines generate --bundle "$T_DIG_DATA/benchmarks/heldout-axes" \
  --config iid-sd35-large.json --assets configs/t2i_models/sd35-large.assets.lock.json \
  --run "$T_DIG_RUNS/baselines/heldout-sd35-iid-k4" --execute --max-new-images 4
```

- Rerun the identical command to resume. A changed config, asset lock or source cannot resume a run, so use one run directory per method, T2I model and K.
- `--limit N` or `--indices` selects a subset. The config uses `cuda:0`; pick the GPU with `CUDA_VISIBLE_DEVICES`.
- Each image is committed atomically with a receipt. Finished sets are at `<run>/images/<5-digit index>/samples/<4-digit slot>.png` with `metadata.jsonl`.
- Latent seeds depend only on the prompt id, the global seed and the slot, so raising K keeps the earlier i.i.d. and CADS seeds.
- The run keeps a copy of its source in `<run>/source/`; run that copy to resume after a code change.

## Scores

VQAScore and axis-conditioned TPIPS, as reported beside T-DIG:

```bash
python -m tdig.score_baselines index --run <run dir> [--run <run dir> ...] --out <score dir>
SCORE_OUT=<score dir> bash tdig/score_baselines.sh
```

The launcher runs one VQAScore and one TPIPS shard per GPU and merges them; rerunning with the same `SCORE_OUT` continues unfinished shards. COCO rows are scored with the axis `overall`.

`python -m baselines metrics --run <run> --out <dir> --assets <lock> --execute` computes CLIP-B/32 alignment and DINOv2 diversity (patch distance, CLS distance, CLS Vendi score). Group Inference selects with CLIP and DINO, so these scores are not independent evidence for it.

## COCO-250 FID

FID and KID on 250 COCO captions (`tdig/coco_fid.py`), following the Scaling Group Inference protocol. The caption pool `$T_DIG_DATA/benchmarks/coco2017-val-5000` holds one caption per validation image, the one with the smallest SHA-256 of `coco-caption-v1:42:image_id:caption_id` (`python -m baselines prepare-coco`, see [SETUP.md](SETUP.md)). The 250 captions are those with the smallest SHA-256 of `coco-fid-v1:42:<image_id>`. The K images per caption are pooled (1,000 at K = 4) against all 5,000 photographs, centre-cropped to a square. KID is reported beside FID, which is biased upward at 1,000 samples.

Checksums: prepared pool `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`, selected rows `191b68f491608caf47f5367c0c41ffe18db36310827ca64d9b2f6dab61090cf3`.

```bash
python -m tdig.coco_fid select --bundle "$T_DIG_DATA/benchmarks/coco2017-val-5000" \
  --out "$T_DIG_DATA/tdig/coco250-v1/subset.json"
# Generate with each method of one T2I model on an 8-GPU node; rerun to continue.
CONFIGS=<iid.json>:<pag.json>:<cads.json>:<group.json> RENDER_OUT=<dir> \
  ASSETS=configs/t2i_models/qwen-image-2512.assets.lock.json bash tdig/render_coco.sh
python -m tdig.coco_fid index --subset "$T_DIG_DATA/tdig/coco250-v1/subset.json" \
  --run <dir>/<method>-gpu<N> [--run ...] --out <fid dir>
FID_OUT=<fid dir> bash tdig/coco_fid.sh   # optional REAL_IMAGES, INCEPTION_DIR
```

`index` writes `coverage.json` with the rows each method still misses. `coco_fid.sh` needs Clean-FID 0.1.35 and the Inception checkpoint, installed by [SETUP.md](SETUP.md).

## Files

| File | Purpose |
|---|---|
| `config.py` | Method and T2I model defaults; config validation |
| `assets.py` | Asset locks: path expansion, SHA-256 verification, seed derivation |
| `runner.py` | `generate`: atomic commits, source snapshots, seeds, resume |
| `backend.py`, `pag_worker.py` | T2I model loading; the PAG child process |
| `sd3.py` | SD3.5 pipeline, CADS hook and Group Inference loop |
| `native_pool.py` | Group Inference and CADS on the Diffusers loop of Qwen and Klein |
| `algorithms.py` | CADS schedule, flow previews, pruning sizes, Group Inference solver |
| `features.py`, `metrics.py` | CLIP and DINOv2 features and metrics |
| `cli.py` | `python -m baselines {config,prepare-coco,lock-assets,generate,metrics}` |

```bash
python -m pytest tests/test_baselines.py tests/test_cads_adapter.py tests/test_cads_reference.py \
  tests/test_benchmarks.py
```
