# T-DIG: Text-Conditioned Diversity for Image Generation

T-DIG makes a text-to-image (T2I) model produce a set of images that vary along a requested axis ("dog breed", "time of day") and otherwise stay faithful to the prompt. A small language model, the *policy* (Qwen3.5-4B, 9B or 35B-A3B with a LoRA adapter), reads the prompt and the axis and writes K = 4 prompts, one per image (generation), or K edit instructions for an existing set (editing). The T2I model is frozen; only the policy is trained, with GRPO. The reward is the diversity of the image set, measured by TPIPS conditioned on the axis text, minus a VQAScore fidelity hinge against the original prompt, minus length, quality (HPSv3) and, for editing, drift (DreamSim) penalties. The paper trains policies for Qwen-Image-2512, Stable Diffusion 3.5 Large, Stable Diffusion 3.5 Medium and FLUX.2 [klein] 9B, and evaluates them on held-out (prompt, axis) pairs that people wrote in the interactive tool included here.

- Paper: https://arxiv.org/abs/2610.29003
- Website: https://image-study.org/t-dig/
- Policies (LoRA adapters): https://huggingface.co/neulab/t-dig
- Interactive test set: https://huggingface.co/datasets/neulab/t-dig-test
- Policy training data: https://huggingface.co/datasets/neulab/t-dig-train

## Train your own policy

One command trains a T-DIG policy on one machine, with whatever GPUs it has, against one of the paper's four T2I models or any Diffusers text-to-image model you have. It downloads the weights and training data it is missing, starts the reward services, trains the policy with GRPO and writes a PEFT LoRA adapter you can load next to the Qwen3.5 base model. The paper's settings are the defaults.

Set up the three Python environments once ([Setup](#setup)); then:

```bash
source scripts/env.sh
# The paper's recipe for one of its T2I models (sd35-medium, sd35-large, flux2-klein-9b, qwen-image-2512).
bash scripts/train_policy.sh --t2i-model sd35-medium --out $T_DIG_RUNS/my-policy            # prints the plan
bash scripts/train_policy.sh --t2i-model sd35-medium --out $T_DIG_RUNS/my-policy --execute  # trains
# Your own T2I model: any Diffusers text-to-image model directory, with its pipeline class and sampling settings.
bash scripts/train_policy.sh --t2i-model /path/to/my-sdxl --pipeline StableDiffusionXLPipeline \
  --steps 30 --guidance 5 --out $T_DIG_RUNS/my-sdxl-policy --execute
```

The adapter is written to `<out>/adapter/lora_adapter`; checkpoints are under `<out>/policy/checkpoints/`. A run that stops (time limit, crash) continues from its last checkpoint with the same command plus `--resume`. `--help` lists every option: policy size (`--policy 2B|4B|9B|35B-A3B`, default 4B), reward weights, batch and group size, step count (`--max-steps`, default 2000; 0 trains one epoch) and memory knobs.

**GPUs.** The launcher uses the GPUs in `CUDA_VISIBLE_DEVICES`, else every GPU of the machine, and places the three processes by count:

| GPUs | Policy (vLLM rollout + LoRA training) | VQAScore | T2I model + TPIPS reward services |
| --- | --- | --- | --- |
| 1 | GPU 0 | GPU 0 | GPU 0 |
| 2 | GPU 0 | GPU 1 | GPU 1 |
| 3 to 7 | GPU 0 | GPU 1 | one per remaining GPU |
| 8 or more | GPUs 0 and 1 | GPU 2 | one per remaining GPU |

Each reward service renders and scores the images of the completions sent to it, so a step's time is set by the images per step (`batch size × group size × 4`, 256 by default) divided over the reward GPUs. Peak GPU memory of each process, measured on an H200 with the defaults:

| Process | Peak memory |
| --- | --- |
| Policy, Qwen3.5-4B with LoRA (FSDP actor) | 21 GB allocated, 27 GB reserved, plus vLLM's budget: half of the memory free when training starts (`--rollout-memory-fraction` sets it; the 4B weights take 8 GB of it) |
| Reward service, SD3.5 Medium + TPIPS (8B) | 34 GB at 512 px; 37 GB rendering 1024 px validation images |
| Reward service, a custom SD2.1-size model (sd-turbo) + TPIPS | 20 GB |
| VQAScore, CLIP-FlanT5-XL | 8 to 11 GB |

Processes that share a GPU add up. VQAScore is CLIP-FlanT5-XXL (about 22 GB) on a dedicated GPU of 40 GB or more, else the XL model (`--vqa-model`). The quality term of the reward (`--quality-weight 0.05`, the paper's value) adds the 17 GB HPSv3 scorer to every reward GPU and is off by default; `--token-batch` lowers the policy's activation memory. Throughput on an H200: one reward service renders 2.3 SD3.5 Medium images per second at 512 px (25 per second for sd-turbo), and sampling a step's completions from the 4B policy takes about 20 s, so the default step takes about two minutes per SD3.5 Medium reward GPU. The two-step runs that produced these numbers used one GPU (SD3.5 Medium) and two GPUs (sd-turbo), and are the launcher's tested configurations; the paper's 9B policies were trained on 2 × 8 H200s.

**Data.** A paper T2I model trains on its rows of [neulab/t-dig-train](https://huggingface.co/datasets/neulab/t-dig-train): 76k (prompt, axis) rows with per-row fidelity thresholds measured on that model's own images. A custom model trains on the same rows without those thresholds, so `--tau` (default 0.9, about the median of the measured thresholds) is the fidelity threshold of every row. To measure thresholds for your model instead, see `tdig/render_iid.sh` in [tdig/README.md](tdig/README.md). `--dataset` points at any directory in that format.

**Using the policy.** The adapter loads with vLLM or PEFT on the base model it was trained from; `tdig.instruction` writes the message the policy expects:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tdig.instruction import instruction
from axis_grpo.training.protocol import parse_prompts

if __name__ == "__main__":   # vLLM starts its engine in a subprocess that re-imports this file
    llm = LLM("<Qwen3.5-4B snapshot>", enable_lora=True, max_lora_rank=32, max_model_len=2048)
    message = [{"role": "user", "content": instruction("a cozy reading nook with an armchair", "interior design style", 4, "sd35-medium", word_cap=32)}]
    text = llm.get_tokenizer().apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = llm.generate([text], SamplingParams(max_tokens=400, temperature=1.0), lora_request=LoRARequest("tdig", 1, "<out>/adapter/lora_adapter"))
    print(parse_prompts(out[0].outputs[0].text, 4))   # four prompts, one per image
```

`axis_framework/` serves such an adapter behind the interactive tool ([step (e)](#e-run-the-interactive-framework-locally)).

## Repository layout

| Path | Contents |
| --- | --- |
| `hpdv2/` | Prompt curation from HPDv2: LLM filtering passes, deduplication, clustering. |
| `iid/` | Verified i.i.d. image sets from a T2I model server. |
| `elicitation/` | Axis elicitation: an LLM looks at four i.i.d. images of a prompt and names axes to vary. |
| `axis_grpo/` | Training-data export (`axis_grpo/`) and GRPO training: trainer, reward and VQAScore services, multi-node fleet (`axis_grpo/training/`). |
| `tdig/` | Policy datasets and instruction, fidelity and quality thresholds, checkpoint validation, checkpoint selection, paper metrics, test-set inputs. |
| `baselines/` | i.i.d. sampling, CADS, Scaling Group Inference and PAG on the four T2I models; COCO-250 FID evaluation. |
| `benchmarks/` | Benchmark prompt bundles (COCO 2017, held-out axes) and their source locks. |
| `axis_framework/` | The interactive web tool (orchestrator, policy, T2I model and suggest servers, web page). |
| `configs/` | Asset locks of the T2I and auxiliary models (`configs/t2i_models/`). |
| `scripts/` | Shared launcher environment (`env.sh`), the one-machine policy trainer (`train_policy.sh`), training-matrix and baseline-batch launchers, downloads. |
| `requirements/` | Pinned packages of the three Python environments. |
| `tests/` | CPU unit tests. |

Each package has its own README.

## Setup

The stages need incompatible torch and transformers versions, so the code runs in three Python 3.12 environments, each with its own requirements file:

| Environment | Variable | Requirements | Used for |
| --- | --- | --- | --- |
| training | `PY_TRAIN` | `requirements/train.txt` (verl at commit 8a69493, torch 2.11, vLLM 0.24, transformers 5.14) | GRPO training, vLLM policy sampling in validation, the framework's policy server |
| reward | `PY_REWARD` | `requirements/reward.txt` (torch 2.7.1, transformers 5.3, diffusers 0.37) | T2I models, TPIPS and image scorers, baselines, paper metrics, the framework's T2I model and suggest servers |
| VQAScore | `PY_VQA` | `requirements/vqa.txt` (torch 2.5.1, transformers 4.49, t2v_metrics 3.0) | VQAScore (CLIP-FlanT5-XXL) |

Each file lists the complete environment the paper's runs used. Some of its packages declare dependencies that conflict with each other, so install without dependency resolution (`--no-deps`), with torch (and, for training, vLLM) from the matching CUDA index:

```bash
T=https://download.pytorch.org/whl
python -m venv envs/train  && envs/train/bin/pip install --no-deps --extra-index-url $T/cu129 \
  --extra-index-url https://wheels.vllm.ai/0.24.0/cu129 -r requirements/train.txt
python -m venv envs/reward && envs/reward/bin/pip install --no-deps --extra-index-url $T/cu126 -r requirements/reward.txt
python -m venv envs/vqa    && envs/vqa/bin/pip install --no-deps --extra-index-url $T/cu121 -r requirements/vqa.txt
export PY_TRAIN=$PWD/envs/train/bin/python PY_REWARD=$PWD/envs/reward/bin/python PY_VQA=$PWD/envs/vqa/bin/python
$PY_TRAIN scripts/patch_verl.py   # one-line fix to verl's V1 trainer that LoRA training needs
```

PAG and Clean-FID need small import overlays; see [baselines/SETUP.md](baselines/SETUP.md). The CPU tests need `pytest`, which the requirements do not include; install it in the reward and training environments. They run in the reward environment, except `tests/test_training_reward.py`, which needs the training environment: `$PY_REWARD -m pytest tests --ignore tests/test_training_reward.py` and `$PY_TRAIN -m pytest tests/test_training_reward.py`.

### Environment variables

Every launcher sources [scripts/env.sh](scripts/env.sh), which sets these defaults. The Python modules read the same variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `T_DIG_ROOT` | the repository root (from the file location) | Set it when the scheduler runs a copy of the script from elsewhere (`sbatch` does). |
| `T_DIG_DATA` | `$T_DIG_ROOT/data` | Datasets: benchmark bundles (`benchmarks/`), downloads (`downloads/`), policy datasets (`tdig/`). |
| `T_DIG_RUNS` | `$T_DIG_ROOT/runs` | Outputs: training runs, validation, scores (`tdig/`). |
| `T_DIG_MODELS` | `$T_DIG_ROOT/models` | Local weights: T2I models (`sd35-large`, `sd35-medium`, `flux2-klein-9b`, `qwen-image-2512`), `clip`, `dino`, `gpt2`, `pag-10000`, `dreamsim_cache`, `aesthetic/`, `all-MiniLM-L6-v2`, released adapters (`t-dig-checkpoints`). |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache; snapshots are read from `HF_HUB_CACHE` (default `$HF_HOME/hub`). Policies are `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-35B-A3B`; TPIPS is `sywang/TPIPS-Embed-Qwen3VL-8B`; VQAScore is `zhiqiulin/clip-flant5-xxl`. |
| `PY_TRAIN`, `PY_REWARD`, `PY_VQA` | `python` | Interpreters of the three environments. |
| `T_DIG_SETUP` | unset | Optional file sourced first by every launcher, for site setup (module loads, `LD_LIBRARY_PATH`, CUDA). |
| `T_DIG_SUBMIT` | `bash` | Submit command stored by `scripts/prepare_training_matrix.py`, e.g. `sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --time=72:00:00`; `bash` runs the job in the foreground until training finishes. |
| `T_DIG_NODEFILE`, `T_DIG_REMOTE_LAUNCH`, `T_DIG_JOB_ID` | under Slurm: the job's node list, `srun --nodes=1 --ntasks=1 -w {host} --`, `$SLURM_JOB_ID` | Hosts of a multi-node training job, the command that starts a process on another node, and the run's job id ([axis_grpo/training/README.md](axis_grpo/training/README.md)). |

### Weights

Jobs read every weight offline (`HF_HUB_OFFLINE=1`), so download them once on a machine with internet access. `hf` is the Hugging Face CLI installed in each environment (`envs/reward/bin/hf`). Stable Diffusion 3.5 and FLUX.2 [klein] are gated: accept their licenses on Hugging Face and run `hf auth login` first.

T2I models, at the revisions the asset locks in `configs/t2i_models/` were made from (every file is hashed against the lock before a model loads):

```bash
hf download stabilityai/stable-diffusion-3.5-large --revision ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f --local-dir $T_DIG_MODELS/sd35-large
hf download stabilityai/stable-diffusion-3.5-medium --revision b940f670f0eda2d07fbb75229e779da1ad11eb80 --local-dir $T_DIG_MODELS/sd35-medium
hf download black-forest-labs/FLUX.2-klein-9B --revision 92196c8e11f7b6cf2b7493e037d8c5345c559216 --local-dir $T_DIG_MODELS/flux2-klein-9b
hf download Qwen/Qwen-Image-2512 --revision 25468b98e3276ca6700de15c6628e51b7de54a26 --local-dir $T_DIG_MODELS/qwen-image-2512
```

Other local weights in `$T_DIG_MODELS` (`clip`, `dino`, `gpt2` and `pag-10000` for the baselines are in [baselines/SETUP.md](baselines/SETUP.md)):

```bash
# Prompt clustering (step a1); the sweep requires the provenance file.
hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir $T_DIG_MODELS/all-MiniLM-L6-v2
echo '{"repository": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"}' > $T_DIG_MODELS/all-MiniLM-L6-v2/tdig-provenance.json
# LAION-Aesthetics V2 head (tracked score and paper metrics).
mkdir -p $T_DIG_MODELS/aesthetic && curl -L -o $T_DIG_MODELS/aesthetic/sac_logos_ava1_l14_linearMSE.pth \
  'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth'
# DreamSim (editing drift and paper metrics); downloads its weights into the cache on first call.
$PY_REWARD -c "from dreamsim import dreamsim; dreamsim(pretrained=True, device='cpu', cache_dir='$T_DIG_MODELS/dreamsim_cache')"
# Released adapters (steps c and e).
hf download neulab/t-dig --local-dir $T_DIG_MODELS/t-dig-checkpoints
```

Hugging Face cache (`HF_HUB_CACHE`): the policy bases, the reward and metric models, the framework's suggestion VLM and the Qwen-Image editor. `scripts/prepare_training_matrix.py` uses the policy snapshot that `refs/main` names, so download the policies without `--revision`; the paper's runs used `Qwen/Qwen3.5-4B@851bf6e`, `Qwen/Qwen3.5-9B@c202236` and `Qwen/Qwen3.5-35B-A3B@59d61f3`.

```bash
for r in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-35B-A3B; do hf download $r; done
hf download sywang/TPIPS-Embed-Qwen3VL-8B --revision 86a1493c60121e656a1c075bc7aed88f25fcd4d4
hf download Qwen/Qwen-Image-Edit-2509 --revision d3968ef930e841f4c73640fb8afa3b306a78167e
for r in zhiqiulin/clip-flant5-xxl google/flan-t5-xxl openai/clip-vit-large-patch14-336 \
  MizzenAI/HPSv3 Qwen/Qwen2-VL-7B-Instruct yuvalkirstain/PickScore_v1 laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  xswu/HPSv2 timm/vit_large_patch14_clip_224.openai openai/clip-vit-base-patch32 bert-base-uncased \
  Qwen/Qwen3-VL-8B-Instruct; do hf download $r; done
# ImageReward reads its checkpoint from ~/.cache/ImageReward.
hf download THUDM/ImageReward ImageReward.pt med_config.json --local-dir ~/.cache/ImageReward
```

To use other copies of the T2I models, write your own lock with `python -m baselines lock-assets` ([baselines/SETUP.md](baselines/SETUP.md)).

### Running on a cluster

The launchers (`*.sh`) carry no scheduler directives; each states its resources in a comment. Run one with `bash` on a GPU machine, or submit it with Slurm and pass the variables through the job environment, for example `sbatch --export=ALL,VAL_OUT=... tdig/validate_checkpoints.sh`. A launcher uses the GPUs in `CUDA_VISIBLE_DEVICES` if set, else every GPU of the machine. The paper's experiments use nodes with eight H200 GPUs; nothing in the code needs them, and [Train your own policy](#train-your-own-policy) runs on one machine with any GPUs. Multi-node training also runs without a scheduler, over ssh; see [axis_grpo/training/README.md](axis_grpo/training/README.md).

## Steps

Run from the repository root after `source scripts/env.sh`.

### (a) Build the training data

1. Curate base prompts from HPDv2 ([hpdv2/README.md](hpdv2/README.md)): `python -m hpdv2 prepare`, the two `filter` passes (paid LLM API calls), deduplication and the clustering sweep (`scripts/run_cluster_sweep.py`). The paper kept one prompt per cluster of the 0.35 cut, drawn from the more central half of its members, 23,020 prompts; that pool is `prompts/curated-pool.json` in [neulab/t-dig-train](https://huggingface.co/datasets/neulab/t-dig-train). To reproduce the paper, skip this step and pass that file to `iid.curated prepare` below.
2. Generate four i.i.d. images per prompt with each T2I model ([iid/README.md](iid/README.md), one 8-GPU node per T2I model) and elicit axes from them ([elicitation/README.md](elicitation/README.md), paid vision-LLM API calls):
   ```bash
   $PY_REWARD -m iid.curated prepare --input <t-dig-train>/prompts/curated-pool.json --out $T_DIG_RUNS/curated-iid --allow-partial-input --execute
   CURATED_RUN=$T_DIG_RUNS/curated-iid T2I_MODEL=flux2-klein-9b bash iid/render_curated.sh   # and the other three
   python -m elicitation prepare --render-root $T_DIG_RUNS/curated-iid --out $T_DIG_RUNS/elicitation   # then python -m elicitation run
   ```
3. Export the axis rows and split them into prompt-disjoint train and held-out sets ([axis_grpo/README.md](axis_grpo/README.md)):
   ```bash
   $PY_REWARD -m axis_grpo.prepare plan  --assignment <assignment.json> --out $T_DIG_DATA/generation-policy-plan --seed 42 --k 4 --execute
   $PY_REWARD -m axis_grpo.prepare build --plan $T_DIG_DATA/generation-policy-plan --elicitation-run <run> \
     --out $T_DIG_DATA/generation-policy-verbatim --reward-text request --execute
   $PY_REWARD -m axis_grpo.training.split --source $T_DIG_DATA/generation-policy-verbatim \
     --out $T_DIG_DATA/generation-policy-val200-v1 --execute
   ```
4. Write the policy datasets with the T-DIG instruction and attach per-row fidelity (VQAScore) and quality (HPSv3) thresholds measured on the i.i.d. images ([tdig/README.md](tdig/README.md), "Dataset pipeline"). The image-input variant needs the split's image twin first: each row plus 512 px copies of the four i.i.d. images the elicitor saw, read from the export's `source-records.jsonl`.
   ```bash
   $PY_REWARD -m axis_grpo.training.image_data --text-dataset $T_DIG_DATA/generation-policy-val200-v1 \
     --records $T_DIG_DATA/generation-policy-verbatim/source-records.jsonl \
     --image-cache $T_DIG_DATA/policy-image-inputs512-v1 \
     --out $T_DIG_DATA/generation-policy-val200-images512-v1 --workers 32 --execute
   $PY_REWARD -m tdig.build_dataset --text-source $T_DIG_DATA/generation-policy-val200-v1 \
     --image-source $T_DIG_DATA/generation-policy-val200-images512-v1 \
     --out $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated --order rotated --execute
   $PY_REWARD -m tdig.iid_vqa index --dataset $T_DIG_DATA/generation-policy-val200-v1 \
     --render-root $T_DIG_RUNS/curated-iid --out $T_DIG_RUNS/tdig/iid-vqa-v1
   VQA_OUT=$T_DIG_RUNS/tdig/iid-vqa-v1 bash tdig/iid_vqa.sh            # 1 node, 8 GPUs
   $PY_REWARD -m tdig.iid_vqa attach --thresholds $T_DIG_RUNS/tdig/iid-vqa-v1/thresholds.jsonl \
     --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated \
     --out $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau
   ```
   Then run `tdig/iid_quality.sh` and `tdig.iid_quality attach` for the quality threshold, `tdig.recap_dataset --relative-budget 20` for the relative word cap, and `tdig.build_edit_dataset` for the editing policy's dataset.
5. COCO-250 rows for validation. These need the COCO caption pool ([baselines/SETUP.md](baselines/SETUP.md), "COCO data") and the 250-caption subset (`python -m tdig.coco_fid select`, [baselines/README.md](baselines/README.md#coco-250-fid)): `python -m tdig.eval_sets --benchmark coco250 --out $T_DIG_DATA/tdig/eval-sets`.

### (b) Train a policy with GRPO

The paper's matrix: one job per (T2I model, policy size), each on two 8-GPU nodes ([axis_grpo/training/README.md](axis_grpo/training/README.md)). On one machine, use [Train your own policy](#train-your-own-policy) instead.

```bash
T_DIG_SUBMIT="sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --time=72:00:00" \
$PY_TRAIN scripts/prepare_training_matrix.py --dataset <policy dataset>/<T2I model variant> \
  --out $T_DIG_RUNS/<matrix> --t2i-models flux2-klein-9b --sizes 9B \
  --tau 0.9 --lam 2.0 --learning-rate 2e-5 --metric vendi --length-weight 0.1 --quality-weight 0.05 --quality-margin 0 \
  --save-freq 50 --save-freq-early 25 --max-steps 2000 --validation offline \
  --track-metrics imagereward hpsv3 clip_score aesthetic --execute
$PY_TRAIN scripts/preflight_training_matrix.py --matrix $T_DIG_RUNS/<matrix>
$PY_TRAIN scripts/submit_training_matrix.py --matrix $T_DIG_RUNS/<matrix> --expected-jobs 1 --execute
```

Without `--execute`, each step prints what it would do. The editing policy uses `--task edit --policy-observation elicitation_images` and an editing dataset; see [tdig/README.md](tdig/README.md).

### (c) Validate checkpoints and compute the paper metrics

On held-out pairs (and COCO-250 rows), one 8-GPU node per run:

```bash
TRAIN_RUN=$T_DIG_RUNS/<matrix>/<job> VAL_OUT=$T_DIG_RUNS/tdig/<validation> STEPS=0+300+600+900 \
SETS='heldout=<dataset>/<variant>/val.parquet;coco250=<eval sets>/<variant>/coco250.parquet' \
  bash tdig/validate_checkpoints.sh
$PY_VQA tdig/select_frontier.py $T_DIG_RUNS/tdig/<validation>/ --out $T_DIG_RUNS/tdig/frontier
```

`validate_checkpoints.sh` merges each checkpoint into a LoRA adapter, samples one completion per row with vLLM, generates its images with the T2I model and scores them with the training run's reward configuration, and writes every paper metric (`tdig/paper_metrics.py`) to `<VAL_OUT>/paper-metrics/summary.json`. `select_frontier.py` orders the checkpoints by the alignment/diversity frontier. The README of the checkpoints repository on Hugging Face gives the rule that selects the released checkpoints.

On the interactive test set:

```bash
hf download neulab/t-dig-test --repo-type dataset --local-dir $T_DIG_DATA/t-dig-test
$PY_REWARD -m tdig.test_set --test $T_DIG_DATA/t-dig-test/data/flux2-klein-9b/test.jsonl \
  --val <dataset>-customtau/flux2-klein-9b/val.parquet --t2i-model flux2-klein-9b --out $T_DIG_DATA/tdig/testset-flux2-klein-9b
TRAIN_RUN=... VAL_OUT=... STEPS=... SETS="testset=$T_DIG_DATA/tdig/testset-flux2-klein-9b/testset.parquet" \
  bash tdig/validate_checkpoints.sh
```

The `bundle/` next to `testset.parquet` holds the same prompts and axes in the baselines' bundle format, for step (d).

### (d) Run the baselines

The held-out axis bundle holds the held-out base prompts of a policy dataset with their axes: `python -m tdig.heldout_bundle --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau --out $T_DIG_DATA/benchmarks/heldout-axes` ([tdig/README.md](tdig/README.md)). Weights, overlays and COCO data are in [baselines/SETUP.md](baselines/SETUP.md).

```bash
$PY_REWARD -m baselines config --method cads --t2i-model sd35-large > cads.json     # or iid, pag, group
$PY_REWARD -m baselines generate --bundle $T_DIG_DATA/benchmarks/heldout-axes --run $T_DIG_RUNS/cads-heldout \
  --config cads.json --assets configs/t2i_models/sd35-large.assets.lock.json --execute --max-new-images 800
```

Score them with the policy's metrics (`tdig.paper_metrics index-baselines`, then `tdig/paper_metrics.sh`), or with VQAScore and TPIPS only (`tdig/score_baselines.sh`). For COCO-250, generate images for the 250 captions with `tdig/render_coco.sh` and score FID and KID with `tdig/coco_fid.sh`. See [baselines/README.md](baselines/README.md) and [baselines/SETUP.md](baselines/SETUP.md).

### (e) Run the interactive framework locally

```bash
# 1. Fake mode: no GPU, no weights; templated prompts and placeholder images drive the whole loop.
FAKE=1 bash axis_framework/scripts/launch_local.sh       # UI on http://127.0.0.1:19100
RUN=local bash axis_framework/scripts/launch_local.sh stop

# 2. Real models: download the released adapters (Setup, "Weights"), build the alpha ladders, serve a T2I model.
CHECKPOINTS=$T_DIG_MODELS/t-dig-checkpoints bash axis_framework/scripts/build_alpha_ladders.sh
SERVE=tdig_flux9b_alpha_gen:tdig_flux9b_edit_alpha bash axis_framework/scripts/launch_local.sh
```

The policy server runs in the training environment; the other processes run in the reward environment. `axis_framework/scripts/launch_pool.sh` spreads the services over a whole node. See [axis_framework/README.md](axis_framework/README.md).

## Development

This code was written with the help of Claude, Anthropic's AI assistant.

## License

Apache License 2.0 ([LICENSE](LICENSE)). Third-party code: `axis_grpo/training/trainer.py` adapts parts of verl (Apache-2.0, [THIRD_PARTY.md](axis_grpo/training/THIRD_PARTY.md)); `axis_framework/vendor/qwen_vl_utils` is qwen-vl-utils 0.0.14 (Apache-2.0). Models, datasets and evaluation code used by this repository are subject to their own licenses.

## Citation

```bibtex
@misc{khanuja2026tdig,
  title={T-DIG: Text-Conditioned Diversity for Image Generation},
  author={Simran Khanuja and Rohit Gandikota and Gaurav Parmar and Graham Neubig},
  year={2026},
  eprint={2610.29003},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.29003}
}
```
