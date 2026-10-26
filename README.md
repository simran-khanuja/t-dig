# T-DIG: Text-Conditioned Diversity for Image Generation
### [Project Website](https://image-study.org/t-dig/) | [Arxiv Preprint](https://arxiv.org/abs/2610.29003) | [Trained Policies](https://huggingface.co/neulab/t-dig) | [Training Data](https://huggingface.co/datasets/neulab/t-dig-train) | [Test Set](https://huggingface.co/datasets/neulab/t-dig-test) <br>
Official code for "T-DIG: Text-Conditioned Diversity for Image Generation".

T-DIG makes a text-to-image (T2I) model produce a set of images that vary along an axis you name ("dog breed", "time of day") and otherwise stay faithful to the prompt. A small LLM *policy* (Qwen3.5 with a LoRA adapter) reads the prompt and the axis and writes one prompt per image; the T2I model is frozen. The policy is trained with GRPO, rewarded for the diversity of the image set (TPIPS conditioned on the axis text) minus a VQAScore fidelity hinge against the prompt. We release policies for Qwen-Image-2512, Stable Diffusion 3.5 Large, Stable Diffusion 3.5 Medium and FLUX.2 [klein] 9B.

## Setup
The stages need incompatible torch versions, so the code runs in three Python 3.12 environments. Install each from its pinned requirements file (without dependency resolution; the pins conflict on paper):
```bash
git clone https://github.com/simran-khanuja/t-dig.git && cd t-dig
T=https://download.pytorch.org/whl
python -m venv envs/train  && envs/train/bin/pip install --no-deps --extra-index-url $T/cu129 --extra-index-url https://wheels.vllm.ai/0.24.0/cu129 -r requirements/train.txt
python -m venv envs/reward && envs/reward/bin/pip install --no-deps --extra-index-url $T/cu126 -r requirements/reward.txt
python -m venv envs/vqa    && envs/vqa/bin/pip install --no-deps --extra-index-url $T/cu121 -r requirements/vqa.txt
export PY_TRAIN=$PWD/envs/train/bin/python PY_REWARD=$PWD/envs/reward/bin/python PY_VQA=$PWD/envs/vqa/bin/python
$PY_TRAIN scripts/patch_verl.py   # one-line fix to verl's V1 trainer that LoRA training needs
source scripts/env.sh             # before every command below; see its header for the data, runs and models directories
```
Stable Diffusion 3.5 and FLUX.2 [klein] are gated on Hugging Face: accept their licenses and run `hf auth login` first.

## Train your own policy
One command trains a policy on one machine with whatever GPUs it has, against one of the paper's T2I models or any Diffusers text-to-image model. It downloads the weights and training data it is missing, starts the reward services, trains with GRPO and writes a PEFT LoRA adapter to `<out>/adapter/lora_adapter`. The paper's settings are the defaults.
```bash
# A paper T2I model: sd35-medium, sd35-large, flux2-klein-9b or qwen-image-2512.
bash scripts/train_policy.sh --t2i-model sd35-medium --out $T_DIG_RUNS/my-policy --execute
# Your own Diffusers model: its directory, pipeline class and sampling settings.
bash scripts/train_policy.sh --t2i-model /path/to/my-sdxl --pipeline StableDiffusionXLPipeline --steps 30 --guidance 5 --out $T_DIG_RUNS/my-sdxl-policy --execute
```
Without `--execute` the command prints its plan. `--resume` continues a stopped run from its last checkpoint. `--help` lists the rest: `--policy 2B|4B|9B|35B-A3B` (default 4B), reward weights, `--batch-size`, `--group-size`, `--max-steps` (default 2000) and memory knobs.

The launcher uses the GPUs in `CUDA_VISIBLE_DEVICES`, else all of them, and places the three processes by count:

| GPUs | Policy (vLLM rollout + LoRA training) | VQAScore | T2I model + TPIPS reward services |
| --- | --- | --- | --- |
| 1 | GPU 0 | GPU 0 | GPU 0 |
| 2 | GPU 0 | GPU 1 | GPU 1 |
| 3 to 7 | GPU 0 | GPU 1 | one per remaining GPU |
| 8 or more | GPUs 0 and 1 | GPU 2 | one per remaining GPU |

Peak memory with the defaults, measured on an H200 (processes sharing a GPU add up):

| Process | Peak memory |
| --- | --- |
| Policy, Qwen3.5-4B + LoRA | 27 GB, plus vLLM's budget: half of the memory free when training starts (`--rollout-memory-fraction`) |
| Reward service, SD3.5 Medium + TPIPS | 34 GB (37 GB at 1024 px validation) |
| Reward service, an SD2.1-size model + TPIPS | 20 GB |
| VQAScore, CLIP-FlanT5-XL | 8 to 11 GB (XXL, used on a dedicated GPU of 40 GB or more: 22 GB) |

A step renders `batch size × group size × 4` images (256 by default) across the reward GPUs, about two minutes per SD3.5 Medium reward GPU on an H200. A paper T2I model trains on its rows of [neulab/t-dig-train](https://huggingface.co/datasets/neulab/t-dig-train) with per-row fidelity thresholds measured on that model; a custom model uses the same rows with `--tau` (default 0.9) as the threshold. Pass `--dataset` to train on your own rows in that format.

## Using a policy
Trained and [released](https://huggingface.co/neulab/t-dig) adapters load with vLLM or PEFT on their Qwen3.5 base; `tdig.instruction` writes the message the policy expects:
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

## Interactive tool
The web tool from the paper's user study: type a prompt and an axis, get a set of images, edit it. `FAKE=1` needs no GPU or weights; with real models it serves released adapters next to a T2I model ([axis_framework/README.md](axis_framework/README.md)).
```bash
FAKE=1 bash axis_framework/scripts/launch_local.sh       # UI on http://127.0.0.1:19100
hf download neulab/t-dig --local-dir $T_DIG_MODELS/t-dig-checkpoints
CHECKPOINTS=$T_DIG_MODELS/t-dig-checkpoints bash axis_framework/scripts/build_alpha_ladders.sh
SERVE=tdig_flux9b_alpha_gen:tdig_flux9b_edit_alpha bash axis_framework/scripts/launch_local.sh
```

## Reproducing the paper
Each launcher (`*.sh`) states its resources in a comment and uses the GPUs in `CUDA_VISIBLE_DEVICES`, else all of them; run it with `bash` or submit it with your scheduler. The paper used nodes of eight H200s; nothing in the code needs them. Jobs read weights offline: download the T2I models at the locked revisions and the other weights first ([Weights](#weights)).

**Training data** ([hpdv2](hpdv2/README.md), [iid](iid/README.md), [elicitation](elicitation/README.md), [axis_grpo](axis_grpo/README.md), [tdig](tdig/README.md)). The curated prompt pool and the finished rows are in [neulab/t-dig-train](https://huggingface.co/datasets/neulab/t-dig-train); to rebuild them, render four i.i.d. images per prompt with each T2I model, elicit axes, export and split the rows, then attach the fidelity and quality thresholds:
```bash
$PY_REWARD -m iid.curated prepare --input <t-dig-train>/prompts/curated-pool.json --out $T_DIG_RUNS/curated-iid --allow-partial-input --execute
CURATED_RUN=$T_DIG_RUNS/curated-iid T2I_MODEL=flux2-klein-9b bash iid/render_curated.sh   # and the other three
python -m elicitation prepare --render-root $T_DIG_RUNS/curated-iid --out $T_DIG_RUNS/elicitation && python -m elicitation run
$PY_REWARD -m axis_grpo.prepare plan  --assignment <assignment.json> --out $T_DIG_DATA/generation-policy-plan --seed 42 --k 4 --execute
$PY_REWARD -m axis_grpo.prepare build --plan $T_DIG_DATA/generation-policy-plan --elicitation-run <run> --out $T_DIG_DATA/generation-policy-verbatim --reward-text request --execute
$PY_REWARD -m axis_grpo.training.split --source $T_DIG_DATA/generation-policy-verbatim --out $T_DIG_DATA/generation-policy-val200-v1 --execute
$PY_REWARD -m axis_grpo.training.image_data --text-dataset $T_DIG_DATA/generation-policy-val200-v1 --records $T_DIG_DATA/generation-policy-verbatim/source-records.jsonl --image-cache $T_DIG_DATA/policy-image-inputs512-v1 --out $T_DIG_DATA/generation-policy-val200-images512-v1 --workers 32 --execute
$PY_REWARD -m tdig.build_dataset --text-source $T_DIG_DATA/generation-policy-val200-v1 --image-source $T_DIG_DATA/generation-policy-val200-images512-v1 --out $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated --order rotated --execute
$PY_REWARD -m tdig.iid_vqa index --dataset $T_DIG_DATA/generation-policy-val200-v1 --render-root $T_DIG_RUNS/curated-iid --out $T_DIG_RUNS/tdig/iid-vqa-v1
VQA_OUT=$T_DIG_RUNS/tdig/iid-vqa-v1 bash tdig/iid_vqa.sh
$PY_REWARD -m tdig.iid_vqa attach --thresholds $T_DIG_RUNS/tdig/iid-vqa-v1/thresholds.jsonl --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated --out $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau
```
Then `tdig/iid_quality.sh` and `tdig.iid_quality attach` for the quality threshold, `tdig.recap_dataset --relative-budget 20` for the word cap, `tdig.build_edit_dataset` for the editing policy and `tdig.eval_sets --benchmark coco250` for the COCO-250 validation rows.

**Training matrix** ([axis_grpo/training/README.md](axis_grpo/training/README.md)): one job per (T2I model, policy size) on two 8-GPU nodes. `T_DIG_SUBMIT` is the submit command, `bash` runs the job in the foreground.
```bash
T_DIG_SUBMIT="sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --time=72:00:00" \
$PY_TRAIN scripts/prepare_training_matrix.py --dataset <policy dataset>/<T2I model variant> --out $T_DIG_RUNS/<matrix> --t2i-models flux2-klein-9b --sizes 9B \
  --tau 0.9 --lam 2.0 --learning-rate 2e-5 --metric vendi --length-weight 0.1 --quality-weight 0.05 --quality-margin 0 \
  --save-freq 50 --save-freq-early 25 --max-steps 2000 --validation offline --track-metrics imagereward hpsv3 clip_score aesthetic --execute
$PY_TRAIN scripts/preflight_training_matrix.py --matrix $T_DIG_RUNS/<matrix>
$PY_TRAIN scripts/submit_training_matrix.py --matrix $T_DIG_RUNS/<matrix> --expected-jobs 1 --execute
```

**Validation and paper metrics** ([tdig/README.md](tdig/README.md)): merge checkpoints, sample with vLLM, render, score every paper metric and pick checkpoints on the alignment/diversity frontier. The interactive test set is scored the same way after `tdig.test_set` converts it.
```bash
TRAIN_RUN=$T_DIG_RUNS/<matrix>/<job> VAL_OUT=$T_DIG_RUNS/tdig/<validation> STEPS=0+300+600+900 \
SETS='heldout=<dataset>/<variant>/val.parquet;coco250=<eval sets>/<variant>/coco250.parquet' bash tdig/validate_checkpoints.sh
$PY_VQA tdig/select_frontier.py $T_DIG_RUNS/tdig/<validation>/ --out $T_DIG_RUNS/tdig/frontier
hf download neulab/t-dig-test --repo-type dataset --local-dir $T_DIG_DATA/t-dig-test
$PY_REWARD -m tdig.test_set --test $T_DIG_DATA/t-dig-test/data/flux2-klein-9b/test.jsonl --val <dataset>-customtau/flux2-klein-9b/val.parquet --t2i-model flux2-klein-9b --out $T_DIG_DATA/tdig/testset-flux2-klein-9b
```

**Baselines** ([baselines/README.md](baselines/README.md), [baselines/SETUP.md](baselines/SETUP.md)): i.i.d. sampling, CADS, Scaling Group Inference and PAG on the held-out axis bundle, scored with the policy's metrics; COCO-250 FID with `tdig/render_coco.sh` and `tdig/coco_fid.sh`.
```bash
python -m tdig.heldout_bundle --dataset $T_DIG_DATA/tdig/generation-policy-tdig-v2-rotated-customtau --out $T_DIG_DATA/benchmarks/heldout-axes
$PY_REWARD -m baselines config --method cads --t2i-model sd35-large > cads.json     # or iid, pag, group
$PY_REWARD -m baselines generate --bundle $T_DIG_DATA/benchmarks/heldout-axes --run $T_DIG_RUNS/cads-heldout --config cads.json --assets configs/t2i_models/sd35-large.assets.lock.json --execute --max-new-images 800
```

### Weights
T2I models at the revisions of the asset locks in `configs/t2i_models/` (every file is hashed against the lock before a model loads; `python -m baselines lock-assets` writes a lock for another copy):
```bash
hf download stabilityai/stable-diffusion-3.5-large --revision ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f --local-dir $T_DIG_MODELS/sd35-large
hf download stabilityai/stable-diffusion-3.5-medium --revision b940f670f0eda2d07fbb75229e779da1ad11eb80 --local-dir $T_DIG_MODELS/sd35-medium
hf download black-forest-labs/FLUX.2-klein-9B --revision 92196c8e11f7b6cf2b7493e037d8c5345c559216 --local-dir $T_DIG_MODELS/flux2-klein-9b
hf download Qwen/Qwen-Image-2512 --revision 25468b98e3276ca6700de15c6628e51b7de54a26 --local-dir $T_DIG_MODELS/qwen-image-2512
```
Policies, reward and metric models, and the framework's VLM and editor go in the Hugging Face cache (`HF_HUB_CACHE`); the paper used `Qwen/Qwen3.5-4B@851bf6e`, `Qwen3.5-9B@c202236` and `Qwen3.5-35B-A3B@59d61f3`:
```bash
for r in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-35B-A3B; do hf download $r; done
hf download sywang/TPIPS-Embed-Qwen3VL-8B --revision 86a1493c60121e656a1c075bc7aed88f25fcd4d4
hf download Qwen/Qwen-Image-Edit-2509 --revision d3968ef930e841f4c73640fb8afa3b306a78167e
for r in zhiqiulin/clip-flant5-xxl google/flan-t5-xxl openai/clip-vit-large-patch14-336 MizzenAI/HPSv3 Qwen/Qwen2-VL-7B-Instruct yuvalkirstain/PickScore_v1 \
  laion/CLIP-ViT-H-14-laion2B-s32B-b79K xswu/HPSv2 timm/vit_large_patch14_clip_224.openai openai/clip-vit-base-patch32 bert-base-uncased Qwen/Qwen3-VL-8B-Instruct; do hf download $r; done
hf download THUDM/ImageReward ImageReward.pt med_config.json --local-dir ~/.cache/ImageReward
# Local scorers: prompt clustering, LAION aesthetic head, DreamSim.
hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir $T_DIG_MODELS/all-MiniLM-L6-v2
echo '{"repository": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"}' > $T_DIG_MODELS/all-MiniLM-L6-v2/tdig-provenance.json
mkdir -p $T_DIG_MODELS/aesthetic && curl -L -o $T_DIG_MODELS/aesthetic/sac_logos_ava1_l14_linearMSE.pth 'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth'
$PY_REWARD -c "from dreamsim import dreamsim; dreamsim(pretrained=True, device='cpu', cache_dir='$T_DIG_MODELS/dreamsim_cache')"
```
`clip`, `dino`, `gpt2` and `pag-10000` for the baselines are in [baselines/SETUP.md](baselines/SETUP.md).

## Repository layout
| Path | Contents |
| --- | --- |
| `scripts/` | `env.sh`, the one-machine trainer `train_policy.sh`, training-matrix launchers, downloads. |
| `axis_grpo/` | Training-data export and GRPO training: trainer, reward and VQAScore services, multi-node fleet. |
| `tdig/` | Policy datasets and instruction, thresholds, checkpoint validation and selection, paper metrics, test-set inputs. |
| `hpdv2/`, `iid/`, `elicitation/` | Prompt curation, i.i.d. image sets, axis elicitation. |
| `baselines/`, `benchmarks/` | Baselines on the four T2I models, COCO-250 FID, benchmark bundles. |
| `axis_framework/` | The interactive web tool. |
| `configs/`, `requirements/`, `tests/` | Asset locks, pinned environments, CPU unit tests (`pytest tests`). |

Each package has its own README. This code was written with the help of Claude, Anthropic's AI assistant.

## Note on license
The code is released under the MIT License ([LICENSE](LICENSE)). `axis_grpo/training/trainer.py` adapts parts of verl and `axis_framework/vendor/qwen_vl_utils` is qwen-vl-utils, both Apache-2.0 ([THIRD_PARTY.md](axis_grpo/training/THIRD_PARTY.md)). The models and datasets you use with this code have their own licenses; please read them.

## Citing our work
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
