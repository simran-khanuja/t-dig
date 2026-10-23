# Baseline environment setup

This file installs what `baselines` needs beyond the reward environment: auxiliary weights, the PAG import overlay, COCO data and the Clean-FID overlay. Run every command from the repository root. The baseline CLI never downloads or installs anything itself.

```bash
source scripts/env.sh
P=$PY_REWARD
export PIP_CACHE_DIR="$PWD/.cache/pip"
```

## Environment

Generation runs in the reward environment (`PY_REWARD`, `requirements/reward.txt`). Scaling Group Inference also needs `gurobipy` 13.0.3 (in the requirements) and a Gurobi license; the restricted license that ships with the pip package covers small problems only. Each run manifest records the installed package versions and checks them on resume.

## Weights

T2I model weights go in `$T_DIG_MODELS/{qwen-image-2512,sd35-large,sd35-medium,flux2-klein-9b}`. The auxiliary assets are pinned:

| Asset | Directory | Source |
|---|---|---|
| CLIP-B/32 | `$T_DIG_MODELS/clip` | `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| DINOv2-B | `$T_DIG_MODELS/dino` | `facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| GPT-2 tokenizer | `$T_DIG_MODELS/gpt2` | `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Trained PAG policy | `$T_DIG_MODELS/pag-10000` | [author Drive release](https://drive.google.com/drive/folders/1NUB7BFDnyS8Xsxq4Uy3tzdLFA9GRjKWS), file `12pJQQsN5Ks8Q2qZJTg0JVpqmvamEyBHD`, `10000.zip` |

- Download the Hugging Face assets with `huggingface_hub.snapshot_download(repo_id, revision=..., local_dir=...)` and no file pattern: the CLIP and DINOv2 weights are `pytorch_model.bin`.
- From the PAG archive extract only `config.json`, `generation_config.json` and `model.safetensors`; do not load its pickled training state.

## Asset locks

`configs/t2i_models/<T2I model>.assets.lock.json` lists the T2I model, `clip`, `dino`, `pag`, `tokenizer` and `pag_runtime` directories with the SHA-256 and size of every file. Paths are stored as `${T_DIG_MODELS}/<name>` and `${T_DIG_ROOT}/.cache/pag-python`. Every file is hashed before any model loads, so the weights must match the lock byte for byte.

For other paths or revisions, write an `asset-paths.json` like `configs/t2i_models/sd35-large.asset-paths.json` and create a new lock (it never overwrites an existing one; a run cannot resume under another lock):

```bash
"$P" -m baselines lock-assets --paths configs/t2i_models/sd35-large.asset-paths.json \
  --out configs/t2i_models/my-sd35-large.assets.lock.json --execute
```

## PAG overlay

PAG's policy needs Transformers 4.46.3, which conflicts with the reward environment's. It goes in an import overlay that only the PAG child process gets on `PYTHONPATH`:

```bash
"$P" -m pip install --target .cache/pag-python --no-deps \
  transformers==4.46.3 tokenizers==0.20.3 huggingface-hub==0.36.2 \
  gdown==5.2.0 beautifulsoup4==4.13.4 soupsieve==2.7
```

The asset locks hash the overlay's source files (`pag_runtime`), so a different install needs a new lock.

## COCO data

`scripts/download_benchmark_data.py` downloads one pinned file per call into a new directory `$T_DIG_DATA/downloads/<kind>`, validates the archive and extracts it. Without `--execute` it prints the URL and target only.

```bash
python scripts/download_benchmark_data.py coco-annotations --execute   # captions_val2017.json
python scripts/download_benchmark_data.py coco-images --execute        # val2017 JPEGs, FID only
python scripts/download_benchmark_data.py fid-inception --execute
"$P" -m baselines prepare-coco \
  --source "$T_DIG_DATA/downloads/coco-annotations/annotations/captions_val2017.json" \
  --out "$T_DIG_DATA/benchmarks/coco2017-val-5000" --seed 42 --execute
```

`prepare-coco` writes the caption pool from which the COCO-250 subset is drawn ([README.md](README.md#coco-250-fid)).

## Clean-FID overlay

Clean-FID 0.1.35 calls `sqrtm(..., disp=False)`, which the reward environment's SciPy lacks, so Clean-FID and SciPy 1.16.3 go in their own overlay:

```bash
"$P" -m pip install --target .cache/metric-python --no-deps clean-fid==0.1.35 scipy==1.16.3
```

`tdig/coco_fid.sh` puts the overlay first on `PYTHONPATH` and loads the Inception weights from `INCEPTION_DIR` (default `$T_DIG_DATA/downloads/fid-inception`, where the download above puts them), so Clean-FID downloads nothing.
