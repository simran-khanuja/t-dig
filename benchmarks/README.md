# Evaluation benchmarks

The `benchmarks` package prepares pinned prompt bundles for the baseline and policy evaluations. It makes no API calls and imports no models. `baselines/README.md` describes the runners that generate and score these bundles, and the upstream code pins of the baselines.

## What the three published baselines evaluate

| Baseline | Published evaluation prompts | Reproducibility limits |
|---|---|---|
| PAG (Prompt Adaptation with GFlowNets) | Lexica, DiffusionDB, COCO and short ChatGPT-generated prompts. The released files hold 64, 256, 256 and 64 prompts. | The paper describes 256 prompts per source dataset; the release has 64 Lexica rows. The released lists are used unchanged. |
| CADS | 1,000 random COCO validation captions, 10 images each. | Caption ids, sampling seed and COCO release year are not given. |
| Scaling Group Inference | 250 COCO validation prompts for FID. | The FID caption ids are not released. |

Sources:

- [PAG, CVPR 2025, Section 4.1](https://openaccess.thecvf.com/content/CVPR2025/papers/Yun_Learning_to_Sample_Effective_and_Diverse_Prompts_for_Text-to-Image_Generation_CVPR_2025_paper.pdf), [appendix B.3](https://arxiv.org/html/2502.11477v1#A2.SS3), [released prompt files](https://github.com/dbsxodud-11/PAG/tree/d5f6ba2d2e896fb4be0660f8a2f1bacf3e5d5608/prompts).
- [CADS, Appendix G and Algorithm 1](https://arxiv.org/html/2310.17347v3#A7).
- [Scaling Group Inference, ICLR 2026, Section 4.1 and Appendix A.3.2](https://proceedings.iclr.cc/paper_files/paper/2026/file/9132fb7713ccc4860399d41695be8622-Paper-Conference.pdf), [author code](https://github.com/GaParmar/group-inference/tree/a1fff209d3796df16d87a7876c5076721ccb1772). The arXiv v1 omits the COCO-FID sentence.

No published protocol fixes a reproducible COCO sample, so all methods share COCO-250: 250 captions of COCO 2017 validation, drawn by a seeded hash of the image id (`tdig/coco_fid.py`) from a pool of one caption per validation image (5,000 rows; `benchmarks/coco.py`). The selection rules are in `baselines/README.md`. Only the 250 subset rows are rendered and scored. K and the T2I model settings are matched across arms, and each K is reported separately.

## Prompt bundles

Bundles live under `$T_DIG_DATA/benchmarks/`:

| Directory | Rows | Built by |
|---|---:|---|
| `pag-lexica` | 64 | `python -m benchmarks prepare --benchmark pag-lexica` |
| `pag-diffusiondb` | 256 | `python -m benchmarks prepare --benchmark pag-diffusiondb` |
| `pag-coco` | 256 | `python -m benchmarks prepare --benchmark pag-coco` |
| `pag-chatgpt` | 64 | `python -m benchmarks prepare --benchmark pag-chatgpt` |
| `coco2017-val-5000` | 5,000 | `python -m baselines prepare-coco` (see `baselines/SETUP.md`); the caption pool of COCO-250 |
| `heldout-axes` | held-out base prompts | `python -m tdig.heldout_bundle --dataset <-customtau dataset> --out <dir>` |

The four pinned PAG bundles each hold `source.jsonl` (exact upstream bytes), `prompts.jsonl` (stable row ids, original prompt and metadata) and `manifest.json` (revision, hashes, count). The URLs, revisions and SHA-256 hashes are in `benchmarks/sources.lock.json`. No length filter, rewriting or deduplication is applied.

`benchmarks/heldout.py` reads the held-out axis bundle: base prompts the T-DIG policy never trains on, with the variation axes collected for each (`metadata.axes`) and per-T2I-model fidelity thresholds (`metadata.align_thresh`). Baselines render from the base prompt alone; `tdig/score_baselines.py` scores each set once per requested axis.

```bash
# Plan only; no network access.
python -m benchmarks prepare --benchmark pag-coco --out "$T_DIG_DATA/benchmarks/pag-coco"

# Download the pinned prompt file; refuses an existing output directory.
python -m benchmarks prepare --benchmark pag-coco --out "$T_DIG_DATA/benchmarks/pag-coco" \
  --execute --allow-download

# Or use local bytes, which must match the SHA-256 in sources.lock.json.
python -m benchmarks prepare --benchmark pag-coco --out <new dir> \
  --source <path to eval_prompt_coco.jsonl> --execute
```

An interrupted preparation does not resume; choose a fresh output directory. `load_bundle()` checks the source against the lock and rebuilds every row before trusting `prompts.jsonl`.

## Files

| File | Purpose |
|---|---|
| `cli.py` | `python -m benchmarks prepare`; bundle loading and digests. |
| `sources.lock.json` | Pinned URLs, revisions and hashes of the four upstream prompt files. |
| `coco.py` | COCO 2017 caption selection and bundle loading. |
| `heldout.py` | Loader of the held-out axis bundle. |

Tests: `python -m pytest tests/test_benchmarks.py` (offline planning, immutable outputs, bundle identity, source corruption).
