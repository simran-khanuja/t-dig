"""Compute every metric the paper reports, for baseline image sets and policy checkpoints.

A *set* is the `k` images one method produced for one request. Baselines and trained policies are
indexed into one format and scored by the same code.

Metrics. "Base" means scored against the unchanged base prompt; "own" means each image against
the prompt that rendered it, reported only when a method rewrites the prompt.

| Key | Level | Definition |
| --- | --- | --- |
| `vqascore_mean`, `vqascore_min` | image, base | CLIP-FlanT5-XXL probability of "yes"; mean and minimum over the set. |
| `vqascore_own_mean`, `vqascore_own_min` | image, own | The same against the rendering prompt. |
| `clipscore`, `clipscore_own` | image | CLIPScore of Hessel et al.: `2.5 * max(cos, 0)` with OpenAI CLIP ViT-B/32; text truncated to 77 tokens. No base prompt reaches 77 tokens; most rewritten prompts of a trained FLUX.2 [klein] policy exceed it, so `clipscore_own` scores truncated text, is written to `rows.jsonl` only, and is left out of the summaries. |
| `tpips_vendi` | set, per axis | Normalised Vendi, `(vendi - 1) / (k - 1)`, of TPIPS embeddings conditioned on the axis text. `tpips_pairwise` is the mean pairwise cosine distance of the same embeddings. |
| `dino_vendi`, `dreamsim_vendi` | set | Normalised Vendi of DINOv2-B CLS and of DreamSim embeddings. Not conditioned on an axis. `*_raw` is the Vendi score itself, from 1 to `k`. |
| `pickscore`, `imagereward`, `hpsv3` | image, base | PickScore v1, ImageReward-v1.0, HPSv3. PickScore reads 77 text tokens and ImageReward 35; HPSv3 reads the whole prompt. Against the base prompt, ImageReward truncates 4.5% of the held-out prompts and 3.1% of the training prompts, and PickScore none. |
| `aesthetic` | image | LAION-Aesthetics V2. Ignores the prompt. |

Subcommands, in order:

- `index-baselines` or `index-validation` (CPU): write `sets.jsonl` and `index.jsonl`.
- `tdig.iid_vqa score` on the same directory (VQAScore, the PY_VQA environment).
- `score` (one shard per GPU, the PY_REWARD environment): everything else.
- `merge` (CPU): per-row table, per-group summary and a CSV.

`tdig/paper_metrics.sh` runs the last three steps on one node.
"""

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

from .iid_vqa import _atomic_text, _read_jsonl

IMAGE_BASE = ("clipscore", "pickscore", "imagereward", "hpsv3", "aesthetic")
SET_LEVEL = ("dino_vendi", "dino_vendi_raw", "dreamsim_vendi", "dreamsim_vendi_raw")
SUMMARY = (
    "vqascore_mean",
    "vqascore_min",
    "vqascore_own_mean",
    "vqascore_own_min",
    "clipscore",
    "tpips_vendi",
    "tpips_pairwise",
    "dino_vendi",
    "dreamsim_vendi",
    "pickscore",
    "imagereward",
    "hpsv3",
    "aesthetic",
    "fidelity_penalty",
    "reward",
)


def _write_index(out, sets):
    """Write `sets.jsonl`, the VQAScore index `index.jsonl` and `index.json` into `out`.

    `index.jsonl` has the format `tdig.iid_vqa score` reads. Its `renderer` field holds the
    set's group followed by `||base` or `||own`, and `base_prompt` holds the text to score
    against: the base prompt, or for `||own` the prompt that rendered that image. Returns
    `{group: {sets, rewritten}}`.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for s in sets:
        for i, path in enumerate(s["paths"]):
            images.append(
                dict(
                    renderer=s["group"] + "||base",
                    prompt_id=s["set_id"],
                    image_index=i,
                    path=path,
                    base_prompt=s["base_prompt"],
                )
            )
            if s.get("own_prompts"):
                images.append(
                    dict(
                        renderer=s["group"] + "||own",
                        prompt_id=s["set_id"],
                        image_index=i,
                        path=path,
                        base_prompt=s["own_prompts"][i],
                    )
                )
    _atomic_text(
        out / "sets.jsonl", "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sets)
    )
    _atomic_text(
        out / "index.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in images)
    )
    groups = {}
    for s in sets:
        g = groups.setdefault(s["group"], dict(sets=0, rewritten=0))
        g["sets"] += 1
        g["rewritten"] += bool(s.get("own_prompts"))
    _atomic_text(
        out / "index.json",
        json.dumps(dict(sets=len(sets), images=len(images), groups=groups), indent=2) + "\n",
    )
    return groups


def index_baselines(runs, out, coco_subset=None):
    """Index complete rows of baseline run directories (`baselines.runner` layout).

    `coco_subset`, a `subset.json` of `tdig.coco_fid select`, restricts rows of the COCO benchmark
    to its indices; rows of other benchmarks are unaffected.

    The group is `<T2I model>|<method>|<benchmark>`. A set's factors are the axes of a held-out
    axis row, else `overall`. `own_prompts` are the `prompts` of the row's `adaptations.json`
    when any of them differs from the base prompt, as for PAG. A row rendered twice keeps its
    first complete copy. Writes the files of `_write_index` and returns its result.
    """
    wanted = (
        {r["index"] for r in json.loads(Path(coco_subset).read_text())["rows"]}
        if coco_subset
        else None
    )
    sets, seen = [], set()
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text())
        config = manifest["config"]
        group = f'{config["renderer"]}|{config["method"]}|{manifest["bundle"]["benchmark"]}'
        for row in json.loads((run / "input.json").read_text()):
            if (
                wanted is not None
                and manifest["bundle"]["benchmark"].startswith("coco")
                and row["index"] not in wanted
            ):
                continue
            if (group, row["id"]) in seen:
                continue
            work = run / "work" / f'{row["index"]:05d}'
            paths = [work / "slots" / f"{s:04d}" / "image.png" for s in range(config["k"])]
            if not all(p.is_file() for p in paths):
                continue
            seen.add((group, row["id"]))
            meta = row.get("metadata", {})
            own = None
            if (work / "adaptations.json").is_file():
                prompts = json.loads((work / "adaptations.json").read_text()).get("prompts") or []
                if len(prompts) == config["k"] and any(p != row["prompt"] for p in prompts):
                    own = prompts
            factors = [
                dict(
                    row_id=x["row_id"],
                    factor=x["axis"],
                    axis_kind=x["axis_kind"],
                    contrastive=x["contrastive"],
                )
                for x in meta.get("axes", [])
            ] or [dict(row_id=row["id"], factor="overall", axis_kind="overall", contrastive=False)]
            sets.append(
                dict(
                    group=group,
                    set_id=row["id"],
                    base_prompt=row["prompt"],
                    own_prompts=own,
                    factors=factors,
                    paths=[str(p.resolve()) for p in paths],
                    align_thresh=meta.get("align_thresh", {}).get(config["renderer"]),
                )
            )
    return _write_index(out, sets)


def index_validation(val_outs, out):
    """Index the images of `tdig.validate_checkpoints` output directories.

    The group is `<training run>|step_<N>|<set>`. Each scored row is one set with one factor, its
    axis. `own_prompts` are the prompts parsed from the completion; they are omitted for an
    editing run, whose completions are edit instructions and not descriptions of the image. A
    malformed completion has no images and is counted in `out/skipped.json`.

    Raises `ValueError` when the images of a scored row are missing from every reward service.
    """
    from axis_grpo.training.protocol import parse_prompts

    sets, skipped = [], {}
    for root in map(Path, val_outs):
        plan = json.loads((root / "plan.json").read_text())
        run = Path(plan["run"])
        name = f"{run.parent.name}/{run.name}"
        editing = "drift_weight" in plan["reward"]
        for path in sorted(root.glob("step_*/*.jsonl")):
            group = f"{name}|{path.parent.name}|{path.stem}"
            for r in _read_jsonl(path):
                if r["request_id"] is None:
                    skipped[group] = skipped.get(group, 0) + 1
                    continue
                folders = [d for d in root.glob(f"reward-*/images/{r['request_id']}") if d.is_dir()]
                if not folders:
                    raise ValueError(f"No saved images for {r['row_id']} of {group}")
                paths = sorted(folders[0].glob("*.png"))
                kind = "overall" if r["axis"] == "overall" else "axis"
                sets.append(
                    dict(
                        group=group,
                        set_id=r["row_id"],
                        base_prompt=r["base_prompt"],
                        own_prompts=None if editing else parse_prompts(r["completion"], r["k"]),
                        paths=[str(p.resolve()) for p in paths],
                        # Rows of sets named heldout* or probe* carry their own threshold;
                        # elsewhere the logged value is the configured fallback, not a
                        # property of the row (tdig/README.md, "Checkpoint validation").
                        align_thresh=(
                            r["metrics"].get("fidelity_threshold")
                            if path.stem.startswith(("heldout", "probe"))
                            else None
                        ),
                        factors=[
                            dict(
                                row_id=r["row_id"],
                                factor=r["axis"],
                                axis_kind=kind,
                                contrastive=path.stem.startswith("probe"),
                            )
                        ],
                    )
                )
    groups = _write_index(out, sets)
    _atomic_text(Path(out) / "skipped.json", json.dumps(skipped, indent=2) + "\n")
    return groups


class Scorers:
    def __init__(self, device, cache, tpips_checkpoint, aesthetic_head, dreamsim_cache, dino_path):
        """Load every scorer except VQAScore on one device.

        Args:
            device: torch device string.
            cache: Hugging Face hub cache (CLIP ViT-B/32, PickScore, HPSv3, its Qwen2-VL base).
            tpips_checkpoint: local snapshot or `repo@revision` of `sywang/TPIPS-Embed-Qwen3VL-8B`.
            aesthetic_head: LAION-Aesthetics V2 linear head file.
            dreamsim_cache: directory holding the DreamSim weights.
            dino_path: local DINOv2-B model directory.
        """
        import torch
        import tpips
        from dreamsim import dreamsim
        from transformers import AutoModel, CLIPModel, CLIPProcessor
        from axis_grpo.training.tracked import TrackedScorers
        from elicitation.tpips_compat import qwen_tokenizer_compat, verify_tokenizer
        from axis_grpo.training import hpsv3_model
        from .scorers import pickscore_model

        self.device, self.torch = device, torch
        self.tracked = TrackedScorers(("imagereward", "aesthetic"), device, aesthetic_head)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.pick, self.pick_processor = pickscore_model.load(device)
        self.hps, self.hps_processor = hpsv3_model.load(device)
        self.pickscore_model, self.hpsv3_model = pickscore_model, hpsv3_model
        self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True).to(device).eval()
        self.dreamsim, self.dreamsim_preprocess = dreamsim(
            pretrained=True, device=device, cache_dir=dreamsim_cache
        )
        from baselines.assets import local_checkpoint

        tpips_checkpoint = local_checkpoint(tpips_checkpoint, cache)
        with qwen_tokenizer_compat():
            self.tpips = tpips.load_model(
                "embedding", tpips_checkpoint, device=device, cache_dir=cache
            )
        verify_tokenizer(self.tpips, tpips_checkpoint, [self.tpips.create_prompt("overall")])

    def clipscore(self, images, texts):
        """Return the CLIPScore of image `i` against text `i`: `2.5 * max(cosine, 0)`, CLIP ViT-B/32."""
        torch = self.torch
        inputs = self.clip_processor(
            text=list(texts),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        with torch.inference_mode():
            out = self.clip(**inputs)
        image = torch.nn.functional.normalize(out.image_embeds.float(), dim=-1)
        text = torch.nn.functional.normalize(out.text_embeds.float(), dim=-1)
        return (2.5 * (image * text).sum(-1).clamp(min=0)).cpu().tolist()

    def dino_embeddings(self, images):
        """Return DINOv2-B CLS embeddings.

        Images are resized to 256 px (bilinear) and ImageNet-normalised, the preprocessing of
        `baselines.features.Features`.
        """
        import numpy as np

        torch = self.torch
        # Resized one at a time so a set may mix sizes; for same-size images this equals
        # resizing the stacked batch.
        pixels = torch.cat(
            [
                torch.nn.functional.interpolate(
                    torch.from_numpy(np.array(im, copy=True))
                    .permute(2, 0, 1)[None]
                    .float()
                    .to(self.device)
                    / 255,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
                for im in images
            ]
        )
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.inference_mode():
            return (
                self.dino(pixel_values=(pixels - mean) / std)
                .last_hidden_state[:, 0]
                .float()
                .cpu()
                .numpy()
            )

    def score(self, item):
        """Return every metric of one set except VQAScore, as a dict; see the module docstring."""
        from PIL import Image
        from axis_grpo.training.metrics import diversity

        torch = self.torch
        images = [Image.open(p).convert("RGB") for p in item["paths"]]
        base = item["base_prompt"]
        result = dict(group=item["group"], set_id=item["set_id"])
        result["clipscore"] = self.clipscore(images, [base] * len(images))
        if item.get("own_prompts"):
            result["clipscore_own"] = self.clipscore(images, item["own_prompts"])
        result["pickscore"] = [
            float(v)
            for v in self.pickscore_model.score(
                self.pick, self.pick_processor, images, base, self.device
            )
        ]
        result["hpsv3"] = [
            float(v)
            for v in self.hpsv3_model.score(self.hps, self.hps_processor, images, base, self.device)
        ]
        result.update(self.tracked.score(images, base))
        d = diversity(self.dino_embeddings(images))
        result.update(dino_vendi=d["normalized_vendi"], dino_vendi_raw=d["vendi"])
        with torch.inference_mode():
            vectors = torch.cat(
                [self.dreamsim.embed(self.dreamsim_preprocess(im).to(self.device)) for im in images]
            )
        d = diversity(vectors.float().cpu().numpy())
        result.update(dreamsim_vendi=d["normalized_vendi"], dreamsim_vendi_raw=d["vendi"])
        result["tpips"] = {}
        for entry in item["factors"]:
            with torch.inference_mode():
                vectors = (
                    self.tpips.embed(images, factor=entry["factor"], normalized=True)
                    .float()
                    .cpu()
                    .numpy()
                )
            d = diversity(vectors)
            result["tpips"][entry["row_id"]] = dict(
                tpips_vendi=d["normalized_vendi"], tpips_pairwise=d["pairwise"]
            )
        return result


def score_shard(out, shard, num_shards, **assets):
    """Score every `num_shards`-th set, starting at `shard`; append to `out/metrics-<shard>.jsonl`.

    Sets already in that file are skipped, so a rerun continues. Writes
    `out/metrics-done-<shard>.json` on completion.
    """
    out = Path(out)
    todo = _read_jsonl(out / "sets.jsonl")[shard::num_shards]
    target = out / f"metrics-{shard:02d}.jsonl"
    done = {(r["group"], r["set_id"]) for r in _read_jsonl(target)} if target.exists() else set()
    todo = [s for s in todo if (s["group"], s["set_id"]) not in done]
    print(f"metrics shard {shard}/{num_shards}: {len(todo)} sets to score", flush=True)
    if todo:
        scorers = Scorers(**assets)
        start = time.monotonic()
        with open(target, "a", encoding="utf-8") as stream:
            for i, item in enumerate(todo, 1):
                stream.write(json.dumps(scorers.score(item), ensure_ascii=False) + "\n")
                if i % 25 == 0 or i == len(todo):
                    stream.flush()
                    rate = i / (time.monotonic() - start)
                    print(
                        f"  {i}/{len(todo)}  {rate:.2f} sets/s  eta {(len(todo) - i) / rate / 60:.0f} min",
                        flush=True,
                    )
    _atomic_text(out / f"metrics-done-{shard:02d}.json", json.dumps(dict(sets=len(todo))) + "\n")


def merge(out, num_shards):
    """Join VQAScore and the other metrics, and summarise per group.

    Reads `thresholds.jsonl` (written by `tdig.iid_vqa merge`) and `metrics-*.jsonl`. Writes
    `out/rows.jsonl`, one line per set and factor with every key of the module docstring plus,
    when the set has an `align_thresh`, `fidelity_penalty` (`mean(max(0, align_thresh - a_i))`
    over base VQAScores) and `reward` (`tpips_vendi - fidelity_penalty`). Writes
    `out/summary.json`, per group the mean of each key over its lines, overall and by kind
    (`axis`, `overall`, `contrastive`), and `out/summary.csv` with one line per
    group. Returns the summary. Raises `ValueError` when a shard is unfinished or a set lacks a
    score.
    """
    out = Path(out)
    vqa = {(r["renderer"], r["prompt_id"]): r for r in _read_jsonl(out / "thresholds.jsonl")}
    scored = {}
    for shard in range(num_shards):
        if not (out / f"metrics-done-{shard:02d}.json").exists():
            raise ValueError(f"Metric shard {shard} is unfinished")
        scored.update(
            {(r["group"], r["set_id"]): r for r in _read_jsonl(out / f"metrics-{shard:02d}.jsonl")}
        )
    mean = lambda values: sum(values) / len(values)
    rows = []
    for item in _read_jsonl(out / "sets.jsonl"):
        key = (item["group"], item["set_id"])
        base = vqa.get((item["group"] + "||base", item["set_id"]))
        if key not in scored or base is None:
            raise ValueError(f"Set without all scores: {key}")
        m = scored[key]
        shared = dict(
            group=item["group"],
            set_id=item["set_id"],
            vqascore_mean=base["align_thresh"],
            vqascore_min=base["score_min"],
            **{k: mean(m[k]) for k in IMAGE_BASE},
            **{k: m[k] for k in SET_LEVEL},
        )
        if item.get("own_prompts"):
            own = vqa[(item["group"] + "||own", item["set_id"])]
            shared.update(
                vqascore_own_mean=own["align_thresh"],
                vqascore_own_min=own["score_min"],
                clipscore_own=mean(m["clipscore_own"]),
            )
        if item.get("align_thresh") is not None:
            shared["fidelity_penalty"] = mean(
                [max(0.0, item["align_thresh"] - a) for a in base["scores"]]
            )
        for entry in item["factors"]:
            line = dict(
                shared,
                row_id=entry["row_id"],
                factor=entry["factor"],
                **m["tpips"][entry["row_id"]],
                kind="contrastive" if entry["contrastive"] else entry["axis_kind"],
            )
            if "fidelity_penalty" in line:
                line["reward"] = line["tpips_vendi"] - line["fidelity_penalty"]
            rows.append(line)
    block = lambda items: dict(
        lines=len(items),
        **{
            k: statistics.mean(r[k] for r in items if k in r)
            for k in SUMMARY
            if any(k in r for r in items)
        },
    )
    summary = {}
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group]
        summary[group] = dict(
            **block(members),
            per_kind={
                k: block([r for r in members if r["kind"] == k])
                for k in sorted({r["kind"] for r in members})
            },
        )
    _atomic_text(
        out / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    _atomic_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    with open(out / "summary.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "lines"] + list(SUMMARY))
        for group, s in summary.items():
            writer.writerow(
                [group, s["lines"]] + [f"{s[k]:.4f}" if k in s else "" for k in SUMMARY]
            )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("index-baselines")
    s.add_argument("--run", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--coco-subset", type=Path, help="subset.json restricting COCO rows")
    s = sub.add_parser("index-validation")
    s.add_argument(
        "--val-out",
        type=Path,
        action="append",
        required=True,
        help="output directory of tdig.validate_checkpoints",
    )
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--num-shards", type=int, required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--tpips-checkpoint", required=True)
    s.add_argument("--aesthetic-head", required=True)
    s.add_argument("--dreamsim-cache", required=True)
    s.add_argument("--dino-path", required=True)
    s.add_argument("--device", default="cuda:0")
    s = sub.add_parser("merge")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--num-shards", type=int, required=True)
    a = p.parse_args()
    if a.command == "index-baselines":
        print(json.dumps(index_baselines(a.run, a.out, a.coco_subset), indent=1))
    elif a.command == "index-validation":
        print(json.dumps(index_validation(a.val_out, a.out), indent=1))
    elif a.command == "score":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE=a.cache)
        score_shard(
            a.out,
            a.shard,
            a.num_shards,
            device=a.device,
            cache=a.cache,
            tpips_checkpoint=a.tpips_checkpoint,
            aesthetic_head=a.aesthetic_head,
            dreamsim_cache=a.dreamsim_cache,
            dino_path=a.dino_path,
        )
    else:
        merge(a.out, a.num_shards)
        print(f"wrote {a.out}/summary.json, summary.csv and rows.jsonl")


if __name__ == "__main__":
    main()
