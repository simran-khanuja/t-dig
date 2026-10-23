"""Score the i.i.d. renders of every dataset prompt and derive per-row fidelity thresholds.

An *i.i.d. render* is an image generated from an unchanged base prompt; each base prompt has up
to four per reward T2I model, from independent seeds. The *fidelity threshold* of a row
(`align_thresh`) is the mean VQAScore of the i.i.d. renders of its base prompt by the T2I model
the policy trains against. The reward penalises an image only when it scores below that
threshold, so each prompt is held to the fidelity its T2I model reaches unaided.

VQAScore is the probability, in [0, 1], that CLIP-FlanT5 answers "yes" to whether the image
shows the base prompt. The scorer is `axis_grpo.training.vqa_service.VQABackend`, the class
the reward service uses.

Subcommands, in order:

- `index`: list every i.i.d. render of every dataset prompt. CPU; needs `pyarrow`.
- `score`: score one shard of the index on one GPU. Needs the `PY_VQA` environment.
- `merge`: combine shard outputs into one threshold table. CPU.
- `attach`: publish copies of the datasets with `align_thresh` on every row. CPU; needs
  `pyarrow`.

The i.i.d. renders use inference settings: 1024 px; 40 steps for SD3.5, 4 for FLUX.2 [klein],
50 for Qwen-Image. `merge` keeps every per-image score, so other statistics need no rescoring.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T2I_MODELS = ("sd35-medium", "sd35-large", "flux2-klein-9b", "qwen-image-2512")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_text(path, text):
    temp = Path(str(path) + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def build_index(dataset, render_roots, out, workers=32):
    """Write `index.jsonl`: one record per i.i.d. render of a dataset prompt.

    Args:
        dataset: directory holding `train.parquet` and `val.parquet`; every `prompt_id` and
            `base_prompt` in their `extra_info` is indexed.
        render_roots: directories searched for `**/images/<prompt_id>/<NNNN>/image.png`. The
            T2I model is the path component that matches one of `T2I_MODELS`; paths with a
            `source` component are skipped.
        out: output directory.
        workers: threads used to read the `receipt.json` beside each image.

    Writes `out/index.jsonl`, sorted by T2I model, prompt and image index, with fields
    `renderer`, `prompt_id`, `image_index`, `path`, `base_prompt`. Writes `out/index.json`
    with image and prompt counts per T2I model and the prompts that have fewer than four images.

    Raises `ValueError` when a receipt's `render_prompt` differs from the dataset's base
    prompt, when one (T2I model, prompt, index) has two files, or when a prompt has no image
    for some T2I model.
    """
    import pyarrow.parquet as pq

    prompts = {}
    for name in ("train.parquet", "val.parquet"):
        for row in pq.read_table(Path(dataset) / name, columns=["extra_info"]).to_pylist():
            e = row["extra_info"]
            if prompts.setdefault(e["prompt_id"], e["base_prompt"]) != e["base_prompt"]:
                raise ValueError(f"Two base prompts share prompt_id {e['prompt_id']}")
    found = {}
    for root in render_roots:
        # `find`, because a recursive pathlib glob is many times slower on the shared filesystem.
        listing = subprocess.run(
            ["find", str(root), "-path", "*/images/*/*/image.png"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for path in map(Path, listing.splitlines()):
            if "source" in path.parts:
                continue
            t2i_model = next((p for p in path.parts if p in T2I_MODELS), None)
            prompt_id, index = path.parts[-3], int(path.parts[-2])
            if t2i_model is None or prompt_id not in prompts:
                continue
            key = (t2i_model, prompt_id, index)
            if key in found:
                raise ValueError(f"Two files for {key}: {found[key]} and {path}")
            found[key] = path.resolve()

    def check(item):
        (t2i_model, prompt_id, index), path = item
        receipt = json.loads((path.parent / "receipt.json").read_text())
        if (
            receipt["prompt_id"] != prompt_id
            or receipt["index"] != index
            or receipt["render_prompt"] != prompts[prompt_id]
        ):
            raise ValueError(f"Receipt does not match the dataset prompt: {path}")
        return dict(
            renderer=t2i_model,
            prompt_id=prompt_id,
            image_index=index,
            path=str(path),
            base_prompt=prompts[prompt_id],
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(check, sorted(found.items())))
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        counts[r["renderer"]][r["prompt_id"]] += 1
    summary = dict(
        dataset=str(Path(dataset).resolve()),
        prompts=len(prompts),
        images=len(records),
        render_roots=[str(Path(r).resolve()) for r in render_roots],
        renderers={},
    )
    for t2i_model in T2I_MODELS:
        missing = sorted(p for p in prompts if counts[t2i_model][p] == 0)
        if missing:
            raise ValueError(f"{t2i_model}: {len(missing)} prompts have no i.i.d. render")
        short = {p: counts[t2i_model][p] for p in sorted(prompts) if counts[t2i_model][p] < 4}
        summary["renderers"][t2i_model] = dict(
            images=sum(counts[t2i_model].values()), prompts_with_fewer_than_4=short
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        out / "index.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    _atomic_text(out / "index.json", json.dumps(summary, indent=2) + "\n")
    return summary


def score_shard(out, shard, num_shards, cache, model, device, batch_sizes, benchmark_pairs=96):
    """Score every `num_shards`-th record of the index, starting at `shard`, on one GPU.

    Args:
        out: directory holding `index.jsonl`; shard files are written beside it.
        shard, num_shards: this worker scores records `shard, shard + num_shards, ...`.
        cache: Hugging Face hub cache holding the CLIP-FlanT5 weights.
        model: `clip-flant5-xxl` or `clip-flant5-xl`.
        device: torch device string.
        batch_sizes: candidate batch sizes. The fastest on the first `benchmark_pairs`
            records is used for the shard.
        benchmark_pairs: records timed per candidate batch size, after one warm-up batch.

    Appends one line per image to `out/scores-<shard>.jsonl` with fields `renderer`,
    `prompt_id`, `image_index`, `score`. Records already in that file are skipped, so a rerun
    continues where the last one stopped. Writes `out/benchmark-<shard>.json` with images per
    second and peak GPU memory for each candidate, and `out/done-<shard>.json` on completion.

    Raises `RuntimeError` from the backend when a score is non-finite or outside [0, 1].
    """
    import torch
    from axis_grpo.training.vqa_service import VQABackend

    out = Path(out)
    records = _read_jsonl(out / "index.jsonl")[shard::num_shards]
    target = out / f"scores-{shard:02d}.jsonl"
    done = set()
    if target.exists():
        done = {(r["renderer"], r["prompt_id"], r["image_index"]) for r in _read_jsonl(target)}
    todo = [r for r in records if (r["renderer"], r["prompt_id"], r["image_index"]) not in done]
    print(f"shard {shard}/{num_shards}: {len(records)} records, {len(todo)} to score", flush=True)
    if not todo:
        _atomic_text(out / f"done-{shard:02d}.json", json.dumps(dict(records=len(records))) + "\n")
        return
    backend = VQABackend(model, device, cache, batch_sizes[0])
    pairs = lambda rs: [[r["path"], r["base_prompt"]] for r in rs]

    sample = todo[:benchmark_pairs]
    backend.batch_size = batch_sizes[0]
    backend._score(pairs(sample[: batch_sizes[0]]))  # warm-up: kernels and page cache
    measured = []
    for size in batch_sizes:
        backend.batch_size = size
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            backend._score(pairs(sample))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            measured.append(dict(batch_size=size, out_of_memory=True))
            continue
        torch.cuda.synchronize()
        measured.append(
            dict(
                batch_size=size,
                images_per_second=len(sample) / (time.monotonic() - start),
                peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
            )
        )
    usable = [m for m in measured if not m.get("out_of_memory")]
    best = max(usable, key=lambda m: m["images_per_second"])
    backend.batch_size = best["batch_size"]
    _atomic_text(
        out / f"benchmark-{shard:02d}.json",
        json.dumps(
            dict(
                pairs=len(sample),
                candidates=measured,
                selected=best["batch_size"],
                note="every candidate reads the same files, so later candidates see a warmer page cache",
            ),
            indent=2,
        )
        + "\n",
    )
    print(f"selected batch {best['batch_size']}: {measured}", flush=True)

    # The backend decodes images serially, so threads prefetch the next chunk's bytes.
    chunk = best["batch_size"] * 8
    touch = lambda r: Path(r["path"]).read_bytes() and None
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool, open(target, "a", encoding="utf-8") as stream:
        ahead = pool.map(touch, todo[:chunk])
        for i in range(0, len(todo), chunk):
            list(ahead)
            ahead = pool.map(touch, todo[i + chunk : i + 2 * chunk])
            block = todo[i : i + chunk]
            for r, s in zip(block, backend._score(pairs(block)), strict=True):
                stream.write(
                    json.dumps(
                        dict(
                            renderer=r["renderer"],
                            prompt_id=r["prompt_id"],
                            image_index=r["image_index"],
                            score=s,
                        )
                    )
                    + "\n"
                )
            stream.flush()
            n = i + len(block)
            rate = n / (time.monotonic() - start)
            print(
                f"  {n}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo) - n) / rate / 60:.0f} min",
                flush=True,
            )
    _atomic_text(
        out / f"done-{shard:02d}.json",
        json.dumps(
            dict(
                records=len(records),
                images_per_second=len(todo) / (time.monotonic() - start),
                batch_size=best["batch_size"],
            )
        )
        + "\n",
    )


def merge(out, num_shards):
    """Combine shard scores into one threshold per T2I model and prompt.

    Args:
        out: directory holding `index.jsonl` and `scores-*.jsonl`.
        num_shards: number of shards the index was scored in.

    Writes `out/thresholds.jsonl`, one line per (T2I model, prompt) with fields `renderer`,
    `prompt_id`, `base_prompt`, `align_thresh` (mean score), `score_min`, `images`, `scores`
    (per-image, ordered by image index). Writes `out/thresholds.json` with the distribution of
    `align_thresh` per T2I model. Returns that summary.

    Raises `ValueError` when a shard is unfinished or the scored records differ from the index.
    """
    out = Path(out)
    index = {
        (r["renderer"], r["prompt_id"], r["image_index"]): r
        for r in _read_jsonl(out / "index.jsonl")
    }
    scores = {}
    for shard in range(num_shards):
        if not (out / f"done-{shard:02d}.json").exists():
            raise ValueError(f"Shard {shard} is unfinished")
        for r in _read_jsonl(out / f"scores-{shard:02d}.jsonl"):
            scores[(r["renderer"], r["prompt_id"], r["image_index"])] = r["score"]
    if set(scores) != set(index):
        raise ValueError(f"Scored records differ from the index: {len(scores)} vs {len(index)}")
    base_prompt = {key[:2]: r["base_prompt"] for key, r in index.items()}
    groups = defaultdict(list)
    for (t2i_model, prompt_id, image_index), score in sorted(scores.items()):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid VQAScore")
        groups[(t2i_model, prompt_id)].append(score)
    lines, by_t2i_model = [], defaultdict(list)
    for (t2i_model, prompt_id), values in sorted(groups.items()):
        mean = sum(values) / len(values)
        by_t2i_model[t2i_model].append(mean)
        lines.append(
            json.dumps(
                dict(
                    renderer=t2i_model,
                    prompt_id=prompt_id,
                    base_prompt=base_prompt[(t2i_model, prompt_id)],
                    align_thresh=mean,
                    score_min=min(values),
                    images=len(values),
                    scores=values,
                ),
                ensure_ascii=False,
            )
        )
    summary = {}
    for t2i_model, values in by_t2i_model.items():
        ordered = sorted(values)
        # Nearest-rank percentile: defined for any number of prompts, including one.
        q = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]
        summary[t2i_model] = dict(
            prompts=len(values),
            mean=statistics.mean(values),
            sd=statistics.pstdev(values),
            p05=q(0.05),
            p25=q(0.25),
            median=q(0.5),
            p75=q(0.75),
            p95=q(0.95),
            below_0p5=sum(v < 0.5 for v in values),
            at_least_0p9=sum(v >= 0.9 for v in values),
        )
    _atomic_text(out / "thresholds.jsonl", "\n".join(lines) + "\n")
    _atomic_text(out / "thresholds.json", json.dumps(summary, indent=2) + "\n")
    return summary


def attach(thresholds, dataset, out):
    """Publish a copy of the tdig datasets with `align_thresh` on every row.

    Args:
        thresholds: `thresholds.jsonl` written by `merge`.
        dataset: directory written by `tdig.build_dataset`, one subdirectory per variant.
        out: new directory to publish.

    For each variant, every parquet file is copied with two additions per row:
    `align_thresh` inside the JSON of `reward_model.ground_truth`, and
    `extra_info.align_thresh` plus `extra_info.align_thresh_images` (number of i.i.d. renders
    averaged). The value is the threshold of the row's `prompt_id` for the variant's
    `instruction_renderer`. Each variant receives a `manifest.json` with file hashes.

    Raises `ValueError` when a row has no threshold.
    """
    import pyarrow.parquet as pq
    from axis_grpo.dataset import export_parquet
    from axis_grpo.prepare import new_directory
    from baselines.assets import file_hash
    from hpdv2.storage import atomic_json, read_json

    table = {(r["renderer"], r["prompt_id"]): r for r in _read_jsonl(thresholds)}
    dataset = Path(dataset)
    with new_directory(out, (dataset,)) as target:
        for variant in sorted(p for p in dataset.iterdir() if p.is_dir()):
            manifest = read_json(variant / "manifest.json")
            t2i_model = manifest["instruction_renderer"]
            (target / variant.name).mkdir()
            for source in sorted(variant.glob("*.parquet")):
                if file_hash(source) != manifest["files"][source.name]:
                    raise ValueError(f"Source changed since its manifest was written: {source}")
                rows = pq.read_table(source).to_pylist()
                for row in rows:
                    e = row["extra_info"]
                    entry = table.get((t2i_model, e["prompt_id"]))
                    if entry is None or entry["base_prompt"] != e["base_prompt"]:
                        raise ValueError(f"No threshold for {t2i_model} / {e['prompt_id']}")
                    spec = json.loads(row["reward_model"]["ground_truth"])
                    spec["align_thresh"] = entry["align_thresh"]
                    row["reward_model"]["ground_truth"] = json.dumps(spec, ensure_ascii=False)
                    e["align_thresh"] = entry["align_thresh"]
                    e["align_thresh_images"] = entry["images"]
                export_parquet(target / variant.name / source.name, rows)
                if pq.read_table(target / variant.name / source.name).to_pylist() != rows:
                    raise RuntimeError("Parquet export/readback differs")
            (target / variant.name / "instruction.txt").write_text(
                (variant / "instruction.txt").read_text()
            )
            manifest.update(
                align_thresh="mean VQAScore of the i.i.d. renders of the base prompt, per T2I model",
                align_thresh_source=str(Path(thresholds).resolve()),
                align_thresh_sha256=file_hash(thresholds),
                source_dataset=str(variant.resolve()),
            )
            manifest["files"] = {
                f.name: file_hash(f)
                for f in sorted((target / variant.name).iterdir())
                if f.is_file()
            }
            atomic_json(target / variant.name / "manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("index")
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--render-root", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--num-shards", type=int, required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--model", default="clip-flant5-xxl")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    s = sub.add_parser("merge")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--num-shards", type=int, required=True)
    s = sub.add_parser("attach")
    s.add_argument("--thresholds", type=Path, required=True)
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "index":
        print(json.dumps(build_index(a.dataset, a.render_root, a.out), indent=2))
    elif a.command == "score":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE=a.cache)
        score_shard(a.out, a.shard, a.num_shards, a.cache, a.model, a.device, a.batch_sizes)
    elif a.command == "merge":
        print(json.dumps(merge(a.out, a.num_shards), indent=2))
    else:
        attach(a.thresholds, a.dataset, a.out)


if __name__ == "__main__":
    main()
