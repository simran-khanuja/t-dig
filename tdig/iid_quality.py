"""Measure the per-prompt quality threshold: the mean HPSv3 score of a T2I model's i.i.d. renders.

The reward's optional quality term (`axis_grpo/training/protocol.py`, `QUALITY_KEYS`) charges a
set whose images score below what the T2I model reaches on the same base prompt unaided. This
module measures that level, `quality_thresh`, and attaches it to the datasets. It mirrors
`iid_vqa.py` (the fidelity threshold `align_thresh`) and reads the image index written there.

HPSv3 resizes every image to about 448 x 448 px, so the 1024 px i.i.d. renders and the 512 px
training renders reach the scorer at the same size.

Commands:

- `score`: score one shard of the index with HPSv3 against the base prompt. One GPU. Writes
  `quality-scores-<shard>.jsonl`; a rerun skips finished images.
- `merge`: write `quality.jsonl`, one line per T2I model and prompt with `quality_thresh` (mean),
  `images` and `scores`, and `quality.json` with per-T2I-model statistics.
- `attach`: publish a copy of a dataset with `quality_thresh` in every row's ground truth and
  `extra_info`. CPU. Only variants whose `instruction_renderer` is in `--t2i-models` are copied.
"""

import argparse
import json
import statistics
from pathlib import Path


def _read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score(index, out, shard, num_shards, t2i_models, cache, batch=16):
    """Score the shard's images and append `{renderer, prompt_id, image_index, hpsv3}` lines.

    Writes `quality-scores-<shard>.jsonl` (skipping images already in it) and, when done,
    `quality-done-<shard>.json`.
    """
    import os

    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from PIL import Image
    from axis_grpo.training import hpsv3_model

    records = [r for r in _read_jsonl(index) if r["renderer"] in t2i_models]
    records = records[shard::num_shards]
    path = Path(out) / f"quality-scores-{shard:02d}.jsonl"
    done = (
        {(r["renderer"], r["prompt_id"], r["image_index"]) for r in _read_jsonl(path)}
        if path.exists()
        else set()
    )
    todo = [r for r in records if (r["renderer"], r["prompt_id"], r["image_index"]) not in done]
    print(f"shard {shard}: {len(records)} images, {len(todo)} to score", flush=True)
    model, processor = hpsv3_model.load("cuda:0")
    with path.open("a", encoding="utf-8") as sink:
        for start in range(0, len(todo), batch):
            chunk = todo[start : start + batch]
            images = [Image.open(r["path"]).convert("RGB") for r in chunk]
            values = hpsv3_model.score(
                model, processor, images, [r["base_prompt"] for r in chunk], "cuda:0"
            )
            for r, v in zip(chunk, values):
                sink.write(
                    json.dumps(
                        dict(
                            renderer=r["renderer"],
                            prompt_id=r["prompt_id"],
                            image_index=r["image_index"],
                            hpsv3=float(v),
                        )
                    )
                    + "\n"
                )
            sink.flush()
            if (start // batch) % 50 == 0:
                print(f"shard {shard}: {start + len(chunk)} of {len(todo)}", flush=True)
    (Path(out) / f"quality-done-{shard:02d}.json").write_text(
        json.dumps(dict(images=len(records))) + "\n"
    )


def merge(index, out, num_shards, t2i_models):
    """Write `quality.jsonl` and `quality.json`. Raises `ValueError` when an indexed image has no score."""
    out = Path(out)
    base = {
        (r["renderer"], r["prompt_id"]): r["base_prompt"]
        for r in _read_jsonl(index)
        if r["renderer"] in t2i_models
    }
    expected = sum(1 for r in _read_jsonl(index) if r["renderer"] in t2i_models)
    scores = {}
    for shard in range(num_shards):
        for r in _read_jsonl(out / f"quality-scores-{shard:02d}.jsonl"):
            scores.setdefault((r["renderer"], r["prompt_id"]), {})[r["image_index"]] = r["hpsv3"]
    if sum(len(v) for v in scores.values()) != expected:
        raise ValueError(
            f"{sum(len(v) for v in scores.values())} scores for {expected} indexed images"
        )
    summary = {}
    with (out / "quality.jsonl").open("w", encoding="utf-8") as sink:
        for (t2i_model, prompt_id), values in sorted(scores.items()):
            ordered = [values[i] for i in sorted(values)]
            sink.write(
                json.dumps(
                    dict(
                        renderer=t2i_model,
                        prompt_id=prompt_id,
                        base_prompt=base[(t2i_model, prompt_id)],
                        quality_thresh=statistics.mean(ordered),
                        images=len(ordered),
                        scores=ordered,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            summary.setdefault(t2i_model, []).append(
                (statistics.mean(ordered), statistics.pstdev(ordered))
            )
    report = {}
    for t2i_model, pairs in summary.items():
        means = sorted(m for m, _ in pairs)
        report[t2i_model] = dict(
            prompts=len(means),
            mean=statistics.mean(means),
            sd_across_prompts=statistics.pstdev(means),
            mean_sd_within_prompt=statistics.mean(s for _, s in pairs),
            p05=means[len(means) // 20],
            median=means[len(means) // 2],
            p95=means[-(len(means) // 20) - 1],
        )
    (out / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def attach(quality, dataset, out, t2i_models):
    """Publish copies of the dataset variants of `renderers` with `quality_thresh` on every row."""
    import pyarrow.parquet as pq
    from axis_grpo.dataset import export_parquet
    from axis_grpo.prepare import new_directory
    from baselines.assets import file_hash
    from hpdv2.storage import atomic_json, read_json

    table = {(r["renderer"], r["prompt_id"]): r for r in _read_jsonl(quality)}
    dataset = Path(dataset)
    with new_directory(out, (dataset,)) as target:
        for variant in sorted(p for p in dataset.iterdir() if p.is_dir()):
            manifest = read_json(variant / "manifest.json")
            t2i_model = manifest["instruction_renderer"]
            if t2i_model not in t2i_models:
                continue
            (target / variant.name).mkdir()
            for source in sorted(variant.glob("*.parquet")):
                if file_hash(source) != manifest["files"][source.name]:
                    raise ValueError(f"Source changed since its manifest was written: {source}")
                rows = pq.read_table(source).to_pylist()
                for row in rows:
                    e = row["extra_info"]
                    entry = table.get((t2i_model, e["prompt_id"]))
                    if entry is None or entry["base_prompt"] != e["base_prompt"]:
                        raise ValueError(f"No quality threshold for {t2i_model} / {e['prompt_id']}")
                    spec = json.loads(row["reward_model"]["ground_truth"])
                    spec["quality_thresh"] = entry["quality_thresh"]
                    row["reward_model"]["ground_truth"] = json.dumps(spec, ensure_ascii=False)
                    e["quality_thresh"] = entry["quality_thresh"]
                export_parquet(target / variant.name / source.name, rows)
                if pq.read_table(target / variant.name / source.name).to_pylist() != rows:
                    raise RuntimeError("Parquet export/readback differs")
            (target / variant.name / "instruction.txt").write_text(
                (variant / "instruction.txt").read_text()
            )
            manifest.update(
                quality_thresh="mean HPSv3 of the i.i.d. renders of the base prompt, per T2I model",
                quality_thresh_source=str(Path(quality).resolve()),
                quality_thresh_sha256=file_hash(quality),
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
    for name in ("score", "merge"):
        s = sub.add_parser(name)
        s.add_argument(
            "--index", type=Path, required=True, help="index.jsonl written by tdig.iid_vqa index"
        )
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--num-shards", type=int, required=True)
        s.add_argument("--t2i-models", nargs="+", required=True)
        if name == "score":
            s.add_argument("--shard", type=int, required=True)
            s.add_argument("--cache", type=Path, required=True)
    s = sub.add_parser("attach")
    s.add_argument("--quality", type=Path, required=True)
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--t2i-models", nargs="+", required=True)
    a = p.parse_args()
    if a.command == "score":
        a.out.mkdir(parents=True, exist_ok=True)
        score(a.index, a.out, a.shard, a.num_shards, set(a.t2i_models), a.cache)
    elif a.command == "merge":
        print(json.dumps(merge(a.index, a.out, a.num_shards, set(a.t2i_models)), indent=2))
    else:
        attach(a.quality, a.dataset, a.out, set(a.t2i_models))


if __name__ == "__main__":
    main()
