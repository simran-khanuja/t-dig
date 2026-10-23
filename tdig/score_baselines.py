"""Score baseline image sets with the two measures the generation policy is trained on.

A *baseline run* is a directory written by `baselines.runner`: `input.json` lists the benchmark
rows, `manifest.json` records the T2I model and method, and image `s` of row `i` is
`work/<i>/slots/<s>/image.png`. A benchmark is split over several such directories, one per GPU.
This module reads them and never writes inside them.

Two measures, both against the unchanged base prompt of the row:

- VQAScore: probability in [0, 1] that CLIP-FlanT5-XXL answers "yes" to whether the image shows
  the prompt. Scored by `tdig.iid_vqa score`, which reads the index this module writes.
- TPIPS diversity of the `k` images of one row: `normalized_vendi` in [0, 1] and `pairwise` in
  [0, 2] from `axis_grpo.training.metrics.diversity`, the reward's diversity term. The embedding
  is conditioned on a *factor*, a text naming what to compare:
  - COCO rows have no variation axis, so the factor is `overall`.
  - A held-out axis row lists the axes requested for its base prompt. The baseline renders one
    set from the base prompt alone, scored once per axis with the axis text as the factor: the
    variation a method delivers along an axis it was not told.

Subcommands, in order: `index` (CPU), `tpips` (one shard per GPU, T2I model environment),
`merge` (CPU, after `tdig.iid_vqa score` and `tpips` have finished).
"""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .iid_vqa import _atomic_text, _read_jsonl


def build_index(runs, out):
    """Write the image index and the set index of every complete row of the given runs.

    Args:
        runs: baseline run directories.
        out: output directory.

    Writes `out/index.jsonl`, one line per image, in the format `tdig.iid_vqa score`
    reads: `renderer` holds `<T2I model>|<method>|<benchmark>`, `prompt_id` the benchmark row
    id, plus `image_index`, `path`, `base_prompt`. Writes `out/sets.jsonl`, one line per row,
    with `group`, `prompt_id`, `base_prompt`, `paths`,
    `align_thresh` (the prompt's fidelity threshold for the group's T2I model, held-out axis
    rows only) and `factors`. `factors` lists what to score the set along: one entry per axis of
    a held-out axis row, else the single entry `overall`. Each entry has `row_id`, `factor`,
    `axis_kind` (`axis` or `overall`) and `contrastive`. A row with a missing image is skipped
    and counted in `out/index.json`.

    Raises `ValueError` when two runs supply the same row of one group.
    """
    images, sets, summary, seen = [], [], defaultdict(lambda: dict(rows=0, complete=0)), set()
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text())
        config = manifest["config"]
        group = f'{config["renderer"]}|{config["method"]}|{manifest["bundle"]["benchmark"]}'
        for row in json.loads((run / "input.json").read_text()):
            summary[group]["rows"] += 1
            paths = [
                run / "work" / f'{row["index"]:05d}' / "slots" / f"{s:04d}" / "image.png"
                for s in range(config["k"])
            ]
            if not all(p.is_file() for p in paths):
                continue
            if (group, row["id"]) in seen:
                raise ValueError(f"Row {row['id']} of {group} appears in two runs")
            seen.add((group, row["id"]))
            summary[group]["complete"] += 1
            meta = row.get("metadata", {})
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
                    prompt_id=row["id"],
                    base_prompt=row["prompt"],
                    paths=[str(p.resolve()) for p in paths],
                    factors=factors,
                    align_thresh=meta.get("align_thresh", {}).get(config["renderer"]),
                )
            )
            images += [
                dict(
                    renderer=group,
                    prompt_id=row["id"],
                    image_index=s,
                    path=str(p.resolve()),
                    base_prompt=row["prompt"],
                )
                for s, p in enumerate(paths)
            ]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        out / "index.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in images)
    )
    _atomic_text(
        out / "sets.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sets)
    )
    _atomic_text(
        out / "index.json",
        json.dumps(dict(images=len(images), sets=len(sets), groups=summary), indent=2) + "\n",
    )
    return dict(images=len(images), sets=len(sets), groups=dict(summary))


def _factors(item):
    """Return the factor entries of one set; a `sets.jsonl` line without `factors` means `overall`."""
    return item.get("factors") or [
        dict(row_id=item["prompt_id"], factor="overall", axis_kind="overall", contrastive=False)
    ]


def score_tpips(out, shard, num_shards, checkpoint, cache, device):
    """Compute TPIPS diversity for every `num_shards`-th set, starting at `shard`, once per factor.

    Args:
        out: directory holding `sets.jsonl`.
        shard, num_shards: this worker scores sets `shard, shard + num_shards, ...`.
        checkpoint: local snapshot of `sywang/TPIPS-Embed-Qwen3VL-8B`.
        cache: Hugging Face hub cache.
        device: torch device string.

    Appends one line per set and factor to `out/tpips-<shard>.jsonl` with `group`, `prompt_id`,
    `row_id`, `factor`, `normalized_vendi`, `vendi`, `pairwise`. Pairs already in that file are
    skipped. Writes `out/tpips-done-<shard>.json` on completion.
    """
    import torch
    import tpips
    from PIL import Image
    from axis_grpo.training.metrics import diversity
    from elicitation.tpips_compat import qwen_tokenizer_compat, verify_tokenizer

    out = Path(out)
    todo = _read_jsonl(out / "sets.jsonl")[shard::num_shards]
    target = out / f"tpips-{shard:02d}.jsonl"
    done = (
        {(r["group"], r["prompt_id"], r.get("row_id", r["prompt_id"])) for r in _read_jsonl(target)}
        if target.exists()
        else set()
    )
    pending = lambda item: [
        f for f in _factors(item) if (item["group"], item["prompt_id"], f["row_id"]) not in done
    ]
    todo = [r for r in todo if pending(r)]
    print(f"tpips shard {shard}/{num_shards}: {len(todo)} sets to score", flush=True)
    if todo:
        from baselines.assets import local_checkpoint

        checkpoint = local_checkpoint(checkpoint, cache)
        with qwen_tokenizer_compat():
            model = tpips.load_model("embedding", checkpoint, device=device, cache_dir=cache)
        verify_tokenizer(model, checkpoint, [model.create_prompt("overall")])
        start = time.monotonic()
        with open(target, "a", encoding="utf-8") as stream:
            for i, item in enumerate(todo, 1):
                images = [Image.open(p).convert("RGB") for p in item["paths"]]
                for entry in pending(item):
                    with torch.inference_mode():
                        vectors = (
                            model.embed(images, factor=entry["factor"], normalized=True)
                            .float()
                            .cpu()
                            .numpy()
                        )
                    stream.write(
                        json.dumps(
                            dict(
                                group=item["group"],
                                prompt_id=item["prompt_id"],
                                row_id=entry["row_id"],
                                factor=entry["factor"],
                                **diversity(vectors),
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if i % 50 == 0 or i == len(todo):
                    stream.flush()
                    rate = i / (time.monotonic() - start)
                    print(
                        f"  {i}/{len(todo)}  {rate:.2f} sets/s  eta {(len(todo) - i) / rate / 60:.0f} min",
                        flush=True,
                    )
    _atomic_text(out / f"tpips-done-{shard:02d}.json", json.dumps(dict(sets=len(todo))) + "\n")


def merge(out, num_shards):
    """Join VQAScore and TPIPS diversity per set and summarise per group.

    Args:
        out: directory holding `sets.jsonl`, `thresholds.jsonl` (written by
            `tdig.iid_vqa merge`) and `tpips-*.jsonl`.
        num_shards: number of TPIPS shards.

    Writes `out/per-set.jsonl`, one line per set and factor with `group`, `prompt_id`, `row_id`,
    `factor`, `axis_kind`, `contrastive`, `vqascore_mean`, `vqascore_min`,
    `normalized_vendi`, `pairwise`, and, when the set has an `align_thresh`, `fidelity_penalty`
    (`mean(max(0, align_thresh - a_i))`) and `reward` (`normalized_vendi - fidelity_penalty`, the
    training reward with Vendi, fidelity weight 1 and no length term). The VQAScore of a set is
    repeated on each of its factor lines. Writes `out/summary.json`: per group, the number of
    lines and the mean of each quantity, overall and, for held-out axis rows,
    per kind: `axis` (named, not contrastive), `overall` and `contrastive`. Returns the summary.
    Raises `ValueError` when a set lacks either score.
    """
    out = Path(out)
    vqa = {(r["renderer"], r["prompt_id"]): r for r in _read_jsonl(out / "thresholds.jsonl")}
    tp = {}
    for shard in range(num_shards):
        if not (out / f"tpips-done-{shard:02d}.json").exists():
            raise ValueError(f"TPIPS shard {shard} is unfinished")
        tp.update(
            {
                (r["group"], r["prompt_id"], r.get("row_id", r["prompt_id"])): r
                for r in _read_jsonl(out / f"tpips-{shard:02d}.jsonl")
            }
        )
    rows = []
    for item in _read_jsonl(out / "sets.jsonl"):
        key = (item["group"], item["prompt_id"])
        if key not in vqa:
            raise ValueError(f"Set without VQAScore: {key}")
        for entry in _factors(item):
            if key + (entry["row_id"],) not in tp:
                raise ValueError(f"Set without TPIPS diversity: {key + (entry['row_id'],)}")
            d = tp[key + (entry["row_id"],)]
            line = dict(
                group=item["group"],
                prompt_id=item["prompt_id"],
                row_id=entry["row_id"],
                factor=entry["factor"],
                axis_kind=entry["axis_kind"],
                contrastive=entry["contrastive"],
                vqascore_mean=vqa[key]["align_thresh"],
                vqascore_min=vqa[key]["score_min"],
                normalized_vendi=d["normalized_vendi"],
                pairwise=d["pairwise"],
            )
            if item.get("align_thresh") is not None:
                line["fidelity_penalty"] = statistics.mean(
                    max(0.0, item["align_thresh"] - a) for a in vqa[key]["scores"]
                )
                line["reward"] = line["normalized_vendi"] - line["fidelity_penalty"]
            rows.append(line)
    names = [
        "vqascore_mean",
        "vqascore_min",
        "normalized_vendi",
        "pairwise",
        "fidelity_penalty",
        "reward",
    ]
    mean = lambda items: dict(
        lines=len(items),
        **{
            n: statistics.mean(r[n] for r in items if n in r)
            for n in names
            if any(n in r for r in items)
        },
    )
    kind = lambda r: "contrastive" if r["contrastive"] else r["axis_kind"]
    summary = {}
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group]
        summary[group] = dict(
            **mean(members),
            per_kind=(
                {
                    k: mean([r for r in members if kind(r) == k])
                    for k in sorted({kind(r) for r in members})
                }
                if any("reward" in r for r in members)
                else {}
            ),
        )
    _atomic_text(out / "per-set.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _atomic_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("index")
    s.add_argument("--run", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("tpips")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--num-shards", type=int, required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--device", default="cuda:0")
    s = sub.add_parser("merge")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--num-shards", type=int, required=True)
    a = p.parse_args()
    if a.command == "index":
        print(json.dumps(build_index(a.run, a.out), indent=2))
    elif a.command == "tpips":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE=a.cache)
        score_tpips(a.out, a.shard, a.num_shards, a.checkpoint, a.cache, a.device)
    else:
        print(json.dumps(merge(a.out, a.num_shards), indent=2))


if __name__ == "__main__":
    main()
