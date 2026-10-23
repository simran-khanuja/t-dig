"""Build the training dataset of the image-editing policy from a published generation dataset.

The *editing policy* sees a base prompt, a variation axis and the `k = 4` i.i.d. renders of the
prompt, and writes `k` edit instructions. Instruction `i` is applied to render `i` by the reward
T2I model's image editor, and the reward scores the edited set: diversity along the axis, a
fidelity hinge against the base prompt, and a drift hinge that keeps each edit close to its source.

Rows, their order, axes and fidelity thresholds come unchanged from one text variant of a
`-customtau` tdig dataset. Each row gains:

- `prompt`: `tdig.instruction.edit_instruction`, with `k` `<image>` placeholders.
- `images`: 512 px copies of the source renders, the policy's input.
- `task: "edit"` and `source_images` in the JSON of `reward_model.ground_truth`: the full-size
  source renders. The reward service resizes them to the render resolution of the row's split.

`--images` chooses the source renders:

- `renderer` (default): the reward T2I model's own i.i.d. renders (`--iid-index`). The row's
  `align_thresh` was measured on these images, so it is the mean VQAScore of its own sources.
- `elicitation`: the renders shown during axis elicitation (`--records`), from one of four
  T2I models per prompt.
- `assigned`: the i.i.d. samples of the T2I model the prompt was assigned at elicitation
  (`extra_info.source_renderer`, a quarter of the prompts each), read from `--iid-index`. Every
  editing run uses this form, so each editor trains on the same sources from all four T2I models.
  `align_thresh` and `quality_thresh` become the mean VQAScore and mean HPSv3 of the row's own
  sources (`--thresholds`, `--quality`).

`--editor` sets only the instruction: a command for an instruction editor, a description of the
finished image for SD3.5's image-to-image editor (`tdig.instruction.EDITOR_KIND`).
"""

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from axis_grpo.dataset import export_parquet
from axis_grpo.prepare import new_directory
from baselines.assets import file_hash
from hpdv2.storage import atomic_json, read_json

from .instruction import EDIT_VERSION, EDITOR_KIND, SDEDIT_VERSION, edit_instruction

SPLITS = ("train", "val", "probe-contrastive")


def t2i_model_sources(index, t2i_model):
    """Return `{prompt_id: [paths ordered by image index]}` for one T2I model from an i.i.d. index.

    Args:
        index: `index.jsonl` written by `tdig.iid_vqa index`.
        t2i_model: T2I model name as written in the index.
    """
    found = {}
    with open(index, encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r["renderer"] == t2i_model:
                found.setdefault(r["prompt_id"], {})[r["image_index"]] = r["path"]
    return {p: [v[i] for i in sorted(v)] for p, v in found.items()}


def elicitation_sources(records):
    """Return `{prompt_id: [paths]}` of the renders shown during axis elicitation.

    Args:
        records: `source-records.jsonl` of the elicitation.
    """
    found = {}
    with open(records, encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            found[r["row"]["id"]] = [image["path"] for image in r["images"]]
    return found


def policy_input(source, target):
    """Write a 512 px RGB PNG copy of `source` to `target` unless it exists; return `(path, sha256)`."""
    from PIL import Image

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.png")
        with Image.open(source) as image:
            image.convert("RGB").resize((512, 512), Image.LANCZOS).save(temporary)
        temporary.replace(target)
    return str(target.resolve()), file_hash(target)


def edit_row(row, sources, inputs, kind="edit", thresholds=None):
    """Return the editing-task copy of one generation row.

    Args:
        row: a row of the source dataset. Not modified.
        sources: full-size source render paths, in image order.
        inputs: `(path, sha256)` of the 512 px copies, in the same order.
        kind: `edit` or `sdedit`, the instruction the editor needs.
        thresholds: optional `dict(align_thresh=, align_thresh_images=, quality_thresh=)` measured
            on `sources`; replaces the thresholds the row carried.

    Raises `ValueError` when the number of images differs from the row's `k`.
    """
    row = copy.deepcopy(row)
    e = row["extra_info"]
    if len(sources) != e["k"] or len(inputs) != e["k"]:
        raise ValueError(f"{e['prompt_id']}: {len(sources)} source images for k={e['k']}")
    spec = json.loads(row["reward_model"]["ground_truth"])
    spec.update(task="edit", source_images=list(sources))
    if thresholds:
        spec.update(
            align_thresh=thresholds["align_thresh"], quality_thresh=thresholds["quality_thresh"]
        )
        e.update(thresholds)
    row["reward_model"]["ground_truth"] = json.dumps(spec, ensure_ascii=False)
    row["prompt"] = [
        dict(role="user", content=edit_instruction(e["base_prompt"], e["axis"], e["k"], kind))
    ]
    row["images"] = [path for path, _ in inputs]
    e.update(
        task="edit",
        policy_observation="elicitation_images",
        instruction_version=SDEDIT_VERSION if kind == "sdedit" else EDIT_VERSION,
        input_image_sha256=[digest for _, digest in inputs],
    )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--dataset", type=Path, required=True, help="one text variant of a -customtau tdig dataset"
    )
    p.add_argument(
        "--t2i-model",
        default="flux2-klein-9b",
        help="reward T2I model; its editor applies the instructions",
    )
    p.add_argument("--images", choices=["renderer", "elicitation", "assigned"], default="renderer")
    p.add_argument(
        "--editor", default=None, help="editor the instruction is written for; default --t2i-model"
    )
    p.add_argument(
        "--thresholds", type=Path, help="thresholds.jsonl of iid_vqa merge; for --images assigned"
    )
    p.add_argument(
        "--quality",
        type=Path,
        nargs="+",
        help="quality.jsonl files of iid_quality merge; for --images assigned",
    )
    p.add_argument(
        "--iid-index", type=Path, help="index.jsonl of i.i.d. renders; for --images renderer"
    )
    p.add_argument(
        "--records",
        type=Path,
        help="source-records.jsonl of the elicitation; for --images elicitation",
    )
    p.add_argument(
        "--input-cache",
        type=Path,
        required=True,
        help="directory receiving the 512 px policy inputs",
    )
    p.add_argument("--out", type=Path, required=True, help="new dataset directory")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    import pyarrow.parquet as pq

    manifest = read_json(a.dataset / "manifest.json")
    kind = EDITOR_KIND[a.editor or a.t2i_model]
    if a.images != "assigned" and (
        manifest.get("instruction_renderer") != a.t2i_model or "align_thresh" not in manifest
    ):
        raise ValueError(
            "Use the -customtau variant of the reward T2I model, so align_thresh matches the editor"
        )
    splits = {}
    for name in SPLITS:
        if file_hash(a.dataset / f"{name}.parquet") != manifest["files"][f"{name}.parquet"]:
            raise ValueError(f"Source changed since its manifest was written: {name}")
        splits[name] = pq.read_table(a.dataset / f"{name}.parquet").to_pylist()
    measured = None
    if a.images == "assigned":
        assigned = {
            r["extra_info"]["prompt_id"]: r["extra_info"]["source_renderer"]
            for rows in splits.values()
            for r in rows
        }
        by_t2i_model = {
            name: t2i_model_sources(a.iid_index, name) for name in sorted(set(assigned.values()))
        }
        sources = {pid: by_t2i_model[name].get(pid, []) for pid, name in assigned.items()}
        vqa = {
            (r["renderer"], r["prompt_id"]): r
            for r in map(json.loads, open(a.thresholds, encoding="utf-8"))
        }
        hps = {
            (r["renderer"], r["prompt_id"]): r
            for path in a.quality
            for r in map(json.loads, open(path, encoding="utf-8"))
        }
        measured = {
            pid: dict(
                align_thresh=vqa[(name, pid)]["align_thresh"],
                align_thresh_images=vqa[(name, pid)]["images"],
                quality_thresh=hps[(name, pid)]["quality_thresh"],
            )
            for pid, name in assigned.items()
        }
    else:
        sources = (
            t2i_model_sources(a.iid_index, a.t2i_model)
            if a.images == "renderer"
            else elicitation_sources(a.records)
        )
    prompts = sorted({r["extra_info"]["prompt_id"] for rows in splits.values() for r in rows})
    missing = [pid for pid in prompts if len(sources.get(pid, [])) != 4]
    report = dict(
        rows={k: len(v) for k, v in splits.items()},
        base_prompts=len(prompts),
        images=a.images,
        renderer=a.t2i_model,
        prompts_without_four_sources=len(missing),
    )
    print(json.dumps(report, indent=2), flush=True)
    if missing:
        raise ValueError(f"{len(missing)} prompts lack four source images, e.g. {missing[0]}")
    if not a.execute:
        return
    cache = a.input_cache.resolve() / a.images
    jobs = [
        (pid, i, Path(path), cache / pid / f"{i}.png")
        for pid in prompts
        for i, path in enumerate(sources[pid])
    ]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        done = list(pool.map(lambda j: policy_input(j[2], j[3]), jobs))
    inputs = {}
    for (pid, i, _, _), item in zip(jobs, done, strict=True):
        inputs.setdefault(pid, []).append(item)
    with new_directory(a.out, (a.dataset, cache)) as out:
        for name, rows in splits.items():
            data = [
                edit_row(
                    r,
                    sources[r["extra_info"]["prompt_id"]],
                    inputs[r["extra_info"]["prompt_id"]],
                    kind,
                    measured[r["extra_info"]["prompt_id"]] if measured else None,
                )
                for r in rows
            ]
            export_parquet(out / f"{name}.parquet", data)
            if pq.read_table(out / f"{name}.parquet").to_pylist() != data:
                raise RuntimeError("Parquet export/readback differs")
        (out / "instruction.txt").write_text(
            edit_instruction("{base_prompt}", "{axis}", 4, kind)
            + "\n\n"
            + "=" * 20
            + " overall rows "
            + "=" * 20
            + "\n\n"
            + edit_instruction("{base_prompt}", "overall", 4, kind)
            + "\n"
        )
        result = {k: v for k, v in manifest.items() if k != "files"}
        result.update(
            task="edit",
            policy_observation="elicitation_images",
            source_images=a.images,
            instruction_renderer=a.editor or a.t2i_model,
            editor_kind=kind,
            quality_thresh=(
                "mean HPSv3 of the row's source images"
                if measured
                else result.get("quality_thresh")
            ),
            instruction_version=SDEDIT_VERSION if kind == "sdedit" else EDIT_VERSION,
            source_dataset=str(a.dataset.resolve()),
            source_manifest_sha256=file_hash(a.dataset / "manifest.json"),
            input_resolution=512,
            input_images_per_prompt=4,
            unique_images=len(jobs),
        )
        result["files"] = {f.name: file_hash(f) for f in sorted(out.iterdir()) if f.is_file()}
        atomic_json(out / "manifest.json", result)


if __name__ == "__main__":
    main()
