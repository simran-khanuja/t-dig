"""Build the tdig training datasets from the shared train/validation split.

Reads the text split (`train.parquet`, `val.parquet`) and its image-bearing twin, and publishes
one directory per policy variant under `--out`. Variants differ only in the instruction text of
the `prompt` column; rows, order and reward targets are identical.

Every variant:

- Drops rows whose axis is a *hard-contrastive request*, one that names something the images
  must not show ("full-body shots rather than portraits"). The reward conditions diversity on
  the request text and scores fidelity against the base prompt only, so it ranks a set that
  includes the excluded thing above one that honours the request. `is_hard_contrastive` defines
  the test.
- Rewrites the `prompt` column with `tdig.instruction.instruction`.

Sources are read-only. `--out` must not exist; `axis_grpo.prepare.new_directory` publishes it
atomically.
"""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from axis_grpo.dataset import export_parquet
from axis_grpo.prepare import new_directory
from baselines.assets import file_hash
from hpdv2.storage import atomic_json, read_json

from .instruction import T2I_MODELS, VERSION, instruction

# "with or without X", "with and/or without X" and "with/without X" name both poles of a pair.
_PAIRED_WITHOUT = re.compile(r"\bwith(?:\s+(?:and|or|and/or)\s+|/)without\b", re.I)
_CONTRAST = re.compile(r"\b(instead of|rather than|without)\b", re.I)
_LEADING_REMOVE = re.compile(r"^\s*remove\b", re.I)

# (directory name, reward T2I model, policy receives images)
VARIANTS = [(r, r, False) for r in T2I_MODELS] + [("flux2-klein-9b-images", "flux2-klein-9b", True)]


def is_hard_contrastive(axis):
    """Return whether an axis names something the images must exclude.

    Args:
        axis: the variation request, verbatim.

    Returns `True` when the axis contains `instead of`, `rather than` or an unpaired
    `without`, or starts with `Remove` (case-insensitive). Returns `False` otherwise, including
    for `overall` and the paired forms `with or without`, `with and without` and
    `with/without`.
    """
    if axis == "overall":
        return False
    if _LEADING_REMOVE.search(axis):
        return True
    return bool(_CONTRAST.search(_PAIRED_WITHOUT.sub(" ", axis)))


def _reindex(rows):
    """Number rows 0..n-1 in `extra_info.index`, keeping the previous index.

    Args:
        rows: row dicts in output order. Modified in place.

    Returns the same list. `extra_info.tdig_source_index` receives the row's index in the
    source split, written once per row.
    """
    for i, row in enumerate(rows):
        e = row["extra_info"]
        e.setdefault("tdig_source_index", e["index"])
        e["index"] = i
    return rows


def rotate_slots(rows):
    """Reorder training rows so every pass mixes all axis slots and repeats no base prompt early.

    Args:
        rows: training rows in source order, where pass `p` holds axis slot `p - 1` of every
            base prompt, in one prompt order shared by all passes. Not modified.

    Returns new rows; every input row appears exactly once. In pass `j` (from 0), a prompt with
    `n` named-axis rows shows the row at position `(h + j) % n` of its slot-ordered rows, for
    `j < n`, where `h` is derived from the SHA-256 of the prompt id. Source T2I models alternate
    with period four in the prompt order, so an offset from the prompt's rank would tie the axis
    slot to the source T2I model.

    Each pass is a body followed by a tail. A prompt whose `overall` row belongs to the pass
    puts that row in the body and its named row in the tail; every other prompt puts its named
    row in the body, so a base prompt appears once per body. `extra_info.pass_number` becomes
    the new pass and `extra_info.tdig_source_pass` keeps the source pass.

    Raises `ValueError` if a base prompt has two `overall` rows or a row is lost or repeated.
    """
    named, overall, order = {}, {}, []
    for row in rows:
        e = row["extra_info"]
        if e["prompt_id"] not in named:
            named[e["prompt_id"]] = []
            order.append(e["prompt_id"])
        if e["axis_kind"] == "overall":
            if e["prompt_id"] in overall:
                raise ValueError("A base prompt has two overall rows")
            overall[e["prompt_id"]] = row
        else:
            named[e["prompt_id"]].append(row)
    result = []
    passes = max(
        [len(v) for v in named.values()]
        + [r["extra_info"]["pass_number"] for r in overall.values()]
    )
    for phase in range(passes):
        body, tail = [], []
        for prompt_id in order:
            axes = sorted(named[prompt_id], key=lambda r: r["extra_info"]["axis_slot"])
            offset = int(hashlib.sha256(prompt_id.encode()).hexdigest()[:8], 16)
            # A prompt can keep only its overall row when the filter removed every named axis.
            choice = axes[(offset + phase) % len(axes)] if phase < len(axes) else None
            extra = overall.get(prompt_id)
            if extra is not None and extra["extra_info"]["pass_number"] == phase + 1:
                body.append(extra)
                if choice is not None:
                    tail.append(choice)
            elif choice is not None:
                body.append(choice)
        for row in body + tail:
            row = copy.deepcopy(row)
            row["extra_info"]["tdig_source_pass"] = row["extra_info"]["pass_number"]
            row["extra_info"]["pass_number"] = phase + 1
            result.append(row)
    if sorted(r["extra_info"]["row_id"] for r in result) != sorted(
        r["extra_info"]["row_id"] for r in rows
    ):
        raise ValueError("Reordering lost or repeated a row")
    return result


def partition(train, val, order="source"):
    """Split source rows into the published sets.

    Args:
        train: rows of the source `train.parquet`, in curriculum order.
        val: rows of the source `val.parquet`.
        order: `source` keeps the source row order; `rotated` applies `rotate_slots` to the
            training rows after filtering.

    Returns a dict of row lists, each reindexed from 0:
        `train`: training rows that are not hard-contrastive, in the order `order` selects.
        `val`: held-out rows that are not hard-contrastive.
        `probe-contrastive`: held-out hard-contrastive rows. Never trained on; measures
            whether a trained policy honours such requests.
        `dropped-contrastive-train`: training rows removed by the filter, for later use with
            a reward that can score them.

    Raises `ValueError` if a removed row is an `overall` row or a pass number decreases.
    """
    sets = {"train": [], "val": [], "probe-contrastive": [], "dropped-contrastive-train": []}
    for source, keep, drop in (
        (train, "train", "dropped-contrastive-train"),
        (val, "val", "probe-contrastive"),
    ):
        for row in source:
            row = copy.deepcopy(row)
            e = row["extra_info"]
            hard = is_hard_contrastive(e["axis"])
            if hard and e["axis_kind"] != "axis":
                raise ValueError("Only named-axis rows can be contrastive")
            sets[drop if hard else keep].append(row)
    if order == "rotated":
        sets["train"] = rotate_slots(sets["train"])
    elif order != "source":
        raise ValueError("order must be 'source' or 'rotated'")
    for rows in sets.values():
        _reindex(rows)
    passes = [r["extra_info"]["pass_number"] for r in sets["train"]]
    if passes != sorted(passes):
        raise ValueError("Filtering broke the pass order")
    return sets


def rewrite(row, t2i_model, images):
    """Return a copy of one row with the tdig instruction in its `prompt` column.

    Args:
        row: a source row. Not modified.
        t2i_model: reward T2I model named in rule 4 of the instruction.
        images: whether the row carries four policy input images.

    Returns the new row. `extra_info.instruction_version` and
    `extra_info.instruction_renderer` record the instruction. Raises `ValueError` when
    `images` disagrees with the row's `images` column.
    """
    row = copy.deepcopy(row)
    e = row["extra_info"]
    if bool(row.get("images")) != images:
        raise ValueError("Row image column does not match the variant")
    row["prompt"] = [
        dict(
            role="user",
            content=instruction(e["base_prompt"], e["axis"], e["k"], t2i_model, images=images),
        )
    ]
    e["instruction_version"] = VERSION
    e["instruction_renderer"] = t2i_model
    if images:
        e["image_instruction_version"] = VERSION
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--text-source",
        type=Path,
        required=True,
        help="directory holding the text split and its manifest.json",
    )
    p.add_argument(
        "--image-source",
        type=Path,
        required=True,
        help="directory holding the image-bearing twin of the same split",
    )
    p.add_argument("--out", type=Path, required=True, help="new directory to publish")
    p.add_argument(
        "--order",
        choices=["source", "rotated"],
        default="source",
        help="rotated: mix all axis slots into every pass (see rotate_slots)",
    )
    p.add_argument("--execute", action="store_true", help="write files; otherwise report counts")
    a = p.parse_args()
    import pyarrow.parquet as pq

    sources = {}
    for images, directory in ((False, a.text_source), (True, a.image_source)):
        manifest = read_json(directory / "manifest.json")
        for name in ("train.parquet", "val.parquet"):
            if file_hash(directory / name) != manifest["files"][name]:
                raise ValueError(
                    f"Source changed since its manifest was written: {directory / name}"
                )
        sources[images] = partition(
            pq.read_table(directory / "train.parquet").to_pylist(),
            pq.read_table(directory / "val.parquet").to_pylist(),
            a.order,
        )
    text, image = sources[False], sources[True]
    for name in text:
        if [r["extra_info"]["row_id"] for r in text[name]] != [
            r["extra_info"]["row_id"] for r in image[name]
        ]:
            raise ValueError(f"Text and image sources disagree on {name}")

    e = lambda rows: [r["extra_info"] for r in rows]
    report = dict(
        complete=True,
        instruction_version=VERSION,
        order=a.order,
        text_source=str(a.text_source.resolve()),
        image_source=str(a.image_source.resolve()),
        text_source_manifest_sha256=file_hash(a.text_source / "manifest.json"),
        image_source_manifest_sha256=file_hash(a.image_source / "manifest.json"),
        rows=len(text["train"]),
        val_rows=len(text["val"]),
        probe_rows=len(text["probe-contrastive"]),
        dropped_train_rows=len(text["dropped-contrastive-train"]),
        base_prompts=len({x["prompt_id"] for x in e(text["train"])}),
        val_base_prompts=len({x["prompt_id"] for x in e(text["val"])}),
        overall_rows=sum(x["axis_kind"] == "overall" for x in e(text["train"])),
        rows_by_pass=dict(sorted(Counter(str(x["pass_number"]) for x in e(text["train"])).items())),
        dropped_by_pass=dict(
            sorted(
                Counter(str(x["pass_number"]) for x in e(text["dropped-contrastive-train"])).items()
            )
        ),
        filter="hard-contrastive axes removed: 'instead of', 'rather than', unpaired 'without', leading 'Remove'",
        reward_text="request",
    )
    print(json.dumps(report, indent=2))
    if not a.execute:
        return

    with new_directory(a.out, (a.text_source, a.image_source)) as out:
        for directory, t2i_model, images in VARIANTS:
            target = out / directory
            target.mkdir()
            for name, rows in sources[images].items():
                data = [rewrite(r, t2i_model, images) for r in rows]
                export_parquet(target / f"{name}.parquet", data)
                if pq.read_table(target / f"{name}.parquet").to_pylist() != data:
                    raise RuntimeError("Parquet export/readback differs")
                for before, after in zip(rows, data, strict=True):
                    if before["reward_model"] != after["reward_model"]:
                        raise RuntimeError("Reward target changed")
            sample = sources[images]["train"][0]["extra_info"]
            (target / "instruction.txt").write_text(
                instruction("{base_prompt}", "{axis}", sample["k"], t2i_model, images=images)
                + "\n\n"
                + "=" * 20
                + " overall rows "
                + "=" * 20
                + "\n\n"
                + instruction("{base_prompt}", "overall", sample["k"], t2i_model, images=images)
                + "\n"
            )
            manifest = dict(
                report,
                instruction_renderer=t2i_model,
                policy_observation="elicitation_images" if images else "text",
            )
            manifest["files"] = {
                f.name: file_hash(f) for f in sorted(target.iterdir()) if f.is_file()
            }
            atomic_json(target / "manifest.json", manifest)
        atomic_json(out / "report.json", report)


if __name__ == "__main__":
    main()
