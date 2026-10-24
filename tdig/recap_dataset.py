"""Publish a copy of dataset variants whose instruction states one short word cap.

The published datasets cap a prompt at 60 words for SD3.5 and 150 for FLUX.2 [klein] and
Qwen-Image, the budgets of the text encoders. This module rewrites the `prompt` column of every
row with `instruction(..., word_cap=<cap>)` (`tdig-v4-2`) and leaves every other column, the row
order, `align_thresh` and `quality_thresh` unchanged. Train with the matching reward cap:
`prepare_training_matrix.py --length-cap <cap>`.

With `--relative-budget B` the cap of a row is the base prompt's word count plus `B`, stated in
its instruction and stored as `length_cap` in its ground truth, where the reward reads it in place
of the configured cap. This budgets the words the policy adds, whatever the base prompt's length.

A cap below the length of a base prompt cannot be met while keeping its wording. With
`--word-cap`, the manifest records how many base prompts exceed the cap and how many leave fewer
than ten words for the change.
"""

import argparse
import copy
from pathlib import Path

from tdig.instruction import VERSION_SHORT, instruction


def recap(row, t2i_model, cap, relative=None):
    """Return a copy of `row` with the short-cap instruction; other fields are untouched.

    With `relative`, the cap is the base prompt's word count plus `relative`, also written to the
    row's ground truth and `extra_info` as `length_cap`.
    """
    import json

    row = copy.deepcopy(row)
    e = row["extra_info"]
    if relative is not None:
        cap = len(e["base_prompt"].split()) + relative
        spec = json.loads(row["reward_model"]["ground_truth"])
        spec["length_cap"] = cap
        row["reward_model"]["ground_truth"] = json.dumps(spec, ensure_ascii=False)
        e["length_cap"] = cap
    row["prompt"] = [
        dict(
            role="user",
            content=instruction(
                e["base_prompt"],
                e["axis"],
                e["k"],
                t2i_model,
                images=bool(row.get("images")),
                word_cap=cap,
            ),
        )
    ]
    e["instruction_version"] = VERSION_SHORT
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--dataset", type=Path, required=True, help="directory with one subdirectory per variant"
    )
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--word-cap", type=int, default=None, help="one cap for every row")
    p.add_argument(
        "--relative-budget", type=int, default=None, help="cap = words of the base prompt + this"
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if (a.word_cap is None) == (a.relative_budget is None):
        raise ValueError("Give exactly one of --word-cap and --relative-budget")
    import pyarrow.parquet as pq
    from axis_grpo.dataset import export_parquet
    from axis_grpo.prepare import new_directory
    from baselines.assets import file_hash
    from hpdv2.storage import atomic_json, read_json

    with new_directory(a.out, (a.dataset,)) as target:
        for name in a.variants:
            variant = a.dataset / name
            manifest = read_json(variant / "manifest.json")
            t2i_model = manifest["instruction_renderer"]
            (target / name).mkdir()
            example = None
            for source in sorted(variant.glob("*.parquet")):
                if file_hash(source) != manifest["files"][source.name]:
                    raise ValueError(f"Source changed since its manifest was written: {source}")
                rows = [
                    recap(r, t2i_model, a.word_cap, a.relative_budget)
                    for r in pq.read_table(source).to_pylist()
                ]
                export_parquet(target / name / source.name, rows)
                if pq.read_table(target / name / source.name).to_pylist() != rows:
                    raise RuntimeError("Parquet export/readback differs")
                if source.name == "train.parquet":
                    example = rows[0]["prompt"][0]["content"]
                    lengths = list(
                        {
                            r["extra_info"]["base_prompt"]: len(
                                r["extra_info"]["base_prompt"].split()
                            )
                            for r in rows
                        }.values()
                    )
                    if a.word_cap is not None:
                        manifest.update(
                            base_prompts_over_cap=sum(n > a.word_cap for n in lengths),
                            base_prompts_within_ten_words_of_cap=sum(
                                n > a.word_cap - 10 for n in lengths
                            ),
                        )
            (target / name / "instruction.txt").write_text(example)
            manifest.update(
                instruction_version=VERSION_SHORT,
                word_cap=a.word_cap,
                relative_budget=a.relative_budget,
                source_dataset=str(variant.resolve()),
            )
            manifest["files"] = {
                f.name: file_hash(f) for f in sorted((target / name).iterdir()) if f.is_file()
            }
            atomic_json(target / name / "manifest.json", manifest)


if __name__ == "__main__":
    main()
