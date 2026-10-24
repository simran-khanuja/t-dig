"""Build the evaluation inputs of the interactive test set for one T2I model.

The test set (https://huggingface.co/datasets/neulab/t-dig-test) holds, per T2I model,
`data/<T2I model>/test.jsonl`: one item per prompt a participant typed, with the first variation
request (`axis`) and its mode (`mode`, `generate` or `edit`). The paper's automatic test-set
metrics use the `generate` items with their first axis, verbatim.

    python -m tdig.test_set --test <t-dig-test>/data/flux2-klein-9b/test.jsonl \
        --val $T_DIG_DATA/tdig/<dataset>-customtau/flux2-klein-9b/val.parquet \
        --t2i-model flux2-klein-9b --out $T_DIG_DATA/tdig/testset-flux2-klein-9b

Writes to `--out`:

- `testset.parquet`: one row per (prompt, axis) in the format of the held-out `val.parquet`,
  with the instruction of `recap_dataset.recap` at a word cap of the base prompt's words plus
  `--relative-budget` (default 20), and k = 4. Score checkpoints on it with
  `tdig/validate_checkpoints.sh`, `SETS=testset=<out>/testset.parquet`.
- `bundle/`: a `heldout-axes` bundle (`benchmarks/heldout.py`), one row per base prompt with its
  axes, for the baselines, which render from the prompt alone.

Axes are the participants' text, verbatim; TPIPS is conditioned on it for policy and baselines
alike. `align_thresh` and `quality_thresh` enter only the validation reward, not the reported
metrics (VQAScore, D_a, HPSv3 and ImageReward from `tdig.paper_metrics`). Both are set to their
medians over the rows of `--val`, a `-customtau` dataset's `val.parquet` (every row carries
`align_thresh`), whose first row is also the row template.
"""

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path

from benchmarks.heldout import NAME
from .recap_dataset import recap


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--test", type=Path, required=True, help="data/<T2I model>/test.jsonl of the test set"
    )
    p.add_argument(
        "--val",
        type=Path,
        required=True,
        help="val.parquet of a -customtau dataset for this T2I model",
    )
    p.add_argument(
        "--t2i-model",
        required=True,
        choices=["qwen-image-2512", "sd35-large", "sd35-medium", "flux2-klein-9b"],
    )
    p.add_argument(
        "--relative-budget", type=int, default=20, help="word cap = base prompt words + this"
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import pandas as pd

    pairs = {}  # (prompt, axis) -> test-set item id, in test-set order
    for r in (json.loads(line) for line in a.test.open() if line.strip()):
        if r["mode"] == "generate":
            pairs.setdefault((r["prompt"].strip(), r["axis"].strip()), r["item_id"])

    val = pd.read_parquet(a.val)
    template = val.iloc[0].to_dict()
    gts = [json.loads(g["ground_truth"]) for g in val["reward_model"]]
    align = statistics.median(g["align_thresh"] for g in gts)
    quality = (
        statistics.median(g["quality_thresh"] for g in gts)
        if all("quality_thresh" in g for g in gts)
        else None
    )

    prompts, rows = {}, []
    for (bp, axis), item in pairs.items():
        pid = hashlib.sha256(bp.encode()).hexdigest()
        slot = len(prompts.get(pid, []))
        prompts.setdefault(pid, []).append((bp, axis))
        kind = "overall" if axis.lower() == "overall" else "axis"
        r = copy.deepcopy(template)
        e = dict(r["extra_info"])
        # split must be "train" or "val" (axis_grpo.training.protocol); the set is named by the
        # validation SETS entry, and item_id keeps the link to the released item.
        e.update(
            index=len(rows),
            row_id=item,
            split="val",
            prompt_id=pid,
            base_prompt=bp,
            axis=axis,
            axis_kind=kind,
            axis_slot=float(slot),
            tpips_factor=axis,
            concise_factor=axis,
            source_renderer=a.t2i_model,
            cluster_id=pid,
            item_id=item,
        )
        for k in ("planned_index", "source_record"):
            e.pop(k, None)
        r["extra_info"] = e
        spec = dict(
            base_prompt=bp, axis=axis, tpips_factor=axis, k=4, align_thresh=align
        )
        if quality is not None:
            spec["quality_thresh"] = quality
        r["reward_model"] = dict(style="model", ground_truth=json.dumps(spec, ensure_ascii=False))
        rows.append(recap(r, a.t2i_model, None, relative=a.relative_budget))

    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(a.out / "testset.parquet")
    b = a.out / "bundle"
    b.mkdir(exist_ok=True)
    lines = []
    for i, (pid, items) in enumerate(prompts.items()):
        axes = [
            dict(
                axis=ax,
                axis_kind="overall" if ax.lower() == "overall" else "axis",
                axis_slot=j,
                contrastive=False,
                row_id=pairs[(bp, ax)],
            )
            for j, (bp, ax) in enumerate(items)
        ]
        lines.append(
            json.dumps(
                dict(
                    id=f"{NAME}:{pid}",
                    index=i,
                    metadata=dict(
                        align_thresh={a.t2i_model: align},
                        axes=axes,
                        prompt_id=pid,
                        source_renderer=a.t2i_model,
                    ),
                    prompt=items[0][0],
                )
            )
            + "\n"
        )
    raw = "".join(lines).encode()
    (b / "prompts.jsonl").write_bytes(raw)
    (b / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                benchmark=NAME,
                count=len(lines),
                rows_sha256=hashlib.sha256(raw).hexdigest(),
                axes=len(rows),
                contrastive_axes=0,
                overall_axes=sum(1 for r in rows if r["extra_info"]["axis_kind"] == "overall"),
                source=a.test.name,
                selection="test-set items whose first request was generate-mode, first axis",
            ),
            indent=2,
        )
    )
    print(f"{len(rows)} (prompt, axis) rows, {len(lines)} prompts")


if __name__ == "__main__":
    main()
