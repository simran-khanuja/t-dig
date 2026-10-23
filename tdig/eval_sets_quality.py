"""Add `quality_thresh` to the COCO-250 policy rows of one T2I model.

A reward with a quality term (`quality_weight`) needs `quality_thresh` on every row it scores.
The threshold of a benchmark prompt is the mean HPSv3 of the T2I model's i.i.d. samples of it,
read from the baseline `rows.jsonl` of `paper_metrics.py` (method `iid`). Rows are otherwise
unchanged. Writes `<out>/<variant>/coco250.parquet` and `<out>/manifest.json`.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--eval-sets", type=Path, required=True)
    p.add_argument(
        "--baseline-rows",
        type=Path,
        required=True,
        help="rows.jsonl of paper_metrics for the baselines",
    )
    p.add_argument("--t2i-model", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import pyarrow.parquet as pq
    from axis_grpo.dataset import export_parquet
    from axis_grpo.prepare import new_directory
    from baselines.assets import file_hash
    from hpdv2.storage import atomic_json

    groups = {f"{a.t2i_model}|iid|coco2017-val-5000": "coco250"}
    quality = {}
    with a.baseline_rows.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["group"] in groups and r.get("hpsv3") is not None:
                quality[(groups[r["group"]], r["row_id"])] = r["hpsv3"]
    with new_directory(a.out, (a.eval_sets,)) as target:
        files = {}
        for variant in a.variants:
            (target / variant).mkdir()
            for name in ("coco250",):
                source = a.eval_sets / variant / f"{name}.parquet"
                rows = pq.read_table(source).to_pylist()
                for row in rows:
                    value = quality.get((name, row["extra_info"]["row_id"]))
                    if value is None:
                        raise ValueError(
                            f"No i.i.d. HPSv3 for {name} / {row['extra_info']['row_id']}"
                        )
                    spec = json.loads(row["reward_model"]["ground_truth"])
                    spec["quality_thresh"] = value
                    row["reward_model"]["ground_truth"] = json.dumps(spec, ensure_ascii=False)
                    row["extra_info"]["quality_thresh"] = value
                export_parquet(target / variant / source.name, rows)
                files[f"{variant}/{source.name}"] = file_hash(target / variant / source.name)
        atomic_json(
            target / "manifest.json",
            dict(
                source=str(a.eval_sets.resolve()),
                renderer=a.t2i_model,
                quality_thresh="mean HPSv3 of the T2I model's i.i.d. samples of the benchmark prompt",
                baseline_rows_sha256=file_hash(a.baseline_rows),
                files=files,
            ),
        )


if __name__ == "__main__":
    main()
