"""Build the held-out axis bundle from a published tdig dataset.

Reads `val.parquet` and `probe-contrastive.parquet` of the four text variants of a `-customtau`
dataset and writes one row per held-out base prompt, in the format
`benchmarks.heldout.load_heldout` reads. The variants hold the same rows; each contributes its
T2I model's threshold.

`metadata.axes` lists, per variation request of the prompt: `row_id`, `axis` (verbatim, also the
TPIPS factor), `axis_kind` (`axis` or `overall`), `axis_slot` and `contrastive` (true for rows of
`probe-contrastive.parquet`). `metadata.align_thresh` maps each T2I model to the prompt's fidelity
threshold.
"""

import argparse
import json
from pathlib import Path

from benchmarks.cli import digest
from benchmarks.heldout import NAME
from hpdv2.storage import atomic_json

T2I_MODELS = ("sd35-medium", "sd35-large", "flux2-klein-9b", "qwen-image-2512")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--dataset", type=Path, required=True, help="a -customtau tdig dataset directory"
    )
    p.add_argument("--out", type=Path, required=True, help="new bundle directory")
    a = p.parse_args()
    import pyarrow.parquet as pq

    prompts, order = {}, []
    for t2i_model in T2I_MODELS:
        for name, contrastive in (("val", False), ("probe-contrastive", True)):
            for row in pq.read_table(a.dataset / t2i_model / f"{name}.parquet").to_pylist():
                e = row["extra_info"]
                if e["prompt_id"] not in prompts:
                    prompts[e["prompt_id"]] = dict(
                        prompt=e["base_prompt"],
                        axes={},
                        align_thresh={},
                        source_renderer=e["source_renderer"],
                        cluster_id=e["cluster_id"],
                    )
                    order.append(e["prompt_id"])
                item = prompts[e["prompt_id"]]
                if item["prompt"] != e["base_prompt"]:
                    raise ValueError("One prompt id with two base prompts")
                item["axes"][e["row_id"]] = dict(
                    row_id=e["row_id"],
                    axis=e["axis"],
                    axis_kind=e["axis_kind"],
                    axis_slot=e["axis_slot"],
                    contrastive=contrastive,
                )
                item["align_thresh"][t2i_model] = e["align_thresh"]
    rows = []
    for index, prompt_id in enumerate(sorted(order)):
        item = prompts[prompt_id]
        axes = sorted(
            item["axes"].values(),
            key=lambda x: (x["axis_kind"] == "overall", x["axis_slot"], x["row_id"]),
        )
        rows.append(
            dict(
                index=index,
                id=f"{NAME}:{prompt_id}",
                prompt=item["prompt"],
                metadata=dict(
                    prompt_id=prompt_id,
                    axes=axes,
                    align_thresh=item["align_thresh"],
                    source_renderer=item["source_renderer"],
                    cluster_id=item["cluster_id"],
                ),
            )
        )
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "prompts.jsonl").write_bytes(raw)
    atomic_json(
        a.out / "manifest.json",
        dict(
            schema_version=1,
            benchmark=NAME,
            count=len(rows),
            rows_sha256=digest(raw),
            axes=sum(len(r["metadata"]["axes"]) for r in rows),
            contrastive_axes=sum(x["contrastive"] for r in rows for x in r["metadata"]["axes"]),
            overall_axes=sum(
                x["axis_kind"] == "overall" for r in rows for x in r["metadata"]["axes"]
            ),
            source=str(a.dataset.resolve()),
            selection="every held-out base prompt, sorted by prompt id",
        ),
    )
    print(json.dumps(json.loads((a.out / "manifest.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
