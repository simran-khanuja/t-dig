"""Build benchmark prompt sets in the input format of the generation policy.

A policy row has a `prompt` (the instruction of `tdig.instruction`), a
`reward_model.ground_truth` and an `extra_info`. The COCO subset is a list of base prompts
without variation requests; this module writes it as policy rows whose axis is `overall`.

One directory is written per policy variant, because the instruction names the variant's reward
T2I model. An image variant also needs four i.i.d. renders of each base prompt as policy input:
`--iid-run` names the baseline runs to take them from, resized to 512 px.
"""

import argparse
import json
import os
from pathlib import Path

from axis_grpo.dataset import export_parquet
from baselines.assets import file_hash
from hpdv2.storage import atomic_json

from .instruction import VERSION, instruction

K = 4
VARIANTS = {
    "sd35-medium": ("sd35-medium", False),
    "sd35-large": ("sd35-large", False),
    "flux2-klein-9b": ("flux2-klein-9b", False),
    "qwen-image-2512": ("qwen-image-2512", False),
    "flux2-klein-9b-images": ("flux2-klein-9b", True),
}


def benchmark_rows(name, root):
    """Return `[(row_id, base_prompt, metadata)]` for `coco250`.

    Args:
        name: `coco250` (the rows of `$T_DIG_DATA/tdig/coco250-v1/subset.json`).
        root: repository root; `<root>/data` stands in for an unset `T_DIG_DATA`.

    Raises `ValueError` for any other name.
    """
    if name == "coco250":
        subset = json.loads(
            (
                Path(os.environ.get("T_DIG_DATA", root / "data")) / "tdig/coco250-v1/subset.json"
            ).read_text()
        )
        return [(r["id"], r["prompt"], dict(benchmark_index=r["index"])) for r in subset["rows"]]
    raise ValueError(f"Unknown benchmark {name!r}")


def iid_images(runs, cache, wanted):
    """Return `{benchmark_index: [four 512 px PNG paths]}` from i.i.d. baseline runs.

    Args:
        runs: baseline run directories whose method is `iid`.
        cache: directory receiving the resized copies, `<cache>/<run name>/<index>-<slot>.png`.
        wanted: set of benchmark indices to prepare; other rows of the runs are ignored.
    """
    from PIL import Image

    found = {}
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["config"]["method"] != "iid":
            raise ValueError(f"Not an i.i.d. run: {run}")
        for row in json.loads((run / "input.json").read_text()):
            if row["index"] not in wanted:
                continue
            sources = [
                run / "work" / f'{row["index"]:05d}' / "slots" / f"{s:04d}" / "image.png"
                for s in range(K)
            ]
            if not all(p.is_file() for p in sources):
                continue
            targets = []
            for slot, source in enumerate(sources):
                target = Path(cache) / run.name / f'{row["index"]:05d}-{slot}.png'
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(source) as image:
                        image.convert("RGB").resize((512, 512), Image.LANCZOS).save(target)
                targets.append(str(target.resolve()))
            found[row["index"]] = targets
    return found


def build(name, variant, root, out, images=None):
    """Write `<out>/<variant>/<name>.parquet` and return the number of rows.

    Args:
        name: benchmark name accepted by `benchmark_rows`.
        variant: key of `VARIANTS`.
        root: repository root.
        out: output directory.
        images: `{benchmark_index: [paths]}` from `iid_images`; required for an image variant.

    Each row has `data_source` `axis_diversity`, the `overall` instruction, a ground truth of
    `base_prompt`, `axis`, `tpips_factor` (both `overall`) and `k`, and `extra_info` with `index`,
    `row_id` (the benchmark row id), `split` `val`, `benchmark` and `benchmark_index`.
    Raises `ValueError` when an image variant lacks the renders of a row.
    """
    t2i_model, visual = VARIANTS[variant]
    rows = []
    for index, (row_id, prompt, meta) in enumerate(benchmark_rows(name, root)):
        spec = dict(base_prompt=prompt, axis="overall", tpips_factor="overall", k=K)
        row = dict(
            data_source="axis_diversity",
            ability="axis_diversity",
            prompt=[
                dict(
                    role="user", content=instruction(prompt, "overall", K, t2i_model, images=visual)
                )
            ],
            reward_model=dict(style="model", ground_truth=json.dumps(spec, ensure_ascii=False)),
            extra_info=dict(
                index=index,
                row_id=row_id,
                split="val",
                base_prompt=prompt,
                axis="overall",
                axis_kind="overall",
                k=K,
                benchmark=name,
                constraints=[],
                benchmark_index=meta["benchmark_index"],
                instruction_version=VERSION,
                instruction_renderer=t2i_model,
            ),
        )
        if visual:
            if not images or meta["benchmark_index"] not in images:
                raise ValueError(f"No i.i.d. renders for {row_id}")
            row["images"] = images[meta["benchmark_index"]]
        rows.append(row)
    target = Path(out) / variant
    target.mkdir(parents=True, exist_ok=True)
    export_parquet(target / f"{name}.parquet", rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--benchmark", choices=["coco250"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--iid-run",
        type=Path,
        action="append",
        default=[],
        help="iid baseline runs supplying policy input images for the image variant",
    )
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    wanted = {meta["benchmark_index"] for _, _, meta in benchmark_rows(a.benchmark, root)}
    images = iid_images(a.iid_run, a.out / "input-images", wanted) if a.iid_run else None
    report = {}
    for variant, (_, visual) in VARIANTS.items():
        if visual and images is None:
            continue
        report[variant] = build(a.benchmark, variant, root, a.out, images)
    files = {
        str(f.relative_to(a.out)): file_hash(f)
        for f in sorted(a.out.rglob(f"{a.benchmark}.parquet"))
    }
    atomic_json(
        a.out / f"{a.benchmark}-manifest.json",
        dict(
            benchmark=a.benchmark,
            rows=report,
            files=files,
            instruction_version=VERSION,
            axis="overall",
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
