"""Compute FID and KID of generated image sets against COCO val2017 on a fixed 250-caption subset.

The subset follows Scaling Group Inference (ICLR 2026), which computes FID on 250 COCO
validation prompts. Five choices fix the protocol; each changes the result:

- The 250 captions are the rows of the `coco2017-val-5000` bundle with the smallest
  SHA-256 of `coco-fid-v1:<seed>:<image_id>`. The choice depends on image ids and the seed only,
  never on which images a method has rendered.
- Generated images are pooled over the `k` images of a caption: 250 captions give 1,000 images
  for `k = 4`.
- The reference set is all 5,000 val2017 photographs, not the 250 paired ones. A covariance of
  2,048 Inception features estimated from 250 images is rank-deficient.
- Each photograph is centre-cropped to a square before Clean-FID's bicubic resize to 299 px.
  Generated images are square; without the crop the photographs alone are distorted.
- KID is reported beside FID. FID is biased upward at 1,000 samples; KID is unbiased.

Subcommands: `select` (CPU), `index` (CPU; baseline images of the subset and what is missing),
`index-validation` (CPU; images of `validate_checkpoints.py`), `score` (GPU; needs
`clean-fid==0.1.35` and its SciPy pin from `.cache/metric-python` on `PYTHONPATH`).
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .iid_vqa import _atomic_text, _read_jsonl

BENCHMARK = "coco2017-val-5000"


def select_prompts(bundle, count=250, seed=42):
    """Return `count` rows of the COCO bundle chosen by a seeded hash of the image id.

    Args:
        bundle: directory holding the bundle's `prompts.jsonl`.
        count: number of captions.
        seed: integer mixed into the hash.

    Returns a list of dicts with `index` (row index in the bundle), `id`, `image_id`,
    `file_name` and `prompt`, ordered by `index`.
    """
    rows = _read_jsonl(Path(bundle) / "prompts.jsonl")
    key = lambda r: hashlib.sha256(
        f'coco-fid-v1:{seed}:{r["metadata"]["image_id"]}'.encode()
    ).hexdigest()
    chosen = sorted(sorted(rows, key=key)[:count], key=lambda r: r["index"])
    return [
        dict(
            index=r["index"],
            id=r["id"],
            image_id=r["metadata"]["image_id"],
            file_name=r["metadata"]["file_name"],
            prompt=r["prompt"],
        )
        for r in chosen
    ]


def build_index(subset, runs, out):
    """List the generated images of the subset found in baseline run directories.

    Args:
        subset: `subset.json` written by `select`.
        runs: baseline run directories (`baselines.runner` layout) of the COCO benchmark.
        out: output directory.

    Writes `out/sets.jsonl`, one line per (group, caption) whose `k` images all exist, with
    `group` (`<T2I model>|<method>`), `index` and `paths`. Writes `out/coverage.json` with, per
    group, the number of captions found and the bundle indices still missing. Returns the
    coverage.
    """
    wanted = {r["index"] for r in json.loads(Path(subset).read_text())["rows"]}
    found = defaultdict(dict)
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["bundle"]["benchmark"] != BENCHMARK:
            continue
        config = manifest["config"]
        group = f'{config["renderer"]}|{config["method"]}'
        for row in json.loads((run / "input.json").read_text()):
            if row["index"] not in wanted:
                continue
            paths = [
                run / "work" / f'{row["index"]:05d}' / "slots" / f"{s:04d}" / "image.png"
                for s in range(config["k"])
            ]
            if all(p.is_file() for p in paths):
                found[group][row["index"]] = [str(p.resolve()) for p in paths]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(dict(group=g, index=i, paths=p))
        for g in sorted(found)
        for i, p in sorted(found[g].items())
    ]
    _atomic_text(out / "sets.jsonl", "\n".join(lines) + "\n")
    coverage = {
        g: dict(found=len(v), missing=sorted(wanted - set(v))) for g, v in sorted(found.items())
    }
    _atomic_text(out / "coverage.json", json.dumps(coverage, indent=2) + "\n")
    return coverage


def index_validation(val_outs, out, name="coco250"):
    """List the images that `validate_checkpoints.py` rendered for the COCO subset.

    Args:
        val_outs: validation directories; each `step_<N>/<name>.jsonl` becomes one group
            `<directory name>|step_<N>`.
        out: directory receiving `sets.jsonl`, in the format `score` reads.
        name: the validation set name of the COCO subset.

    Writes `out/sets.jsonl` and `out/coverage.json`; returns `{group: captions}`. Raises
    `ValueError` when a row's image folder does not hold `k` images. A completion that did not
    parse has no images and is left out, so a group can hold fewer than 250 captions;
    `score --include-incomplete` scores such a group on the captions it has.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines, counts = [], {}
    for val_out in map(Path, val_outs):
        for path in sorted(val_out.glob(f"step_*/{name}.jsonl")):
            group = f"{val_out.name}|{path.parent.name}"
            for n, row in enumerate(_read_jsonl(path)):
                if row["metrics"].get("malformed"):
                    continue
                folders = [
                    f for f in val_out.glob(f"reward-*/images/{row['request_id']}") if f.is_dir()
                ]
                images = sorted(folders[0].glob("*.png")) if len(folders) == 1 else []
                if len(images) != row["k"]:
                    raise ValueError(f"{group}: {len(images)} images for {row['row_id']}")
                lines.append(
                    json.dumps(
                        dict(
                            group=group,
                            index=n,
                            row_id=row["row_id"],
                            paths=[str(i.resolve()) for i in images],
                        )
                    )
                )
                counts[group] = counts.get(group, 0) + 1
    _atomic_text(out / "sets.jsonl", "\n".join(lines) + "\n")
    _atomic_text(out / "coverage.json", json.dumps(counts, indent=2) + "\n")
    return counts


def crop_square(source, target):
    """Save a centre-cropped square PNG copy of one photograph, unless it already exists."""
    from PIL import Image

    if target.exists():
        return
    with Image.open(source) as image:
        image = image.convert("RGB")
        side = min(image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.png")
        image.crop((left, top, left + side, top + side)).save(temporary)
        temporary.replace(target)


def score(out, real_images, inception_dir, device, complete_only=True, expected=250):
    """Compute FID and KID of every group in `out/sets.jsonl` against the val2017 photographs.

    Args:
        out: directory holding `sets.jsonl`; results are written beside it.
        real_images: directory holding the 5,000 `val2017` JPEG files.
        inception_dir: directory holding Clean-FID's `inception-2015-12-05.pt`.
        device: torch device string.
        complete_only: skip a group that has fewer than `expected` captions.
        expected: number of captions in the subset.

    Writes `out/real-cropped/` (square PNG copies), `out/real-features.npy`, one
    `out/features-<group>.npy` per group, and `out/fid.json` with, per group, `captions`,
    `images`, `fid`, `kid` and `kid_x1000`. Returns that table.
    """
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from cleanfid.fid import frechet_distance, get_files_features, kernel_distance
    from cleanfid.inception_torchscript import InceptionV3W

    out = Path(out)
    model = InceptionV3W(str(inception_dir), download=False, resize_inside=False).to(device).eval()

    def features(paths):
        with torch.inference_mode():
            f = get_files_features(
                [str(p) for p in paths],
                model,
                num_workers=8,
                batch_size=64,
                device=torch.device(device),
                mode="clean",
                verbose=False,
            )
        if f.shape != (len(paths), 2048) or not np.isfinite(f).all():
            raise ValueError("Invalid Inception feature matrix")
        return f

    cache = out / "real-features.npy"
    if cache.exists():
        real = np.load(cache)
    else:
        sources = sorted(Path(real_images).glob("*.jpg"))
        if len(sources) != 5000:
            raise ValueError(f"Expected 5,000 val2017 photographs, found {len(sources)}")
        targets = [out / "real-cropped" / (p.stem + ".png") for p in sources]
        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(crop_square, sources, targets))
        real = features(targets)
        np.save(cache, real)
    real_mean, real_cov = real.mean(0), np.cov(real, rowvar=False)

    groups = defaultdict(list)
    for item in _read_jsonl(out / "sets.jsonl"):
        groups[item["group"]].append(item)
    table = {}
    for group, items in sorted(groups.items()):
        if complete_only and len(items) < expected:
            table[group] = dict(captions=len(items), skipped=f"fewer than {expected} captions")
            continue
        paths = [p for item in items for p in item["paths"]]
        name = out / f'features-{group.replace("|", "-")}.npy'
        f = np.load(name) if name.exists() and len(np.load(name)) == len(paths) else features(paths)
        np.save(name, f)
        kid = float(kernel_distance(f, real))
        table[group] = dict(
            captions=len(items),
            images=len(paths),
            fid=float(frechet_distance(f.mean(0), np.cov(f, rowvar=False), real_mean, real_cov)),
            kid=kid,
            kid_x1000=kid * 1000,
        )
        print(group, table[group], flush=True)
    _atomic_text(out / "fid.json", json.dumps(table, indent=2) + "\n")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True, help="subset.json to write")
    s.add_argument("--count", type=int, default=250)
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("index")
    s.add_argument("--subset", type=Path, required=True)
    s.add_argument("--run", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("index-validation")
    s.add_argument("--val-out", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--real-images", type=Path, required=True)
    s.add_argument("--inception-dir", type=Path, required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--include-incomplete", action="store_true")
    a = p.parse_args()
    if a.command == "select":
        rows = select_prompts(a.bundle, a.count, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        _atomic_text(
            a.out,
            json.dumps(
                dict(
                    benchmark=BENCHMARK,
                    count=len(rows),
                    seed=a.seed,
                    rule="smallest SHA-256 of coco-fid-v1:<seed>:<image_id>",
                    rows=rows,
                ),
                indent=1,
            )
            + "\n",
        )
        print(f"selected {len(rows)} captions -> {a.out}")
    elif a.command == "index":
        coverage = build_index(a.subset, a.run, a.out)
        for group, c in coverage.items():
            print(f"{group:32s} {c['found']:4d} captions, {len(c['missing'])} missing")
    elif a.command == "index-validation":
        for group, count in index_validation(a.val_out, a.out).items():
            print(f"{group:48s} {count:4d} captions")
    else:
        score(
            a.out, a.real_images, a.inception_dir, a.device, complete_only=not a.include_incomplete
        )


if __name__ == "__main__":
    main()
