"""Render K i.i.d. samples of each prompt at evaluation settings and write the image index.

The thresholds `align_thresh` and `quality_thresh` of a training row are means over the
T2I model's i.i.d. samples of its base prompt. This module renders those samples for prompts
that have none, with the T2I model, assets and `render.val` settings of a reward configuration
(1024 px, the model card's sampler), matching the samples of the published thresholds.

Image `j` of a prompt starts from its own noise, seeded by SHA-256 of `<prompt_id>|<j>`. Output:
`<out>/images/<prompt_id>/<jjjj>.png` and `<out>/index-<shard>.jsonl`, one line per image with
the fields `iid_vqa.py score` and `iid_quality.py score` read (`renderer`, `prompt_id`,
`image_index`, `path`, `base_prompt`). Concatenate the shard files into `index.jsonl` when every
shard has written `render-done-<shard>.json`. A rerun skips prompts whose images exist.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--prompts", type=Path, required=True, help="JSON lines with prompt_id and base_prompt"
    )
    p.add_argument(
        "--reward-config",
        type=Path,
        required=True,
        help="reward-config.json of a run on the T2I model",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--k", type=int, default=4)
    a = p.parse_args()
    import torch
    from baselines.assets import verify_assets
    from baselines.config import t2i_model_defaults
    from baselines.native_pool import NativeT2IModel

    config = json.loads(a.reward_config.read_text())
    cfg = t2i_model_defaults("iid", config["renderer"])
    cfg.update(config["render"]["val"], device="cuda:0")
    t2i_model = NativeT2IModel(verify_assets(Path(config["renderer_assets"]), ["renderer"]), cfg)
    t2i_model.pipe.vae.enable_slicing()
    rows = [json.loads(line) for line in a.prompts.open()][a.shard :: a.num_shards]
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / f"index-{a.shard:02d}.jsonl").open("w", encoding="utf-8") as index:
        for n, row in enumerate(rows):
            folder = a.out / "images" / row["prompt_id"]
            paths = [folder / f"{j:04d}.png" for j in range(a.k)]
            if not all(path.exists() for path in paths):
                folder.mkdir(parents=True, exist_ok=True)
                seeds = [
                    int(hashlib.sha256(f"{row['prompt_id']}|{j}".encode()).hexdigest()[:8], 16)
                    % (2**31 - 1)
                    for j in range(a.k)
                ]
                kwargs = t2i_model.kwargs([row["base_prompt"]] * a.k, seeds[0])
                kwargs["generator"] = [torch.Generator("cpu").manual_seed(s) for s in seeds]
                kwargs["num_images_per_prompt"] = 1
                if isinstance(kwargs.get("negative_prompt"), str):
                    kwargs["negative_prompt"] = [kwargs["negative_prompt"]] * a.k
                with torch.inference_mode():
                    images = t2i_model.pipe(**kwargs).images
                for path, image in zip(paths, images):
                    temporary = path.with_suffix(".tmp.png")
                    image.save(temporary)
                    temporary.replace(path)
            for j, path in enumerate(paths):
                index.write(
                    json.dumps(
                        dict(
                            renderer=config["renderer"],
                            prompt_id=row["prompt_id"],
                            image_index=j,
                            path=str(path.resolve()),
                            base_prompt=row["base_prompt"],
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if n % 100 == 0:
                print(f"shard {a.shard}: {n + 1} of {len(rows)}", flush=True)
    (a.out / f"render-done-{a.shard:02d}.json").write_text(
        json.dumps(dict(prompts=len(rows), k=a.k)) + "\n"
    )


if __name__ == "__main__":
    main()
