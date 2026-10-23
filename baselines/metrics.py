"""Durable per-prompt alignment/diversity metrics for either benchmark."""

from pathlib import Path
import numpy as np
from hpdv2.storage import atomic_json, digest, read_json, run_lock
from .assets import verify_assets
from .runner import inspect, code_hash, environment


def set_metrics(features):
    unary = np.asarray(features["clip"], dtype=float)
    pair = np.asarray(features["pairwise"], dtype=float)
    cls = np.asarray(features["dino_cls"], dtype=float)
    k = len(unary)
    if (
        k < 1
        or pair.shape != (k, k)
        or cls.shape[0] != k
        or not all(np.isfinite(v).all() for v in (unary, pair, cls))
    ):
        raise ValueError("Invalid/nonfinite evaluation features")
    norms = np.linalg.norm(cls, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero feature vector")
    cls = cls / norms
    kernel = cls @ cls.T
    eigen = np.linalg.eigvalsh(kernel / k)
    if eigen.min() < -1e-8:
        raise ValueError("Invalid PSD Vendi kernel")
    eigen = np.clip(eigen, 0, None)
    eigen /= eigen.sum()
    positive = eigen[eigen > 0]
    return {
        "clip_base_alignment": float(unary.mean()),
        "dino_patch_pair_distance": (float(pair[np.triu_indices(k, 1)].mean()) if k > 1 else None),
        "dino_cls_pair_distance": (
            float((1 - kernel)[np.triu_indices(k, 1)].mean()) if k > 1 else None
        ),
        "dino_cls_vendi": float(np.exp(-(positive * np.log(positive)).sum())),
        "k": k,
        "clip_per_image": unary.tolist(),
        "clip_text_truncated": bool(features.get("clip_text_truncated", False)),
        "clip_text_tokens_before_truncation": features.get("clip_text_tokens_before_truncation"),
    }


def evaluate(run, out, asset_path, execute=False, allow_partial=False, device="cuda:0"):
    run, out = Path(run).resolve(), Path(out).resolve()
    manifest = read_json(run / "manifest.json")
    rows = read_json(run / "input.json")
    records = inspect(run, rows, manifest)
    complete = [r for r in rows if all(records[r["id"]])]
    if not allow_partial and (len(complete) != len(rows) or not manifest["full_benchmark"]):
        raise ValueError(
            "Full metrics require the complete full benchmark; use --allow-partial for a labeled smoke audit"
        )
    assets = read_json(asset_path)
    identity = {
        "schema_version": 1,
        "run_hash": digest(manifest),
        "scoring_assets": {k: assets["assets"][k] for k in ["clip", "dino"]},
        "images": {r["id"]: [v["sha256"] for v in records[r["id"]]] for r in complete},
        "allow_partial": allow_partial,
        "code_hash": code_hash(),
        "environment": environment(),
        "device": device,
        "metrics": "CLIP-B32 cosine; DINOv2-B patch pair distance; DINO-CLS cosine/Vendi; v1",
    }
    if not execute:
        return {
            "execute": False,
            "complete_sets": len(complete),
            "expected_sets": len(rows),
            "out": str(out),
        }
    with run_lock(out):
        if (out / "manifest.json").exists():
            if read_json(out / "manifest.json") != identity:
                raise ValueError("Metric manifest changed; use a new output directory")
        elif any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError("Existing metrics lack manifest")
        else:
            atomic_json(out / "manifest.json", identity)
        verified = verify_assets(asset_path, ["clip", "dino"])
        from .features import Features
        import torch
        from PIL import Image

        scorer = None
        values = []
        for row in complete:
            path = out / "sets" / f'{row["index"]:05d}.json'
            if path.exists():
                value = read_json(path)
                if (
                    value.get("manifest_hash") != digest(identity)
                    or value.get("prompt_id") != row["id"]
                ):
                    raise ValueError("Metric record identity mismatch")
            else:
                if scorer is None:
                    scorer = Features(
                        verified["assets"]["clip"]["path"],
                        verified["assets"]["dino"]["path"],
                        device,
                    )
                images = []
                for i in range(manifest["config"]["k"]):
                    with Image.open(
                        run / "work" / f'{row["index"]:05d}' / "slots" / f"{i:04d}" / "image.png"
                    ) as im:
                        images.append(
                            torch.from_numpy(np.array(im.convert("RGB"), copy=True))
                            .permute(2, 0, 1)[None]
                            .float()
                            / 127.5
                            - 1
                        )
                value = {
                    "manifest_hash": digest(identity),
                    "prompt_id": row["id"],
                    "base_prompt": row["prompt"],
                    **set_metrics(scorer.extract(images, row["prompt"])),
                }
                atomic_json(path, value)
            values.append(value)
        keys = [
            "clip_base_alignment",
            "dino_patch_pair_distance",
            "dino_cls_pair_distance",
            "dino_cls_vendi",
        ]
        summary = {
            "full_benchmark_complete": manifest["full_benchmark"] and len(complete) == len(rows),
            "scored_sets": len(values),
            "expected_sets": len(rows),
            "full_benchmark_sets": manifest["bundle"]["count"],
            "k": manifest["config"]["k"],
            "macro_over_completed_sets": {
                key: (
                    float(np.mean([v[key] for v in values]))
                    if values and values[0][key] is not None
                    else None
                )
                for key in keys
            },
            "clip_truncated_captions": sum(v["clip_text_truncated"] for v in values),
            "note": "CLIP and DINO patch metrics are also used for GI selection; report independent alignment such as VQAScore separately.",
        }
        atomic_json(out / "summary.json", summary)
        return summary
