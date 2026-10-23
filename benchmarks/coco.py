"""COCO 2017 validation: one deterministically selected caption per image."""

import hashlib
import json

from hpdv2.storage import atomic_json
from .cli import digest

NAME = "coco2017-val-5000"


def select_rows(data, seed, expected_images=5000):
    images = data["images"]
    if len(images) != expected_images or len({i["id"] for i in images}) != expected_images:
        raise ValueError(f"Expected {expected_images} unique COCO validation images")
    by_image = {i["id"]: [] for i in images}
    seen = set()
    for caption in data["annotations"]:
        ident = caption["id"]
        if type(ident) is not int or ident in seen or caption["image_id"] not in by_image:
            raise ValueError("Duplicate/invalid caption ID or foreign image ID")
        if not isinstance(caption["caption"], str) or not caption["caption"].strip():
            raise ValueError("Empty COCO caption")
        seen.add(ident)
        by_image[caption["image_id"]].append(caption)
    rows = []
    for index, image in enumerate(sorted(images, key=lambda i: i["id"])):
        ident = image["id"]
        if type(ident) is not int or image["file_name"] != f"{ident:012d}.jpg":
            raise ValueError("Unexpected COCO image identity/filename")
        captions = by_image[ident]
        if len(captions) < 5:
            raise ValueError(f"Image {ident} has fewer than five reference captions")
        # Hash ordering avoids dependence on JSON order or Python RNG version.
        selected = min(
            captions,
            key=lambda c: (
                hashlib.sha256(f'coco-caption-v1:{seed}:{ident}:{c["id"]}'.encode()).hexdigest(),
                c["id"],
            ),
        )
        metadata = {
            "image_id": ident,
            "caption_id": selected["id"],
            "file_name": image["file_name"],
            "prompt": selected["caption"],
            "reference_captions": sorted(captions, key=lambda c: c["id"]),
        }
        rows.append(
            {
                "index": index,
                "id": f"coco2017:{ident:012d}",
                "prompt": selected["caption"],
                "metadata": metadata,
            }
        )
    return rows


def prepare_coco(source, out, seed=42, execute=False):
    if type(seed) is not int:
        raise ValueError("Caption seed must be an integer")
    if not execute:
        return {
            "execute": False,
            "benchmark": NAME,
            "count": 5000,
            "caption_seed": seed,
            "source": str(source),
            "out": str(out),
        }
    if out.exists():
        raise ValueError("COCO bundle already exists; use a new directory")
    raw = source.read_bytes()
    rows = select_rows(json.loads(raw), seed)
    row_bytes = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
    ).encode()
    manifest = {
        "schema_version": 1,
        "benchmark": NAME,
        "count": 5000,
        "caption_seed": seed,
        "source_sha256": digest(raw),
        "rows_sha256": digest(row_bytes),
        "selection": "all 5000 images sorted by image ID; minimum SHA256(coco-caption-v1:seed:image_id:caption_id)",
        "source_url": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "source_member": "annotations/captions_val2017.json",
        "protocol": "shared-study-protocol; not the unpublished CADS/GI caption subsets",
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.json").write_bytes(raw)
    (out / "prompts.jsonl").write_bytes(row_bytes)
    atomic_json(out / "manifest.json", manifest)
    return manifest


def load_coco(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("benchmark") != NAME
        or manifest.get("count") != 5000
    ):
        raise ValueError("Unexpected COCO manifest")
    raw = (path / "source.json").read_bytes()
    row_bytes = (path / "prompts.jsonl").read_bytes()
    if digest(raw) != manifest["source_sha256"] or digest(row_bytes) != manifest["rows_sha256"]:
        raise ValueError("COCO content checksum mismatch")
    rows = [json.loads(line) for line in row_bytes.decode().splitlines()]
    if rows != select_rows(json.loads(raw), manifest["caption_seed"]):
        raise ValueError("COCO selection does not reproduce from original annotations")
    return manifest, rows
