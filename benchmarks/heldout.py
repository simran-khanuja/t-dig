"""Held-out axis prompts: base prompts the generation policy never trains on, with their axes.

A bundle directory holds `prompts.jsonl` and `manifest.json`. Each row has `index`, `id`,
`prompt` (the base prompt) and `metadata`, whose `axes` lists the variation requests collected
for that prompt. Baseline methods render from `prompt` alone; the axes are used afterwards to
measure diversity along each requested axis. `tdig/heldout_bundle.py` builds the bundle.
"""

import json

from .cli import digest

NAME = "heldout-axes"


def load_heldout(path):
    """Return `(manifest, rows)` of a held-out axis bundle.

    Raises `ValueError` when the manifest is not a schema-1 `heldout-axes` manifest, when
    `prompts.jsonl` does not hash to `rows_sha256`, or when row indices are not 0..count-1.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("benchmark") != NAME:
        raise ValueError("Unexpected held-out axis manifest")
    raw = (path / "prompts.jsonl").read_bytes()
    if digest(raw) != manifest["rows_sha256"]:
        raise ValueError("Held-out axis content checksum mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if [r["index"] for r in rows] != list(range(manifest["count"])):
        raise ValueError("Held-out axis rows must be indexed 0..count-1")
    if any(
        not isinstance(r["prompt"], str) or not r["prompt"].strip() or not r["metadata"]["axes"]
        for r in rows
    ):
        raise ValueError("Held-out axis row without a prompt or axes")
    return manifest, rows
