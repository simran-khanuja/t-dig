"""Build immutable prompt pools and validate stage inputs."""

import hashlib
from pathlib import Path

from .storage import atomic_json, digest, prompt_id, read_json

SCHEMA = 2


def validate_prompts(rows):
    if not isinstance(rows, list):
        raise ValueError("prompts must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each prompt must be an object")
        text = row.get("prompt")
        if not isinstance(text, str) or not text or text != text.strip():
            raise ValueError("Prompt must be a nonempty, stripped string")
        if row.get("id") != prompt_id(text) or row["id"] in seen:
            raise ValueError("Invalid or duplicate prompt ID")
        seen.add(row["id"])
    return rows


def prepare(source, output, split, provenance=None):
    if Path(output).exists():
        raise ValueError("Output already exists; choose a new versioned pool path")
    raw = Path(source).read_bytes()
    data = read_json(source)
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Expected HPDv2 list or an object with a data list")
    rows, seen, blanks = [], set(), 0
    for number, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid dataset row {number}")
        text = item.get("prompt") or item.get("caption") or ""
        if not isinstance(text, str):
            raise ValueError(f"Non-string prompt in row {number}")
        text = text.strip()
        if not text:
            blanks += 1
        elif text not in seen:
            seen.add(text)
            rows.append({"id": prompt_id(text), "prompt": text})
    if not rows:
        raise ValueError("Dataset has no usable prompt strings")
    result = {
        "schema_version": SCHEMA,
        "kind": "prompt_pool",
        "metadata": {
            "split": split,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source": str(Path(source).resolve()),
            "provenance": provenance,
            "n_rows": len(items),
            "n_blank": blanks,
            "n_unique": len(rows),
            "normalization": "strip outer whitespace; case-sensitive exact dedup; first occurrence",
        },
        "prompts": rows,
        "pool_hash": digest(rows),
    }
    atomic_json(output, result)
    return result


def load_input(path, stage, allow_partial=False):
    data = read_json(path)
    if data.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported input schema; legacy artifacts cannot be resumed implicitly")
    if stage == "pass1":
        if data.get("kind") != "prompt_pool":
            raise ValueError("Pass 1 requires a prepared prompt pool")
        rows = validate_prompts(data.get("prompts"))
        if data.get("pool_hash") != digest(rows):
            raise ValueError("Pool hash mismatch")
    else:
        expected = "pass1" if stage == "pass2" else "pass2"
        if data.get("kind") != "filter_summary" or data.get("stage") != expected:
            raise ValueError(f"{stage} requires a new-format {expected} summary")
        if data.get("metadata", {}).get("full_pipeline_complete") is not True and not allow_partial:
            raise ValueError(
                "Upstream stage is incomplete; use --allow-partial-input only for a pilot"
            )
        rows = validate_prompts(data.get("kept"))
        from .policy import validate_decisions

        validate_decisions(rows, rows, expected, include_prompt=True)
    return data, [{"id": r["id"], "prompt": r["prompt"]} for r in rows]
