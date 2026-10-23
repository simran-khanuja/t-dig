"""Deterministic, stratified manual-review samples in escaped standalone HTML."""

from collections import defaultdict
from html import escape
from pathlib import Path
import random

from .storage import digest, read_json
from .dataset import SCHEMA


def build_review(source, output, per_group=20, seed=42):
    if per_group < 1:
        raise ValueError("per_group must be positive")
    if Path(output).exists():
        raise ValueError("Review output exists; choose a new path")
    data = read_json(source)
    if data.get("schema_version") != SCHEMA or data.get("kind") not in {
        "filter_summary",
        "clustered_pool",
    }:
        raise ValueError("Review requires a new-format filter summary or clustered pool")
    groups = defaultdict(list)
    error_ids = {r["id"] for r in data.get("errors", [])}
    for state in ("kept", "rejected", "errors", "pending"):
        for row in data.get(state, []):
            # Failed rows also appear in pending; show them only once.
            if state == "pending" and row["id"] in error_ids:
                continue
            groups[state].append(row)
    rng = random.Random(seed)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>HPDv2 review</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}"
        "article{border:1px solid #ccc;padding:16px;margin:12px 0}small{overflow-wrap:anywhere}"
        "blockquote{white-space:pre-wrap}label{margin-right:16px}</style>",
        "<h1>HPDv2 manual review</h1>",
        f"<p>Source: {escape(str(source))}<br>SHA256: {digest(data)}<br>Seed: {seed}</p>",
        "<p>Stratified inspection sample, not a measured precision/recall score. "
        "Review rejects as well as keeps. Check policy fit, prompt identity, axes, and participant suitability. "
        "Checkboxes are local visual aids and are not saved; record adjudications separately by ID.</p>",
    ]
    for group in sorted(groups):
        candidates = sorted(groups[group], key=lambda r: r["id"])
        selected = rng.sample(candidates, min(per_group, len(candidates)))
        parts.append(f"<h2>{escape(group)} — {len(selected)} of {len(candidates)}</h2>")
        for row in selected:
            reason = row.get("reason", row.get("error", "Not processed"))
            axes = "".join(
                f"<li><strong>{escape(axis['dimension'])}</strong>: "
                + " / ".join(escape(v) for v in axis["values"])
                + "</li>"
                for axis in row.get("variation_axes", [])
            )
            parts.append(
                f"<article><small>{escape(row['id'])}</small>"
                f"<blockquote>{escape(row['prompt'])}</blockquote>"
                f"<p>{escape(reason)}</p><p>Candidate axes (screening evidence only):</p><ul>{axes}</ul>"
                "<label><input type='checkbox'>Agree</label><label><input type='checkbox'>Disagree</label>"
                "<label><input type='checkbox'>Needs discussion</label></article>"
            )
    parts.append("</html>")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(parts))
    return {
        "n_groups": len(groups),
        "n_sampled": sum(min(per_group, len(v)) for v in groups.values()),
        "output": str(output),
    }
