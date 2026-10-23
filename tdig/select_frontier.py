"""Select the checkpoints of a run that form its alignment-diversity frontier.

Reads the per-row validation records that `validate_checkpoints.py` writes,
`<validation>/step_<N>/<set>.jsonl`, and for every validated step computes two means over the
rows of the set:

- alignment: `fidelity_mean`, the mean VQAScore of the set's images against the base prompt;
- diversity: `normalized_vendi`, the TPIPS Vendi score on the requested axis, from 0 to 1.

A step is on the frontier when no other step beats it. Step B beats step A when B is better on
one measure by more than `--margin` and no worse on the other by more than `--margin`. The
margin absorbs the noise of the set mean (standard error about 0.008 over 379 rows). The
frontier, ordered from the most aligned step to the most diverse, is the strength scale:
strength 1 loads the first adapter, strength X the last.

Selection and reporting use disjoint prompts. A hash of the base prompt's text assigns it to the
selection or the report half; the frontier is chosen on the selection half and its values are
read from the report half. `--select-fraction 1` uses the whole set for both, which overstates
the frontier and suits inspection only.

A policy can raise VQAScore without being more faithful by asking the T2I model to print the
base prompt inside the image ("Text at the bottom reads: '...'"). `asks_for_text` flags a
completion that requests rendered text when its base prompt does not. A step whose share of
flagged completions exceeds `--max-text-share` (default 0.10; untrained policies score 0 to
0.06) stays in the outputs but cannot enter the frontier.

A malformed completion has no images. It counts as diversity 0 and is left out of the
alignment mean; `malformed` reports its share of rows.

Validation directories with the same name are one run: their steps are pooled, so checkpoints
scored by a later job into another directory join the run's frontier. A step present in two
directories is read from the first.

Outputs, in `--out`: `frontier.json` (every step and the ordered frontier of each run),
`frontier.csv` (one line per run and step) and `frontier.png` (one panel per run).
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ALIGNMENT, DIVERSITY = "fidelity_mean", "normalized_vendi"
EXTRA = (
    "score",
    "fidelity_min",
    "imagereward",
    "hpsv3",
    "clip_score",
    "aesthetic",
    "words_per_prompt",
    "drift_mean",
)
SURFACE, INK, MUTED, GRID, AXIS, ACCENT = (
    "#fcfcfb",
    "#0b0b0b",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#2a78d6",
)


TEXT_REQUEST = re.compile(
    r"(text|caption|words?|label|banner|sign|title|subtitle|typography|lettering|inscription)[^.<]{0,40}"
    r"\b(reads?|reading|says?|saying|stating|states|written|that read)\b|\b(reads?|says?)\s*[:\"'\u201c\u2018]",
    re.I,
)


def asks_for_text(completion, base_prompt):
    """Return whether the completion asks for rendered text and the base prompt does not."""
    return bool(TEXT_REQUEST.search(completion)) and not TEXT_REQUEST.search(base_prompt)


def half(base_prompt, fraction):
    """Return `select` or `report` for a base prompt; stable across runs and steps."""
    value = int(hashlib.sha256(base_prompt.encode()).hexdigest()[:8], 16) / 2**32
    return "select" if value < fraction else "report"


def summarize(rows):
    """Return the means of one step over `rows`; None when `rows` is empty."""
    if not rows:
        return None
    valid = [r["metrics"] for r in rows if not r["metrics"].get("malformed")]
    out = {
        "rows": len(rows),
        "malformed": 1 - len(valid) / len(rows),
        "text_share": sum(
            asks_for_text(r.get("completion", ""), r.get("base_prompt", "")) for r in rows
        )
        / len(rows),
        "diversity": sum(m[DIVERSITY] for m in valid) / len(rows),
        "alignment": sum(m[ALIGNMENT] for m in valid) / len(valid) if valid else 0.0,
    }
    for key in EXTRA:
        values = [m[key] for m in valid if m.get(key) is not None]
        out[key] = sum(values) / len(values) if values else None
    return out


def frontier(points, margin=0.0):
    """Return the steps no other step beats, ordered from most aligned to most diverse.

    Args:
        points: dict of step to a dict holding `alignment` and `diversity`.
        margin: a difference no larger than this counts as a tie on that measure.
    """

    def dominated(a):
        return any(
            b["alignment"] >= a["alignment"] - margin
            and b["diversity"] >= a["diversity"] - margin
            and (
                b["alignment"] > a["alignment"] + margin or b["diversity"] > a["diversity"] + margin
            )
            for b in points.values()
        )

    kept = [s for s, p in points.items() if not dominated(p)]
    return sorted(kept, key=lambda s: (-points[s]["alignment"], points[s]["diversity"], s))


def load(validation, name, fraction):
    """Return `{step: {"select": ..., "report": ..., "all": ...}}` for one validation directory."""
    steps = {}
    for path in sorted(Path(validation).glob(f"step_*/{name}.jsonl")):
        if not (path.parent / f"{name}.summary.json").exists():
            continue  # the step is still being scored
        rows = [json.loads(line) for line in path.open()]
        parts = {"select": [], "report": []}
        for row in rows:
            parts[half(row["base_prompt"], fraction)].append(row)
        step = int(re.fullmatch(r"step_(\d+)", path.parent.name).group(1))
        steps[step] = {
            "select": summarize(parts["select"]),
            "report": summarize(parts["report"]),
            "all": summarize(rows),
        }
    return steps


def plot(results, path, reported):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(3, n)
    lines = -(-n // cols)
    fig, axes = plt.subplots(
        lines, cols, figsize=(4.6 * cols, 4.0 * lines), squeeze=False, facecolor=SURFACE
    )
    for ax, (name, result) in zip(axes.flat, results.items()):
        ax.set_facecolor(SURFACE)
        points = {s: v[reported] for s, v in result["steps"].items()}
        ax.scatter(
            [p["alignment"] for p in points.values()],
            [p["diversity"] for p in points.values()],
            s=22,
            color=MUTED,
            zorder=2,
        )
        chosen = result["frontier"]
        ax.plot(
            [points[s]["alignment"] for s in chosen],
            [points[s]["diversity"] for s in chosen],
            color=ACCENT,
            linewidth=1.6,
            marker="o",
            markersize=6,
            zorder=3,
        )
        for s, p in points.items():
            ax.annotate(
                str(s),
                (p["alignment"], p["diversity"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color=INK if s in chosen else MUTED,
            )
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("VQAScore against the base prompt", fontsize=9, color=INK)
        ax.set_ylabel("TPIPS Vendi on the axis, 0 to 1", fontsize=9, color=INK)
        ax.grid(color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Checkpoints by training step, {reported} prompts; blue marks the frontier chosen on the "
        "selection prompts",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("validations", nargs="+", help="validation directories, each holding step_<N>/")
    p.add_argument("--set", default="heldout", help="validation set name")
    p.add_argument(
        "--select-fraction", type=float, default=0.5, help="share of base prompts used to choose"
    )
    p.add_argument("--margin", type=float, default=0.005, help="difference treated as a tie")
    p.add_argument(
        "--max-text-share",
        type=float,
        default=0.10,
        help="steps with a larger share of completions asking for rendered text are excluded",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    if not 0 < a.select_fraction <= 1:
        raise ValueError("--select-fraction must be in (0, 1]")
    reported = "report" if a.select_fraction < 1 else "select"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, runs = {}, {}
    for validation in a.validations:
        runs.setdefault(Path(validation).name, []).append(str(validation))
    for name, validations in runs.items():
        steps = {}
        for validation in validations:
            for step, values in load(validation, a.set, a.select_fraction).items():
                steps.setdefault(step, values)
        if not steps:
            print(f"{name}: no scored step for set {a.set}")
            continue
        steps = dict(sorted(steps.items()))
        clean = {
            s: v["select"] for s, v in steps.items() if v["all"]["text_share"] <= a.max_text_share
        }
        chosen = frontier(clean, a.margin)
        results[name] = {
            "validation": validations,
            "set": a.set,
            "steps": steps,
            "frontier": chosen,
            "strength": {i + 1: s for i, s in enumerate(chosen)},
        }
    (out / "frontier.json").write_text(
        json.dumps(
            {
                "select_fraction": a.select_fraction,
                "margin": a.margin,
                "reported": reported,
                "runs": results,
            },
            indent=2,
        )
    )
    with (out / "frontier.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "step",
                "strength",
                "rows",
                "alignment",
                "diversity",
                "malformed",
                "text_share",
                *EXTRA,
                "select_alignment",
                "select_diversity",
            ]
        )
        for name, result in results.items():
            rank = {s: i for i, s in result["strength"].items()}
            for step, v in sorted(result["steps"].items()):
                r = v[reported]
                w.writerow(
                    [
                        name,
                        step,
                        rank.get(step, ""),
                        r["rows"],
                        r["alignment"],
                        r["diversity"],
                        r["malformed"],
                        v["all"]["text_share"],
                        *[r[k] for k in EXTRA],
                        v["select"]["alignment"],
                        v["select"]["diversity"],
                    ]
                )
    for name, result in results.items():
        print(f"\n{name}: {len(result['steps'])} steps scored, frontier {result['frontier']}")
        for i, s in result["strength"].items():
            r = result["steps"][s][reported]
            print(
                f"  strength {i}: step {s:5d}  VQAScore {r['alignment']:.3f}  Vendi {r['diversity']:.3f}  "
                f"ImageReward {r['imagereward'] if r['imagereward'] is None else round(r['imagereward'], 3)}"
            )
    if results and not a.no_plot:
        plot(results, out / "frontier.png", reported)


if __name__ == "__main__":
    main()
