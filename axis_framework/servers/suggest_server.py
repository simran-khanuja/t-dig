"""Suggest server: show a VLM the four i.i.d. images and ask what the T2I model held fixed.

    POST /suggest {base_prompt, images: [4], k, values} -> {axes: [{name, why, example_values}]}
    POST /rephrase {base_prompt, axis} -> {kind, axis, add}   (see REPHRASE_SYSTEM)
    GET  /health

The elicitation prompt, user template and parser come from backend/suggest.py (SYSTEM,
USER_TMPL, parse), so suggestions use the same "what is the same across all four that did not
have to be?" framing the training axes were elicited under. Only the generate loop is
re-implemented here.

`example_values` come from a second VLM call over the chosen axis phrases. The elicitation
prompt forbids values, since a value list caps diversity; the values are UI decoration on the
chip and never reach the policy, the reward or the audit.

    $PY_REWARD \\
        axis_framework/servers/suggest_server.py --port 19103
    ... --fake                       # no GPU: canned axes, for driving the loop
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axis_framework import config  # noqa: E402
from axis_framework.backend import suggest  # noqa: E402
from axis_framework.http_util import HttpError, serve  # noqa: E402

_STATE = {}
_GPU_LOCK = threading.Lock()

# Second pass, values only, for an axis that is already chosen.
VALUES_SYSTEM = """You are given an image-generation prompt and ONE diversity axis \
that has already been chosen. Name {n} clearly different values of that axis which would \
suit this prompt.

Answer with {n} short noun phrases on one line, separated by " | ", and nothing else. \
No numbering, no explanation."""

VALUES_USER = """Prompt: "{prompt}"
Axis: {axis}"""

FAKE_AXES = [
    {
        "name": "cultural setting",
        "why": "(fake) all four look like the same place",
        "example_values": ["Lagos", "Osaka", "Oaxaca"],
    },
    {
        "name": "camera angle",
        "why": "(fake) all four are eye-level medium shots",
        "example_values": ["low angle", "overhead", "tight close-up"],
    },
    {
        "name": "time of day",
        "why": "(fake) all four are in flat midday light",
        "example_values": ["dawn", "dusk", "night"],
    },
    {
        "name": "the people shown",
        "why": "(fake) all four show the same kind of person",
        "example_values": ["an elderly man", "a teenage girl", "a woman in her forties"],
    },
]


def _vlm():
    if "model" not in _STATE:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = time.time()
        print(f"[suggest] loading {_STATE['model_id']} ...", flush=True)
        _STATE["model"] = Qwen3VLForConditionalGeneration.from_pretrained(
            _STATE["model_id"], torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        _STATE["proc"] = AutoProcessor.from_pretrained(_STATE["model_id"])
        print(f"[suggest] loaded in {time.time() - t0:.0f}s", flush=True)
    return _STATE["model"], _STATE["proc"]


def _ask(messages, max_new_tokens, temperature):
    import torch
    from qwen_vl_utils import process_vision_info

    vlm, proc = _vlm()
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(
        vlm.device
    )
    with torch.no_grad():
        out = vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or None,
            top_p=0.9,
        )
    return proc.tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def _example_values(base_prompt, axis, n=3, temperature=0.7):
    """Return up to *n* example values for one axis (UI decoration only)."""
    raw = _ask(
        [
            {"role": "system", "content": VALUES_SYSTEM.format(n=n)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VALUES_USER.format(prompt=base_prompt, axis=axis)}
                ],
            },
        ],
        max_new_tokens=80,
        temperature=temperature,
    )
    vals = [v.strip(" .*\"") for v in raw.split("|")]
    return [v for v in vals if v][:n]


# /rephrase: the policy was trained on short axis names ("time of day"), but users also type
# constraints for all four images ("no bench"), exclusions or complaints. This call classifies
# the request and restates it as an axis plus an optional "add" clause. The orchestrator
# decides whether to offer it; the recorded axis is always what the participant chose.
REPHRASE_SYSTEM = """A user of an image-generation tool typed a request into the "axis" box. The tool \
makes four versions of the user's prompt that differ along the axis, so EVERY request here is a request \
for variety: your job is to name the thing the user wants to see differ between the four images, in a \
short noun phrase, and to keep any detail that must hold in every image separate.

Rules:
- Read exclusions and corrections as the axis they point at: "no suits, some other attire" means vary the \
  attire (other than suits); "no bench" means vary where he sits; "make her have curly hair" means vary \
  the hairstyle.
- Shorten the wording only. If the user lists several things, keep every one of them.
- "add" is usually empty. Use it only when the REQUEST itself states something that must be true of \
  all four images (a count, an object that must be present, a fixed camera view), as a clause that can \
  be appended to the prompt. Never copy details from the prompt into "add".
- When the request asks for something to be SHOWN or INCLUDED ("show a couple", "include a dog", "with a \
  red car") that the prompt does not already have, put that thing in "add". The axis is then only what \
  the request says should differ, and it is empty when the request names nothing to vary.
- kind is for the record: axis (already a request for variety), exclusion (an axis stated as what NOT to \
  show), constraint (a change wanted in all four images), complaint (a comment on what went wrong).

Answer with exactly three lines, "kind: ...", "axis: ..." and "add: ...", and nothing else."""

REPHRASE_USER = 'Prompt: "{prompt}"\nRequest: "{axis}"'

REPHRASE_SHOTS = [
    (
        "a photo of a man sitting on a chair",
        "no suits, some other attire",
        "exclusion",
        "attire, other than suits",
        "",
    ),
    (
        "a baker decorating a square cake. Focus on a rustic wedding theme with white roses and wooden accents.",
        "if you could create different themed cakes that would be great",
        "axis",
        "cake theme",
        "",
    ),
    (
        "An middle-aged man, in a blue pair of shorts and a blue T-shirt, scrolling his phone sitting under a tree in a park.",
        "No bench asked for.",
        "exclusion",
        "where he sits, not a bench",
        "",
    ),
    (
        "The lone woman is waiting for the next available bus.",
        "Make the woman have curly hair, and carry a tote bag.",
        "constraint",
        "hairstyle and bag",
        "",
    ),
    (
        "a young person looks into a mirror and sees a fantasy creature staring back.",
        "gender, design of creature, frame material, image composition",
        "axis",
        "gender, design of creature, frame material, image composition",
        "",
    ),
    (
        "i want to see grades assigned on a written paper, with large red letters",
        "I am not seeing a grade or a total score",
        "complaint",
        "the grade written on the paper",
        "a large red letter grade and a total score written on the paper",
    ),
    (
        "Create an image of a proficient performer juggling five balls whilst balancing on a unicycle atop a small table.",
        "Add more variety to the position of the balls, they should not all be in a straight arced line. "
        "And make sure all five balls are different colors.",
        "constraint",
        "position of the balls",
        "five balls of different colours",
    ),
    ("a dog playing in a park", "dog breed", "axis", "dog breed", ""),
    ("a photo of a beach", "show a lifeguard tower", "constraint", "", "a lifeguard tower"),
    (
        "a street market at dusk",
        "types of stalls and include a street musician",
        "constraint",
        "types of stalls",
        "a street musician",
    ),
    (
        "baseball game from the point of view of someone sitting behind home plate",
        "Change the sky to different times of days",
        "axis",
        "time of day",
        "",
    ),
    (
        "A family of 4 (1 male adult, 1 female adult, 2 twins) having breakfast in a 5 star restaurant.",
        "Exclude one female adult.",
        "constraint",
        "who is in the family",
        "only one female adult",
    ),
    (
        "a man hanging a guitar on a wall-rack alongside other instruments.",
        "types of instruments, wall finish, viewing angle",
        "axis",
        "types of instruments, wall finish, viewing angle",
        "",
    ),
    (
        "Generate a photo realistic picture of a skateboarder getting ready to drop in on a vert ramp",
        "using this image change the spectators all to teenagers cheering on the skateboarder.",
        "constraint",
        "the spectators",
        "the spectators are teenagers cheering",
    ),
]


def _rephrase_messages(base_prompt, axis):
    m = [{"role": "system", "content": REPHRASE_SYSTEM}]
    for prompt, req, kind, ax, add in REPHRASE_SHOTS:
        m.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": REPHRASE_USER.format(prompt=prompt, axis=req)}
                ],
            }
        )
        m.append({"role": "assistant", "content": f"kind: {kind}\naxis: {ax}\nadd: {add}"})
    m.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": REPHRASE_USER.format(prompt=base_prompt, axis=axis)}
            ],
        }
    )
    return m


def _parse_rephrase(raw):
    import re

    out = {"kind": None, "axis": "", "add": ""}
    for line in raw.splitlines():
        mm = re.match(r"\s*(kind|axis|add)\s*:\s*(.*)$", line, re.I)
        if mm:
            out[mm.group(1).lower()] = mm.group(2).strip().strip("\"'` ").rstrip(".")
    if out["kind"] not in ("axis", "exclusion", "constraint", "complaint", "mixed"):
        return None
    if out["axis"].lower() in ("none", "empty", "-", "n/a"):
        out["axis"] = ""
    if out["add"].lower() in ("none", "empty", "-", "n/a"):
        out["add"] = ""
    return out


def route_rephrase(p):
    """Classify and restate a typed axis request (text only, temperature 0)."""
    base_prompt = (p.get("base_prompt") or "").strip()
    axis = (p.get("axis") or "").strip()
    if not axis:
        raise HttpError(400, "axis is required")
    if _STATE["fake"]:
        words = axis.split()
        fake = (
            {"kind": "exclusion", "axis": " ".join(words[-2:]), "add": ""}
            if axis.lower().startswith(("no ", "make ", "don't"))
            else {"kind": "axis", "axis": " ".join(words[-3:]), "add": ""}
        )
        return dict(fake, raw="(fake)", fake=True)
    with _GPU_LOCK:
        raw, found = "", None
        for _ in range(2):
            raw = _ask(_rephrase_messages(base_prompt, axis), max_new_tokens=80, temperature=0.0)
            found = _parse_rephrase(raw)
            if found is not None:
                break
    if found is None:
        raise HttpError(502, f"VLM reply unparseable: {raw[:200]}")
    return dict(found, raw=raw, model=_STATE["model_id"])


def route_suggest(p):
    base_prompt = (p.get("base_prompt") or "").strip()
    images = p.get("images") or []
    k = int(p.get("k", 4))
    want_values = bool(p.get("values", True))
    exclude = {str(x).strip().lower() for x in (p.get("exclude") or [])}
    if not base_prompt:
        raise HttpError(400, "base_prompt is required")

    if _STATE["fake"]:
        axes = [a for a in FAKE_AXES if a["name"].lower() not in exclude][:k]
        return {"axes": axes, "raw": "(fake)", "fake": True}

    missing = [fp for fp in images if not os.path.exists(fp)]
    if missing:
        raise HttpError(400, f"images not found: {missing[:3]}")
    if not images:
        raise HttpError(
            400,
            "suggestion needs the i.i.d. images -- the diagnostic "
            "framing is about what the four renders have in common, and "
            "prompt text alone does not carry that (suggest.SYSTEM)",
        )

    retries = int(p.get("retries", 3))
    # `exclude` lists rejected axes. Ask for k + len(exclude) so they can be dropped.
    ask_k = k + len(exclude)
    content = [{"type": "image", "image": f"file://{os.path.abspath(fp)}"} for fp in images]
    content.append({"type": "text", "text": suggest.USER_TMPL.format(prompt=base_prompt)})
    messages = [
        {"role": "system", "content": suggest.SYSTEM.format(k=ask_k)},
        {"role": "user", "content": content},
    ]

    with _GPU_LOCK:
        found, raw = None, ""
        for _ in range(retries + 1):
            raw = _ask(messages, max_new_tokens=400, temperature=float(p.get("temperature", 0.7)))
            found = suggest.parse(raw)
            if found is not None:
                break
        if found is None:
            raise HttpError(502, f"VLM reply unparseable after {retries + 1} tries: {raw[:300]}")

        axes = [
            {"name": f["axis"], "why": f["observed"], "example_values": []}
            for f in found
            if f["axis"].strip().lower() not in exclude
        ][:k]
        if want_values:
            for a in axes:
                try:
                    a["example_values"] = _example_values(base_prompt, a["name"])
                except Exception as e:  # decoration must never fail the suggestion
                    print(f"[suggest] values failed for {a['name']!r}: {e}", flush=True)

    return {"axes": axes, "raw": raw, "model": _STATE["model_id"]}


def route_health(_p):
    return {
        "status": "ok",
        "ready": _STATE.get("ready", False),
        "fake": _STATE.get("fake"),
        "model": _STATE.get("model_id"),
        "loaded": "model" in _STATE,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19103)
    ap.add_argument("--model", default=config.SUGGEST_VLM)
    ap.add_argument(
        "--preload",
        action="store_true",
        help="load the VLM at startup instead of on the first request",
    )
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    _STATE["fake"] = a.fake
    _STATE["model_id"] = a.model
    if a.preload and not a.fake:
        _vlm()
    _STATE["ready"] = True

    serve(
        {
            ("GET", "/health"): route_health,
            ("POST", "/suggest"): route_suggest,
            ("POST", "/rephrase"): route_rephrase,
        },
        a.host,
        a.port,
        "suggest",
        ready_msg=f"(fake={a.fake})",
    )


if __name__ == "__main__":
    main()
