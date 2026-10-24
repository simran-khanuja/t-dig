"""Policy server: the GRPO-trained prompt-set expander behind HTTP.

    POST /expand      {base_prompt, axis, k, arm, off_axis, exp, model, images}
                                                          -> {prompts: [K], raw}
    POST /expand_many {items: [{base_prompt, axis}], k, arm, off_axis, exp, model}
                                                          -> {results: [...]}
    POST /edit_plan   {base_prompt, axis_a, axis_b, source_prompts, source_images, arm,
                       off_axis, exp, model}              -> {instructions: [K], raw}
    GET  /loras                                          -> the off-axis ladder
    GET  /health

vLLM loads the base model once and serves the GRPO checkpoint as a LoRA adapter, not a
merge, so checkpoints hot-swap and arm B ("baseline") is the same process with the adapter
off. Both arms then share weights, sampler, tokenizer and chat template; they differ only in
the LoRA and the instruction template.

Off-axis ladder (config.off_axis_ladder): each rung is one checkpoint of a single GRPO run.
All rungs stay in host memory (`max_cpu_loras` = number of rungs), so a slider move is an
adapter swap, not a read from disk. `max_loras = 1` (adapters per batch) is enough because
`_GPU_LOCK` serialises generation. A rung whose adapter is missing is an error; the server
never answers from a neighbouring checkpoint.

Vision policies (profiles.policy_is_vlm) get images with every request, built into model
input the way verl builds a training row: split the content on `<image>`, render with the
processor's chat template, load images with `qwen_vl_utils.process_vision_info`. Round one
is shown the i.i.d. renders; a layered round is shown the round-one set it edits.

/edit_plan serves round-two policies (backend/layered.py): given the round-one prompts
and images, write one edit instruction per image for the render server's editor. Its ladder
(config.layer_exp_for) shares the vLLM process with round one's.

Completions are parsed with `rollout.parse_prompts`, as during training. A completion that
does not parse to exactly K items is resampled up to `retries` times (request field, default
config.EXPAND_RETRIES), then replaced by a templated fallback flagged `degraded: true` so
analysis can drop it.

    $PY_TRAIN \\
        axis_framework/servers/policy_server.py --port 19101 --lora axis_grpo/ckpt/<EXP>
    ... --fake                       # no GPU: templated prompts, for driving the loop
"""

import argparse
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axis_framework import config, policy_prompt  # noqa: E402
from axis_framework.http_util import HttpError, serve  # noqa: E402

_STATE = {}
# Chat-template kwargs used in training: enable_thinking from $ENABLE_THINKING (default False).
# Ignored by templates that do not reference it.
_TEMPLATE_KW = {"enable_thinking": os.environ.get("ENABLE_THINKING", "False").lower() == "true"}
_GPU_LOCK = threading.Lock()


def _served_models():
    from axis_framework import profiles

    return list(profiles.served())


def _wording(exp):
    """The image-preamble wording run *exp* trained with ("named" if unknown)."""
    if not exp:
        return "named"
    from axis_framework import profiles

    try:
        return profiles.run_info(exp).get("image_axis_wording") or "named"
    except Exception:
        return "named"


def _as_shown(paths, exp):
    """Load source images as PIL images, resized to run *exp*'s training resolution.

    A live grid can be larger than the resolution the layered policy trained on (512 for the
    FLUX.2-Klein runs); resizing keeps the policy's image-token count on-distribution.
    """
    from PIL import Image
    from axis_framework import profiles

    res = profiles.run_info(exp)["train"]["resolution"] if exp else None
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if res and im.size != (res, res):
            im = im.resize((res, res), Image.LANCZOS)
        out.append(im)
    return out


def resolve_lora(path):
    """Resolve a verl checkpoint dir to the adapter directory vLLM can load.

    Accepts:
      axis_grpo/ckpt/<EXP>                     -> latest global_step_N -> actor/lora_adapter
      axis_grpo/ckpt/<EXP>/global_step_N       -> actor/lora_adapter
      .../lora_adapter (or any dir with adapter_config.json) -> itself

    `latest_checkpointed_iteration.txt` is preferred over the directory listing, since a
    `global_step_N` dir can be half-written after a preemption.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"--lora {path} does not exist")

    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path

    latest = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest):
        step = open(latest).read().strip()
        cand = os.path.join(path, f"global_step_{step}")
        if os.path.isdir(cand):
            return resolve_lora(cand)

    steps = sorted(
        (d for d in os.listdir(path) if re.fullmatch(r"global_step_\d+", d)),
        key=lambda d: int(d.split("_")[-1]),
    )
    if steps:
        return resolve_lora(os.path.join(path, steps[-1]))

    for sub in ("actor/lora_adapter", "lora_adapter", "actor/huggingface", "actor"):
        cand = os.path.join(path, sub)
        if os.path.exists(os.path.join(cand, "adapter_config.json")):
            return cand
    raise FileNotFoundError(
        f"no LoRA adapter under {path}. Expected either adapter_config.json here, or a "
        f"global_step_N/actor/lora_adapter (what verl writes with "
        f"trainer.default_local_dir). Serve the untuned base model by omitting --lora."
    )


def build_ladder(rungs, first_id=2):
    """Attach a LoRARequest to each config rung, marking rungs with no adapter unavailable.

    Ids start at 2 because id 1 is the `--lora` adapter. vLLM keys its adapter cache on the
    int id, so two adapters sharing one id would silently serve whichever came first.
    """
    from vllm.lora.request import LoRARequest

    out = []
    for i, r in enumerate(rungs):
        rung = dict(r)
        if not r["available"] or r.get("base"):
            # A `base` rung is the untrained policy: served, with no adapter (_use_lora).
            rung["lora_request"] = None
            out.append(rung)
            continue
        try:
            path = resolve_lora(r["path"])
        except FileNotFoundError as e:
            print(f"[policy] rung step={r['step']} unusable: {e}", flush=True)
            rung["available"] = False
            rung["why"] = str(e)
            rung["lora_request"] = None
            out.append(rung)
            continue
        rung["path"] = path
        rung["lora_request"] = LoRARequest(r["exp"], first_id + i, path)
        out.append(rung)
    return out


def pick_rung(ladder, off_axis):
    """Return the rung nearest to *off_axis*, ties to the lower rung.

    Rungs are distinct checkpoints, so there is no interpolation. The UI snaps the slider to
    the same stops.
    """
    if not ladder:
        return None
    return min(ladder, key=lambda r: (abs(r["off_axis"] - off_axis), r["off_axis"]))


def _ladder_for(exp=None):
    """Return the rungs for experiment *exp*, or the default ladder when *exp* is empty.

    An unknown *exp* is a 503, not a fallback to the default ladder.
    """
    ladders = _STATE.get("ladders") or {}
    if not exp:
        return _STATE.get("ladder") or []
    if exp not in ladders:
        raise HttpError(
            503,
            f"no ladder for experiment {exp!r}. Loaded: "
            f"{sorted(ladders)}. Export it with "
            f"axis_framework/scripts/export_ladder.sh, or point the session at a "
            f"T2I model whose ladder is loaded.",
        )
    return ladders[exp]


def _public_ladder(exp=None):
    return [{k: v for k, v in r.items() if k != "lora_request"} for r in _ladder_for(exp)]


def route_loras(p):
    """Return the ladder for one experiment, or every loaded ladder when none is named."""
    exp = (p or {}).get("exp")
    if exp:
        lam, tau = config.off_axis_reward(exp)
        return {
            "ladder": _public_ladder(exp),
            "exp": exp,
            "lam": lam,
            "tau": tau,
            "default_off_axis": config.OFF_AXIS_DEFAULT,
            "lora": _STATE.get("lora_path"),
        }
    out = {}
    for e in sorted(_STATE.get("ladders") or {}):
        lam, tau = config.off_axis_reward(e)
        out[e] = {"ladder": _public_ladder(e), "lam": lam, "tau": tau}
    lam, tau = config.off_axis_reward(_STATE.get("default_exp"))
    return {
        "ladders": out,
        "ladder": _public_ladder(),
        "exp": _STATE.get("default_exp"),
        "lam": lam,
        "tau": tau,
        "default_off_axis": config.OFF_AXIS_DEFAULT,
        "lora": _STATE.get("lora_path"),
    }


def _sampling(max_tokens, temperature, top_p=None, top_k=None, seed=None):
    """Build SamplingParams for one request.

    Defaults top_p=1.0, top_k=-1 match verl's training rollouts (with temperature=1.0), so
    a served completion comes from the distribution the checkpoint was trained on.
    """
    from vllm import SamplingParams

    return SamplingParams(
        temperature=temperature,
        top_p=1.0 if top_p is None else top_p,
        top_k=-1 if top_k is None else top_k,
        max_tokens=max_tokens,
        seed=seed,
    )


def _model_input(messages, images=None):
    """Build one vLLM prompt: {"prompt": text}, plus "multi_modal_data" when given images.

    Text-only uses the tokenizer's chat template. With images it follows verl's multimodal
    path (rl_dataset._build_messages, the processor's chat template, then
    qwen_vl_utils.process_vision_info at the processor's patch size).
    """
    if not images:
        text = _STATE["tok"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **_TEMPLATE_KW
        )
        return {"prompt": text}
    import re as _re
    from qwen_vl_utils import process_vision_info

    images = list(images)
    msgs, used = [], 0
    for m in messages:
        parts = []
        for seg in (x for x in _re.split("(<image>)", m["content"]) if x != ""):
            if seg == "<image>":
                if used >= len(images):
                    raise HttpError(
                        400,
                        f"instruction has more <image> placeholders than "
                        f"the {len(images)} images sent",
                    )
                img = images[used]
                # A path or a PIL image (_as_shown); qwen_vl_utils takes either.
                parts.append(
                    {"type": "image", "image": img if hasattr(img, "size") else os.fspath(img)}
                )
                used += 1
            else:
                parts.append({"type": "text", "text": seg})
        msgs.append({"role": m["role"], "content": parts})
    if used != len(images):
        raise HttpError(400, f"{len(images)} images sent for {used} <image> placeholders")
    proc = _STATE["processor"]
    text = proc.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, **_TEMPLATE_KW
    )
    patch = getattr(getattr(proc, "image_processor", None), "patch_size", 14)
    imgs, _ = process_vision_info(msgs, image_patch_size=patch)
    return {"prompt": text, "multi_modal_data": {"image": imgs}}


def _generate_inputs(
    inputs, max_tokens, temperature, use_lora, lora_request=None, top_p=None, top_k=None, seeds=None
):
    """Generate completions for prebuilt model inputs in one vLLM call.

    `seeds` has one entry per input (None = unseeded). The orchestrator reuses the seed when
    only the diversity slider moved, so the two sets differ by adapter, not by random draw.

    Returns:
        [(text, finish_reason)] in input order.
    """
    if seeds and any(x is not None for x in seeds):
        sp = [
            _sampling(max_tokens, temperature, top_p, top_k, seed=None if x is None else int(x))
            for x in seeds
        ]
    else:
        sp = _sampling(max_tokens, temperature, top_p, top_k)
    kw = {}
    if use_lora:
        req = lora_request if lora_request is not None else _STATE.get("lora_request")
        if req is not None:
            kw["lora_request"] = req
    with _GPU_LOCK:
        outs = _STATE["llm"].generate(inputs, sp, **kw)
    return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]


def route_expand(p):
    base_prompt = (p.get("base_prompt") or "").strip()
    axis = (p.get("axis") or "").strip()
    seed = None if p.get("seed") is None else int(p["seed"])
    if not base_prompt:
        raise HttpError(400, "base_prompt is required")
    if not axis:
        raise HttpError(400, "axis is required")
    arm = p.get("arm", "ours")
    if arm not in ("ours", "baseline"):
        raise HttpError(400, f"unknown arm {arm!r}")
    k = int(p.get("k", config.K))
    temperature = float(p.get("temperature", config.TEMPERATURE))
    top_p = None if p.get("top_p") is None else float(p["top_p"])
    top_k = None if p.get("top_k") is None else int(p["top_k"])
    max_tokens = int(p.get("max_tokens", config.MAX_TOKENS))
    retries = int(p.get("retries", config.EXPAND_RETRIES))
    # Optional; see policy_prompt.build_instruction for why an untrained guidance line hurts.
    guidance = p.get("guidance")
    model = p.get("model")
    images = list(p.get("images") or [])
    # Image input is a property of the answering run, not the base model: one multimodal
    # base can serve a text-input round-one run and an image-input editor.
    run_vlm = config.run_is_vlm(p.get("exp") or _STATE.get("default_exp"))
    if run_vlm and not _STATE["fake"] and not images:
        raise HttpError(
            400,
            "this is a vision policy: /expand needs `images` (the i.i.d. "
            "renders it was trained to look at)",
        )
    if images and not run_vlm:
        raise HttpError(400, "this is a text policy; it cannot be sent images")

    # off_axis selects the rung (adapter); the reply reports the rung that ran. Absent, the
    # `--lora` adapter answers.
    off_axis = p.get("off_axis")
    rung, off_axis_note = None, None
    if off_axis is not None:
        if arm == "baseline":
            # Arm B has no adapter, so off_axis is dropped and the reply says so.
            off_axis_note = "arm 'baseline' runs the untuned base model, which has no lam"
            off_axis = None
        else:
            ladder = _ladder_for(p.get("exp"))
            if not ladder:
                raise HttpError(
                    503,
                    "this policy server has no off-axis ladder: no adapters under "
                    f"{os.path.join(config.FRAMEWORK, 'adapters')} match "
                    f"{p.get('exp') or _STATE.get('default_exp')}_step*. Run "
                    f"axis_framework/scripts/export_ladder.sh.",
                )
            rung = pick_rung(ladder, float(off_axis))
            if not _STATE["fake"] and rung.get("lora_request") is None and not rung.get("base"):
                have = [r["step"] for r in ladder if r["lora_request"] is not None]
                raise HttpError(
                    503,
                    f"off_axis={off_axis} selects step {rung['step']}, whose adapter is not "
                    f"exported ({rung['path']}). Exported rungs: {have}. Refusing to answer "
                    f"from a neighbouring checkpoint -- that would be a different policy "
                    f"than the one asked for, and nothing downstream could tell.",
                )
            off_axis = rung["off_axis"]

    messages = policy_prompt.build_messages(
        base_prompt,
        axis,
        k=k,
        arm=arm,
        guidance=guidance,
        model=model,
        n_images=len(images),
        image_wording=_wording(p.get("exp")),
        exp=p.get("exp") or _STATE.get("default_exp"),
    )
    axis_factor = policy_prompt.resolve_axis(axis)["factor"]
    t0 = time.time()

    if _STATE["fake"]:
        prompts = policy_prompt.naive_expansion(base_prompt, axis, k)
        return {
            "prompts": prompts,
            "raw": "(fake)",
            "degraded": False,
            "attempts": 0,
            "arm": arm,
            "axis": axis,
            "axis_factor": axis_factor,
            "k": k,
            "fake": True,
            "off_axis": off_axis,
            "step": (rung or {}).get("step"),
            "rung": (rung or {}).get("label"),
            "band": (rung or {}).get("band"),
            "off_axis_note": off_axis_note,
            "instruction": messages[0]["content"],
            "seconds": 0.0,
        }

    parse_prompts = _STATE["parse_prompts"]
    raw, reason, prompts = "", "", []
    attempts = 0
    for attempts in range(1, retries + 2):
        # Arm B runs with the adapter off. Each retry advances the seed.
        raw, reason = _generate_inputs(
            [_model_input(messages, images)],
            max_tokens,
            temperature,
            use_lora=_use_lora(arm, rung),
            lora_request=(rung or {}).get("lora_request"),
            top_p=top_p,
            top_k=top_k,
            seeds=[None if seed is None else seed + attempts - 1],
        )[0]
        prompts = parse_prompts(raw)
        if len(prompts) == k:
            break

    degraded = len(prompts) != k
    if degraded:
        print(
            f"[policy] degraded after {attempts} attempts: parsed {len(prompts)}/{k} "
            f"(finish={reason}) for axis={axis!r}",
            flush=True,
        )
        prompts = policy_prompt.naive_expansion(base_prompt, axis, k)

    return {
        "prompts": prompts,
        "raw": raw,
        "degraded": degraded,
        "attempts": attempts,
        "finish_reason": reason,
        "arm": arm,
        "axis": axis,
        "k": k,
        # The phrase the audit conditions TPIPS on: for a study-vocabulary key, its
        # canonical tpips_factor.
        "axis_factor": axis_factor,
        # The resolved rung, not the requested value; the event log keys on it.
        "off_axis": off_axis,
        "step": (rung or {}).get("step"),
        "rung": (rung or {}).get("label"),
        "band": (rung or {}).get("band"),
        "off_axis_note": off_axis_note,
        "lora": (
            None
            if (rung or {}).get("base")
            else ((rung or {}).get("path") or _STATE.get("lora_path"))
        ),
        "model": _STATE.get("base_model"),
        "instruction": messages[0]["content"],
        "temperature": temperature,
        "seed": seed,
        **({} if top_p is None else {"top_p": top_p}),
        **({} if top_k is None else {"top_k": top_k}),
        "seconds": round(time.time() - t0, 2),
    }


def _use_lora(arm, rung):
    """Return True when the request should use an adapter: arm "ours" on a non-base rung.

    A `base` rung (step 0) is the untrained model under the trained instruction; it runs with
    no adapter, not with `--lora`.
    """
    return arm == "ours" and not (rung or {}).get("base")


def _resolve_off_axis(off_axis, arm, exp=None):
    """Resolve off_axis to (rung, note, resolved_off_axis) on *exp*'s ladder.

    Used by /expand_many and /edit_plan.
    """
    if off_axis is None:
        return None, None, None
    if arm == "baseline":
        return None, "arm 'baseline' runs the untuned base model, which has no lam", None
    ladder = _ladder_for(exp) if exp else (_STATE.get("ladder") or [])
    if not ladder:
        raise HttpError(
            503,
            "this policy server has no off-axis ladder: no adapters under "
            f"{os.path.join(config.FRAMEWORK, 'adapters')} match "
            f"{exp or _STATE.get('default_exp')}_step*. Run axis_framework/scripts/export_ladder.sh.",
        )
    rung = pick_rung(ladder, float(off_axis))
    if not _STATE["fake"] and rung.get("lora_request") is None and not rung.get("base"):
        have = [r["step"] for r in ladder if r["lora_request"] is not None]
        raise HttpError(
            503,
            f"off_axis={off_axis} selects step {rung['step']}, whose adapter is not "
            f"exported ({rung['path']}). Exported rungs: {have}. Refusing to answer "
            f"from a neighbouring checkpoint -- that would be a different policy than "
            f"the one asked for, and nothing downstream could tell.",
        )
    return rung, None, rung["off_axis"]


def route_expand_many(p):
    """Run N independent /expand requests in one vLLM batch.

    Each item gets its own K prompts and `degraded` flag; the rung and sampling params are
    shared. Items that fail to parse are retried together in the next batch.
    """
    items = p.get("items") or []
    if not items:
        raise HttpError(400, "items is required")
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise HttpError(400, "items must be a list of {base_prompt, axis}")
    arm = p.get("arm", "ours")
    if arm not in ("ours", "baseline"):
        raise HttpError(400, f"unknown arm {arm!r}")
    k = int(p.get("k", config.K))
    temperature = float(p.get("temperature", config.TEMPERATURE))
    top_p = None if p.get("top_p") is None else float(p["top_p"])
    top_k = None if p.get("top_k") is None else int(p["top_k"])
    max_tokens = int(p.get("max_tokens", config.MAX_TOKENS))
    retries = int(p.get("retries", config.EXPAND_RETRIES))
    guidance = p.get("guidance")

    run_vlm = config.run_is_vlm(p.get("exp") or _STATE.get("default_exp"))
    specs, item_images = [], []
    # Per-item seed, else the batch seed + item index (layered rounds: one draw per source).
    item_seeds = [
        (
            int(it["seed"])
            if it.get("seed") is not None
            else (None if p.get("seed") is None else int(p["seed"]) + i)
        )
        for i, it in enumerate(items)
    ]
    for i, it in enumerate(items):
        bp = (it.get("base_prompt") or "").strip()
        ax = (it.get("axis") or p.get("axis") or "").strip()
        if not bp or not ax:
            raise HttpError(400, f"items[{i}] needs base_prompt and axis")
        imgs = list(it.get("images") or p.get("images") or [])
        if run_vlm and not _STATE["fake"] and not imgs:
            raise HttpError(400, f"items[{i}]: a vision policy needs images")
        specs.append((bp, ax))
        item_images.append(imgs if run_vlm else [])

    rung, note, off_axis = _resolve_off_axis(p.get("off_axis"), arm, p.get("exp"))
    msgs = [
        policy_prompt.build_messages(
            bp,
            ax,
            k=k,
            arm=arm,
            guidance=guidance,
            model=p.get("model"),
            n_images=len(item_images[i]),
            image_wording=_wording(p.get("exp")),
            exp=p.get("exp") or _STATE.get("default_exp"),
        )
        for i, (bp, ax) in enumerate(specs)
    ]
    t0 = time.time()

    if _STATE["fake"]:
        out = [
            {
                "prompts": policy_prompt.naive_expansion(bp, ax, k),
                "raw": "(fake)",
                "degraded": False,
                "axis": ax,
                "base_prompt": bp,
                "axis_factor": policy_prompt.resolve_axis(ax)["factor"],
            }
            for bp, ax in specs
        ]
        return {
            "results": out,
            "fake": True,
            "k": k,
            "arm": arm,
            "off_axis": off_axis,
            "step": (rung or {}).get("step"),
            "rung": (rung or {}).get("label"),
            "band": (rung or {}).get("band"),
            "off_axis_note": note,
            "seconds": 0.0,
        }

    parse_prompts = _STATE["parse_prompts"]
    results = [{"prompts": [], "raw": "", "finish_reason": None} for _ in specs]
    todo = list(range(len(specs)))
    attempts = 0
    lr = (rung or {}).get("lora_request")
    for attempts in range(1, retries + 2):
        gen = _generate_inputs(
            [_model_input(msgs[i], item_images[i]) for i in todo],
            max_tokens,
            temperature,
            use_lora=_use_lora(arm, rung),
            lora_request=lr,
            top_p=top_p,
            top_k=top_k,
            seeds=[None if item_seeds[i] is None else item_seeds[i] + attempts - 1 for i in todo],
        )
        nxt = []
        for i, (raw, reason) in zip(todo, gen):
            results[i]["raw"] = raw
            results[i]["finish_reason"] = reason
            got = parse_prompts(raw)
            if len(got) == k:
                results[i]["prompts"] = got
            else:
                nxt.append(i)
        todo = nxt
        if not todo:
            break

    out = []
    for i, (bp, ax) in enumerate(specs):
        degraded = len(results[i]["prompts"]) != k
        if degraded:
            print(f"[policy] batch item {i} degraded after {attempts} attempts", flush=True)
            results[i]["prompts"] = policy_prompt.naive_expansion(bp, ax, k)
        out.append(
            {
                "prompts": results[i]["prompts"],
                "raw": results[i]["raw"],
                "degraded": degraded,
                "base_prompt": bp,
                "axis": ax,
                "finish_reason": results[i]["finish_reason"],
                "axis_factor": policy_prompt.resolve_axis(ax)["factor"],
            }
        )

    return {
        "results": out,
        "k": k,
        "arm": arm,
        "attempts": attempts,
        "off_axis": off_axis,
        "step": (rung or {}).get("step"),
        "rung": (rung or {}).get("label"),
        "band": (rung or {}).get("band"),
        "off_axis_note": note,
        "lora": (rung or {}).get("path"),
        "model": _STATE.get("base_model"),
        "n": len(specs),
        "seconds": round(time.time() - t0, 2),
    }


def route_edit_plan(p):
    """Write K edit instructions for an existing set (the round-two policy's task).

    Takes the round-one prompts and images, the axis they varied (A) and the axis to add (B).
    A vision policy sees images interleaved with prompts, as in its training rows.
    Retries and the `degraded` fallback work as in /expand.
    """
    base_prompt = (p.get("base_prompt") or "").strip()
    axis_a = (p.get("axis_a") or "").strip()
    axis_b = (p.get("axis_b") or "").strip()
    seed = None if p.get("seed") is None else int(p["seed"])
    srcp = list(p.get("source_prompts") or [])
    srci = list(p.get("source_images") or [])
    if not base_prompt or not axis_a or not axis_b:
        raise HttpError(400, "base_prompt, axis_a and axis_b are required")
    if not srcp:
        raise HttpError(400, "source_prompts is required")
    vlm = config.run_is_vlm(p.get("exp")) if p.get("exp") else config.policy_is_vlm()
    if vlm and len(srci) != len(srcp):
        raise HttpError(
            400, f"vision policy: {len(srci)} source_images for " f"{len(srcp)} source_prompts"
        )
    arm = p.get("arm", "ours")
    if arm not in ("ours", "baseline"):
        raise HttpError(400, f"unknown arm {arm!r}")
    exp = p.get("exp")
    if not exp:
        raise HttpError(400, "exp is required: name the round-two ladder to answer from")
    k = len(srcp)
    temperature = float(p.get("temperature", config.TEMPERATURE))
    top_p = None if p.get("top_p") is None else float(p["top_p"])
    top_k = None if p.get("top_k") is None else int(p["top_k"])
    max_tokens = int(p.get("max_tokens", config.MAX_TOKENS))
    retries = int(p.get("retries", config.EXPAND_RETRIES))

    rung, note, off_axis = _resolve_off_axis(p.get("off_axis"), arm, exp)
    try:
        messages = policy_prompt.build_layer_messages(
            base_prompt, axis_a, axis_b, srcp, model=p.get("model"), vlm=vlm, exp=exp
        )
    except ValueError as e:
        raise HttpError(400, str(e))
    common = {
        "arm": arm,
        "axis_a": axis_a,
        "axis_b": axis_b,
        "k": k,
        "exp": exp,
        "off_axis": off_axis,
        "step": (rung or {}).get("step"),
        "rung": (rung or {}).get("label"),
        "band": (rung or {}).get("band"),
        "off_axis_note": note,
        "lora": (rung or {}).get("path"),
        "model": _STATE.get("base_model"),
        "instruction": messages[0]["content"],
    }
    t0 = time.time()
    if _STATE["fake"]:
        return {
            "instructions": policy_prompt.naive_edit_instructions(axis_b, k),
            "raw": "(fake)",
            "degraded": False,
            "attempts": 0,
            "fake": True,
            "seconds": 0.0,
            **common,
        }

    inp = _model_input(messages, _as_shown(srci, exp) if vlm else None)
    parse_prompts = _STATE["parse_prompts"]
    raw, reason, got, attempts = "", "", [], 0
    for attempts in range(1, retries + 2):
        raw, reason = _generate_inputs(
            [inp],
            max_tokens,
            temperature,
            use_lora=_use_lora(arm, rung),
            lora_request=(rung or {}).get("lora_request"),
            top_p=top_p,
            top_k=top_k,
            seeds=[None if seed is None else seed + attempts - 1],
        )[0]
        got = parse_prompts(raw)
        if len(got) == k:
            break
    degraded = len(got) != k
    if degraded:
        print(
            f"[policy] edit_plan degraded after {attempts} attempts: parsed "
            f"{len(got)}/{k} (finish={reason})",
            flush=True,
        )
        got = policy_prompt.naive_edit_instructions(axis_b, k)
    return {
        "instructions": got,
        "raw": raw,
        "degraded": degraded,
        "attempts": attempts,
        "finish_reason": reason,
        "temperature": temperature,
        "seconds": round(time.time() - t0, 2),
        **common,
    }


def route_health(_p):
    ladder = _public_ladder()
    return {
        "status": "ok",
        "ready": _STATE.get("ready", False),
        "fake": _STATE.get("fake"),
        "model": _STATE.get("base_model"),
        "lora": _STATE.get("lora_path"),
        "exp": _STATE.get("default_exp"),
        "ladders": {
            e: [r["step"] for r in v if r["available"]]
            for e, v in (_STATE.get("ladders") or {}).items()
        },
        "ladder": [r["step"] for r in ladder if r["available"]],
        "ladder_missing": [r["step"] for r in ladder if not r["available"]],
        "vlm": config.policy_is_vlm(),
        "layer_exps": sorted(e for e in (config.layer_exp_for(m) for m in _served_models()) if e),
        "profile": _STATE.get("profile"),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19101)
    ap.add_argument(
        "--model", default=None, help="base model; default: the one every served run was trained on"
    )
    ap.add_argument(
        "--lora",
        default=None,
        help="verl checkpoint dir (axis_grpo/ckpt/<EXP>) or an adapter dir. Omitted, the "
        "untuned base is served, so arm A equals arm B; /health reports this.",
    )
    ap.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="default: the largest lora_rank among the served runs",
    )
    ap.add_argument(
        "--cudagraphs",
        action="store_true",
        help="capture CUDA graphs instead of running eager: faster decoding, at the cost of "
        "startup time and GPU memory. Off by default because this server shares a GPU "
        "with the T2I model and the suggest VLM.",
    )
    ap.add_argument(
        "--no-ladder",
        action="store_true",
        help="ignore adapters/ and serve only --lora; /expand then refuses off_axis",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="default: the largest max_model_len among the served runs",
    )
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument(
        "--fake",
        action="store_true",
        help="no GPU: templated prompts, for driving the loop end to end",
    )
    a = ap.parse_args()

    from axis_framework import profiles

    if not a.fake:
        profiles.require_served()
    a.model = a.model or profiles.policy_base()
    a.lora_rank = a.lora_rank or profiles.policy_lora_rank()
    a.max_model_len = a.max_model_len or profiles.policy_max_model_len()
    _STATE["profile"] = {
        "base": a.model,
        "lora_rank": a.lora_rank,
        "max_model_len": a.max_model_len,
        "vlm": config.policy_is_vlm(),
        "max_images": config.policy_max_images(),
        "serve": profiles.served(),
    }
    print(f"[policy] profile from served runs: {_STATE['profile']}", flush=True)

    _STATE["fake"] = a.fake
    _STATE["base_model"] = a.model
    # One ladder per served T2I model, all as adapters on one vLLM engine.
    served = _served_models()
    default_exp = config.off_axis_exp_for(served[0]) if served else ""
    _STATE["default_exp"] = default_exp
    exps = (
        []
        if a.no_ladder
        else sorted(
            {e for m in served for e in (config.off_axis_exp_for(m), config.layer_exp_for(m)) if e}
        )
    )
    ladders = {e: config.off_axis_ladder(e) for e in exps}
    ladders = {e: r for e, r in ladders.items() if r}
    rungs = ladders.get(default_exp) or (next(iter(ladders.values())) if ladders else [])
    if a.fake:
        # Fake mode still advertises the rungs so the slider and event log work without weights.
        _STATE["ladders"] = {e: [dict(r, lora_request=None) for r in v] for e, v in ladders.items()}
        _STATE["ladder"] = [dict(r, lora_request=None) for r in rungs]
    if not a.fake:
        from axis_framework.backend.rollout import parse_prompts

        _STATE["parse_prompts"] = parse_prompts

        from transformers import AutoTokenizer
        from vllm import LLM

        lora_path = resolve_lora(a.lora) if a.lora else None
        _STATE["lora_path"] = lora_path
        if lora_path is None:
            print(
                "[policy] WARNING: no --lora. Arm A is the UNTUNED base model, so an "
                "A/B run would compare the baseline against itself.",
                flush=True,
            )

        n_rungs = sum(1 for v in ladders.values() for r in v if r["available"])
        # Keep every adapter in host memory (+1 for --lora). vLLM requires
        # max_cpu_loras >= max_loras, hence the floor of 2.
        n_cpu = max(2, n_rungs + (1 if lora_path else 0))

        t0 = time.time()
        _STATE["tok"] = AutoTokenizer.from_pretrained(a.model)
        mm = {}
        if config.policy_is_vlm():
            from transformers import AutoProcessor

            _STATE["processor"] = AutoProcessor.from_pretrained(a.model)
            # vLLM allows one image per prompt by default (training sets this via LIMIT_IMAGES).
            mm = {"limit_mm_per_prompt": {"image": config.policy_max_images()}}
        _STATE["llm"] = LLM(
            model=a.model,
            gpu_memory_utilization=a.gpu_util,
            max_model_len=a.max_model_len,
            tensor_parallel_size=1,
            enforce_eager=not a.cudagraphs,
            enable_lora=(lora_path is not None or n_rungs > 0),
            max_lora_rank=a.lora_rank,
            max_loras=1,
            max_cpu_loras=n_cpu,
            **mm,
        )
        if lora_path is not None:
            from vllm.lora.request import LoRARequest

            _STATE["lora_request"] = LoRARequest("axis_grpo", 1, lora_path)
        # Adapter ids must be unique across all ladders (vLLM caches adapters by id).
        _STATE["ladders"] = {}
        next_id = 2
        for e, v in ladders.items():
            built = build_ladder(v, first_id=next_id)
            _STATE["ladders"][e] = built
            next_id += len(v)
        _STATE["ladder"] = _STATE["ladders"].get(default_exp) or (
            next(iter(_STATE["ladders"].values())) if _STATE["ladders"] else []
        )
        ok = [r["step"] for r in _STATE["ladder"] if r.get("lora_request") is not None]
        missing = [
            r["step"]
            for r in _STATE["ladder"]
            if r.get("lora_request") is None and not r.get("base")
        ]
        print(
            f"[policy] up in {time.time() - t0:.0f}s (lora={lora_path}, ladder "
            f"{default_exp}: {len(ok)} steps {ok[:3]}..{ok[-3:] if ok else []}"
            + (f", missing {missing}" if missing else "")
            + ")",
            flush=True,
        )
    if not a.fake and config.policy_is_vlm():
        # Warm up the image path before reporting ready: vLLM defers multimodal set-up to the
        # first image request, which is then ~15 s slower than later ones.
        try:
            from PIL import Image

            t1 = time.time()
            n = max(1, min(config.policy_max_images(), config.K_TRAINED))
            blank = [Image.new("RGB", (512, 512), (127, 127, 127)) for _ in range(n)]
            msg = [
                {
                    "role": "user",
                    "content": "Describe these images in five words.\n"
                    + "\n".join(f"Image {i + 1}: <image>" for i in range(n)),
                }
            ]
            _generate_inputs([_model_input(msg, blank)], 8, 0.0, use_lora=False)
            print(f"[policy] image path warmed up in {time.time() - t1:.1f}s", flush=True)
        except Exception as e:  # a failed warm-up must not stop the server
            print(f"[policy] image warm-up skipped: {type(e).__name__}: {e}", flush=True)
    _STATE["ready"] = True

    serve(
        {
            ("GET", "/health"): route_health,
            ("GET", "/loras"): route_loras,
            ("POST", "/expand"): route_expand,
            ("POST", "/expand_many"): route_expand_many,
            ("POST", "/edit_plan"): route_edit_plan,
        },
        a.host,
        a.port,
        "policy",
        ready_msg=f"(fake={a.fake}, lora={_STATE.get('lora_path')})",
    )


if __name__ == "__main__":
    main()
