"""The prompt the live policy sees, byte-identical to the one GRPO trained on.

A described (`tdig`) run gets its instruction from `tdig/instruction.py`. Any other run gets
the round-one templates of `backend/instructions.py` and `backend/layered.py`, which are the
exact text of its training rows.

Two axis paths, as in the round-one training data:

  * vocabulary axis: one of the seven study keys, with a display name, a `tpips_factor` and
    worked `examples`. Rendered as "Axis (People): the apparent age, ... \n For example ...".
  * free-text axis: the phrase the VLM or the user named, used as its own factor with no
    examples. Rendered as "Axis: <phrase>". This is the common path.

Arm B (the baseline) is defined here too. It asks the same untuned base model for the same K
prompts in the same `<prompt>` format, so the arms differ only in the expander.
"""

import threading

from . import config

_bd = None
_vocab = None
_mod_lock = threading.Lock()


def _mod():
    """Return (`backend.instructions`, its axis vocabulary), loaded once.

    Both are assigned together, under a lock, only after both load. Concurrent requests
    (ThreadingHTTPServer, speculative expansions) would otherwise see a module with no vocab,
    and a failed `load_axis_vocab` (it raises when the training and study axes disagree) would
    be retried on the next call rather than masked.
    """
    global _bd, _vocab
    if _bd is None or _vocab is None:
        with _mod_lock:
            if _bd is None or _vocab is None:
                from axis_framework.backend import instructions as bd

                vocab = bd.load_axis_vocab()
                _bd, _vocab = bd, vocab
    return _bd, _vocab


def resolve_axis(axis):
    """Return {key, name, factor, block, expand_phrase} for free text or a study axis key.

    `factor` is the string the reward conditions on and rule 3 of the instruction quotes; the
    audit conditions TPIPS on it too.
    """
    _, vocab = _mod()
    axis = (axis or "").strip()
    if axis in vocab:
        v = vocab[axis]
        return {
            "key": axis,
            "name": v["name"],
            "factor": v["tpips_factor"],
            "block": f"Axis ({v['name']}): {v['tpips_factor']}\n{v['examples']}",
            "expand_phrase": v["baseline_expand_phrase"],
        }
    # Free text: the phrase is the factor, as in build_rows_elicited, with no examples.
    return {
        "key": None,
        "name": axis,
        "factor": axis,
        "block": f"Axis: {axis}",
        "expand_phrase": axis,
    }


OVERALL_AXIS = "overall"


def _family_template(model):
    """Return the v4 instruction for T2I model *model*'s prompt budget."""
    bd, _ = _mod()
    return bd.INSTRUCTIONS.get("v4_" + config.length_family(model), bd.INSTRUCTION)


def build_instruction(base_prompt, axis, k=None, guidance=None, model=None):
    """Return the policy's user-turn content, from the template its run trained on.

    `model` is the T2I model the session draws with. It selects rule 4's prompt budget
    (config.length_family): FLUX and Qwen-Image runs trained on "150 WORDS ... 512 tokens",
    SD3.5 runs on "60 WORDS ... 77 tokens". Omitted, the SD3 wording is used.

    The "overall" axis uses instructions.INSTRUCTION_OVERALL, as in its training rows,
    whose rules 3 and 5 say "do not collapse" instead of "vary ONLY <axis>".

    `guidance` is passed to `instructions.guidance_block`, which states the T2I model's CFG
    scale. Pass it only if the served checkpoint was trained with --guidance-set; the servers
    leave it off by default.
    """
    bd, _ = _mod()
    k = config.K if k is None else int(k)
    tmpl = _family_template(model)
    if (axis or "").strip().lower() == OVERALL_AXIS:
        from axis_framework.backend import instructions as ao

        # Rule 4 comes from this family's v4 wording, retargeted as in the training rows.
        probe = tmpl.format(
            base_prompt="x",
            axis_block="Axis: x",
            guidance_block=bd.guidance_block(guidance),
            axis_factor="x",
            k=k,
        )
        m = ao.RULE4_RE.search(probe)
        rule4 = m.group(0).replace(*ao.RULE4_TAIL) if m else ao.FALLBACK_RULE4
        return ao.INSTRUCTION_OVERALL.format(
            base_prompt=base_prompt, guidance_block=bd.guidance_block(guidance), k=k, rule4=rule4
        )
    a = resolve_axis(axis)
    return tmpl.format(
        base_prompt=base_prompt,
        axis_block=a["block"],
        guidance_block=bd.guidance_block(guidance),
        axis_factor=a["factor"],
        k=k,
    )


def with_images(content, axis, n, wording="named"):
    """Splice round one's image preamble into a text instruction, as the training rows did.

    The preamble and its insertion point are `instructions.IMAGE_PREAMBLE` / `SPLICE_BEFORE`.
    Returns content
    carrying `n` `<image>` placeholders; policy_server replaces the i-th with the i-th image
    (verl's rl_dataset._build_messages convention).

    Raises:
        ValueError: *content* has no `instructions.SPLICE_BEFORE` marker.
    """
    from axis_framework.backend import instructions as bv

    block = "\n".join(f"Image {i + 1}: <image>" for i in range(n))
    # `wording` is the served run's axis wording: "named" quotes the axis,
    # "generic" says "the Axis". Both placeholders are passed so either preamble version formats.
    axis_ref = "the Axis" if wording == "generic" else '"%s"' % axis
    pre = bv.IMAGE_PREAMBLE.format(k=n, images=block, axis=axis, axis_ref=axis_ref)
    cut = content.find(bv.SPLICE_BEFORE)
    if cut < 0:
        raise ValueError(f"instruction has no {bv.SPLICE_BEFORE!r} to splice the images at")
    return content[:cut] + pre + content[cut:]


# --------------------------------------------------------------------------- arm B

# Baseline instruction: same base model, task, K, output format and "same scene" constraint as
# arm A. It omits the trained instruction's ordered rules, vary-within-a-fixed-detail clause and
# 60-word budget, which were tuned against the reward. It is generic prompt expansion.
BASELINE_INSTRUCTION = """You are helping a user get more variety out of an \
image-generation model.

Base prompt: {base_prompt}

Write {k} expanded versions of this prompt that differ from each other in \
{expand_phrase}. Each one should still describe the same scene as the base prompt.

Output exactly {k} prompts, each wrapped in <prompt></prompt> tags, and nothing else."""


def build_baseline_instruction(base_prompt, axis, k=None):
    k = config.K if k is None else int(k)
    a = resolve_axis(axis)
    return BASELINE_INSTRUCTION.format(
        base_prompt=base_prompt, k=k, expand_phrase=a["expand_phrase"]
    )


def _tdig(exp):
    """Return (descriptor, tdig.instruction module) for a described "tdig" run *exp*, else None."""
    if not exp:
        return None
    from . import profiles

    d = profiles.descriptor(exp)
    if not d or d.get("format") != "tdig":
        return None
    return d, config.import_tdig_instruction(d.get("instruction_root"))


def tdig_word_cap(base_prompt, d):
    """Return rule 4's word cap: base prompt words plus the run's `relative_budget`, or None.

    Same count as `tdig/recap_dataset.py` wrote into training rows (`len(base_prompt.split())
    + budget`).
    """
    budget = d.get("relative_budget")
    return None if budget is None else len(base_prompt.split()) + int(budget)


def build_messages(
    base_prompt,
    axis,
    k=None,
    arm="ours",
    guidance=None,
    model=None,
    n_images=0,
    image_wording="named",
    exp=None,
):
    """Return the chat messages for one expansion request.

    One user turn and no system turn, matching the training rows
    (`prompt: [{role: user, content: INSTRUCTION}]`).

    `n_images` > 0 adds round one's image preamble (a vision policy sees the i.i.d. renders).
    Arm B gets the same images.

    Raises:
        ValueError: *arm* is not "ours" or "baseline".
    """
    fb = _tdig(exp) if arm == "ours" else None
    if fb:
        # A described run: tdig.instruction places its own image context. The axis is passed as
        # written; a study-vocabulary key resolves to its factor.
        d, fi = fb
        ax = (axis or "").strip()
        ax = OVERALL_AXIS if ax.lower() == OVERALL_AXIS else resolve_axis(ax)["factor"]
        content = fi.instruction(
            base_prompt,
            ax,
            config.K if k is None else int(k),
            d["tdig_renderer"],
            images=bool(n_images),
            word_cap=tdig_word_cap(base_prompt, d),
        )
        return [{"role": "user", "content": content}]
    if arm == "baseline":
        content = build_baseline_instruction(base_prompt, axis, k)
    elif arm == "ours":
        content = build_instruction(base_prompt, axis, k, guidance, model=model)
    else:
        raise ValueError(f"unknown arm {arm!r} (expected 'ours' or 'baseline')")
    if n_images:
        content = with_images(content, axis, n_images, image_wording)
    return [{"role": "user", "content": content}]


def build_layer_messages(
    base_prompt, axis_a, axis_b, source_prompts, model=None, vlm=False, exp=None
):
    """Return the chat messages for a layered round, from the `backend.layered` templates.

    The policy gets the round-one prompts (and, for a vision policy, the images) and writes one
    edit instruction per image. The template is chosen as the dataset builder chose it: by the
    T2I model's editor kind (instruction editor or sdedit), by whether axis A or B is "overall",
    and by whether the policy sees images.

    Raises:
        ValueError: both axes are "overall" (never in training).
    """
    fb = _tdig(exp)
    if fb:
        # A described editor run was trained on (base prompt, axis, K source images) only, so
        # axis A and the source prompts are not in its instruction. "overall" is a trained axis.
        d, fi = fb
        ax = (axis_b or "").strip()
        ax = OVERALL_AXIS if ax.lower() == OVERALL_AXIS else resolve_axis(ax)["factor"]
        content = fi.edit_instruction(
            base_prompt, ax, len(source_prompts), kind=d.get("editor_kind", "edit")
        )
        return [{"role": "user", "content": content}]
    from axis_framework.backend import layered as bl

    kind = bl.edit_kind(config.model_family(model))
    a_overall = (axis_a or "").strip().lower() == OVERALL_AXIS
    b_overall = (axis_b or "").strip().lower() == OVERALL_AXIS
    if a_overall and b_overall:
        raise ValueError(
            "layering 'overall' onto a set that already varied 'overall' is "
            "not a task the round-two policy was trained on"
        )
    if b_overall:
        tmpl = (bl.OVERALL_B_VLM_FOR if vlm else bl.OVERALL_B_FOR)[kind]
    elif a_overall:
        tmpl = (bl.OVERALL_A_VLM_FOR if vlm else bl.OVERALL_A_FOR)[kind]
    else:
        tmpl = (bl.INSTRUCTION_VLM_FOR if vlm else bl.INSTRUCTION_FOR)[kind]
    k = len(source_prompts)
    block = bl.source_block(list(source_prompts), [None] * k, vlm)
    content = tmpl.format(
        k=k, base_prompt=base_prompt, axis_a=axis_a, axis_b=axis_b, source_block=block
    )
    return [{"role": "user", "content": content}]


def naive_edit_instructions(axis_b, k):
    """Return K templated edit instructions, used only when the round-two policy fails to parse.

    The caller flags the trial `degraded: true`.
    """
    ax = axis_b.strip()
    ax = ax[4:] if ax.lower().startswith("the ") else ax
    return [
        f"change the {ax} to a clearly different value (variation {i + 1} of {k})"
        for i in range(int(k))
    ]


# --------------------------------------------------------------------------- fallback


def naive_expansion(base_prompt, axis, k=None):
    """Return K templated prompts, used only when the policy emits no K parseable prompts.

    Keeps a session going after a malformed completion. Each such trial is logged with
    `degraded: true` so the analysis can drop it.
    """
    k = config.K if k is None else int(k)
    a = resolve_axis(axis)
    return [f"{base_prompt}, variation {i + 1} of {a['factor']}" for i in range(k)]
