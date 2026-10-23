"""Round-one policy instructions: the text a non-`tdig` generation policy was trained on.

- `INSTRUCTION_V4` / `INSTRUCTIONS`: a named axis, with a per-T2I-model word cap
  (`LENGTH_BUDGETS`).
- `INSTRUCTION_OVERALL`: the axis "overall"; its rule 4 is copied from the v4 instruction.
- `IMAGE_PREAMBLE`: inserted before `SPLICE_BEFORE` for a vision policy.
- `load_axis_vocab`: the seven study axes, checked against `axes.py`.

The templates are byte-identical to the training data; `tdig/instruction.py` holds the
instructions of the released (`tdig`) policies.
"""

import re

# key -> (display name, TPIPS factor, examples). The factor is copied into ground_truth and
# is the phrase the reward conditions on. `examples` illustrates it and ends with a "not ..."
# clause naming what to hold fixed.
AXES = {
    "demographic": (
        "People",
        "the apparent age, gender, ethnicity, and body type of the people shown",
        "For example a young East Asian woman, an elderly Black man, a heavyset teenager, "
        "a wiry man in his forties. Change who the people appear to be -- not what they "
        "are doing, wearing, or where they are.",
    ),
    "physical_attributes": (
        "Object appearance",
        "the colour, material, shape, and design of the main object",
        "For example a chair could be a carved oak dining chair, a moulded red plastic "
        "stool, or a worn leather armchair. Change how the main object looks -- not which "
        "object it is, and not the setting around it.",
    ),
    "compositional": (
        "Framing",
        "the camera angle, shot distance, and framing",
        "For example a tight close-up, a wide establishing shot, a low angle looking up, "
        "the subject pushed to one side of the frame. Change how the scene is "
        "photographed -- not what is in it.",
    ),
    "mode_coverage": (
        "What it depicts",
        "what the image fundamentally depicts",
        "This axis is the category or kind of object or concept shown. For example "
        "'a musical instrument' spans guitar, drum, and flute; 'a crane' spans the bird "
        "and the construction machine; 'a bat' spans the animal and the sports equipment. "
        "Change which category of thing is depicted -- not its colour, styling, or framing.",
    ),
    "cultural_geographic": (
        "Culture and place",
        "the culture, region, or place depicted",
        "For example a street market in Lagos, in Osaka, in Oaxaca, or in Marrakesh. "
        "Change the cultural setting, dress, and surroundings -- not the subject itself.",
    ),
    "artistic_style": (
        "Visual style",
        "the artistic style or medium",
        "For example a photograph, an oil painting, a pencil sketch, a 3D render. "
        "Change how the image is rendered -- not what is depicted.",
    ),
    "mood_atmosphere": (
        "Mood and lighting",
        "the lighting, mood, and atmosphere",
        "For example harsh midday sun, soft overcast light, moody blue dusk, warm "
        "candlelight. Change the light and the feeling -- not the subject or the framing.",
    ),
}

# Instruction for a named axis. `{length_rule}` is filled per T2I model family (INSTRUCTIONS).
INSTRUCTION_V4 = """You are given an image-generation prompt and a diversity axis.

Base prompt: {base_prompt}
{axis_block}{guidance_block}

Write exactly {k} image-generation prompts which vary the main prompt along the given axis.

RULES, in order of importance:

1. THE BASE PROMPT'S CHOICES ARE CONSTRAINTS, NOT VARIABLES. Anything it states
   explicitly must survive in every prompt: every subject, its count, its kind, and every
   attribute it names. If the base prompt says "two men", every prompt says two men -- not
   a woman, not three men, not a crowd.
2. WHERE THE BASE PROMPT ALREADY FIXES SOMETHING, VARY WITHIN IT, NEVER AWAY FROM IT.
   The axis can only open up what the prompt left open. Contradicting a stated detail is
   the single worst thing you can do here.
3. Vary ONLY {axis_factor}, and together cover clearly distinct values of it.
4. {length_rule}
5. Hold everything else as close to the base prompt as you can -- ideally reuse its exact
   wording and only substitute the part the axis changes.

Output exactly {k} prompts, each wrapped in <prompt></prompt> tags, and nothing else."""

# Rule 4 is the only T2I-model-specific part of v4: its word cap follows the T2I model's
# text-encoder budget. render._render_chunk passes no ``max_sequence_length``, so the
# __call__ defaults of the pinned diffusers (PY_REWARD) apply:
#
#   sd3   StableDiffusion3Pipeline  CLIP-L and CLIP-G truncate at tokenizer_max_length=77;
#                                   only the T5-XXL branch reaches max_sequence_length=256.
#   flux  Flux2KleinPipeline        one Qwen3 text encoder, max_sequence_length=512.
#   qwen  QwenImagePipeline         Qwen2.5-VL, __call__ max_sequence_length=512 (the
#                                   tokenizer itself allows 1024).
#
# Only the two numbers differ between families. The 512-token families get 150 words
# (~200 tokens), below their ceiling, to stop the policy padding.
LENGTH_RULE = """KEEP EACH PROMPT UNDER {words} WORDS. The renderer's text encoder reads only the
   first {tokens} tokens; everything after that is dropped from its conditioning and never
   reaches the image. Extra description past that point is wasted, and it crowds out the
   base prompt. Add the axis change and stop."""

# family -> (word cap, that T2I model's token budget).
LENGTH_BUDGETS = {"sd3": (60, 77), "flux": (150, 512), "qwen": (150, 512)}

LENGTH_RULES = {
    fam: LENGTH_RULE.format(words=w, tokens=t) for fam, (w, t) in LENGTH_BUDGETS.items()
}

# ``{length_rule}`` is substituted with str.replace so the other fields stay open for
# .format(). The key is v4_<family> of LENGTH_BUDGETS.
INSTRUCTIONS = {
    "v4_" + fam: INSTRUCTION_V4.replace("{length_rule}", rule) for fam, rule in LENGTH_RULES.items()
}

# Bare "v4" is an alias for v4_sd3.
INSTRUCTIONS["v4"] = INSTRUCTIONS["v4_sd3"]

# policy_prompt.py falls back to this when a T2I model family has no v4_<family> key.
INSTRUCTION = INSTRUCTIONS["v4"]


def guidance_block(g):
    """Return the instruction line stating guidance scale `g`, or "" when `g` is None.

    The line lets a policy trained with it condition its wording on guidance. It says what
    guidance does and gives no strategy; the reward teaches the strategy.
    """
    if g is None:
        return ""
    return (
        f"\nRenderer guidance scale: {g:.1f} (higher makes the renderer follow the "
        f"wording more literally, lower makes it follow the wording more loosely)"
    )


def load_axis_vocab():
    """Return {axis_key: {name, tpips_factor, examples, baseline_expand_phrase}}.

    Raises KeyError if the keys of AXES differ from those in axes.py (the human-study axes).
    """
    from axis_framework.backend import axes as axes_mod

    study_keys = {a["key"] for a in axes_mod.AXES}
    if set(AXES) != study_keys:
        raise KeyError(
            f"axis keys {sorted(set(AXES) ^ study_keys)} differ between this file and "
            f"{axes_mod.__file__}; the training axes must match the human-study axes"
        )

    return {
        k: {
            "name": name,
            "tpips_factor": factor,
            "examples": examples,
            # prompt-expansion baseline only; training does not read it.
            "baseline_expand_phrase": axes_mod.AXIS_EXPAND_PHRASE[k],
        }
        for k, (name, factor, examples) in AXES.items()
    }


# Rule 4 of a v4 instruction, up to rule 5. Its word cap is T2I-model-specific
# (LENGTH_BUDGETS), so INSTRUCTION_OVERALL copies it from the family's v4 instruction.
RULE4_RE = re.compile(r"^4\. KEEP EACH PROMPT UNDER .*?(?=^5\. )", re.M | re.S)

# Used when an input has no rule 4. 60 words / 77 tokens is the smallest cap (SD3.5-L's
# CLIP branch), so the fallback never overruns a T2I model.
FALLBACK_RULE4 = """4. KEEP EACH PROMPT UNDER 60 WORDS. The renderer's text encoder reads only the
   first 77 tokens; everything after that is dropped from its conditioning and never
   reaches the image. Extra description past that point is wasted, and it crowds out the
   base prompt. Add the change and stop.
"""

# Rules 1, 2 and 4 match v4; rules 3 and 5 differ. {rule4} is filled from the v4 instruction.
INSTRUCTION_OVERALL = """You are given an image-generation prompt and a request for variety.

Base prompt: {base_prompt}
Axis: overall{guidance_block}

Write exactly {k} image-generation prompts which are derived from the base prompt and \
which are as different from one another as the base prompt allows.

"overall" names no single dimension, so there is nothing here to sweep. Use whatever the \
base prompt leaves open -- who or what is shown, the setting, the framing, the light, the \
style -- and spread the {k} prompts across as much of it as you can.

RULES, in order of importance:

1. THE BASE PROMPT'S CHOICES ARE CONSTRAINTS, NOT VARIABLES. Anything it states
   explicitly must survive in every prompt: every subject, its count, its kind, and every
   attribute it names. If the base prompt says "two men", every prompt says two men -- not
   a woman, not three men, not a crowd.
2. WHERE THE BASE PROMPT ALREADY FIXES SOMETHING, VARY WITHIN IT, NEVER AWAY FROM IT.
   A prompt free to differ everywhere else is still not free to contradict a stated
   detail, and doing so is the single worst thing you can do here.
3. THE {k} PROMPTS MUST NOT COLLAPSE, AND MUST NOT ALL DIFFER IN THE SAME ONE WAY.
   Writing the same prompt {k} times is the obvious failure; {k} prompts that differ only
   in, say, the time of day is the one to actually watch for, because it looks like an
   answer. Spend more than one degree of freedom.
{rule4}5. Keep every detail the base prompt states, in its own wording. Everything it does not
   state is yours to change.

Output exactly {k} prompts, each wrapped in <prompt></prompt> tags, and nothing else."""

# v4's rule 4 ends "Add the axis change and stop."; "overall" rows have no axis, so that
# sentence is replaced.
RULE4_TAIL = ("Add the axis change and stop.", "Add what you are changing and stop.")


# Inserted into a round-one instruction for a vision policy, which is otherwise unchanged.
# {axis_ref} is either the quoted axis name or "the Axis".
IMAGE_PREAMBLE = """These {k} images are what the image generator produced for the base \
prompt above, using {k} random seeds. The axis is what the user said they want to see vary \
across the images: your job is to write {k} prompts that vary it along {axis_ref}.

{images}

"""

# The preamble goes after the base prompt and axis, before the task, so "the base prompt
# above" refers back to text already read.
SPLICE_BEFORE = "Write exactly"
