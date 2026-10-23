"""Build the policy instructions of the tdig datasets.

Templates:

- `INSTRUCTION_V4`: rows with a named axis.
- `INSTRUCTION_OVERALL`: rows whose axis is the literal string `overall`.
- `IMAGE_CONTEXT`: inserted into either template when the policy also receives the four
  i.i.d. images of the base prompt.
- `INSTRUCTION_EDIT`, `INSTRUCTION_SDEDIT`: the editing task, per editor kind.

`INSTRUCTION_V4`, `LENGTH_RULE`, `LENGTH_BUDGETS` and `INSTRUCTION_OVERALL` are verbatim copies
of the constants of the same names in `axis_framework/backend/instructions.py`.
`tests/test_tdig.py` compares them byte for byte.

Rule 3 of `INSTRUCTION_V4` is filled with the fixed phrase `the given axis`, because axes can be
full sentences ("Show the figure facing the camera") that do not read as a slot filler.
"""

T2I_MODELS = ("sd35-medium", "sd35-large", "flux2-klein-9b", "qwen-image-2512")

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

LENGTH_RULE = """KEEP EACH PROMPT UNDER {words} WORDS. The renderer's text encoder reads only the
   first {tokens} tokens; everything after that is dropped from its conditioning and never
   reaches the image. Extra description past that point is wasted, and it crowds out the
   base prompt. Add the axis change and stop."""

# Rule 4 when the dataset sets a word cap below the T2I model's budget (`word_cap`). The text
# encoder reads far more than such a cap, so the rule gives drift from the base prompt as the
# reason. It ends with the same sentence as `LENGTH_RULE`, which `RULE4_TAIL` rewrites for
# `overall` rows.
SHORT_LENGTH_RULE = """KEEP EACH PROMPT UNDER {words} WORDS. A short prompt keeps the renderer on the base
   prompt; every extra description competes with it and pulls the image away from what was
   asked. Add the axis change and stop."""
VERSION_SHORT = "tdig-v4-2"

# family -> (word cap, that T2I model's token budget).
LENGTH_BUDGETS = {"sd3": (60, 77), "flux": (150, 512), "qwen": (150, 512)}

FAMILY = {
    "sd35-medium": "sd3",
    "sd35-large": "sd3",
    "flux2-klein-9b": "flux",
    "qwen-image-2512": "qwen",
}

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

# An overall row has no axis to add, so the closing clause of rule 4 names the change instead.
RULE4_TAIL = ("Add the axis change and stop.", "Add what you are changing and stop.")

IMAGE_CONTEXT = (
    "These four images are vanilla outputs for the base prompt. The variation "
    "request was written after viewing them. Use the images to understand the "
    "request and decide what to vary. The base prompt remains the source of "
    "required details; do not treat incidental image details as requirements.\n\n"
    + "\n".join(f"Image {i+1}: <image>" for i in range(4))
    + "\n\n"
)

VERSION = "tdig-v4-1"

# The editing task, adapted from `INSTRUCTION` and `_RULE3_EDIT_OVERALL` of
# `axis_framework/backend/layered.py`. The sources are the k i.i.d. renders of the base prompt,
# so the round-two list of first-axis prompts is omitted; rules 1, 4 and 5 are verbatim.
EDIT_WORD_CAP = 30

INSTRUCTION_EDIT = """You are given a base prompt, the {k} images generated from it, and a diversity axis.

Base prompt: {base_prompt}
Axis: {axis}

{image_block}

The {k} images will be passed to an image-editing model together with one instruction each,
so that the edited images differ from one another along the axis. Write those {k} edit
instructions.

RULES, in order of importance:

1. ONE INSTRUCTION PER IMAGE, IN ORDER. Instruction i is applied to image i. Address what
   that image contains.
2. EDIT, DO NOT REPLACE. Each instruction changes one aspect and leaves the rest of the
   image alone: an instruction that rewrites the whole scene destroys the image it was given.
{rule3}
4. THE BASE PROMPT'S CHOICES ARE CONSTRAINTS. Whatever it states explicitly stays true of
   every image after the edit: every subject, its count, its kind, and every attribute it
   names.
5. Write each instruction as a direct command to an editing model, naming what to change
   and what it becomes: "change the lighting to late-afternoon sun through a window".
   Keep it under {words} words.

Output exactly {k} instructions, each wrapped in <prompt></prompt> tags, and nothing else."""

RULE3_EDIT = """3. Vary ONLY the given axis between the {k} instructions, and together cover clearly distinct
   values of it."""

RULE3_EDIT_OVERALL = """3. THE {k} INSTRUCTIONS MUST NOT COLLAPSE, AND MUST NOT ALL CHANGE THE SAME ONE THING.
   Writing the same instruction {k} times is the obvious failure; {k} instructions that all
   change, say, the lighting is the one to actually watch for, because it looks like an
   answer. Give each image a DIFFERENT kind of change -- one the setting, one the framing,
   one who or what is shown, one the light or the style -- so that the images move apart in
   several directions at once rather than further along a single one.
   This is still an edit and not a replacement: a different change per image, not a bigger
   one. Rule 2 holds."""

EDIT_VERSION = "tdig-edit-1"

# Editor kinds. An instruction editor (`edit`: FLUX.2 [klein], Qwen-Image) applies a command to
# the image. SD3.5 (`sdedit`) re-generates the image from a noised copy, guided by a prompt that
# describes the finished image, so its policy writes descriptions. `INSTRUCTION_SDEDIT` is adapted
# from the constant of that name in `axis_framework/backend/layered.py`.
EDITOR_KIND = {
    "flux2-klein-9b": "edit",
    "sd35-medium": "sdedit",
    "sd35-large": "sdedit",
    "qwen-image-2512": "edit",
}
SDEDIT_WORD_CAP = 60
SDEDIT_VERSION = "tdig-sdedit-1"

INSTRUCTION_SDEDIT = """You are given a base prompt, the {k} images generated from it, and a diversity axis.

Base prompt: {base_prompt}
Axis: {axis}

{image_block}

Each image will be re-generated from its own current version, guided by a prompt that describes
how the finished image should look, so that the re-generated images differ from one another
along the axis. Write those {k} prompts.

RULES, in order of importance:

1. ONE PROMPT PER IMAGE, IN ORDER. Prompt i describes the finished version of image i.
   Describe the whole finished image, not the change: the generator reads your prompt as the
   description of what it should produce.
2. KEEP WHAT EACH IMAGE ALREADY IS. Prompt i keeps the subject, the setting and the composition
   that image i already shows, and changes one aspect of it.
{rule3}
4. THE BASE PROMPT'S CHOICES ARE CONSTRAINTS. Whatever it states explicitly stays true of
   every prompt: every subject, its count, its kind, and every attribute it names.
5. KEEP EACH PROMPT UNDER {words} WORDS. The generator's text encoder reads only the first 77
   tokens; anything after that never reaches the image.

Output exactly {k} prompts, each wrapped in <prompt></prompt> tags, and nothing else."""

RULE3_SDEDIT = """3. Vary ONLY the given axis between the {k} prompts, and together cover clearly distinct values
   of it."""

RULE3_SDEDIT_OVERALL = """3. THE {k} PROMPTS MUST NOT COLLAPSE, AND MUST NOT ALL CHANGE THE SAME ONE THING.
   Writing near-identical prompts is the obvious failure; {k} prompts that all change, say, the
   lighting is the one to actually watch for, because it looks like an answer. Give each image a
   DIFFERENT kind of change -- one the setting, one the framing, one who or what is shown, one
   the light or the style. A different change per image, not a bigger one. Rule 2 holds."""


def length_rule(t2i_model):
    """Return rule 4 for one reward T2I model.

    Args:
        t2i_model: one of `T2I_MODELS`.

    Returns the text of rule 4 without its leading `4. `, carrying that T2I model's word cap
    and text-encoder token budget from `LENGTH_BUDGETS`. Raises `ValueError` for a T2I model
    outside `T2I_MODELS`.
    """
    if t2i_model not in FAMILY:
        raise ValueError(f"Unknown reward T2I model: {t2i_model!r}")
    words, tokens = LENGTH_BUDGETS[FAMILY[t2i_model]]
    return LENGTH_RULE.format(words=words, tokens=tokens)


def instruction(base_prompt, axis, k, t2i_model, images=False, word_cap=None):
    """Return the user message shown to the policy for one dataset row.

    Args:
        base_prompt: the image-generation prompt the policy expands. Non-empty string.
        axis: the variation request, verbatim. The literal `overall` selects
            `INSTRUCTION_OVERALL`; any other non-empty string selects `INSTRUCTION_V4`.
        k: number of prompts the policy writes. Integer, at least 2.
        t2i_model: the reward T2I model the policy trains against, one of `T2I_MODELS`. Sets
            the word cap and token budget in rule 4.
        images: when true, insert `IMAGE_CONTEXT`, which carries four `<image>` placeholders,
            before the sentence that starts `Write exactly`.
        word_cap: when set, rule 4 is `SHORT_LENGTH_RULE` with this cap, for every T2I model, and
            the instruction version is `VERSION_SHORT`.

    Returns the instruction string. Raises `ValueError` on an empty prompt or axis, `k` below
    2, or an unknown T2I model.
    """
    for name, value in (("base_prompt", base_prompt), ("axis", axis)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Empty {name}")
    if type(k) is not int or k < 2:
        raise ValueError("k must be an integer of at least 2")
    rule = length_rule(t2i_model)
    if word_cap is not None:
        if type(word_cap) is not int or word_cap < 1:
            raise ValueError("word_cap must be a positive integer")
        rule = SHORT_LENGTH_RULE.format(words=word_cap)
    if axis == "overall":
        text = INSTRUCTION_OVERALL.format(
            base_prompt=base_prompt,
            guidance_block="",
            k=k,
            rule4="4. " + rule.replace(*RULE4_TAIL) + "\n",
        )
    else:
        text = INSTRUCTION_V4.replace("{length_rule}", rule).format(
            base_prompt=base_prompt,
            axis_block=f"Axis: {axis}",
            guidance_block="",
            axis_factor="the given axis",
            k=k,
        )
    if images:
        marker = f"Write exactly {k} image-generation"
        if text.count(marker) != 1:
            raise ValueError("Instruction template lost its insertion point")
        text = text.replace(marker, IMAGE_CONTEXT + marker, 1)
    return text


def edit_instruction(base_prompt, axis, k, kind="edit"):
    """Return the user message of the editing task for one dataset row.

    Args:
        base_prompt: the prompt the k source images were generated from. Non-empty string.
        axis: the variation request, verbatim; the literal `overall` selects `RULE3_EDIT_OVERALL`.
        k: number of source images, and of edit instructions the policy writes. At least 2.
        kind: `edit` for an instruction editor, `sdedit` for an image-to-image editor whose
            prompt describes the finished image (`EDITOR_KIND` maps T2I models to kinds).

    Returns the instruction with k placeholders `Image i: <image>`, in order; the n-th
    placeholder pairs with the n-th entry of the row's `images` column. Raises `ValueError` on an
    empty prompt or axis, or `k` below 2.
    """
    for name, value in (("base_prompt", base_prompt), ("axis", axis)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Empty {name}")
    if type(k) is not int or k < 2:
        raise ValueError("k must be an integer of at least 2")
    if kind == "sdedit":
        rule3 = (RULE3_SDEDIT_OVERALL if axis == "overall" else RULE3_SDEDIT).format(k=k)
        return INSTRUCTION_SDEDIT.format(
            base_prompt=base_prompt,
            axis=axis,
            k=k,
            rule3=rule3,
            words=SDEDIT_WORD_CAP,
            image_block="\n".join(f"Image {i + 1}: <image>" for i in range(k)),
        )
    if kind != "edit":
        raise ValueError(f"Unknown editor kind: {kind!r}")
    rule3 = (RULE3_EDIT_OVERALL if axis == "overall" else RULE3_EDIT).format(k=k)
    return INSTRUCTION_EDIT.format(
        base_prompt=base_prompt,
        axis=axis,
        k=k,
        rule3=rule3,
        words=EDIT_WORD_CAP,
        image_block="\n".join(f"Image {i + 1}: <image>" for i in range(k)),
    )
