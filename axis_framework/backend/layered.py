"""Round-two (layered) policy instructions: K edit instructions for a set of round-one images.

`INSTRUCTION_FOR[kind]` and `INSTRUCTION_VLM_FOR[kind]` take axis A (round one) and axis B (the
edit); `OVERALL_A_*` and `OVERALL_B_*` are the variants where one axis is "overall". `kind` is
"edit" or "sdedit" (`edit_kind`). The templates are byte-identical to the training data.
"""

# Round-two instruction for an "edit" editor. It follows instructions.INSTRUCTION's rule
# order and <prompt></prompt> format, adding rule 1 and the numbered input set.
INSTRUCTION = """You are given a base prompt, the {k} prompts written from it, and a new \
diversity axis.

Base prompt: {base_prompt}

These {k} prompts were used to generate {k} images. They differ from each other along: {axis_a}
{source_block}
The user wants those {k} images to differ along a further axis: {axis_b}

The {k} images will be passed to an image-editing model together with one instruction each.
Write those {k} edit instructions.

RULES, in order of importance:

1. ONE INSTRUCTION PER IMAGE, IN ORDER. Instruction i is applied to image i, which was
   generated from prompt i. Address what that image contains.
2. EDIT, DO NOT REPLACE. Each instruction changes one aspect and leaves the rest of the
   image alone. The image already differs from the others along {axis_a}, and that
   difference must survive the edit: an instruction that rewrites the whole scene destroys
   it.
3. Vary ONLY {axis_b} between the {k} instructions, and together cover clearly distinct
   values of it.
4. THE BASE PROMPT'S CHOICES ARE CONSTRAINTS. Whatever it states explicitly stays true of
   every image after the edit: every subject, its count, its kind, and every attribute it
   names.
5. Write each instruction as a direct command to an editing model, naming what to change
   and what it becomes: "change the lighting to late-afternoon sun through a window".
   Keep it under 30 words.

Output exactly {k} instructions, each wrapped in <prompt></prompt> tags, and nothing else."""

INSTRUCTION_SDEDIT = """You are given a base prompt, the {k} prompts written from it, and \
a new diversity axis.

Base prompt: {base_prompt}

These {k} prompts were used to generate {k} images. They differ from each other along: {axis_a}
{source_block}
The user wants those {k} images to differ along a further axis: {axis_b}

Each image will be re-generated from its own current version, guided by a prompt that
describes how the finished image should look. Write those {k} prompts.

RULES, in order of importance:

1. ONE PROMPT PER IMAGE, IN ORDER. Prompt i describes the finished version of image i,
   which was generated from prompt i above. Describe the whole finished image, not the
   change: the generator reads your prompt as the description of what it should produce.
2. KEEP WHAT EACH IMAGE ALREADY IS. Prompt i keeps the subject, the setting and the value
   of {axis_a} that image i already had. If image 2 showed an elderly man and image 3 a
   young woman, prompt 2 still describes an elderly man and prompt 3 a young woman.
3. Vary ONLY {axis_b} between the {k} prompts, and together cover clearly distinct values
   of it.
4. THE BASE PROMPT'S CHOICES ARE CONSTRAINTS. Whatever it states explicitly stays true of
   every prompt: every subject, its count, its kind, and every attribute it names.
5. KEEP EACH PROMPT UNDER 60 WORDS. The generator's text encoder reads only the first 77
   tokens; anything after that never reaches the image.

Output exactly {k} prompts, each wrapped in <prompt></prompt> tags, and nothing else."""

# The same instruction for a policy that is also shown the {k} images.
INSTRUCTION_VLM = INSTRUCTION.replace(
    "You are given a base prompt, the {k} prompts written from it, and a new " "diversity axis.",
    "You are given a base prompt, the {k} prompts written from it, the {k} images those "
    "prompts generated, and a new diversity axis. The images are supplied in the same "
    "order as the prompts.",
)


def source_block(prompts, images, vlm):
    """Return the numbered round-one prompts, each followed by `Image i: <image>` if `vlm`.

    verl's RLHFDataset (rl_dataset._build_messages) pairs the n-th `<image>` placeholder with
    the n-th entry of the row's `images` column and asserts the counts match. Image i sits
    beside prompt i because instruction i applies to image i.
    """
    out = []
    for i, p in enumerate(prompts):
        out.append(f"{i + 1}. {p}")
        if vlm:
            out.append(f"   Image {i + 1}: <image>")
    return "\n".join(out)


INSTRUCTION_FOR = {"edit": INSTRUCTION, "sdedit": INSTRUCTION_SDEDIT}

INSTRUCTION_VLM_FOR = {"edit": INSTRUCTION_VLM, "sdedit": INSTRUCTION_SDEDIT}

# Axis A = "overall" (round-one sets written for the axis "overall"): the named-axis wording
# ("differ from each other along: overall", rule 2) does not fit. These templates replace
# the two passages that name axis A; rule 2 differs per template.
_DIFFER_ALONG = (
    "These {k} prompts were used to generate {k} images. They differ from "
    "each other along: {axis_a}"
)

_DIFFER_ALONG_OVERALL = (
    "These {k} prompts were used to generate {k} images. They were written to be as "
    "different from one another as the base prompt allows, rather than to vary along any "
    "one named dimension."
)

_RULE2_EDIT = """2. EDIT, DO NOT REPLACE. Each instruction changes one aspect and leaves the rest of the
   image alone. The image already differs from the others along {axis_a}, and that
   difference must survive the edit: an instruction that rewrites the whole scene destroys
   it."""

_RULE2_EDIT_OVERALL = """2. EDIT, DO NOT REPLACE. Each instruction changes one aspect and leaves the rest of the
   image alone. Each image already differs from the others in whatever the base prompt left
   open -- who is shown, the setting, the framing, the light, the style -- and those
   differences must survive the edit: an instruction that rewrites the whole scene destroys
   them."""

_RULE2_SDEDIT = """2. KEEP WHAT EACH IMAGE ALREADY IS. Prompt i keeps the subject, the setting and the value
   of {axis_a} that image i already had. If image 2 showed an elderly man and image 3 a
   young woman, prompt 2 still describes an elderly man and prompt 3 a young woman."""

_RULE2_SDEDIT_OVERALL = """2. KEEP WHAT EACH IMAGE ALREADY IS. Prompt i keeps whatever image i already showed -- its
   subject, its setting, its framing, its style. If image 2 showed an elderly man in a
   kitchen and image 3 a young woman outdoors, prompt 2 still describes the elderly man in
   the kitchen and prompt 3 the young woman outdoors."""


def _overall_a(tmpl, rule2, rule2_overall):
    """Return *tmpl* with both passages naming axis A replaced by the "overall" wording.

    Raises AssertionError if a passage is missing from *tmpl*.
    """
    for old, new in ((_DIFFER_ALONG, _DIFFER_ALONG_OVERALL), (rule2, rule2_overall)):
        if old not in tmpl:
            raise AssertionError(
                "layered: the instruction template changed and the "
                f"'overall' variant can no longer be derived from it. Missing:\n{old}"
            )
        tmpl = tmpl.replace(old, new, 1)
    return tmpl


INSTRUCTION_OVERALL_A = _overall_a(INSTRUCTION, _RULE2_EDIT, _RULE2_EDIT_OVERALL)

INSTRUCTION_VLM_OVERALL_A = _overall_a(INSTRUCTION_VLM, _RULE2_EDIT, _RULE2_EDIT_OVERALL)

INSTRUCTION_SDEDIT_OVERALL_A = _overall_a(INSTRUCTION_SDEDIT, _RULE2_SDEDIT, _RULE2_SDEDIT_OVERALL)

OVERALL_A_FOR = {"edit": INSTRUCTION_OVERALL_A, "sdedit": INSTRUCTION_SDEDIT_OVERALL_A}

OVERALL_A_VLM_FOR = {"edit": INSTRUCTION_VLM_OVERALL_A, "sdedit": INSTRUCTION_SDEDIT_OVERALL_A}

# Axis B = "overall": more variety with no dimension named, as instructions.INSTRUCTION_OVERALL
# asks in round one. Rule 3 is that instruction's rule 3, adapted to edits: a different change
# per image, not a bigger one.
_WANTS_AXIS_B = "The user wants those {k} images to differ along a further axis: {axis_b}"

_WANTS_OVERALL_B = (
    "The user now wants those {k} images to differ from one another MORE BROADLY than they "
    "already do, without naming any one dimension -- in whatever the base prompt leaves "
    "open."
)

_RULE3_EDIT = """3. Vary ONLY {axis_b} between the {k} instructions, and together cover clearly distinct
   values of it."""

_RULE3_EDIT_OVERALL = """3. THE {k} INSTRUCTIONS MUST NOT COLLAPSE, AND MUST NOT ALL CHANGE THE SAME ONE THING.
   Writing the same instruction {k} times is the obvious failure; {k} instructions that all
   change, say, the lighting is the one to actually watch for, because it looks like an
   answer. Give each image a DIFFERENT kind of change -- one the setting, one the framing,
   one who or what is shown, one the light or the style -- so that the four move apart in
   several directions at once rather than further along a single one.
   This is still an edit and not a replacement: a different change per image, not a bigger
   one. Rule 2 holds."""

_RULE3_SDEDIT = """3. Vary ONLY {axis_b} between the {k} prompts, and together cover clearly distinct values
   of it."""

_RULE3_SDEDIT_OVERALL = """3. THE {k} PROMPTS MUST NOT COLLAPSE, AND MUST NOT ALL DIFFER IN THE SAME ONE WAY.
   Writing the same prompt {k} times is the obvious failure; {k} prompts that differ only
   in, say, the time of day is the one to actually watch for, because it looks like an
   answer. Move each finished image in a DIFFERENT direction -- the setting, the framing,
   who or what is shown, the light, the style -- while rule 2 keeps each one recognisably
   the image it already was."""

# The opener names an axis these rows do not have (cf. instructions.INSTRUCTION_OVERALL).
_OPENER_AXIS = "and a new diversity axis."

_OPENER_OVERALL = "and a request for more variety."


def _overall_b(tmpl, rule3, rule3_overall):
    """Return *tmpl* with the axis-B ask, opener and rule 3 replaced by the "overall" wording.

    Raises AssertionError if a passage is missing from *tmpl*.
    """
    for old, new in (
        (_WANTS_AXIS_B, _WANTS_OVERALL_B),
        (rule3, rule3_overall),
        (_OPENER_AXIS, _OPENER_OVERALL),
    ):
        if old not in tmpl:
            raise AssertionError(
                "layered: the instruction template changed and the "
                f"axis-B 'overall' variant can no longer be derived from it. Missing:\n{old}"
            )
        tmpl = tmpl.replace(old, new, 1)
    return tmpl


INSTRUCTION_OVERALL_B = _overall_b(INSTRUCTION, _RULE3_EDIT, _RULE3_EDIT_OVERALL)

INSTRUCTION_VLM_OVERALL_B = _overall_b(INSTRUCTION_VLM, _RULE3_EDIT, _RULE3_EDIT_OVERALL)

INSTRUCTION_SDEDIT_OVERALL_B = _overall_b(INSTRUCTION_SDEDIT, _RULE3_SDEDIT, _RULE3_SDEDIT_OVERALL)

OVERALL_B_FOR = {"edit": INSTRUCTION_OVERALL_B, "sdedit": INSTRUCTION_SDEDIT_OVERALL_B}

OVERALL_B_VLM_FOR = {"edit": INSTRUCTION_VLM_OVERALL_B, "sdedit": INSTRUCTION_SDEDIT_OVERALL_B}


def edit_kind(family):
    """Return "edit" or "sdedit" for *family*, from render.EDIT_MODELS.

    Raises ``SystemExit`` if the family has no editor. EDIT_MODELS is imported so the
    template and the serving pipeline agree.
    """
    from axis_framework.backend.render import EDIT_MODELS

    spec = EDIT_MODELS.get(family)
    if spec is None:
        raise SystemExit(f"no editor for family {family!r}; have {sorted(EDIT_MODELS)}")
    return spec["instruction"]
