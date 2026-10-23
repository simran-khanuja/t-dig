"""Policy instruction for the generation task: per-T2I-model word caps, optional image context."""

VERSION = "generation-v2-v4"


def instruction(base_prompt, axis, k=4, t2i_model='sd35-medium'):
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("Empty base prompt")
    if not isinstance(axis, str) or not axis.strip():
        raise ValueError("Empty axis")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    limits = {'sd35-medium': 60, 'sd35-large': 60, 'flux2-klein-9b': 150, 'qwen-image-2512': 150}
    if t2i_model not in limits:
        raise ValueError('Unknown reward T2I model for instruction length limit')
    task = (
        'Create variety across multiple attributes that the base prompt leaves open. '
        'The axis "overall" does not name a single attribute.'
        if axis == 'overall'
        else 'Vary ONLY the given axis, and together cover clearly distinct values of it.'
    )
    length = (
        "SD3.5's CLIP text encoders have a 77-token limit; its T5 encoder has a larger budget."
        if t2i_model.startswith('sd35')
        else "The renderer's text encoder has a 512-token limit."
    )
    return f'''You are given an image-generation prompt and a diversity axis.

Base prompt: {base_prompt}
Axis: {axis}

Write exactly {k} image-generation prompts which vary the main prompt along the given axis.

RULES, in order of importance:

1. THE BASE PROMPT'S CHOICES ARE CONSTRAINTS, NOT VARIABLES. Preserve every explicit requirement
   in the base prompt: every subject, its count, its kind, and every attribute it names.
   If the base prompt says "two men", every prompt says two men -- not a woman,
   not three men, not a crowd.
2. WHERE THE BASE PROMPT ALREADY FIXES SOMETHING, VARY WITHIN IT, NEVER AWAY FROM IT.
   The axis can only open up what the prompt left open. Contradicting a stated detail
   is the single worst thing you can do here.
3. {task}
4. KEEP EACH PROMPT UNDER {limits[t2i_model]} WORDS. {length}
   Put the base prompt's required details and the axis change first. Avoid extra description
   that crowds out the base prompt. Add the axis change and stop.
5. Hold everything else as close to the base prompt as you can -- ideally reuse its exact
   wording and only substitute the part the axis changes. Include the base prompt's wording
   in each expanded prompt wherever possible.

Output exactly {k} prompts, each wrapped in <prompt></prompt> tags, and nothing else.'''
