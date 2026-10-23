"""Axis-suggestion prompt and parser for the suggest server.

A VLM is shown the four i.i.d. images of a prompt (`SYSTEM`, `USER_TMPL`) and answers with
`OBSERVED: ... | AXIS: ...` lines, which `parse` reads.
"""

import re

# SYSTEM forbids axes the prompt fixes ("dog count" for "six dogs"), which no policy can
# satisfy.
SYSTEM = """You are reviewing four images generated from the same text prompt. Your task \
is to diagnose under-variation: identify choices the model made once and then repeated \
across all four images.

Report the {k} most striking repeated choices.

The images are labelled Picture 1, Picture 2 and so on, so you can tell them apart. Those \
labels are ours, not part of what the generator drew.

OUTPUT REQUIREMENTS
- Output exactly {k} lines and nothing else.
- Do not use bullets, numbering, or headings.
- Use this format for every line:
  OBSERVED: <what is essentially the same across all four images> | AXIS: <what should vary>
- Keep each AXIS short: one to six words.
- Name the dimension, not possible values. Write "plate colour", not "plate colour (red, blue, white)".

THE ORIGINAL PROMPT IS OFF LIMITS
Do not report anything that the original generation prompt explicitly required. A detail \
repeated because the prompt specified it is successful prompt-following, not a failure to \
vary.

Before including any line, check:
1. Is this detail essentially identical across all four images?
2. Did the original prompt leave it unspecified?
3. Could it vary without contradicting the prompt?

Include the line only if all three answers are yes.

Apply prompt constraints precisely:
- If the prompt says "a red plate", do not report "plate colour".
- If it says "three kettles on a shelf", do not report "kettle count".
- If it says "a doorway with a brass handle", do not report "handle material".
- Constraints apply only to the object they describe. "A red plate beside a kettle" fixes the plate's colour, not the kettle's colour.
- Other dimensions remain eligible. "A red plate" still leaves its size, material, wear, and placement open.

WHAT TO LOOK FOR
Choose repeated details that a viewer would genuinely expect or want to vary. A diversity axis can \
be about concrete visual details like color, pose etc., but also more abstract details like \
changing the cultural background, demographics, or the entire scene composition can be interpreted \
differently for open-ended prompts. Some things you can look at include:
- objects and their properties
- people and their visible characteristics
- actions and poses
- relationships and spatial arrangement
- setting and background
- composition, viewpoint, framing, and lighting
- what the scene depicts
- how an open-ended prompt is interpreted
- the cultural background

SPREAD YOUR {k} LINES ACROSS DIFFERENT KINDS. {k} lines that are all surface visual detail \
is a weak answer, and so is {k} lines that are all abstract. Draw on several of the kinds \
above in one set -- something concrete, something relational, something about the people or \
the world the image belongs to -- as far as the four images honestly support it. Do not \
force a kind the images do not show.

LET THE LENGTH FOLLOW THE AXIS. Many axes fit in two words and should stay there. A \
relation between two things usually needs four or five, and squeezing it into two loses \
the relation. Vary the length as the axis requires instead of writing every line to the \
same shape.

Do not invent missing variation. DO NOT VIOLATE THE ORIGINAL PROMPT.

WORDING
- Be direct and concise.
- Avoid unnecessary articles.
- Write "kettle placement", not "where the kettle is standing".
- Write "doorway presence", not "whether there is a doorway".
- Identify subjects briefly; do not redescribe the entire scene.

Every example above -- the plate, the kettle, the doorway -- is there for your reference \
only. Each shows the SHAPE of an answer, never its content, and none of \
those subjects will appear in your images. Do not reuse their wording.

Ground every line you write in the four images in front of you and in the base prompt you \
were given."""

USER_TMPL = """The prompt was: "{prompt}"

The four attached images are what the model produced. What did it fix in place that \
could have varied?"""


def parse(text):
    """Return [{observed, axis}] from OBSERVED | AXIS lines, or None if none parse."""
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not re.search(r"AXIS\s*:", line, re.I):
            continue
        obs = ""
        m = re.match(r"^\**OBSERVED\**\s*:\s*(.*?)\s*\|", line, re.I)
        if m:
            obs = m.group(1).strip()
        # Skip a line that ends at "AXIS:" with nothing after it.
        m = re.search(r"AXIS\s*:\s*(.+)$", line, re.I)
        if not m:
            continue
        ax = m.group(1).strip().strip('"').rstrip(".").strip("*").strip()
        if ax and ax.lower() != "none":
            out.append({"observed": obs, "axis": ax})
    return out or None
