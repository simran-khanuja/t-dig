"""Diversity axis definitions shown to study participants, and per-axis expansion phrasing.

Keys match the pipeline's `primary_axis` labels. Descriptions are short and neutral: they say
what an axis means without suggesting it is the right choice for a given prompt.
"""

AXES = [
    {
        "key": "demographic",
        "name": "People",
        "desc": "Who appears — their age, gender, ethnicity, body type, or ability.",
    },
    {
        "key": "physical_attributes",
        "name": "Object appearance",
        "desc": "The same object shown in different colors, materials, shapes, or "
        "designs — e.g. 'a chair' could be a wooden dining chair, a plastic "
        "stool, or a leather armchair.",
    },
    {
        "key": "compositional",
        "name": "Framing & layout",
        "desc": "How the picture is taken and arranged — e.g. a close-up vs. a wide "
        "shot, seen from above vs. at eye level, or the subject centered vs. "
        "off to one side.",
    },
    {
        "key": "mode_coverage",
        "name": "What it depicts",
        "desc": "Different images could show fundamentally different things (not just "
        "the same thing restyled) — e.g. 'a musical instrument' could be a guitar, "
        "a drum, or a flute; 'a crane' could be the bird or the machine.",
    },
    {
        "key": "cultural_geographic",
        "name": "Culture & place",
        "desc": "Cultural or regional variation — clothing, food, customs, or setting "
        "across different cultures or regions.",
    },
    {
        "key": "artistic_style",
        "name": "Visual style",
        "desc": "Rendering style or medium — photo, painting, illustration, sketch, 3D, etc.",
    },
    {
        "key": "mood_atmosphere",
        "name": "Mood & lighting",
        "desc": "Overall lighting tone, emotional feel, or ambiance — e.g. eerie, "
        "peaceful, dramatic, or warm.",
    },
]


# Per-axis phrasing for the prompt-expansion baseline.
AXIS_EXPAND_PHRASE = {
    "demographic": "the people shown (age, gender, ethnicity, body type)",
    "physical_attributes": "the color, shape, material, and texture of the objects",
    "compositional": "the camera angle, shot distance, and arrangement of the scene within the frame",
    "mode_coverage": "the fundamental interpretation of what is depicted",
    "cultural_geographic": "cultural and regional variation",
    "artistic_style": "the visual rendering style or medium",
    "mood_atmosphere": "the lighting tone, mood, and atmosphere",
}
