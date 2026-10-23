"""Paths and defaults of the interactive framework.

Rendering, scoring and the policy prompt templates live in `axis_framework/backend/`.

Site paths: T_DIG_MODELS (T2I model weights; AXIS_MODELS overrides) and HF_HOME, as in the
rest of the repository (README, Environment variables).
"""

import os
import re
import sys

# Anchored on this file (<repo>/axis_framework/config.py), not the cwd, which a batch job can set
# to another directory.
REPO = os.environ.get("T_DIG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK = os.path.join(REPO, "axis_framework")
# Outputs of round-one training runs, read only when serving such a run by name:
# ckpt/<EXP>/, logs/all_*.out and iid/<T2I model>/. The released policies are served from
# served_runs/*.json descriptors instead and need none of this.
T_DIG_RUNS = os.environ.get("T_DIG_RUNS", os.path.join(REPO, "runs"))
LEGACY_RUNS = os.environ.get("AXIS_FW_LEGACY_RUNS", os.path.join(T_DIG_RUNS, "axis_grpo"))

if os.environ.get("AXIS_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["AXIS_HF_HOME"])
MODELS = os.environ.get("AXIS_MODELS", os.environ.get("T_DIG_MODELS", os.path.join(REPO, "models")))

# Sessions, images and the event log, one directory per study run.
RUNS = os.environ.get("AXIS_FW_RUNS", os.path.join(FRAMEWORK, "runs"))

# --------------------------------------------------------------------------- services

# Localhost ports above the trainer's (19095/19096), so the framework and a training run can
# share a node.
POLICY_URL = os.environ.get("AXIS_FW_POLICY_URL", "http://127.0.0.1:19101")
T2I_URL = os.environ.get("AXIS_FW_T2I_URL", "http://127.0.0.1:19102")
SUGGEST_URL = os.environ.get("AXIS_FW_SUGGEST_URL", "http://127.0.0.1:19103")

# Interpreter per service. Policy: the training environment (PY_TRAIN, requirements/train.txt).
# Render and suggest: the reward environment (PY_REWARD, requirements/reward.txt).
PY_POLICY = os.environ.get("AXIS_FW_PY_POLICY", os.environ.get("PY_TRAIN", "python"))
PY_T2I = os.environ.get("AXIS_FW_PY_T2I", os.environ.get("PY_REWARD", "python"))
PY_SUGGEST = os.environ.get("AXIS_FW_PY_SUGGEST", os.environ.get("PY_REWARD", "python"))

# --------------------------------------------------------------------------- defaults

# Set size the policy was trained at. A dev flag on the servers; the UI's choices are
# K_CHOICES below.
K = int(os.environ.get("AXIS_FW_K", "4"))

# Resolution, steps and guidance differ by T2I model and stage (grid, iid, edit, accept) and are
# read from the served training run: see render_settings() below.

# Initial diversity alpha in [0, 1]: the fraction of per-sample independent noise (latents.py).
# 0 = the shared latent used in training, so every difference comes from the wording.
INIT_DIVERSITY = float(os.environ.get("AXIS_FW_INIT_DIVERSITY", "0.0"))
# The i.i.d. stage renders at fully independent noise, matching gen_iid.py's four seeds.
INIT_DIVERSITY_IID = 1.0
N_IID = int(os.environ.get("AXIS_FW_N_IID", "4"))

# --------------------------------------------------------------------------- models

# The policy base model, whether it sees images, its LoRA rank and context length are read from
# the served runs (profiles.policy_base etc.). AXIS_FW_POLICY_BASE overrides, for serving an
# untuned base with no runs.
# Round-one LoRA checkpoints live under ckpt/$EXP. Point --lora at the experiment dir (latest
# step is resolved) or at a specific global_step_N/actor/lora_adapter.
SUGGEST_VLM = os.environ.get("AXIS_FW_SUGGEST_VLM", "Qwen/Qwen3-VL-8B-Instruct")

# --------------------------------------------------------------------------- T2I models


# The T2I models the UI may offer, from axis_framework/t2i_models.json (AXIS_FW_T2I_MODELS
# overrides). `family` selects the entry in `backend.render.MODEL_FAMILIES` (pipeline class,
# extra __call__ kwargs, verified shared-latent packing). Two fields change what the UI offers:
#
#   shared_latents  False: initial-diversity alpha < 1 is refused for that model.
#                   backend.render enforces this because wrong-shaped noise does not raise and
#                   corrupts every axis-distance number. FLUX.1's 2D-patch latent packing is
#                   unverified against the pinned diffusers.
#   guidance        False: the model is guidance-distilled, so the Text guidance slider is
#                   disabled.
#
# vram_gb is the estimated resident footprint, used only to warn when a model does not fit
# beside the policy and suggest servers on one GPU.
def _load_t2i_models():
    """Return t2i_models.json with absolute weight paths, skipping keys that start with '_'.

    Loaded here rather than in profiles.py because profiles imports config, and config must
    not import profiles at load time.
    """
    import json

    fp = os.environ.get("AXIS_FW_T2I_MODELS", os.path.join(FRAMEWORK, "t2i_models.json"))
    out = {}
    for k, v in json.load(open(fp)).items():
        if k.startswith("_"):
            continue
        v = dict(v)
        if not os.path.isabs(v["path"]):
            v["path"] = os.path.join(MODELS, v["path"])
        out[k] = v
    return out


T2I_MODELS = _load_t2i_models()


def default_model():
    """Return the T2I model a session opens on: AXIS_FW_MODEL, else the first one served."""
    from . import profiles

    return profiles.default_model()


# Pre-rendered i.i.d. sets, keyed by (model, prompt, render settings); see cache.py.
CACHE = os.environ.get("AXIS_FW_CACHE", os.path.join(FRAMEWORK, "cache"))

# How many pre-rendered prompts the first screen offers. The cache holds more: route_prompt
# adds every typed prompt on a miss.
N_GALLERY = int(os.environ.get("AXIS_FW_N_GALLERY", "8"))

# The curated start-screen prompts. prefetch.py renders them and /prompts lists them first, in
# this order; other cached prompts (typed by users) are cache hits, not gallery cards. Drawn from
# the project's 100-prompt evaluation set, covering people, occupations, scenes and one
# under-specified prompt.
GALLERY_PROMPTS = [
    "a photo of a family at the dinner table",
    "a doctor examining a patient",
    "a teacher in a classroom",
    "a firefighter on duty",
    "a person praying",
    "a street festival",
    "kids playing in the snow",
    "a person meditating",
]

# Optional: the reward's fidelity scorer, for the audit metrics only. Unset, the audit reports
# diversity without VQAScore; no other scorer is substituted.
VQA_URL = os.environ.get("AXIS_FW_VQA_URL", "")

# --------------------------------------------------------------------------- off-axis

# The off-axis diversity ladder: the rungs of one served run, picked by the slider. A rung is a
# saved checkpoint (index-spaced) or, for an alpha ladder, the adapter at a LoRA scale. This
# slider changes what the policy writes, so moving it re-expands; the other two sliders only
# re-render. Within one run the objective `dist - lam * mean_k max(0, tau - fid_k)` is fixed, so
# rungs differ only in how far it was optimised: lower rungs stay closer to the base prompt.
#
# AXIS_FW_SERVE (profiles.py) picks the run each T2I model serves, for both the expand and the
# layered ladder. A policy is served only through the T2I model it was trained against.


def off_axis_exp_for(model=None):
    """Return the round-one ladder run served for T2I model *model* ("" if none)."""
    from . import profiles

    return profiles.round1_exp(model or default_model())


def layer_exp_for(model=None):
    """Return the layered-round ladder run for T2I model *model*, or "" when it has none.

    With one, "Add this to these images" uses a policy trained for that task
    (backend/layered.py): K edit instructions for the images on screen, applied by the
    T2I model's image editor. Without one, the re-expand-and-regenerate path runs.
    """
    from . import profiles

    return profiles.layer_exp(model or default_model())


def off_axis_reward(exp=None):
    """Return (lam, tau) of a ladder's run, parsed from its name, for display and the log."""
    from . import profiles

    return profiles.off_axis_reward(exp or off_axis_exp_for())


def length_family(model=None):
    """Return the instructions.LENGTH_BUDGETS key (prompt length budget) for *model*."""
    from . import profiles

    return profiles.length_family(model or default_model())


def model_family(model=None):
    """Return the render.MODEL_FAMILIES / EDIT_MODELS key for T2I model *model*."""
    from . import profiles

    return profiles.family(model or default_model())


def render_settings(model, stage):
    """Return {resolution, steps, guidance, source} for T2I model *model* at *stage*."""
    from . import profiles

    return profiles.render_settings(model or default_model(), stage)


def policy_is_vlm():
    from . import profiles

    return profiles.policy_is_vlm()


def run_is_vlm(exp):
    from . import profiles

    return profiles.run_is_vlm(exp)


def policy_max_images():
    from . import profiles

    return profiles.policy_max_images()


_LADDER_RE = re.compile(r"_step(\d+)$")

# Coarse slider labels, assigned by a rung's position on the ladder. Offline evaluation scores
# describe a fixed prompt set, not the user's prompt, so they are not shown here; `/audit`
# measures the images on screen.
_BANDS = ["least varied", "less", "middle", "more", "most varied"]


def off_axis_ladder(exp):
    """Return the slider rungs of `exp`, as a list of dicts in slider order.

    A described run (profiles.descriptor with a "ladder") lists its rungs; each is marked
    `available` if its adapter dir exists. Otherwise the rungs are the `<exp>_step<N>` dirs
    under adapters/, in training order.

    `off_axis` is the rung's position on the slider: alpha (= step / 100) for an alpha ladder,
    otherwise index / (n - 1), so stops are evenly spaced even when the checkpoint interval
    varies. The slider snaps to rungs and the server reports which one ran.
    """
    if not exp:
        return []
    root = os.path.join(FRAMEWORK, "adapters")
    from . import profiles

    d = profiles.descriptor(exp)
    if d and d.get("ladder"):
        # A described run names its rungs in slider order (sorted by diversity, not training
        # order). Step 0 is the untrained policy: no adapter, always available.
        chosen = [int(st) for st in d["ladder"]]
        n = len(chosen)
        # An alpha ladder ("alpha_ladder": true): rung N is the adapter at LoRA scale
        # alpha = N/100 and sits at off_axis = alpha. Other ladders use index spacing.
        alpha = bool(d.get("alpha_ladder"))
        return [
            {
                "off_axis": (
                    round(st / 100, 6) if alpha else (round(i / (n - 1), 6) if n > 1 else 0.0)
                ),
                "alpha": round(st / 100, 6) if alpha else None,
                "step": st,
                "of": max(chosen),
                "exp": exp,
                "label": (
                    "untrained" if st == 0 else (f"alpha {st / 100:.2f}" if alpha else f"step {st}")
                ),
                # An alpha ladder's band follows alpha on the slider's 0..1 scale (rungs above 1 are "most varied").
                "band": (
                    _BANDS[min(len(_BANDS) - 1, int(min(st / 100, 0.999) * len(_BANDS)))]
                    if alpha
                    else _BANDS[min(len(_BANDS) - 1, int(i / n * len(_BANDS)))]
                ),
                "path": None if st == 0 else os.path.join(root, f"{exp}_step{st}"),
                "base": st == 0,
                "available": st == 0 or os.path.isdir(os.path.join(root, f"{exp}_step{st}")),
            }
            for i, st in enumerate(chosen)
        ]
    steps = []
    if os.path.isdir(root):
        for d in os.listdir(root):
            if not d.startswith(exp + "_step"):
                continue
            m = _LADDER_RE.search(d)
            if m and os.path.isdir(os.path.join(root, d)):
                steps.append((int(m.group(1)), os.path.join(root, d)))
    steps.sort()
    if not steps:
        return []

    n = len(steps)
    last = steps[-1][0]
    return [
        {
            "off_axis": round(i / (n - 1), 6) if n > 1 else 0.0,
            "step": st,
            "of": last,
            "exp": exp,
            "label": f"step {st}",
            "band": _BANDS[min(len(_BANDS) - 1, int(i / n * len(_BANDS)))],
            "path": path,
            "available": os.path.isdir(path),
        }
        for i, (st, path) in enumerate(steps)
    ]


# The slider position a session opens on, as an off_axis value (the UI snaps to the nearest
# rung, rungIndexFor). It opens near the fidelity end, so the first set stays close to the
# prompt and the user drags right for more variation. On an index-spaced ladder 0.2 is in the
# "least varied" band; on an alpha ladder it is alpha 0.20. Default-position trials do not use
# the last checkpoint, so they are not directly comparable to offline sweep scores; every trial
# logs its resolved `off_axis` and `step`.
OFF_AXIS_DEFAULT = float(os.environ.get("AXIS_FW_OFF_AXIS", "0.2"))

# --------------------------------------------------------------------------- chips

# Chips offered for every prompt, ahead of the VLM's suggestions. The elicitation prompt asks
# what is the same across a set, which "overall" never answers, so it is added here. Like any
# typed axis, "overall" becomes its own tpips_factor and the audit conditions on it verbatim.
ALWAYS_AXES = (("overall", "vary the whole scene -- nothing in particular held fixed"),)

# --------------------------------------------------------------------------- set size

# K_TRAINED is the set size the policies were trained at and the default. K_CHOICES are the
# sizes the UI offers; a trial with k != K_TRAINED is out of distribution and is logged with
# `k_off_distribution`.
K_TRAINED = 4
K_CHOICES = tuple(int(x) for x in os.environ.get("AXIS_FW_K_CHOICES", "2,4,8,16").split(","))

# Speculative expansion: how many suggested axes to expand, in one /expand_many batch, before
# the user picks one. Decode is memory-bandwidth-bound (~206 tok/s per sequence for 1 to 4
# sequences), so the batch costs about one expansion and saves ~2 s on a chip pick. 0 disables
# it. A typed axis that arrives during the batch waits ~2-3 s behind the policy server's lock.
SPECULATE = int(os.environ.get("AXIS_FW_SPECULATE", "3"))

# Expander sampling: temperature 1.0 is verl's rollout temperature; max_tokens is the round-one
# launcher's MAX_RESPONSE.
TEMPERATURE = float(os.environ.get("AXIS_FW_TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.environ.get("AXIS_FW_MAX_TOKENS", "1024"))
EXPAND_RETRIES = int(os.environ.get("AXIS_FW_EXPAND_RETRIES", "3"))


# --------------------------------------------------------------------------- imports


def _add(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def import_tdig_instruction(instruction_root=None):
    """Return `tdig.instruction`: the instruction a described run (served_runs/*.json) used.

    By default this repository's `tdig` package. A descriptor may name another checkout in
    "instruction_root"; its `tdig/instruction.py` is then loaded by file path, without adding
    the checkout to sys.path.
    """
    if not instruction_root:
        _add(REPO)
        from tdig import instruction  # noqa: E402

        return instruction
    import importlib.util

    name = "_tdig_instruction"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(instruction_root, "tdig", "instruction.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod
