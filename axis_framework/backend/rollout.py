"""Completion parsing and latent seeds, as used by the policy's GRPO training rollouts.

`parse_prompts` reads the <prompt> entries of a policy completion; `seed_for` gives the latent
seed of a base prompt. The framework imports both, so a live grid matches training.
"""

import hashlib
import os
import re

PROMPT_RE = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL | re.IGNORECASE)

# Shifts every latent in the run by a constant, to repeat a config on different noise.
SEED_EPOCH = int(os.environ.get("AXIS_SEED_EPOCH", "0"))

# Off by default, so rows sharing a base prompt draw the same latent whatever their axis.
# On gives each (base_prompt, axis) pair its own latent. See seed_for.
SEED_PER_AXIS = os.environ.get("AXIS_SEED_PER_AXIS", "0") == "1"


def parse_prompts(solution_str):
    """Return the non-empty <prompt>...</prompt> entries of a completion, in order."""
    if not solution_str:
        return []
    return [m.strip() for m in PROMPT_RE.findall(solution_str) if m.strip()]


def _stable_hash(s):
    """Return a hash that is stable across processes, unlike Python's salted hash()."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def seed_for(base_prompt, group_size=None, factor=""):
    """Return the latent seed for a row, in [0, 2**31 - 1).

    The seed is a pure function of its key, so every completion in a GRPO group draws the
    same latent in any process, step or epoch. By default the key is `base_prompt`, so rows
    that share a prompt but differ in axis share a latent and their diversity scores stay
    comparable. With AXIS_SEED_PER_AXIS=1 the key is `(base_prompt, factor)`, giving each axis
    its own latent. AXIS_SEED_EPOCH is added to the hash. `group_size` is ignored.
    """
    key = f"{base_prompt}\x1f{factor}" if SEED_PER_AXIS else base_prompt
    return (_stable_hash(key) + SEED_EPOCH) % (2**31 - 1)
