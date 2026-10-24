"""Add one guard to the installed verl's V1 trainer so that it matches runtime.lock.json.

With LoRA the reference policy is the actor with its adapter disabled (`ref_in_actor`), so verl
registers no separate reference worker group. `init_workers` in `trainer/ppo/v1/trainer_base.py`
still looks that group up, and a LoRA run with `use_kl_loss` or `use_kl_in_reward` stops at
startup with `KeyError: 'actor_rollout_ref'`. verl's V0 trainer has the `not self.ref_in_actor`
guard; this script adds it to the V1 trainer. The release configurations use neither KL term, so
the guard is never reached; it is applied because runtime.lock.json hashes the patched file.

Run it once with the training interpreter after installing requirements/train.txt:

    $PY_TRAIN scripts/patch_verl.py

The script is idempotent. It checks the patched file against axis_grpo/training/runtime.lock.json,
which `axis_grpo/training/run.py` and the training preflight verify before training.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import verl

LOCK = Path(__file__).resolve().parents[1] / "axis_grpo/training/runtime.lock.json"
TARGET = "trainer/ppo/v1/trainer_base.py"
LOOKUP = "            self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]\n"
UNPATCHED = "        if self.use_reference_policy:\n" + LOOKUP
PATCHED = (
    "        # T-DIG patch: with LoRA the reference policy is the actor (ref_in_actor), and no\n"
    "        # separate reference worker group exists (scripts/patch_verl.py).\n"
    "        if self.use_reference_policy and not self.ref_in_actor:\n" + LOOKUP
)
# An earlier form of the same patch: a comment block followed by the guarded lookup.
EARLIER = re.compile(
    r"        # LOCAL PATCH \(axis_grpo\).*?\n        if self\.use_reference_policy and "
    r"not self\.ref_in_actor:\n" + re.escape(LOOKUP),
    re.S,
)


def main():
    path = Path(verl.__file__).parent / TARGET
    text = path.read_text()
    if PATCHED not in text:
        if text.count(UNPATCHED) == 1:
            text = text.replace(UNPATCHED, PATCHED)
        elif len(EARLIER.findall(text)) == 1:
            text = EARLIER.sub(lambda m: PATCHED, text)
        else:
            sys.exit(f"{path}: the expected verl code was not found; is verl at the pinned commit?")
        path.write_text(text)
        print(f"patched {path}")
    else:
        print(f"already patched: {path}")
    expected = json.loads(LOCK.read_text())["framework_files"][TARGET]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        sys.exit(f"{path} does not match {LOCK.name}; is verl at the pinned commit?")


if __name__ == "__main__":
    main()
