"""Cache of pre-rendered i.i.d. sets, so a curated prompt's first screen needs no render.

A cold i.i.d. stage takes ~13 s on SD3.5-M and minutes on the larger T2I models. Prompts a
user types still render live.

The key includes the render settings as well as the prompt: a set drawn at 8 steps must not
be served to a session running at 40, or the audit comparison (audit_iid) stops being
like-for-like. A settings mismatch is a miss.

Layout: cache/<model>/<prompt_hash>/{iid_0..N-1.png, meta.json}. `meta.json` carries the
prompt text, so an entry can be read without an index.
"""

import hashlib
import json
import os
import time

from . import config

# The settings that make two renders of the same prompt different images. The used `seed` is
# not keyed: it is the training seed of the base prompt (rollout.seed_for) unless the caller
# pins one; `seed_requested` is the pinned seed, None by default.
KEYED = ("n", "resolution", "steps", "guidance", "init_diversity", "seed_requested")


def prompt_hash(prompt):
    """Return the md5 of the stripped prompt, 12 hex chars (same as gen_iid.prompt_key)."""
    return hashlib.md5(prompt.strip().encode("utf-8")).hexdigest()[:12]


def entry_dir(model, prompt, root=None):
    return os.path.join(root or config.CACHE, model, prompt_hash(prompt))


def settings(
    model, n=None, resolution=None, steps=None, guidance=None, init_diversity=None, seed=None
):
    """Return the cache key for *model*'s i.i.d. set, using that T2I model's own settings.

    Defaults come from config.render_settings(model, "iid"), so they differ per T2I model
    (e.g. FLUX.2-Klein: 1024px, 4 steps, guidance 1.0; SD3.5-Large: 1024px, 40 steps,
    guidance 4.5). Explicit arguments override them.
    """
    v = config.render_settings(model, "iid")
    return {
        "n": n or config.N_IID,
        "resolution": resolution or v["resolution"],
        "steps": steps or v["steps"],
        "guidance": v["guidance"] if guidance is None else float(guidance),
        "init_diversity": (
            config.INIT_DIVERSITY_IID if init_diversity is None else float(init_diversity)
        ),
        "seed_requested": seed,
    }


def lookup(model, prompt, want, root=None):
    """Return the cached meta for these settings, or None. `want` is a settings() dict."""
    d = entry_dir(model, prompt, root)
    fp = os.path.join(d, "meta.json")
    if not os.path.exists(fp):
        return None
    try:
        meta = json.load(open(fp))
    except json.JSONDecodeError:
        return None  # half-written by an interrupted prefetch; treat as absent
    if any(meta.get(k) != want[k] for k in KEYED):
        return None
    if not all(os.path.exists(p) for p in meta.get("images", [])):
        return None
    return meta


def store(model, prompt, render, want, root=None):
    """Copy a /render result into the cache and return its meta.

    The PNGs are copied because the session directory they come from is transient.
    """
    import shutil

    d = entry_dir(model, prompt, root)
    os.makedirs(d, exist_ok=True)
    images = []
    for i, src in enumerate(render["images"]):
        dst = os.path.join(d, f"iid_{i}.png")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        images.append(dst)
    meta = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "rendered": time.time(),
        # The seed the render used; recorded but not keyed (see KEYED).
        "seed": render.get("seed"),
        "model_path": render.get("model_path"),
        **{k: want[k] for k in KEYED},
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def catalogue(model=None, root=None):
    """Return the meta of every complete cache entry, sorted by prompt.

    The UI offers these as prompt chips; sorting keeps the chip order stable between loads.
    """
    root = root or config.CACHE
    out = []
    for m in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if model and m != model:
            continue
        mdir = os.path.join(root, m)
        for h in sorted(os.listdir(mdir)):
            fp = os.path.join(mdir, h, "meta.json")
            if not os.path.exists(fp):
                continue
            try:
                meta = json.load(open(fp))
            except json.JSONDecodeError:
                continue
            if all(os.path.exists(p) for p in meta.get("images", [])):
                out.append(meta)
    return sorted(out, key=lambda m: m["prompt"])
