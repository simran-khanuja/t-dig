"""Pre-render the i.i.d. stage for a fixed prompt set, so the UI opens without a render.

Renders through the T2I model server over HTTP, so a cached set is bit-for-bit what the live
path produces.

    # the default 8 prompts, on the model the loop opens with
    python axis_framework/prefetch.py

    # every model the T2I model server will load (it swaps between them; 15-60s each)
    python axis_framework/prefetch.py --models all

    python axis_framework/prefetch.py --prompts my_prompts.txt --model flux1-dev

Resumable: a (model, prompt) already cached at these settings is skipped. `--force`
re-renders it.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axis_framework import cache, client, config  # noqa: E402

# config.GALLERY_PROMPTS is also the list /prompts orders the start screen by.
DEFAULT_PROMPTS = list(config.GALLERY_PROMPTS)


def load_prompts(path):
    if not path:
        return list(DEFAULT_PROMPTS)
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--t2i-url", default=config.T2I_URL)
    ap.add_argument(
        "--prompts", default=None, help="file, one per line; default: config.GALLERY_PROMPTS"
    )
    ap.add_argument(
        "--model", default=None, help="T2I model key; default: the first one AXIS_FW_SERVE serves"
    )
    ap.add_argument(
        "--models",
        default=None,
        help="'all' for every model the T2I model server offers, or a "
        "comma-separated list. Overrides --model.",
    )
    ap.add_argument("--n", type=int, default=config.N_IID)
    # Unset: each model's own i.i.d. settings (config.render_settings(model, "iid")), which
    # orchestrator.route_prompt uses for cache lookups. A set value applies to every model.
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    r = client.T2IClient(a.t2i_url)
    available = {m["key"]: m for m in r.models()["models"]}
    if a.models == "all":
        models = sorted(available)
    elif a.models:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
    else:
        models = [a.model or config.default_model()]
    missing = [m for m in models if m not in available]
    if missing:
        raise SystemExit(f"T2I model server cannot load {missing}; it offers {sorted(available)}")

    prompts = load_prompts(a.prompts)
    scratch = os.path.join(config.RUNS, "prefetch")
    os.makedirs(scratch, exist_ok=True)
    print(
        f"{len(prompts)} prompts x {len(models)} models, n={a.n}, each at its own " f"iid settings",
        flush=True,
    )

    # Model-major order: the T2I model server holds one model at a time and a swap takes 15-60 s.
    done = skipped = 0
    for m in models:
        want = cache.settings(m, n=a.n, resolution=a.resolution, steps=a.steps, guidance=a.guidance)
        print(
            f"  {m}: {want['resolution']}px/{want['steps']} steps, guidance " f"{want['guidance']}",
            flush=True,
        )
        for i, prompt in enumerate(prompts):
            if not a.force and cache.lookup(m, prompt, want):
                skipped += 1
                print(f"  [{m}] {i + 1}/{len(prompts)} cached: {prompt}", flush=True)
                continue
            t0 = time.time()
            res = r.render(
                prompts=[prompt] * a.n,
                base_prompt=prompt,
                out_dir=scratch,
                tag=f"{m}_{cache.prompt_hash(prompt)}",
                model=m,
                init_diversity=want["init_diversity"],
                guidance_scale=want["guidance"],
                steps=want["steps"],
                resolution=want["resolution"],
            )
            meta = cache.store(m, prompt, res, want)
            done += 1
            print(
                f"  [{m}] {i + 1}/{len(prompts)} {time.time() - t0:5.1f}s  "
                f"{prompt}  -> {os.path.dirname(meta['images'][0])}",
                flush=True,
            )

    print(f"\n{done} rendered, {skipped} already cached -> {config.CACHE}")


if __name__ == "__main__":
    main()
