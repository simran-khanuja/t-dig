"""T2I model server: any t2i_models.json model behind HTTP, with the two UI knobs wired in.

    POST /render {prompts, base_prompt, model, guidance_scale, init_diversity, steps,
                  index_offset, n_total, ...}          -> {images: [path], seed, ...}
    POST /edit   {sources, instructions, model, steps, guidance_scale, resolution,
                  out_dir, tag}                        -> {images: [path], ...}
    POST /audit  {images, prompts, base_prompt, axis} -> the metrics the study logs
    GET  /models -> the T2I models this server may load, and their capabilities
    GET  /health

All model calls go through `axis_framework/backend/render.py` (`_shared_latent` /
`_one_latent` for noise, `_render_chunk` for the pipeline, `tpips_embed` + `vendi_score` for
the audit). This file adds only the initial-diversity mix (`axis_framework/latents.py`) and HTTP.

The seed is the request's `seed`, else `rollout.seed_for(base_prompt)`, the latent a GRPO group
drew for that prompt. The orchestrator renders the i.i.d. set with the training seed and each
policy grid with the policy draw's seed (see axis_framework/README.md, "Seeds").

Two knobs, both render-only (a re-render reuses the policy's K prompts):
  guidance_scale  passed to the pipeline as its CFG scale.
  init_diversity  alpha in [0,1], the fraction of per-sample independent noise; see
                  latents.py (alpha=0 shares one latent across the K images, as in training).

    $PY_REWARD \\
        axis_framework/servers/t2i_server.py --port 19102 --vqa-url http://127.0.0.1:19096
    ... --fake                      # no GPU: placeholder images, for driving the loop
"""

import argparse
import gc
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axis_framework import config, latents  # noqa: E402
from axis_framework.http_util import HttpError, serve  # noqa: E402

_STATE = {}
# Serialise GPU work: concurrent requests would multiply peak memory.
_GPU_LOCK = threading.Lock()

# Factor for the axis-blind audit reading: the TPIPS backbone with its axis LoRA disabled
# (`base=True`), i.e. untuned Qwen3-VL-8B. It is the denominator of the audit's
# axis_vendi / blind_vendi ratio: steering along the axis raises only the numerator, generic
# chaos raises both. Same embedding space as the numerator, so the ratio is unit-consistent.
BLIND_FACTOR = "the overall appearance of the image"


# --------------------------------------------------------------------------- models


def _model_spec(key):
    spec = config.T2I_MODELS.get(key)
    if spec is None:
        raise HttpError(400, f"unknown model {key!r}; have {sorted(_STATE['models'])}")
    if key not in _STATE["models"]:
        raise HttpError(
            400,
            f"model {key!r} was not enabled on this server "
            f"(--models {','.join(sorted(_STATE['models']))})",
        )
    return spec


def _ensure_model(key):
    """Make `key` the loaded T2I model, evicting the previous one.

    One model is resident at a time (they do not fit together beside the policy and suggest
    servers). A switch costs a model load (15-60 s); the pre-rendered cache (cache.py)
    avoids it for cached prompts. Callers hold _GPU_LOCK.

    Returns:
        The new cfg, rebuilt rather than mutated so cfg always matches the loaded model.
    """
    import torch

    if _STATE.get("loaded") == key:
        return _STATE["cfg"]
    spec = _model_spec(key)
    ok, why = _availability(key)
    if not ok:
        raise HttpError(400, f"{key} cannot run here: {why}")
    rs = _STATE["rs"]

    if _STATE.get("loaded") is not None:
        before = torch.cuda.memory_reserved() / 2**30
        rs._STATE.pop("pipe", None)
        # render.load_editor caches the editor separately; for some families it is
        # the pipeline just dropped, so drop it too.
        rs._STATE.pop("editor", None)
        rs._STATE.pop("editor_spec", None)
        # gc.collect() is required: a diffusers pipeline holds reference cycles, so without
        # it `empty_cache()` frees nothing and the next model loads on top of the old one.
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved() / 2**30
        print(
            f"[render] evicted {_STATE['loaded']} for {key}: "
            f"{before:.1f} -> {after:.1f} GiB reserved",
            flush=True,
        )
        _STATE["loaded"] = None

    cfg = {**_STATE["cfg"], "model_path": spec["path"], "model_family": spec["family"]}
    t0 = time.time()

    # A model that does not fit on the card (e.g. Qwen-Image, ~47 GiB) is loaded with
    # diffusers' CPU offload: weights in host RAM, several times slower.
    def _offload_load():
        """Load the pipeline with diffusers' CPU-offload hooks instead of on the card.

        Only `.to(device)` differs from `load_models`; downstream code reads the pipe from
        `rs._STATE` as usual, and `_execution_device` still reports cuda, so latents are
        built on the GPU.
        """
        import diffusers

        pipe_cls = getattr(diffusers, rs.MODEL_FAMILIES[spec["family"]]["pipeline_cls"])
        pipe = pipe_cls.from_pretrained(spec["path"], torch_dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_slicing()
        pipe.enable_model_cpu_offload()
        rs._STATE["pipe"] = pipe
        _STATE["offloaded"] = key

    free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
    need = spec["vram_gb"] * 1.1  # headroom for activations + decode
    if free and need > free:
        print(
            f"[render] {key} wants ~{need:.0f} GiB, {free:.0f} GiB free -> CPU offload", flush=True
        )
        _offload_load()
    else:
        try:
            rs.load_models(cfg)
            _STATE["offloaded"] = None
        except torch.OutOfMemoryError:
            # The vram_gb estimate can be too low, so an OOM also falls back to offload.
            print(f"[render] {key} OOMed loading resident -> retrying with CPU offload", flush=True)
            rs._STATE.pop("pipe", None)
            torch.cuda.empty_cache()
            _offload_load()

    _STATE["cfg"] = cfg
    _STATE["loaded"] = key
    print(
        f"[render] {key} loaded in {time.time() - t0:.0f}s"
        f"{' (offloaded)' if _STATE.get('offloaded') == key else ''}",
        flush=True,
    )
    return cfg


def _capacity():
    """Return (GPU GiB, host GiB) this process may use.

    Host RAM is the job's cgroup limit when there is one, since /proc/meminfo reports the
    whole node.
    """
    import torch

    gpu = (
        torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
    )
    # Walk from this process's cgroup up its ancestors: the leaf is usually "max" and the
    # job's memory limit sits on an ancestor (e.g. .../slurmstepd.scope/<job>/memory.max).
    host = 0
    try:
        rel = open("/proc/self/cgroup").read().strip().split(":")[-1]
    except OSError:
        rel = ""
    d = os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
    while d.startswith("/sys/fs/cgroup") and not host:
        try:
            v = open(os.path.join(d, "memory.max")).read().strip()
            if v.isdigit():
                host = int(v) / 2**30
        except OSError:
            pass
        if d == "/sys/fs/cgroup":
            break
        d = os.path.dirname(d)
    if not host:  # cgroup v1, or no limit set anywhere
        try:
            v = open("/sys/fs/cgroup/memory/memory.limit_in_bytes").read().strip()
            host = int(v) / 2**30
        except OSError:
            host = 0
    if not host or host > 2**20:
        host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return gpu, host


def _availability(key):
    """Return (ok, reason): whether this machine can run model *key*.

    The UI shows an unusable model as a disabled option with the reason.
    """
    spec = config.T2I_MODELS[key]
    gpu, host = _STATE.get("capacity", (0, 0))
    if not gpu:
        return True, ""
    if spec["vram_gb"] * 1.1 <= gpu:
        return True, ""
    # Offload needs host RAM for the weights plus staging buffers and the other services in
    # this cgroup; 1.6x the model size (1.36x was too little for Qwen-Image under 64 GiB).
    if spec["vram_gb"] * 1.6 <= host:
        return True, f"offloaded to host RAM — slow (~{spec['vram_gb']}GB model)"
    return False, (
        f"needs ~{spec['vram_gb']}GB: more than this {gpu:.0f}GB GPU, and "
        f"offloading it needs more than the {host:.0f}GB host RAM this job has"
    )


def route_models(_p):
    """List the T2I models for the UI's model box, with availability and capabilities.

    The UI disables the Initial diversity slider when `shared_latents` is false and the Text
    guidance slider when `guidance` is false (guidance-distilled models).
    """

    def entry(k):
        ok, why = _availability(k)
        return {"available": ok, "why": why}

    return {
        "models": [
            {
                **entry(k),
                "key": k,
                "label": config.T2I_MODELS[k]["label"],
                "family": config.T2I_MODELS[k]["family"],
                "shared_latents": config.T2I_MODELS[k]["shared_latents"],
                "guidance": config.T2I_MODELS[k]["guidance"],
                "vram_gb": config.T2I_MODELS[k]["vram_gb"],
                "note": config.T2I_MODELS[k]["note"],
                "loaded": _STATE.get("loaded") == k,
                "offloaded": _STATE.get("offloaded") == k,
            }
            for k in sorted(_STATE["models"])
        ],
        "loaded": _STATE.get("loaded"),
        "default": _STATE.get("default_model"),
    }


# --------------------------------------------------------------------------- fake mode


def _fake_images(prompts, out_dir, tag, resolution, index_offset=0):
    """Write placeholder PNGs, labelled FAKE, for running the loop without a GPU."""
    from PIL import Image, ImageDraw

    paths = []
    for j, p in enumerate(prompts):
        i = index_offset + j
        shade = 40 + (i * 47) % 160
        im = Image.new("RGB", (resolution, resolution), (shade, 60 + shade // 2, 200 - shade))
        d = ImageDraw.Draw(im)
        d.text(
            (10, 10),
            "\n".join(p[j : j + 44] for j in range(0, min(len(p), 400), 44)),
            fill=(255, 255, 255),
        )
        d.text((resolution // 2 - 30, resolution // 2), "FAKE", fill=(255, 0, 0))
        fp = os.path.join(out_dir, f"{tag}_{i}.png")
        im.save(fp)
        paths.append(fp)
    return paths


# --------------------------------------------------------------------------- render


def _cfg_view(guidance_scale, steps, resolution):
    """Return a copy of the service cfg with this request's guidance, steps and resolution.

    Nothing else (model, dtype, batching) can be changed over HTTP.
    """
    return {
        **_STATE["cfg"],
        "guidance": float(guidance_scale),
        "steps": int(steps),
        "resolution": int(resolution),
    }


def _render(
    prompts, seed, guidance_scale, init_diversity, steps, resolution, index_offset=0, n_total=None
):
    """Render `prompts`, which may be a contiguous slice of a grid of `n_total`.

    Latents are built for the whole grid and then sliced, because at alpha>0 image i's noise
    depends on its grid index. A shard therefore gets the same noise and prompt as a single
    server would. The denoising batch size can differ, which may change results at GPU
    numerical precision; launch every replica with `--render-batch 1` for exact agreement.
    """
    rs = _STATE["rs"]
    view = _cfg_view(guidance_scale, steps, resolution)
    n_total = int(n_total or (index_offset + len(prompts)))
    z = latents.mixed_latents(rs, seed, n_total, init_diversity, view)
    z = z[index_offset : index_offset + len(prompts)]

    bs = max(1, int(view["render_batch"]))
    images = []
    for st in range(0, len(prompts), bs):
        images.extend(rs._render_chunk(prompts[st : st + bs], z[st : st + bs], view))
    return images


def _settings(model, stage):
    """config.render_settings, or a neutral stand-in in --fake mode (no run to read)."""
    if _STATE.get("fake"):
        try:
            return config.render_settings(model, stage)
        except Exception:
            return {"resolution": 512, "steps": 4, "guidance": 1.0}
    return config.render_settings(model, stage)


def _startup_settings(model):
    g = _settings(model, "grid")
    return {"resolution": g["resolution"], "steps": g["steps"], "guidance": g["guidance"]}


def route_render(p):
    prompts = [str(x) for x in (p.get("prompts") or []) if str(x).strip()]
    if not prompts:
        raise HttpError(400, "no prompts")
    base_prompt = p.get("base_prompt") or prompts[0]
    model = p.get("model") or _STATE.get("loaded") or _STATE.get("default_model")
    # Unset render settings default to this model's served grid settings.
    grid = _settings(model, "grid")
    guidance_scale = float(p.get("guidance_scale", grid["guidance"]))
    init_diversity = float(p.get("init_diversity", config.INIT_DIVERSITY))

    if not _STATE["fake"]:
        spec = _model_spec(model)
        # A family with unverified shared-latent packing would render wrong noise without
        # an error, so refuse init_diversity < 1.0 for it.
        if init_diversity < 1.0 and not spec["shared_latents"]:
            raise HttpError(
                400,
                f"{model} ({spec['family']}) has no verified shared-latent packing, so "
                f"init_diversity must be 1.0 (got {init_diversity}). See MODEL_FAMILIES "
                f"in axis_framework/backend/render.py.",
            )
    steps = int(p.get("steps", grid["steps"]))
    resolution = int(p.get("resolution", grid["resolution"]))
    # A caller may pin the seed (e.g. a re-render); otherwise use the training seed.
    seed = int(p["seed"]) if p.get("seed") is not None else _STATE["seed_for"](base_prompt)

    out_dir = p.get("out_dir") or os.path.join(config.RUNS, "adhoc")
    os.makedirs(out_dir, exist_ok=True)
    tag = p.get("tag") or f"r{int(time.time())}"

    # Sharding (orchestrator._render_pooled): `prompts` may be the slice
    # [index_offset, index_offset+len) of a grid of n_total. Filenames use the grid index,
    # so shards write tag_0..tag_{n-1} without collisions. Absent, the request is a whole grid.
    n_total = int(p.get("n_total") or len(prompts))
    index_offset = int(p.get("index_offset") or 0)
    if index_offset < 0 or index_offset + len(prompts) > n_total:
        raise HttpError(
            400,
            f"shard [{index_offset}, {index_offset + len(prompts)}) "
            f"does not fit a grid of {n_total}",
        )
    sharded = len(prompts) != n_total

    t0 = time.time()
    if _STATE["fake"]:
        paths = _fake_images(prompts, out_dir, tag, resolution, index_offset)
    else:
        with _GPU_LOCK:
            _ensure_model(model)
            images = _render(
                prompts,
                seed,
                guidance_scale,
                init_diversity,
                steps,
                resolution,
                index_offset=index_offset,
                n_total=n_total,
            )
        paths = []
        for i, im in enumerate(images):
            fp = os.path.join(out_dir, f"{tag}_{index_offset + i}.png")
            im.save(fp)
            paths.append(fp)

    meta = {
        "images": paths,
        "prompts": prompts,
        "seed": seed,
        "model": model,
        "guidance_scale": guidance_scale,
        "init_diversity": init_diversity,
        "steps": steps,
        "resolution": resolution,
        "index_offset": index_offset,
        "n_total": n_total,
        "model_path": _STATE["cfg"]["model_path"],
        "model_family": _STATE["cfg"]["model_family"],
        "fake": _STATE["fake"],
        "seconds": round(time.time() - t0, 2),
    }
    # Sidecar JSON so a trial can be rebuilt from disk. A shard writes <tag>.part<offset>.json;
    # the orchestrator writes the merged <tag>.json.
    name = f"{tag}.part{index_offset}.json" if sharded else f"{tag}.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# --------------------------------------------------------------------------- edit


def route_edit(p):
    """Apply instruction i to source image i (a layered round's T2I model).

    Uses `render.edit_images`, the editor the layered reward trained against (picked
    from EDIT_MODELS by family; for FLUX.2-Klein, the resident generator). Defaults come from
    config.render_settings(model, "edit"). Outputs are <tag>_<i>.png, i = the source's grid
    index.
    """
    # A source is an image path, or a list of reference paths for one output ("Combine";
    # multi-reference editors such as FLUX.2-Klein and Qwen-Image-Edit).
    sources = [
        [str(y) for y in x] if isinstance(x, (list, tuple)) else str(x)
        for x in (p.get("sources") or [])
    ]
    instructions = [str(x) for x in (p.get("instructions") or [])]
    if not sources:
        raise HttpError(400, "no sources")
    if len(sources) != len(instructions):
        raise HttpError(400, f"{len(sources)} sources for {len(instructions)} instructions")
    flat = [y for x in sources for y in (x if isinstance(x, list) else [x])]
    missing = [x for x in flat if not os.path.isfile(x)]
    if missing:
        raise HttpError(400, f"source images not found: {missing[:2]}")
    model = p.get("model") or _STATE.get("loaded") or _STATE.get("default_model")
    ed = _settings(model, "edit")
    steps = int(p.get("steps", ed["steps"]))
    guidance = float(p.get("guidance_scale", ed["guidance"]))
    resolution = int(p.get("resolution", ed["resolution"]))
    out_dir = p.get("out_dir") or os.path.join(config.RUNS, "adhoc")
    os.makedirs(out_dir, exist_ok=True)
    tag = p.get("tag") or f"e{int(time.time())}"

    t0 = time.time()
    if _STATE["fake"]:
        paths = _fake_images(instructions, out_dir, tag, resolution, 0)
    else:
        rs = _STATE["rs"]
        fam = _model_spec(model)["family"]
        if fam not in rs.EDIT_MODELS:
            raise HttpError(
                400, f"{model} ({fam}) has no image editor; have " f"{sorted(rs.EDIT_MODELS)}"
            )
        with _GPU_LOCK:
            _ensure_model(model)
            view = {
                **_STATE["cfg"],
                "edit_steps": steps,
                "edit_guidance": guidance,
                "resolution": resolution,
            }
            if ed.get("strength") is not None:
                view["edit_strength"] = float(ed["strength"])
            images = rs.edit_images(sources, instructions, view)
        # `index_offset` is set when the orchestrator shards edits (orchestrator._edit_grid);
        # file indices stay the source's grid index.
        off = int(p.get("index_offset") or 0)
        paths = []
        for i, im in enumerate(images):
            fp = os.path.join(out_dir, f"{tag}_{off + i}.png")
            im.save(fp)
            paths.append(fp)

    meta = {
        "images": paths,
        "sources": sources,
        "instructions": instructions,
        "model": model,
        "steps": steps,
        "guidance_scale": guidance,
        "resolution": resolution,
        "task": "edit",
        "model_path": _STATE["cfg"]["model_path"],
        "model_family": _STATE["cfg"]["model_family"],
        "fake": _STATE["fake"],
        "seconds": round(time.time() - t0, 2),
    }
    if p.get("index_offset") is None:
        # Shards skip the sidecar; the orchestrator writes one for the whole set.
        with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return meta


# --------------------------------------------------------------------------- audit


def route_audit(p):
    """Compute server-side metrics for one grid. Never shown to the user.

    Returned keys:
      axis_dist     mean pairwise TPIPS distance conditioned on the axis phrase (the
                    reward's `dist`, same scale as training).
      axis_vendi    Vendi over the same embeddings: effective number of distinct images,
                    in [1, K]; robust to a single outlier.
      blind_vendi   the same with axis conditioning off (see BLIND_FACTOR).
      ratio         axis_vendi / blind_vendi, the headline metric.
      vqa_base/own  VQAScore of each image against the base prompt and against the policy's
                    prompt. Absent, with a `notes` entry, when no VQA service is configured.
    """
    paths = p.get("images") or []
    if len(paths) < 2:
        raise HttpError(400, "audit needs at least 2 images")
    if _STATE["fake"]:
        return {"fake": True, "notes": ["fake mode: no metrics computed"]}
    if not _STATE["audit"]:
        raise HttpError(400, "this T2I model server was launched with --no-audit")

    from PIL import Image

    rs = _STATE["rs"]
    axis = p.get("axis") or ""
    base_prompt = p.get("base_prompt") or ""
    prompts = p.get("prompts") or []
    images = [Image.open(fp).convert("RGB") for fp in paths]
    notes = []

    with _GPU_LOCK:
        E = rs.tpips_embed(images, axis, _STATE["cfg"])
        out = {
            "axis": axis,
            "axis_dist": rs.pairwise_distance(E),
            "axis_vendi": rs.vendi_score(E),
            "axis_vendi_norm": rs.normalised_vendi(E),
        }
        E0 = rs.tpips_embed(images, BLIND_FACTOR, _STATE["cfg"], base=True)
        out["blind_vendi"] = rs.vendi_score(E0)
        out["blind_dist"] = rs.pairwise_distance(E0)
        out["ratio"] = out["axis_vendi"] / out["blind_vendi"] if out["blind_vendi"] > 1e-6 else None

        if _STATE["cfg"].get("vqa_url"):
            out["vqa_base"] = [float(x) for x in rs.vqa_align(images, base_prompt, _STATE["cfg"])]
            if len(prompts) == len(images):
                out["vqa_own"] = [
                    float(x) for x in rs.vqa_pairs(images, list(prompts), _STATE["cfg"])
                ]
            out["vqa_base_min"] = min(out["vqa_base"])
        else:
            notes.append("no --vqa-url: fidelity not measured (no substitute scorer)")

    out["notes"] = notes
    return out


def route_health(_p):
    return {
        "status": "ok",
        "ready": _STATE.get("ready", False),
        "fake": _STATE.get("fake"),
        "audit": _STATE.get("audit"),
        "loaded": _STATE.get("loaded"),
        "models": sorted(_STATE.get("models", [])),
        "model_path": _STATE.get("cfg", {}).get("model_path"),
        "vqa_url": _STATE.get("cfg", {}).get("vqa_url") or None,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19102)
    ap.add_argument(
        "--models",
        default=None,
        help="comma-separated config.T2I_MODELS keys this server may load (default: every "
        "one whose weights are on disk). One is resident at a time; /render switches by "
        "reloading.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="which to load at startup (default: the first served T2I model)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-batch", type=int, default=8)
    ap.add_argument(
        "--vqa-url",
        default=config.VQA_URL,
        help="official VQAScore service (axis_grpo/training/vqa_service.py). "
        "Without it the audit reports diversity but no fidelity.",
    )
    ap.add_argument(
        "--no-audit",
        action="store_true",
        help="do not load TPIPS (~16 GiB). /render still works; /audit 400s.",
    )
    ap.add_argument(
        "--fake",
        action="store_true",
        help="no GPU: placeholder images, for driving the loop end to end",
    )
    a = ap.parse_args()
    a.model = a.model or config.default_model()
    _STATE["default_model"] = a.model

    _STATE["fake"] = a.fake
    _STATE["audit"] = not a.no_audit and not a.fake
    from axis_framework.backend.rollout import seed_for

    _STATE["seed_for"] = seed_for

    if a.models:
        keys = [k.strip() for k in a.models.split(",") if k.strip()]
        unknown = [k for k in keys if k not in config.T2I_MODELS]
        if unknown:
            raise SystemExit(f"unknown --models {unknown}; " f"have {sorted(config.T2I_MODELS)}")
    else:
        # Default: every model whose weights are on disk.
        keys = [k for k, v in config.T2I_MODELS.items() if a.fake or os.path.isdir(v["path"])]
    _STATE["models"] = set(keys)
    if a.model not in _STATE["models"]:
        raise SystemExit(f"--model {a.model} is not among --models {sorted(_STATE['models'])}")

    _STATE["capacity"] = (0, 0)
    if a.fake:
        _STATE["cfg"] = {
            "model_path": "(fake)",
            "model_family": config.T2I_MODELS[a.model]["family"],
            "render_batch": a.render_batch,
            "vqa_url": "",
        }
        _STATE["loaded"] = a.model
        _STATE["ready"] = True
    else:
        from axis_framework.backend import render as rs

        _STATE["rs"] = rs
        spec = config.T2I_MODELS[a.model]
        cfg = rs.default_cfg(
            device=a.device,
            model_path=spec["path"],
            model_family=spec["family"],
            render_batch=a.render_batch,
            **_startup_settings(a.model),
            vqa_url=a.vqa_url,
            # VQA is optional here (audit only); /audit notes when it is missing.
            require_vqa=bool(a.vqa_url),
            require_tpips=_STATE["audit"],
        )
        _STATE["cfg"] = cfg
        t0 = time.time()
        rs.load_models(cfg)
        _STATE["loaded"] = a.model
        _STATE["capacity"] = _capacity()
        gpu, host = _STATE["capacity"]
        print(
            f"[render] {a.model} + scorers up in {time.time() - t0:.0f}s "
            f"(capacity: {gpu:.0f}GB GPU, {host:.0f}GB host)",
            flush=True,
        )
        for k in sorted(_STATE["models"]):
            ok, why = _availability(k)
            if not ok:
                print(f"[render] {k} UNAVAILABLE: {why}", flush=True)
        _STATE["ready"] = True

    serve(
        {
            ("GET", "/health"): route_health,
            ("GET", "/models"): route_models,
            ("POST", "/render"): route_render,
            ("POST", "/edit"): route_edit,
            ("POST", "/audit"): route_audit,
        },
        a.host,
        a.port,
        "render",
        ready_msg=f"(fake={a.fake}, audit={_STATE['audit']}, "
        f"loaded={_STATE['loaded']}, models={sorted(_STATE['models'])})",
    )


if __name__ == "__main__":
    main()
