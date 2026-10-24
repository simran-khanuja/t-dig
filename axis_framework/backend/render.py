"""Rendering, editing and audit scoring for the interactive framework's T2I model server.

The T2I model server holds one diffusers pipeline (`_STATE["pipe"]`), optionally TPIPS for the
audit, and talks to a VQAScore service (`axis_grpo/training/vqa_service.py`) over HTTP. A render
config `cfg` (see `default_cfg`) selects the model family, resolution, steps and guidance.
TPIPS needs `tpips` 0.1.2 with transformers 5.3.0 (PY_REWARD).
"""

import inspect
import json
import os
import threading
import time

import numpy as np

TPIPS_CKPT = "sywang/TPIPS-Embed-Qwen3VL-8B"

# Editor for each family's images, for the layered task (image + instruction -> edited
# image). `pipeline` is the diffusers class. `path` is None when the editor is the generator's
# own checkpoint, else a Hugging Face repo id. `instruction` says what `prompt` means:
#
#   "edit"   a command describing a change ("change the lighting to dusk"); no `strength`.
#   "sdedit" a description of the finished image; `strength` sets how far the result may move
#            from the input. A command written here is rendered as a picture of the command.
#
# layered.INSTRUCTION_FOR picks the round-two template by `instruction`.
EDIT_MODELS = {
    # Flux2KleinPipeline takes `image`, so the generator edits with no second checkpoint.
    "flux2klein": {
        "pipeline": "Flux2KleinPipeline",
        "path": None,
        "instruction": "edit",
        "defaults": {"num_inference_steps": 4, "guidance_scale": 1.0},
    },
    # QwenImageEditPlusPipeline needs a `processor` component that models/qwen-image-2512
    # does not ship, so editing uses the separate Qwen-Image-Edit weights (QwenImageImg2Img
    # on the generator's weights is sdedit). Qwen-Image-Edit-2509 cannot generate from text
    # alone, so a qwen run holds two checkpoints. Defaults are from its model card.
    "qwen-image": {
        "pipeline": "QwenImageEditPlusPipeline",
        "path": "Qwen/Qwen-Image-Edit-2509",
        "instruction": "edit",
        "guidance_kwarg": "true_cfg_scale",
        "defaults": {"num_inference_steps": 40, "true_cfg_scale": 4.0, "negative_prompt": " "},
    },
    # SD3.5 has no instruction-editing model; img2img on the generator's weights is sdedit.
    "sd3": {
        "pipeline": "StableDiffusion3Img2ImgPipeline",
        "path": None,
        "instruction": "sdedit",
        "defaults": {"num_inference_steps": 40, "guidance_scale": 4.5, "strength": 0.6},
    },
}

# cfg["model_family"] -> diffusers pipeline class, extra __call__ kwargs, and latent shape info
# for the shared latent. A family with shared_latents=False raises rather than guess a
# latent shape.
MODEL_FAMILIES = {
    "sd3": {
        "pipeline_cls": "StableDiffusion3Pipeline",
        "extra_kwargs": lambda cfg: {"guidance_scale": cfg["guidance"]},
        "shared_latents": True,
        "latent_channels": lambda pipe: pipe.transformer.config.in_channels,
        "channels_kwarg": "num_channels_latents",
    },
    "flux2klein": {
        "pipeline_cls": "Flux2KleinPipeline",
        # FLUX.2-Klein is step-distilled; its model card specifies 4 steps and
        # guidance_scale 1.0. Forwarding cfg["guidance"] lets the grid settings
        # override the pipeline default of 4.0.
        "extra_kwargs": lambda cfg: {"guidance_scale": cfg["guidance"]},
        "shared_latents": True,
        "latent_channels": lambda pipe: pipe.transformer.config.in_channels // 4,
        "channels_kwarg": "num_latents_channels",  # unlike the other families
        # The pipeline packs the latent itself, so it takes the unpacked shape.
        "unpacked_shape": lambda pipe, res: (
            1,
            pipe.transformer.config.in_channels,
            res // (pipe.vae_scale_factor * 2),
            res // (pipe.vae_scale_factor * 2),
        ),
    },
    "qwen-image": {
        "pipeline_cls": "QwenImagePipeline",
        # cfg["guidance"] is Qwen's true_cfg_scale.
        "extra_kwargs": lambda cfg: {"true_cfg_scale": cfg["guidance"], "negative_prompt": ""},
        "shared_latents": True,
        "latent_channels": lambda pipe: pipe.transformer.config.in_channels // 4,
        "channels_kwarg": "num_channels_latents",
    },
}

_STATE = {}

# Defaults of a render config; the T2I model server sets the model (model_path, model_family),
# the grid settings and the VQAScore URL. require_tpips / require_vqa (default True) are read
# with .get().
DEFAULTS = {
    "device": "cuda",
    "model_family": "sd3",
    "resolution": 512,
    "steps": 16,
    "guidance": 4.5,
    "render_batch": 16,
    "vqa_url": None,
    "edit_steps": 16,
    "edit_guidance": 4.5,
    "edit_strength": 0.6,
}


def default_cfg(**overrides):
    """Return DEFAULTS updated with `overrides`."""
    return {**DEFAULTS, **overrides}


def _patch_processor_kwarg():
    """Drop the ``fix_mistral_regex`` kwarg tpips 0.1.2 passes, which transformers rejects."""
    from transformers import Qwen3VLProcessor

    orig = Qwen3VLProcessor.from_pretrained
    if getattr(orig, "_patched", False):
        return

    def from_pretrained(*args, **kw):
        kw.pop("fix_mistral_regex", None)
        return orig(*args, **kw)

    from_pretrained._patched = True
    Qwen3VLProcessor.from_pretrained = from_pretrained


def load_models(cfg):
    """Load the T2I model and TPIPS on cfg["device"], and check the VQAScore service."""
    import torch

    dev, dtype = cfg["device"], torch.bfloat16

    if "pipe" not in _STATE:
        import diffusers

        family = cfg.get("model_family", "sd3")
        if family not in MODEL_FAMILIES:
            raise NotImplementedError(
                f"--model-family {family!r} is unknown. Wired up: {sorted(MODEL_FAMILIES)}."
            )
        pipe_cls = getattr(diffusers, MODEL_FAMILIES[family]["pipeline_cls"])

        t0 = time.time()
        pipe = pipe_cls.from_pretrained(cfg["model_path"], torch_dtype=dtype)
        pipe = pipe.to(dev)
        pipe.set_progress_bar_config(disable=True)
        # Decode one image at a time to avoid the batched VAE decode's memory spike;
        # denoising stays batched.
        pipe.vae.enable_slicing()
        _STATE["pipe"] = pipe
        print(
            f"[load] {family} ({MODEL_FAMILIES[family]['pipeline_cls']}) in "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )

    # require_tpips=False (default True) skips TPIPS (~16 GiB of the ~34 GiB) for callers that
    # only render, such as axis_framework's T2I model server launched --no-audit.
    if cfg.get("require_tpips", True) and "tpips" not in _STATE:
        import tpips as tpips_mod

        t0 = time.time()
        _patch_processor_kwarg()
        _STATE["tpips"] = tpips_mod.load_model("embedding", TPIPS_CKPT, device=dev)
        print(f"[load] TPIPS in {time.time() - t0:.0f}s", flush=True)

    # VQAScore runs in its own process because t2v_metrics and tpips need incompatible
    # transformers versions. Check here that it is reachable.
    import urllib.request

    # require_vqa=False (default True) skips the check, for a T2I model server without an audit.
    if not cfg.get("require_vqa", True):
        return
    url = (cfg.get("vqa_url") or "").rstrip("/")
    if not url:
        raise RuntimeError("--vqa-url is required (see axis_grpo/training/vqa_service.py)")
    try:
        with urllib.request.urlopen(url + "/health", timeout=30) as r:
            h = json.loads(r.read())
    except Exception as e:
        raise RuntimeError(f"vqa_service at {url} is not reachable: {e}") from e
    if not h.get("loaded"):
        raise RuntimeError(f"vqa_service at {url} has no model loaded: {h}")
    # The service scores only images inside its --staging directory, so write them there.
    if h.get("staging"):
        cfg["vqa_tmpdir"] = h["staging"]
    print(f"[load] VQA service OK at {url} ({h.get('model')})", flush=True)

    if torch.cuda.is_available():
        print(f"[load] GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)


def _one_latent(seed, cfg):
    """Return one noise sample [1, ...] in the latent layout of cfg's model family.

    Only sd3 has a plain [1,C,h,w] latent; the other families' layouts come from the
    pipeline. `seed` goes to manual_seed unchanged; callers derive their own seeds.
    """
    import torch

    pipe = _STATE["pipe"]
    h = w = cfg["resolution"]
    family = cfg.get("model_family", "sd3")
    g = torch.Generator(device=cfg["device"]).manual_seed(int(seed))

    if family == "sd3":
        return torch.randn(
            (
                1,
                pipe.transformer.config.in_channels,
                h // pipe.vae_scale_factor,
                w // pipe.vae_scale_factor,
            ),
            generator=g,
            device=cfg["device"],
            dtype=torch.bfloat16,
        )

    # Other families pack their latents differently, so the shape comes from the pipeline.
    spec = MODEL_FAMILIES[family]
    if "unpacked_shape" in spec:
        return torch.randn(
            spec["unpacked_shape"](pipe, h), generator=g, device=cfg["device"], dtype=torch.bfloat16
        )

    kw = {
        "batch_size": 1,
        "height": h,
        "width": w,
        "dtype": torch.bfloat16,
        "device": cfg["device"],
        "generator": g,
        "latents": None,
        spec["channels_kwarg"]: spec["latent_channels"](pipe),
    }
    z = pipe.prepare_latents(**kw)
    if isinstance(z, tuple):
        z = z[0]  # some pipelines also return latent_image_ids, recomputed by __call__
    return z


def _shared_latent(seed, k, cfg):
    """Return one noise sample repeated k times, [k,C,h,w], so the set shares a latent.

    Raises NotImplementedError for a family with shared_latents=False (see MODEL_FAMILIES).
    """
    family = cfg.get("model_family", "sd3")
    if not MODEL_FAMILIES[family]["shared_latents"]:
        raise NotImplementedError(
            f"--model-family {family!r} has no verified shared-latent packing (see "
            f"MODEL_FAMILIES)."
        )

    z = _one_latent(seed, cfg)
    return z.repeat(k, *([1] * (z.ndim - 1)))


def _render_chunk(prompts, latents, cfg):
    import torch

    family = cfg.get("model_family", "sd3")
    with torch.inference_mode():
        return _STATE["pipe"](
            prompt=list(prompts),
            latents=latents,
            num_inference_steps=cfg["steps"],
            height=cfg["resolution"],
            width=cfg["resolution"],
            **MODEL_FAMILIES[family]["extra_kwargs"](cfg),
        ).images


def pairwise_distance(E):
    """Return the mean pairwise distance (1 - cosine) over L2-normalised rows of E.

    Cosine is unclipped, matching tpips' ``model.distance``.
    """
    E = E.astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    K = E @ E.T
    iu = np.triu_indices(K.shape[0], k=1)
    return float(np.mean(1.0 - K[iu]))


def vendi_score(E):
    """Return the Vendi score, the effective number of distinct images, in [1, K].

    exp(Shannon entropy of the eigenvalues of the K x K cosine-similarity matrix / K).
    1.0 means all images identical, K means all mutually orthogonal. A single outlier image
    moves it less than it moves mean pairwise distance.
    """
    E = E.astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    S = E @ E.T
    w = np.linalg.eigvalsh(S / len(E))
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def normalised_vendi(E):
    """Return Vendi mapped to [0, 1] as (VS - 1) / (K - 1); 0.0 when K < 2.

    0 means all identical, 1 all orthogonal.
    """
    k = len(E)
    return 0.0 if k < 2 else (vendi_score(E) - 1.0) / (k - 1.0)


def tpips_embed(images, factor, cfg, base=False):
    """Return [n,d] float32 TPIPS embeddings of *images* conditioned on *factor*.

    ``base=True`` disables the TPIPS LoRA adapter, giving the untuned Qwen3-VL-8B backbone's
    embedding at no extra memory.
    """
    import torch

    model = _STATE["tpips"]
    with torch.inference_mode():
        if base:
            lm = model.backbone.model.language_model
            with lm.disable_adapter():
                E = model.embed(list(images), factor=factor, normalized=True)
        else:
            E = model.embed(list(images), factor=factor, normalized=True)
    return E.float().cpu().numpy()


class ScorerDown(RuntimeError):
    """Raised when the VQAScore service fails or returns a malformed reply."""


def vqa_align(images, text, cfg):
    """Return the official VQAScore of each image against one text (the fidelity term)."""
    return vqa_pairs(images, [text] * len(images), cfg)


def vqa_pairs(images, texts, cfg):
    """Return [n] official VQAScores, image i scored against text i, in one request.

    Images go to cfg["vqa_tmpdir"] and are sent as paths, since the VQA service runs on the same
    node. `load_models` sets it to the service's staging directory; the fallback is
    $TMPDIR/axis_grpo_vqa. Raises ScorerDown on any failure; there is no fallback scorer.
    """
    import urllib.error
    import urllib.request

    if len(images) != len(texts):
        raise ScorerDown(f"vqa_pairs got {len(images)} images and {len(texts)} texts")

    url = cfg.get("vqa_url")
    if not url:
        raise ScorerDown("--vqa-url is required (see axis_grpo/training/vqa_service.py)")

    tmp = cfg.get("vqa_tmpdir") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "axis_grpo_vqa")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    try:
        # Write each distinct image once, even if it's scored against multiple texts.
        seen = {}
        for im in images:
            if id(im) not in seen:
                fp = os.path.join(tmp, f"vqa_{os.getpid()}_{threading.get_ident()}_{len(seen)}.png")
                im.save(fp)
                paths.append(fp)
                seen[id(im)] = fp
        body = json.dumps({"pairs": [[seen[id(im)], t] for im, t in zip(images, texts)]}).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/score", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Read the body to report the server-side error, not just the status code.
            detail = e.read().decode("utf-8", "replace")[:500]
            raise ScorerDown(f"vqa_service at {url} returned {e.code}: {detail}") from e
        if "error" in resp:
            raise ScorerDown(f"vqa_service error: {resp['error']}")
        scores = resp["scores"]
        if len(scores) != len(images):
            raise ScorerDown(
                f"vqa_service returned {len(scores)} scores for " f"{len(images)} pairs"
            )
        return np.array(scores, dtype=np.float32)
    except urllib.error.URLError as e:
        raise ScorerDown(f"vqa_service at {url} unreachable ({e})") from e
    finally:
        for fp in paths:
            try:
                os.remove(fp)
            except OSError:
                pass


def load_editor(cfg):
    """Load this family's image editor into ``_STATE["editor"]`` and return the pipeline.

    Args:
        cfg: render config. Reads ``model_family`` and ``model_path``.

    Raises ``NotImplementedError`` if the family has no entry in ``EDIT_MODELS``.

    A family whose editor is the generator reuses the resident pipeline: FLUX.2-Klein is
    ~28 GiB, and a second copy would leave no room for TPIPS.
    """
    if "editor" in _STATE:
        return _STATE["editor"]

    import torch

    family = cfg.get("model_family", "sd3")
    spec = EDIT_MODELS.get(family)
    if spec is None:
        raise NotImplementedError(
            f"no image editor is wired up for family {family!r}. "
            f"Families with one: {sorted(EDIT_MODELS)}."
        )

    path = spec["path"] or cfg["model_path"]
    same_weights = spec["path"] is None
    gen_cls = MODEL_FAMILIES[family]["pipeline_cls"]

    if same_weights and spec["pipeline"] == gen_cls and "pipe" in _STATE:
        _STATE["editor"] = _STATE["pipe"]
        _STATE["editor_spec"] = spec
        return _STATE["editor"]

    import diffusers

    cls = getattr(diffusers, spec["pipeline"])
    t0 = time.time()
    if same_weights and "pipe" in _STATE:
        # Same checkpoint, different pipeline class: reuse the resident components.
        pipe = cls(
            **{
                k: v
                for k, v in _STATE["pipe"].components.items()
                if k in inspect.signature(cls.__init__).parameters
            }
        )
    else:
        pipe = cls.from_pretrained(path, torch_dtype=torch.bfloat16)
    pipe = pipe.to(cfg["device"])
    pipe.set_progress_bar_config(disable=True)
    _STATE["editor"] = pipe
    _STATE["editor_spec"] = spec
    print(f"[load] editor {spec['pipeline']} from {path} in {time.time() - t0:.0f}s", flush=True)
    return pipe


def edit_images(sources, instructions, cfg, seed=None):
    """Apply one instruction to each source image and return the edited PIL images.

    Args:
        sources: file paths or PIL images, one per instruction. An entry may be a list of
            references to combine into one output (edit pipelines only).
        instructions: what the editor is told to do, in the same order.
        cfg: render config. Reads ``edit_steps``, ``edit_guidance``, ``edit_strength`` and
            ``resolution``.

    Raises ``ValueError`` if the two lists differ in length, or if an sdedit editor gets a
    list of sources.

    Each image is edited in its own call so that instruction i is applied to image i.
    """
    from PIL import Image

    if len(sources) != len(instructions):
        raise ValueError(f"{len(sources)} sources against {len(instructions)} instructions")

    pipe = load_editor(cfg)
    spec = _STATE["editor_spec"]
    res = cfg["resolution"]
    # The editor's published defaults (EDIT_MODELS), overridden by cfg edit_steps /
    # edit_guidance when set (the render server's /edit always sets both).
    kw = dict(spec.get("defaults") or {})
    if cfg.get("edit_steps") is not None:
        kw["num_inference_steps"] = cfg["edit_steps"]
    if cfg.get("edit_guidance") is not None:
        # Qwen-Image's CFG knob is true_cfg_scale, as for its generator (MODEL_FAMILIES).
        kw[spec.get("guidance_kwarg", "guidance_scale")] = cfg["edit_guidance"]
    if spec["instruction"] == "sdedit" and cfg.get("edit_strength") is not None:
        # Only sdedit pipelines take strength.
        kw["strength"] = cfg["edit_strength"]
    if seed is not None:
        import torch

        # Every edit of a set uses the set's seed, as in training (axis_grpo.training.backend).
        kw["generator"] = torch.Generator("cpu").manual_seed(int(seed))

    def load(src):
        im = Image.open(src).convert("RGB") if isinstance(src, str) else src
        return im.resize((res, res), Image.LANCZOS) if im.size != (res, res) else im

    out = []
    for src, instr in zip(sources, instructions):
        # A list of sources (the framework's "Combine") is several references for one output.
        # FLUX.2-Klein and Qwen-Image-Edit-2509 accept a list; an sdedit pipeline does not.
        if isinstance(src, (list, tuple)):
            if spec["instruction"] == "sdedit":
                raise ValueError("this editor redraws one image and cannot combine several")
            im = [load(x) for x in src]
        else:
            im = load(src)
        out.append(pipe(image=im, prompt=instr, height=res, width=res, **kw).images[0])
    return out
