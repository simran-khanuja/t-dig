"""Explicit shared-T2I-model configuration and pure validation."""

import math

T2I_MODELS = {"sd35-large", "sd35-medium", "flux2-klein-9b", "qwen-image-2512"}


def t2i_model_defaults(method, t2i_model):
    if t2i_model not in T2I_MODELS:
        raise ValueError("Unknown T2I model")
    cfg = defaults(method)
    cfg["renderer"] = t2i_model
    if t2i_model == "flux2-klein-9b":
        cfg.update(steps=4, guidance=1.0, max_sequence_length=512)
        cfg["cads"]["conditioning"] = "token-only-conditional"
    elif t2i_model == "qwen-image-2512":
        cfg.update(steps=50, guidance=4.0, max_sequence_length=512)
    return cfg


def defaults(method):
    cfg = {
        "method": method,
        "renderer": "sd35-large",
        "k": 4,
        "seed": 42,
        "resolution": 1024,
        "steps": 40,
        "guidance": 4.5,
        "max_sequence_length": 256,
        "dtype": "bfloat16",
        "device": "cuda:0",
        "negative_prompt": "",
        "cads": {
            "tau1": 0.6,
            "tau2": 0.9,
            "noise_scale": 0.25,
            "psi": 1.0,
            "conditioning": "token-only-both-branches",
            "time": "sampling-progress",
        },
        "group": {
            "candidates": 128,
            "retention": 0.75,
            "lambda": 1.0,
            "solver_seconds": 30.0,
            "accept_incumbent": False,
            "preview": "pre-step-x-minus-sigma-v",
            "decoder": "renderer-vae",
            "first_step_cfg": True,
        },
        "pag": {
            "decoding": "diverse-beam-released",
            "length_penalty": -1.0,
            "diversity_penalty": 1.0,
            "min_new_tokens": 15,
            "max_new_tokens": 75,
        },
    }
    return cfg


def validate(cfg):
    if set(cfg) != set(defaults(cfg.get("method"))):
        raise ValueError("Configuration has missing or unknown fields")
    if cfg["method"] not in {"iid", "pag", "cads", "group"} or cfg["renderer"] not in T2I_MODELS:
        raise ValueError("Unsupported method or T2I model")
    for key in ("k", "resolution", "steps", "max_sequence_length"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["resolution"] % 16 or cfg["max_sequence_length"] > 512:
        raise ValueError("Resolution must be divisible by 16; sequence length must be <=512")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**63:
        raise ValueError("Seed must be a nonnegative 63-bit integer")
    if cfg["dtype"] not in {"bfloat16", "float32"} or not isinstance(cfg["negative_prompt"], str):
        raise ValueError("Invalid dtype or negative prompt")
    if not isinstance(cfg["device"], str) or not cfg["device"].startswith("cuda:"):
        raise ValueError("Select an explicit CUDA device, e.g. cuda:0")
    for key, template in [
        ("cads", defaults("cads")["cads"]),
        ("group", defaults("group")["group"]),
        ("pag", defaults("pag")["pag"]),
    ]:
        if set(cfg[key]) != set(template):
            raise ValueError(f"Missing/unknown {key} settings")
    c, g, p = cfg["cads"], cfg["group"], cfg["pag"]
    numeric = [
        cfg["guidance"],
        c["tau1"],
        c["tau2"],
        c["noise_scale"],
        c["psi"],
        g["retention"],
        g["lambda"],
        g["solver_seconds"],
        p["length_penalty"],
        p["diversity_penalty"],
    ]
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in numeric):
        raise ValueError("All numeric settings must be finite")
    if cfg["renderer"] == "flux2-klein-9b":
        if cfg["guidance"] != 1 or cfg["steps"] != 4 or cfg["negative_prompt"]:
            raise ValueError(
                "Distilled Klein requires four steps, guidance=1, empty negative prompt"
            )
        if c["conditioning"] != "token-only-conditional":
            raise ValueError("Distilled Klein has one conditional branch")
    elif cfg["guidance"] <= 1:
        raise ValueError("Non-distilled comparisons require CFG >1")
    if not 0 <= c["tau1"] < c["tau2"] <= 1 or c["noise_scale"] < 0 or not 0 <= c["psi"] <= 1:
        raise ValueError("Invalid CADS schedule/noise/rescale")
    if c["conditioning"] not in {
        "token-only-both-branches",
        "token-and-pooled-both-branches",
        "token-only-conditional",
    } or c["time"] not in {"sampling-progress", "scheduler-timestep/1000"}:
        raise ValueError("Unsupported CADS variant")
    if cfg["renderer"] != "flux2-klein-9b" and c["conditioning"] == "token-only-conditional":
        raise ValueError("Conditional-only CADS is reserved for distilled Klein")
    if cfg["renderer"] == "qwen-image-2512" and c["conditioning"] != "token-only-both-branches":
        raise ValueError("Qwen has token conditioning, no SD3 pooled projections")
    if not cfg["renderer"].startswith("sd35") and not g["first_step_cfg"]:
        raise ValueError("Native pool adapter preserves native guidance at every step")
    if (
        type(g["candidates"]) is not int
        or g["candidates"] < 1
        or (cfg["method"] == "group" and g["candidates"] < cfg["k"])
        or not 0 < g["retention"] < 1
    ):
        raise ValueError("Require M>=K and 0<retention<1")
    if g["lambda"] < 0 or g["solver_seconds"] <= 0 or type(g["accept_incumbent"]) is not bool:
        raise ValueError("Invalid group objective/solver settings")
    if (
        g["preview"] != "pre-step-x-minus-sigma-v"
        or g["decoder"] != "renderer-vae"
        or type(g["first_step_cfg"]) is not bool
    ):
        raise ValueError("Unsupported group preview/decoder/CFG setting")
    if p["decoding"] != "diverse-beam-released" or p["diversity_penalty"] <= 0:
        raise ValueError("Unsupported PAG decoding")
    if (
        any(type(p[v]) is not int for v in ["min_new_tokens", "max_new_tokens"])
        or not 0 < p["min_new_tokens"] <= p["max_new_tokens"]
    ):
        raise ValueError("Invalid PAG generation lengths")
    return cfg
