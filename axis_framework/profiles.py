"""Model-specific serving settings, read from the training runs being served.

A checkpoint is served the way it was trained: same base model, T2I model, resolution, steps,
guidance, image editor and prompt budget. Only the runs to serve are configured; everything
else is read from each run's descriptor (served_runs/<exp>.json) or, for a round-one run, its
training log.

    AXIS_FW_SERVE=<round-one run>[:<layered run>][+<round-one run>[:<layered run>]...]

`+` separates T2I models (sbatch --export splits on commas). The T2I model each run trained against is
read from the run, not named here.

Derived values and their sources in a round-one run's log:
  T2I model        --model-path in the run's reward_service.py flags, matched to t2i_models.json
  policy base     actor_rollout_ref.model.path   (and so whether it is a vision policy)
  LoRA rank       actor_rollout_ref.model.lora_rank
  context         actor_rollout_ref.rollout.max_model_len / limit_images
  grid render     --val-resolution / --val-steps / --val-guidance of the round-one run (the
                  settings its checkpoints were scored at)
  i.i.d. render   the sidecars in axis_grpo/iid/<T2I model>/ -- the sets the axes (and a
                  vision policy's round-one input) came from
  edit render     --edit-steps / --edit-guidance / --train-resolution of the layered run
  accept render   the i.i.d. resolution at the grid's steps and guidance
  prompt budget   the T2I model family (instructions.LENGTH_BUDGETS)
  editor kind     the T2I model family (render.EDIT_MODELS)

Overrides: AXIS_FW_<STAGE>_{RESOLUTION,STEPS,GUIDANCE} (STAGE = GRID, IID, EDIT, ACCEPT)
replace a derived value for every T2I model; render_settings() lists them under `overridden`.

A run with no descriptor or log, or served runs that disagree on a shared value (one policy
process holds one base model), is an error at startup.
"""

import functools
import glob
import json
import os
import re

from . import config

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("grid", "iid", "edit", "accept")


# ------------------------------------------------------------------------ T2I models


def t2i_models():
    """Return the T2I model registry (t2i_models.json, loaded by config)."""
    return config.T2I_MODELS


def t2i_model_for_path(path):
    """Return the t2i_models.json key whose weights are at *path*, or None."""
    want = os.path.realpath(path)
    for k, v in t2i_models().items():
        if os.path.realpath(v["path"]) == want:
            return k
    return None


def family(model):
    spec = t2i_models().get(model)
    if spec is None:
        raise KeyError(f"unknown T2I model {model!r}; t2i_models.json has {sorted(t2i_models())}")
    return spec["family"]


def length_family(model):
    """Return the instructions.LENGTH_BUDGETS key (text encoder prompt budget) for *model*."""
    from axis_framework.backend import instructions as bd

    fam = family(model)
    for key in sorted(bd.LENGTH_BUDGETS, key=len, reverse=True):
        if fam.startswith(key):
            return key
    raise KeyError(
        f"T2I model family {fam!r} has no entry in instructions.LENGTH_BUDGETS "
        f"({sorted(bd.LENGTH_BUDGETS)}); add one before serving it"
    )


# ------------------------------------------------------------------------ run logs


def _log_candidates():
    return glob.glob(os.path.join(config.LEGACY_RUNS, "logs", "all_*.out")) + glob.glob(
        os.path.join(config.LEGACY_RUNS, "*.o[0-9]*")
    )


# The trainer and reward service command lines are printed in the first ~200 KB of a log
# (logs reach 74 MB), so only the first 1 MiB of each log is read, once.
_HEAD_BYTES = 1 << 20
_EXP_RE = re.compile(r"trainer\.experiment_name=['\"]?([^\s'\"\\]+)")


@functools.lru_cache(maxsize=1)
def _log_index():
    """Return {experiment name: [(mtime, path, head text), ...]} over every training log."""
    idx = {}
    for f in _log_candidates():
        try:
            with open(f, "rb") as fh:
                head = fh.read(_HEAD_BYTES).decode("utf-8", "replace")
        except OSError:
            continue
        # Require --model-path: a job log can name the experiment without the flags.
        if "reward_service.py" not in head or "--model-path" not in head:
            continue
        for exp in set(_EXP_RE.findall(head)):
            idx.setdefault(exp, []).append((os.path.getmtime(f), f, head))
    return idx


def run_log(exp):
    """Return (path, head text) of the newest log of run *exp* that launched its reward service.

    Raises:
        FileNotFoundError: no such log.
    """
    hits = _log_index().get(exp)
    if not hits:
        raise FileNotFoundError(
            f"no training log for run {exp!r} (looked for trainer.experiment_name={exp} in "
            f"{config.LEGACY_RUNS}/logs/all_*.out and {config.LEGACY_RUNS}/*.o*). The framework reads what to "
            f"serve from the run's own log, so it cannot serve a run without one."
        )
    _, path, head = max(hits, key=lambda h: h[0])
    return path, head


def _flag(txt, name, cast=str):
    m = re.search(r"--" + re.escape(name) + r"[ =]+['\"]?([^\s'\"\\]+)", txt)
    return cast(m.group(1)) if m else None


def _kv(txt, name, cast=str):
    m = re.search(r"(?:^|[\s'\"+])" + re.escape(name) + r"=['\"]?([^\s'\"\\]+)", txt, re.M)
    return cast(m.group(1)) if m else None


@functools.lru_cache(maxsize=None)
def descriptor(exp):
    """Return served_runs/<exp>.json as a dict, or None if absent.

    Runs trained with this repository's `axis_grpo` (the paper's policies) have no round-one log,
    so each is described by this file, carrying the fields run_info() returns plus:

      format            "tdig": policy_prompt builds the instruction with tdig.instruction
      instruction_root  optional: another checkout whose tdig/instruction.py to use
      tdig_renderer     the T2I model's name in the tdig package
      relative_budget   rule 4's word cap is the base prompt's words plus this
      ladder            the steps to serve, in slider order; 0 is the untrained policy
      alpha_ladder      optional: true if step N means LoRA scale N/100
      vlm               whether this run's policy is shown images (Qwen3.5 is multimodal, so
                        the base model's name does not say)

    A run with no descriptor is read from its log.
    """
    fp = os.path.join(HERE, "served_runs", f"{exp}.json")
    if not exp or not os.path.exists(fp):
        return None
    d = json.load(open(fp))
    d.setdefault("exp", exp)
    d["log"] = fp
    return d


@functools.lru_cache(maxsize=None)
def run_info(exp):
    """Return run *exp*'s policy, T2I model, render and edit settings, and task."""
    d = descriptor(exp)
    if d is not None:
        if d["renderer"] not in t2i_models():
            raise KeyError(
                f"run {exp!r} names T2I model {d['renderer']!r}, which t2i_models.json "
                f"does not have"
            )
        d.setdefault("image_axis_wording", None)
        d.setdefault("edit", {"steps": None, "guidance": None})
        return d
    path, txt = run_log(exp)
    # Read T2I model flags from the reward service's own command line (one line under set -x),
    # so a same-named flag elsewhere in the log is not picked up.
    rs = next(
        (ln for ln in txt.splitlines() if "reward_service.py" in ln and "--model-path" in ln), txt
    )
    info = {
        "exp": exp,
        "log": path,
        "policy_base": _kv(txt, "actor_rollout_ref.model.path"),
        "lora_rank": _kv(txt, "actor_rollout_ref.model.lora_rank", int),
        "max_model_len": _kv(txt, "actor_rollout_ref.rollout.max_model_len", int),
        "max_prompt_length": _kv(txt, "data.max_prompt_length", int),
        "limit_images": _kv(txt, "++actor_rollout_ref.rollout.limit_images", int)
        or _kv(txt, "actor_rollout_ref.rollout.limit_images", int)
        or 0,
        "renderer_path": _flag(rs, "model-path"),
        "model_family": _flag(rs, "model-family"),
        "train": {
            "resolution": _flag(rs, "train-resolution", int),
            "steps": _flag(rs, "train-steps", int),
            "guidance": _flag(rs, "train-guidance", float),
        },
        "val": {
            "resolution": _flag(rs, "val-resolution", int),
            "steps": _flag(rs, "val-steps", int),
            "guidance": _flag(rs, "val-guidance", float),
        },
        "edit": {
            "steps": _flag(rs, "edit-steps", int),
            "guidance": _flag(rs, "edit-guidance", float),
        },
        "train_file": _kv(txt, "data.train_files"),
    }
    missing = [k for k in ("policy_base", "renderer_path") if not info[k]]
    if missing:
        raise ValueError(f"run {exp!r}: could not read {missing} from {path}")
    info["renderer"] = t2i_model_for_path(info["renderer_path"])
    if info["renderer"] is None:
        raise KeyError(
            f"run {exp!r} trained against {info['renderer_path']}, which no "
            f"t2i_models.json entry points at"
        )
    info["vlm"] = "-VL-" in info["policy_base"]
    info["task"] = _task_of(info["train_file"])
    info["image_axis_wording"] = _image_axis_wording(info["train_file"])
    return info


def _image_axis_wording(train_file):
    """Return how a vision run's image preamble named the axis: "named", "generic" or None.

    A round-one vision run's preamble says `vary it along "<axis>"` (named) or
    `vary it along the Axis` (generic); only the run's first training row records which. None
    for a run with no image preamble (a text policy or a layered run).
    """
    if not train_file or not os.path.exists(train_file):
        return None
    try:
        import pyarrow.parquet as pq

        t = pq.ParquetFile(train_file).read_row_group(0, columns=["prompt"])
        content = t.column("prompt")[0].as_py()[0]["content"]
    except Exception:
        return None
    if "vary it along the Axis" in content:
        return "generic"
    if 'vary it along "' in content:
        return "named"
    return None


def _task_of(train_file):
    """Return "layered" if the run trained on layered (round-two) rows, else "expand"."""
    if not train_file or not os.path.exists(train_file):
        return "expand"
    try:
        import pyarrow.parquet as pq

        t = pq.ParquetFile(train_file).read_row_group(0, columns=["reward_model"])
        gt = t.column("reward_model")[0].as_py()["ground_truth"]
        gt = json.loads(gt) if isinstance(gt, str) else gt
        return "layered" if gt.get("task") == "layered" else "expand"
    except Exception:
        return "layered" if "layered" in os.path.basename(train_file) else "expand"


# ------------------------------------------------------------------------ what is served


@functools.lru_cache(maxsize=1)
def served():
    """Return {T2I model: {"round1": exp, "layer": exp or None}} from AXIS_FW_SERVE."""
    spec = os.environ.get("AXIS_FW_SERVE", "").strip()
    out = {}
    for item in [x for x in re.split(r"[+\s]+", spec) if x]:
        r1, _, layer = item.partition(":")
        i1 = run_info(r1)
        if i1["task"] != "expand":
            raise ValueError(
                f"{r1} is a {i1['task']} run; the part before ':' must be a " f"round-one run"
            )
        entry = {"round1": r1, "layer": None}
        if layer:
            il = run_info(layer)
            if il["task"] != "layered":
                raise ValueError(
                    f"{layer} is not a layered run (its data has no " f"task=layered rows)"
                )
            if il["renderer"] != i1["renderer"]:
                raise ValueError(
                    f"{layer} trained against {il['renderer']} but {r1} against "
                    f"{i1['renderer']}: a layer run edits ITS T2I model's images"
                )
            entry["layer"] = layer
        if i1["renderer"] in out:
            raise ValueError(
                f"two round-one runs for T2I model {i1['renderer']}: "
                f"{out[i1['renderer']]['round1']} and {r1}"
            )
        out[i1["renderer"]] = entry
    return out


def served_runs():
    return [e for v in served().values() for e in (v["round1"], v["layer"]) if e]


def require_served():
    if not served() and os.environ.get("AXIS_FW_FAKE") != "1":
        raise SystemExit(
            "nothing to serve: set AXIS_FW_SERVE=<round-one run>[:<layered run>]"
            " (see axis_framework/profiles.py)"
        )


def default_model():
    """Return the T2I model a session opens on: AXIS_FW_MODEL, else the first one served."""
    m = os.environ.get("AXIS_FW_MODEL")
    if m:
        return m
    if served():
        return next(iter(served()))
    if os.environ.get("AXIS_FW_FAKE") == "1":
        return next(iter(t2i_models()))
    require_served()


def round1_exp(model):
    return (served().get(model) or {}).get("round1") or ""


def layer_exp(model):
    return (served().get(model) or {}).get("layer") or ""


def off_axis_reward(exp):
    """Return (lam, tau) parsed from a run name, for display; None for each part absent."""
    m = re.search(r"_lam(\d+p\d+)_tau(\d+p\d+)", exp or "")
    if m:
        return tuple(float(g.replace("p", ".")) for g in m.groups())
    m = re.search(r"_lam(\d+p\d+)", exp or "")
    return (float(m.group(1).replace("p", ".")), None) if m else (None, None)


# ------------------------------------------------------------------------ the policy


def _one(field):
    vals = {run_info(e)[field] for e in served_runs()}
    vals.discard(None)
    if len(vals) > 1:
        raise ValueError(
            f"served runs disagree on {field}: "
            f"{ {e: run_info(e)[field] for e in served_runs()} }. One policy "
            f"process holds one base model; serve compatible runs together."
        )
    return next(iter(vals), None)


def policy_base():
    """Return the base model of every served run's adapters (AXIS_FW_POLICY_BASE overrides)."""
    b = os.environ.get("AXIS_FW_POLICY_BASE") or _one("policy_base")
    if not b:
        require_served()
    return b


def run_is_vlm(exp):
    """Return whether run *exp*'s policy is shown images; False for an unknown or empty run."""
    if not exp:
        return False
    try:
        return bool(run_info(exp)["vlm"])
    except (KeyError, ValueError):
        return False


def policy_is_vlm():
    """Return whether the policy process must load a processor and accept images.

    True when any served run is shown images. Whether a request carries images depends on the
    run answering it (run_is_vlm): one Qwen3.5 base serves a text-input round-one run and an
    image-input editor run side by side.
    """
    if os.environ.get("AXIS_FW_POLICY_BASE"):
        return "-VL-" in (policy_base() or "")
    return any(run_is_vlm(e) for e in served_runs())


def policy_lora_rank():
    ranks = [run_info(e)["lora_rank"] for e in served_runs() if run_info(e)["lora_rank"]]
    return max(ranks) if ranks else 32


def policy_max_model_len():
    lens = [run_info(e)["max_model_len"] for e in served_runs() if run_info(e)["max_model_len"]]
    return max(lens) if lens else 4096


def policy_max_images():
    n = [run_info(e)["limit_images"] for e in served_runs()]
    return max(
        n + [config.N_IID if policy_is_vlm() else 0, config.K_TRAINED if policy_is_vlm() else 0]
    )


# ------------------------------------------------------------------------ render settings


def _iid_meta(model):
    spec = t2i_models().get(model, {})
    d = os.path.join(config.LEGACY_RUNS, "iid", spec.get("iid_dir", model))
    for fp in sorted(glob.glob(os.path.join(d, "*.json")))[:20]:
        try:
            m = json.load(open(fp))
        except (OSError, ValueError):
            continue
        g = m.get("guidance_scale", m.get("guidance"))
        if m.get("resolution") and m.get("steps") and g is not None:
            return {
                "resolution": int(m["resolution"]),
                "steps": int(m["steps"]),
                "guidance": float(g),
            }
    return None


def _override(stage, out):
    for field, cast in (("resolution", int), ("steps", int), ("guidance", float)):
        v = os.environ.get(f"AXIS_FW_{stage.upper()}_{field.upper()}")
        if v not in (None, ""):
            out[field] = cast(v)
            out.setdefault("overridden", []).append(field)
    return out


@functools.lru_cache(maxsize=None)
def _derived(model, stage):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    r1 = round1_exp(model)
    if not r1:
        if os.environ.get("AXIS_FW_FAKE") == "1":
            return {"resolution": 512, "steps": 4, "guidance": 1.0, "source": "fake"}
        raise KeyError(
            f"T2I model {model!r} is not served (AXIS_FW_SERVE serves "
            f"{sorted(served())}); it has no trained settings to render with"
        )
    i1 = run_info(r1)
    grid = {k: v for k, v in i1["val"].items() if v is not None} or {
        k: v for k, v in i1["train"].items() if v is not None
    }
    grid = {**{k: v for k, v in i1["train"].items() if v is not None}, **grid}
    if stage == "grid":
        return {**grid, "source": f"{r1} validation settings"}
    van = _iid_meta(model)
    if stage == "iid":
        return (
            {**van, "source": f"axis_grpo/iid/{t2i_models()[model].get('iid_dir')}"}
            if van
            else {**grid, "source": f"no i.i.d. sidecars; {r1} grid settings"}
        )
    if stage == "accept":
        res = max(grid["resolution"], (van or grid)["resolution"])
        return {**grid, "resolution": res, "source": "iid resolution, grid steps/guidance"}
    lay = layer_exp(model)
    if not lay:
        raise KeyError(f"T2I model {model!r} has no layered run, so no edit settings")
    il = run_info(lay)
    return {
        "resolution": il["edit"].get("resolution")
        or il["train"]["resolution"]
        or grid["resolution"],
        "steps": il["edit"]["steps"] or il["train"]["steps"] or grid["steps"],
        "guidance": (
            il["edit"]["guidance"]
            if il["edit"]["guidance"] is not None
            else il["train"]["guidance"] or grid["guidance"]
        ),
        # sdedit only: edit strength (None = service default).
        "strength": il["edit"].get("strength"),
        "source": f"{lay} editor settings",
    }


def render_settings(model, stage):
    """Return {resolution, steps, guidance, source[, overridden]} for *model* at *stage*."""
    return _override(stage, dict(_derived(model, stage)))


def summary():
    """Return every derived value, for /health and the launch log."""
    out = {"serve": served(), "policy_base": None, "vlm": None, "renderers": {}}
    try:
        out["policy_base"] = policy_base()
        out["vlm"] = policy_is_vlm()
        out["lora_rank"] = policy_lora_rank()
        out["max_model_len"] = policy_max_model_len()
    except Exception as e:
        out["policy_error"] = str(e)
    for m, v in served().items():
        st = {}
        for stage in STAGES:
            try:
                st[stage] = render_settings(m, stage)
            except KeyError:
                pass
        out["renderers"][m] = {
            **v,
            "family": family(m),
            "length_family": length_family(m),
            "settings": st,
        }
    return out


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2, default=str))
