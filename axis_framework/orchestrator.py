"""Orchestrator: the loop state machine, the session store, and the event log.

    GET  /health                                            -> run settings + service health
    GET  /models                                            -> T2I models + capabilities
    GET  /prompts?model=...                                 -> the pre-rendered prompt set
    POST /session/new      {user_id?, model?}               -> {session_id, state}
    POST /session/prompt   {session_id, base_prompt}        -> i.i.d. grid, state=GATE
    POST /session/gate     {session_id, needs_diversity}    -> state=CHOOSE_AXIS | DONE
    POST /session/suggest  {session_id, from?, images?, background?} -> {axes: [chips]}
                             (for a chosen subset; automatic suggestions arrive on the
                              session as `suggested`)
    POST /session/rephrase {session_id, axis}               -> {show, kind, axis, add, axis_raw}
    POST /session/reprompt {session_id, base_prompt, axis?} -> new i.i.d. grid, CHOOSE_AXIS
    POST /session/axis     {session_id, axis, source, off_axis?, k?, mode?} -> the grid, REVIEW
    POST /session/respin   {session_id, axis?, off_axis?, k?, mode?}
                                                            -> a new trial, still REVIEW
    POST /session/rerender {session_id, guidance_scale, init_diversity}
    POST /session/restore  {session_id, trial_id}           -> an earlier round made current
    POST /session/fix      {session_id, image, instruction} -> K edits of one picture, REVIEW
    POST /session/merge    {session_id, images, instruction?} -> K combinations, REVIEW
    POST /session/review   {session_id, decision}           -> state=CHOOSE_AXIS | DONE
    POST /download         {session_id, images}             -> a zip of those images
    GET  /loras?model=...                                   -> the off-axis ladder
    GET  /session/get?session_id=...                        -> the whole session
    POST /client_event     {session_id?, client_id, events} -> UI-only interactions, logged as "ui"
    POST /client_error     {session_id?, message, stack, ...} -> logged as "client_error"
    POST /study/consent, /study/questionnaire; GET /study/instructions  -> study mode
    GET  /media?path=...&w=...                              -> a rendered image
    GET  /                                                  -> the web UI, if built

The orchestrator uses no GPU. It holds the state machine (session.py), calls the three
services over HTTP and writes the event log, so a T2I model crash costs the render in flight,
not the session.

Blinding is enforced here: responses carry images, prompts and slider values, never the arm
or the audit metrics (see _public). Both go to the event log only.

Controls by route:

  * Render-side (text guidance, initial diversity): /rerender re-renders the same prompts
    without calling the policy. The UI does not offer these sliders; live renders use the
    trained defaults (config.render_settings, config.INIT_DIVERSITY). Each render records
    its slider values.
  * Policy-side (off-axis diversity, axis wording, K): /respin re-expands and starts a new
    trial, so a trial is one (axis, arm, K, rung) with one prompt set. Mode `fresh` varies
    the original prompt again; `layer` varies this round's prompts.

Off-axis is offered only when --arms is 'ours' (arm B has no adapter, so a slider that did
nothing would reveal the arm), --no-off-axis is unset and a ladder is exported. /loras and
/health report the reason when it is withheld.

Latency:

  * --t2i-pool shards a grid across replicas of one model; each shard slices the whole
    grid's latents (t2i_server._render), so every image gets the noise it would have had
    on one GPU.
  * _start_speculate expands the suggested axes in one batch before the user picks one. The
    draw is keyed on axis, K, rung, arm and seed and consumed on use.
  * _start_suggest asks the VLM for axis chips whenever a set of images exists (the i.i.d.
    set, then every grid), concurrently with the policy and T2I models. /session/suggest is
    for a suggestion over a user-selected subset.
"""

import argparse
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axis_framework import cache, client, config, profiles, session as S  # noqa: E402
from axis_framework.http_util import HttpError, Raw, serve  # noqa: E402

_STATE = {}

# Background axis suggestions, keyed by session. Kept off the session document because the
# store writes whole documents and a background writer would overwrite a request's changes.
# The request thread drains this into the document (_drain_suggest).
_SUGGEST = {}
_SUGGEST_LOCK = threading.Lock()

# Speculative expansions (_start_speculate), kept off the session for the same reason as
# _SUGGEST. They carry the arm, so keeping them off the session also keeps them out of _public.
_SPEC = {}
_SPEC_LOCK = threading.Lock()


def _render_for(model):
    """Return the T2I model server that owns *model*, else the default server.

    With one server per model, each on its own GPU, no model is evicted and switching costs
    no load (15-120 s otherwise). The default server covers one-GPU runs (`--t2i-url`
    alone, switching models by eviction).
    """
    return _STATE["renders"].get(model) or _STATE["render"]


def _pool_for(model):
    """Return every T2I model server holding *model*, for splitting one grid across GPUs.

    `renders` maps a model to its one owner; a pool is N replicas of the same model.
    `--t2i-pool sd35-large=urlA,urlB,urlC` registers three replicas; `_render_for` names
    the first as the owner, used by /models, /audit and single-shard paths.
    """
    return _STATE["pool"].get(model) or []


def _shards(n, w):
    """Return contiguous [start, stop) slices of a grid of n over w workers, largest first.

    Sizes differ by at most one (4 over 3 is 2/1/1), with the remainder on the first shards.
    """
    w = max(1, min(int(w), int(n)))
    base, rem = divmod(int(n), w)
    out, start = [], 0
    for i in range(w):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out


def _seed_for(base_prompt):
    """Return the training seed for this prompt (`rollout.seed_for`).

    Resolved once here for a sharded grid and sent to every shard, so all shards use the same
    noise even if one server's computation diverged.
    """
    if "seed_for" not in _STATE:
        from axis_framework.backend.rollout import seed_for

        _STATE["seed_for"] = seed_for
    return _STATE["seed_for"](base_prompt)


def _render_grid(model, prompts, base_prompt, out_dir, tag, seed=None, **kw):
    """Render K prompts, split across this model's replica pool when there is one.

    Returns the same dict shape as a single /render, plus `shards`. Each shard draws on the
    whole grid's latents (t2i_server._render) and writes `<tag>.part<offset>.json`; the merged
    `<out_dir>/<tag>.json` is written here. `seconds` is wall clock, i.e. the slowest shard.
    """
    pool = _pool_for(model)
    if len(pool) < 2 or len(prompts) < 2:
        return _render_for(model).render(
            prompts=prompts,
            base_prompt=base_prompt,
            out_dir=out_dir,
            tag=tag,
            seed=seed,
            model=model,
            **kw,
        )

    n = len(prompts)
    cuts = _shards(n, len(pool))
    seed = _seed_for(base_prompt) if seed is None else int(seed)
    t0 = time.time()

    def one(i):
        lo, hi = cuts[i]
        return pool[i].render(
            prompts=prompts[lo:hi],
            base_prompt=base_prompt,
            out_dir=out_dir,
            tag=tag,
            seed=seed,
            model=model,
            index_offset=lo,
            n_total=n,
            **kw,
        )

    with ThreadPoolExecutor(max_workers=len(cuts)) as ex:
        # list() re-raises the first shard failure; a partial grid fails the render.
        parts = list(ex.map(one, range(len(cuts))))

    out = dict(parts[0])
    out["images"] = [p for part in parts for p in part["images"]]
    out["prompts"] = list(prompts)
    out["seed"] = seed
    out.pop("index_offset", None)
    out["n_total"] = n
    out["shards"] = [
        {"url": pool[i].url, "range": list(cuts[i]), "seconds": parts[i].get("seconds")}
        for i in range(len(cuts))
    ]
    out["seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _edit_grid(model, sources, instructions, out_dir, tag, seed):
    """Apply K edits, split across the replica pool as _render_grid does for K prompts.

    Each edit is its own pipeline call (~10 s for four 1024px edits on one server). Writes
    `<out_dir>/<tag>.json` for a pooled edit. `seconds` is the slowest shard's wall clock.
    """
    pool = _pool_for(model)
    if len(pool) < 2 or len(sources) < 2:
        return _render_for(model).edit(
            sources=sources,
            instructions=instructions,
            out_dir=out_dir,
            tag=tag,
            model=model,
            seed=seed,
        )
    cuts = _shards(len(sources), len(pool))
    t0 = time.time()

    def one(i):
        lo, hi = cuts[i]
        return pool[i].edit(
            sources=sources[lo:hi],
            instructions=instructions[lo:hi],
            out_dir=out_dir,
            tag=tag,
            model=model,
            index_offset=lo,
            seed=seed,
        )

    with ThreadPoolExecutor(max_workers=len(cuts)) as ex:
        parts = list(ex.map(one, range(len(cuts))))  # re-raises the first shard failure
    out = dict(parts[0])
    out["images"] = [p for part in parts for p in part["images"]]
    out["sources"], out["instructions"] = list(sources), list(instructions)
    out["shards"] = [
        {"url": pool[i].url, "range": list(cuts[i]), "seconds": parts[i].get("seconds")}
        for i in range(len(cuts))
    ]
    out["seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _caps(model):
    """Return `model`'s capabilities as its T2I model server reports them, cached per model.

    Falls back to the t2i_models.json entry when the server cannot be reached.
    """
    if model not in _STATE["caps"]:
        try:
            for m in _render_for(model).models().get("models", []):
                _STATE["caps"][m["key"]] = m
        except Exception as e:
            print(f"[orch] could not read capabilities for {model}: {e}", flush=True)
            return config.T2I_MODELS.get(model, {})
    return _STATE["caps"].get(model, config.T2I_MODELS.get(model, {}))


def _clamp_alpha(model, alpha):
    """Return *alpha*, or 1.0 if the T2I model lacks verified shared-latent packing.

    The config default alpha=0 would otherwise get a 400 from such a T2I model (e.g. FLUX.1).
    """
    if alpha < 1.0 and _caps(model).get("shared_latents") is False:
        return 1.0
    return alpha


def _can_merge(model):
    """An editing run is served and its editor takes several references ("Combine")."""
    return bool(config.layer_exp_for(model)) and config.model_family(model) in (
        "flux2klein",
        "qwen-image",
    )


def _sess(p):
    sid = p.get("session_id") or ""
    s = _STATE["store"].get(sid)
    if s is None:
        raise HttpError(404, f"no session {sid!r}")
    return s


def _save(s):
    _STATE["store"].put(s)
    return s


def _public(s):
    """Return the session as the participant's browser may see it.

    Arm, audit metrics, raw completions and the instruction are stripped from every trial.
    Anything that bypasses this function unblinds the study.
    """
    st = _drain_suggest(s)
    out = {k: v for k, v in s.items() if k != "trials"}
    out["trials"] = [
        {k: v for k, v in t.items() if k not in ("arm", "audit", "audit_iid", "raw", "instruction")}
        for t in s["trials"]
    ]
    # can_edit: a layered run is served for this T2I model, so "Edit" edits the pictures on
    # screen; otherwise it regenerates from their prompts.
    out["can_edit"] = bool(config.layer_exp_for(s.get("model") or _STATE["model"]))
    out["can_keep"] = out["can_edit"]  # edit rounds honour `keep` (see _render_trial)
    out["can_focus"] = out["can_edit"]  # "Vary this" and "Fix this" on single pictures
    out["can_merge"] = _can_merge(s.get("model") or _STATE["model"])
    # suggest_pending lets the UI show "still looking" instead of an empty chip list.
    out["suggest_pending"] = st["pending"]
    out["suggest_error"] = st["error"]
    return out


# --------------------------------------------------------- automatic axis suggestions

# Axis chips are requested from the suggest server (its own GPU) as soon as a set of images
# exists: the i.i.d. set, then each rendered grid. A Qwen3-VL decode takes ~10-20 s and runs
# while the participant reads the screen and the policy and T2I models work.
#
# Suggestions are best-effort: a failure is recorded in the slot's `error` and the log, never
# in a route's reply. Free-text axis entry does not depend on them.


def _suggest_slot(sid):
    return _SUGGEST.setdefault(sid, {"pending": False, "results": [], "error": None, "gen": 0})


def _with_always(axes, exclude):
    """Return config.ALWAYS_AXES followed by the VLM's axes.

    An always-on axis is skipped when it is in *exclude* or already in *axes*.
    """
    ex = {str(x).strip().lower() for x in (exclude or [])}
    axes = list(axes or [])
    have = {str(a.get("name", "")).strip().lower() for a in axes}
    extra = [
        {"name": n, "why": w, "example_values": [], "always": True}
        for n, w in config.ALWAYS_AXES
        if n.strip().lower() not in ex and n.strip().lower() not in have
    ]
    return extra + axes


def _start_suggest(s, src, images, exclude, k=4, selected=False):
    """Ask the VLM for axis chips in a background thread and return immediately.

    A newer call bumps the slot's `gen`; an older thread discards its answer when it
    finishes, so a slow suggestion cannot replace chips for a newer grid.
    """
    if not images or s.get("suggestions") == "off":
        # No-suggestions condition: no chips, and so no speculative expansion either.
        return
    sid = s["session_id"]
    base_prompt = s.get("base_prompt")
    with _SUGGEST_LOCK:
        slot = _suggest_slot(sid)
        slot["gen"] += 1
        gen = slot["gen"]
        slot["pending"] = True
        slot["error"] = None
    images, exclude = list(images), list(exclude or [])

    def run():
        t0 = time.time()
        try:
            r = _STATE["suggest"].suggest(base_prompt, images, k=k, exclude=exclude)
            rec = {
                "id": uuid.uuid4().hex[:12],
                "ts": time.time(),
                "axes": _with_always(r["axes"], exclude),
                "excluded": exclude,
                "from": src,
                "n_images": len(images),
                "selected": bool(selected),
                "auto": True,
            }
            err = None
        except Exception as e:
            rec, err = None, f"{type(e).__name__}: {e}"
            print(f"[orch] auto-suggest ({src}) failed: {err}", flush=True)

        with _SUGGEST_LOCK:
            slot = _suggest_slot(sid)
            if slot["gen"] != gen:
                return  # superseded while in flight; its answer is about old images
            slot["pending"] = False
            slot["error"] = err
            if rec:
                slot["results"].append(rec)
        # Logged here, not at drain time, so the log records when the suggestion was ready.
        _STATE["log"].write(
            "suggest",
            session_id=sid,
            axes=(rec or {}).get("axes"),
            excluded=exclude,
            source=src,
            images=images,
            selected=bool(selected),
            auto=True,
            error=err,
            seconds=round(time.time() - t0, 2),
        )
        # Pre-expand the chips while the participant reads them (_start_speculate).
        if rec:
            _maybe_speculate(sid, src, rec["axes"])

    threading.Thread(target=run, name=f"suggest-{sid[:8]}", daemon=True).start()


# ------------------------------------------------------------- speculative expansion
#
# Expanding four prompts takes ~2.2 s. Decode is memory-bandwidth-bound (~206 tok/s per
# sequence for 1 to 4 sequences), so expanding the suggested axes in one batch costs about one
# expansion. A chip pick then skips the policy; a typed axis expands live.
#
# The key holds every input that changes what the policy writes: axis, K, off-axis rung, arm
# and policy seed. An entry is consumed on use, so a re-roll samples again.


def _spec_slot(sid):
    return _SPEC.setdefault(sid, {"gen": 0, "inflight": False, "entries": {}})


def _spec_key(axis, k, off_axis, arm, seed=None):
    return (
        str(axis).strip().lower(),
        int(k),
        None if off_axis is None else round(float(off_axis), 6),
        str(arm),
        seed,
    )


def _apply_keep(t, p):
    """Set `keep` on trial *t*: grid indices to leave unchanged in an edit round (_render_trial)."""
    n = len(t.get("source_images") or [])
    keep = sorted(
        {int(i) for i in (p.get("keep") or []) if str(i).lstrip("-").isdigit() and 0 <= int(i) < n}
    )
    if keep:
        t["keep"] = keep
        t["prompts"] = [
            ("(kept as it was)" if i in keep else q) for i, q in enumerate(t["prompts"])
        ]


def _policy_seed(s, axis, kind="gen", bump=False):
    """Return the policy's sampling seed for this session, prompt, axis and *kind*.

    Moving the diversity slider keeps the seed, so consecutive sets differ by adapter scale
    only. A re-roll passes bump=True to advance the draw counter. `kind` separates generate,
    edit and layered draws.
    """
    key = "|".join(
        (
            " ".join(str(s.get("base_prompt") or "").lower().split()),
            " ".join(str(axis).lower().split()),
            kind,
        )
    )
    draws = s.setdefault("policy_draws", {})
    if bump:
        draws[key] = int(draws.get(key, 0)) + 1
    h = hashlib.sha256(f"{s['session_id']}|{key}|{int(draws.get(key, 0))}".encode()).hexdigest()
    return int(h[:8], 16)


def _served_models():
    """Return the T2I models AXIS_FW_SERVE serves, in order."""
    return list(profiles.served())


def _grid(s, stage="grid"):
    """Return the render settings of this session's T2I model at *stage* (config.render_settings)."""
    return config.render_settings(s.get("model") or _STATE["model"], stage)


def _policy_images(s):
    """Return the images a round-one request shows the policy.

    For a vision round-one run, the i.i.d. set in order, as in its training rows; for a
    text run, an empty list. Decided per run, not per process (profiles.run_is_vlm).
    """
    if not config.run_is_vlm(config.off_axis_exp_for(s.get("model") or _STATE["model"])):
        return []
    return list((s.get("iid") or {}).get("images") or [])[: config.policy_max_images()]


def _start_speculate(s, axes, k, off_axis, arm, src):
    """Expand the top `config.SPECULATE` suggested axes in the background, in one batch.

    At most one batch is in flight per session, so a second batch never queues ahead of the
    participant's own request on the policy server's lock.
    """
    n = _STATE["speculate"]
    if not n or not axes:
        return
    sid = s["session_id"]
    base_prompt = s.get("base_prompt")
    if not base_prompt:
        return
    model = s.get("model") or _STATE["model"]
    imgs = _policy_images(s)
    rejected = {a.strip().lower() for a in s["rejected_axes"]}
    want = []
    with _SPEC_LOCK:
        slot = _spec_slot(sid)
        if slot["inflight"]:
            return
        for a in axes:
            name = (a.get("name") or "").strip()
            if not name or name.lower() in rejected:
                continue
            if _spec_key(name, k, off_axis, arm, _policy_seed(s, name)) in slot["entries"]:
                continue
            want.append(name)
            if len(want) >= n:
                break
        if not want:
            return
        slot["gen"] += 1
        gen = slot["gen"]
        slot["inflight"] = True

    seeds = {a: _policy_seed(s, a) for a in want}

    def run():
        t0 = time.time()
        err = None
        try:
            # exp and model select the session T2I model's ladder, not the default one.
            r = _STATE["policy"].expand_many(
                [
                    {"base_prompt": base_prompt, "axis": a, "images": imgs, "seed": seeds[a]}
                    for a in want
                ],
                k=k,
                arm=arm,
                off_axis=off_axis,
                exp=config.off_axis_exp_for(model),
                model=model,
            )
            results = r.get("results") or []
        except Exception as e:
            results, err = [], f"{type(e).__name__}: {e}"
            print(f"[orch] speculative expand failed: {err}", flush=True)

        shared = (
            {
                kk: r.get(kk)
                for kk in (
                    "k",
                    "arm",
                    "attempts",
                    "off_axis",
                    "step",
                    "rung",
                    "band",
                    "off_axis_note",
                    "lora",
                    "model",
                )
            }
            if not err
            else {}
        )
        with _SPEC_LOCK:
            slot = _spec_slot(sid)
            slot["inflight"] = False
            if slot["gen"] != gen:
                return  # a reject (or a new round) landed; these are about old state
            for a, res in zip(want, results):
                # A degraded item is the template fallback; drop it so the axis expands live
                # if picked.
                if res.get("degraded"):
                    continue
                slot["entries"][_spec_key(a, k, off_axis, arm, seeds[a])] = {
                    "at": time.time(),
                    "seconds": r.get("seconds"),
                    "e": {
                        **shared,
                        "prompts": res["prompts"],
                        "raw": res.get("raw"),
                        "degraded": False,
                        "axis": res.get("axis"),
                        "axis_factor": res.get("axis_factor"),
                        "finish_reason": res.get("finish_reason"),
                    },
                }
            # Keep at most 16 unused draws per session.
            if len(slot["entries"]) > 16:
                for key in sorted(slot["entries"], key=lambda x: slot["entries"][x]["at"])[:-16]:
                    slot["entries"].pop(key, None)
        _STATE["log"].write(
            "speculate",
            session_id=sid,
            axes=want,
            source=src,
            k=k,
            arm=arm,
            off_axis=off_axis,
            error=err,
            kept=[a for a, res in zip(want, results) if not res.get("degraded")],
            seconds=round(time.time() - t0, 2),
        )

    threading.Thread(target=run, name=f"spec-{sid[:8]}", daemon=True).start()


def _take_speculation(s, axis, k, off_axis, arm, seed=None):
    """Pop and return the speculative expansion for exactly these inputs, or None."""
    if not _STATE["speculate"]:
        return None
    with _SPEC_LOCK:
        return _spec_slot(s["session_id"])["entries"].pop(
            _spec_key(axis, k, off_axis, arm, seed), None
        )


def _drop_speculation(s):
    """Discard every pending draw for this session and supersede any batch in flight.

    Called when an axis is rejected (the rejected axis must not be served back) and when the
    session finishes.
    """
    with _SPEC_LOCK:
        slot = _spec_slot(s["session_id"])
        slot["gen"] += 1
        slot["entries"].clear()


def _forget(sid):
    """Drop this session's suggestion and speculation state once the loop is over.

    Neither map is otherwise bounded per process. The session document and event log are the
    durable record.
    """
    with _SPEC_LOCK:
        _SPEC.pop(sid, None)
    with _SUGGEST_LOCK:
        _SUGGEST.pop(sid, None)


def _maybe_speculate(sid, src, axes):
    """Expand a landed suggestion's chips under the settings the next request will use.

    The arm is derived as the consuming route derives it: for a suggestion from the i.i.d.
    set, `assign_arm` for the next trial index; for one from the current grid, the current
    trial's arm (a respin inherits it).
    """
    if not _STATE["speculate"] or not axes:
        return
    try:
        s = _STATE["store"].get(sid)
        if not s or not s.get("base_prompt"):
            return
        if src == "current":
            prev = S.current_trial(s)
            if not prev:
                return
            arm = prev["arm"]
            k = prev.get("k") or config.K
            off_axis = prev.get("off_axis")
        else:
            arm = S.assign_arm(s["arm_mode"], _STATE["arm_salt"], sid, len(s["trials"]))
            k = s.get("k") or config.K
            off_axis = _off_axis_for({}, s.get("off_axis"))
        _start_speculate(s, axes, k, off_axis, arm, src)
    except Exception as e:  # speculation must never fail the triggering request
        print(f"[orch] could not start speculation: {type(e).__name__}: {e}", flush=True)


def _drain_suggest(s):
    """Copy finished background suggestions into the session document and return the slot.

    Call from the request thread only, so the document has one writer. Results are
    de-duplicated by id, so a drain without a save (/session/get) is repeated next time.
    """
    with _SUGGEST_LOCK:
        slot = dict(_suggest_slot(s["session_id"]))
        slot["results"] = list(slot["results"])
    have = {x.get("id") for x in (s.get("suggested") or [])}
    for r in slot["results"]:
        if r["id"] not in have:
            s.setdefault("suggested", []).append(r)
    return slot


def _annotate_sidecar(s, t, tag, why):
    """Add the round's study context as `study` to the T2I model's <tag>.json beside the images.

    The render server's sidecar records how the pixels were made; this adds the axis, its
    source, generate vs edit, K and the diversity setting, so an image folder reads on its own.
    Best-effort: a failure is printed, never raised.
    """
    fp = os.path.join(_session_dir(s), f"{tag}.json")
    try:
        with open(fp) as f:
            side = json.load(f)
        side["study"] = {
            "session_id": s["session_id"],
            "participant_id": s.get("user_id"),
            "base_prompt": s.get("base_prompt"),
            "trial": t["index"],
            "why": why,
            "axis": t.get("axis"),
            "axis_source": t.get("axis_source"),
            "mode": "edit" if t.get("edit") else "generate",
            "k": t.get("k"),
            "off_axis": t.get("off_axis"),
            "checkpoint_step": t.get("step"),
            "respin_of": t.get("respin_of"),
            "prompts": t.get("prompts"),
            "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = fp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(side, f, indent=1, ensure_ascii=False, default=str)
        os.replace(tmp, fp)
    except Exception as e:
        print(f"[orch] could not annotate {fp}: {type(e).__name__}: {e}", flush=True)


def _session_dir(s):
    d = os.path.join(_STATE["run_dir"], s["session_id"])
    os.makedirs(d, exist_ok=True)
    return d


def _log(event, s=None, **kw):
    _STATE["log"].write(
        event, session_id=(s or {}).get("session_id"), state=(s or {}).get("state"), **kw
    )


# --------------------------------------------------------------------------- routes


def route_new(p):
    model = p.get("model") or _STATE["model"]
    if model not in config.T2I_MODELS:
        raise HttpError(400, f"unknown model {model!r}")
    served = _served_models()
    if served and model not in served:
        # A T2I model with no served run has no trained checkpoint or render settings.
        raise HttpError(400, f"{model} is not served by this run (serving {served})")
    client = _client_info(p)
    s = S.new_session(
        user_id=client["participant_id"],
        arm_mode=_STATE["arm_mode"],
        meta={"run": _STATE["run"], "arm_salt": _STATE["arm_salt"], "client": client},
    )
    # The T2I model is fixed per session, so the i.i.d. images, grids and audit_iid reference
    # come from one generator.
    s["model"] = model
    s["suggestions"] = _suggestions_for(client)
    try:
        s["study_picture"] = None if p.get("study_picture") is None else int(p["study_picture"])
    except (TypeError, ValueError):
        s["study_picture"] = None
    _save(s)
    _log(
        "session_new",
        s,
        user_id=s["user_id"],
        arm_mode=s["arm_mode"],
        model=model,
        client=client,
        suggestions=s["suggestions"],
        study_picture=s["study_picture"],
        study=bool((_STATE.get("study") or {}).get("on")),
    )
    return _public(s)


def route_prompt(p):
    """START -> SAMPLE_IID -> GATE: render the i.i.d. set, with no axis and no policy.

    Renders N_IID images at init_diversity=1 (independent noise), matching gen_iid.py's
    independent draws. A cached set with matching settings is served without rendering.
    """
    s = _sess(p)
    if s["state"] != S.START:
        raise HttpError(409, f"prompt is only accepted in START (session is {s['state']})")
    base_prompt = (p.get("base_prompt") or "").strip()
    if not base_prompt:
        raise HttpError(400, "base_prompt is required")

    model = s.get("model") or _STATE["model"]
    want = cache.settings(model)
    t0 = time.time()

    # Cache lookup; the key includes the render settings (cache.py).
    hit = cache.lookup(model, base_prompt, want) if _STATE["cache"] else None
    if hit:
        r = {
            "images": hit["images"],
            "prompts": [base_prompt] * len(hit["images"]),
            "seed": hit["seed"],
            "model": model,
            "cached": True,
            "guidance_scale": want["guidance"],
            "init_diversity": want["init_diversity"],
            "steps": want["steps"],
            "resolution": want["resolution"],
            "model_path": hit.get("model_path"),
            "seconds": 0.0,
        }
    else:
        r = _render_grid(
            model,
            prompts=[base_prompt] * want["n"],
            base_prompt=base_prompt,
            out_dir=_session_dir(s),
            tag="iid",
            init_diversity=want["init_diversity"],
            guidance_scale=want["guidance"],
            steps=want["steps"],
            resolution=want["resolution"],
        )
        r["cached"] = False
        if _STATE["cache"]:
            # Populate on miss.
            try:
                cache.store(model, base_prompt, r, want)
            except OSError as e:
                print(f"[orch] could not cache {base_prompt!r}: {e}", flush=True)

    s["base_prompt"] = base_prompt
    s["iid"] = r
    S.advance(s, S.GATE)
    _save(s)
    _log(
        "iid",
        s,
        base_prompt=base_prompt,
        images=r["images"],
        seed=r["seed"],
        model=model,
        cached=r["cached"],
        seconds=round(time.time() - t0, 2),
    )
    # Start the axis chips now, two screens before they are shown.
    _start_suggest(s, "iid", r["images"], s["rejected_axes"])
    return _public(s)


def route_gate(p):
    """GATE -> CHOOSE_AXIS | DONE on the user's "does this need diversity?" answer.

    Logged as a `gate` event for both answers; a "no" records which prompts people consider
    varied enough.
    """
    s = _sess(p)
    if s["state"] != S.GATE:
        raise HttpError(409, f"gate is only accepted in GATE (session is {s['state']})")
    if "needs_diversity" not in p:
        raise HttpError(400, "needs_diversity (bool) is required")
    s["needs_diversity"] = bool(p["needs_diversity"])
    s["gate_reason"] = p.get("reason")
    S.advance(s, S.CHOOSE_AXIS if s["needs_diversity"] else S.DONE)
    _save(s)
    _log("gate", s, needs_diversity=s["needs_diversity"], reason=s.get("gate_reason"))
    return _public(s)


def _validate_k(k):
    """Return *k* as an int if it is in config.K_CHOICES; otherwise raise HttpError 400.

    Refused rather than clamped: an unexpected K would otherwise surface later as a degraded
    (policy) failure.
    """
    try:
        k = int(k)
    except (TypeError, ValueError):
        raise HttpError(400, f"k must be an integer, got {k!r}")
    if k not in config.K_CHOICES:
        raise HttpError(400, f"k must be one of {list(config.K_CHOICES)}, got {k}")
    return k


def _off_axis_for(p, fallback):
    """Return the requested off_axis clamped to [0, 2], *fallback*, or the default.

    Returns None when the run offers no ladder (an A/B run, or no exported adapters); the
    policy server is then called without off_axis and answers from its single --lora.
    """
    if not _STATE["off_axis"]:
        return None
    v = p.get("off_axis")
    if v is None:
        return fallback if fallback is not None else config.OFF_AXIS_DEFAULT
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise HttpError(400, f"off_axis must be a number in [0,2], got {v!r}")
    # An alpha ladder reaches off_axis 2 (alpha 2, typed in the page's alpha box); other
    # ladders top out at 1. The nearest rung is used either way.
    return max(0.0, min(2.0, v))


def _layer_expand(s, sources, axis, k, off_axis, arm, samp=None, seed=None):
    """Return (prompts, meta): K prompts, each one of `sources` varied along `axis`.

    The "build on these" path: image i keeps its round-one axis-A value and gains an axis-B
    one. Source i is expanded and prompt i of its set is kept, so the K results differ along
    B. Each expansion asks for K_TRAINED, since only one prompt is kept from each. All K go in
    one /expand_many batch, which vLLM decodes concurrently; this is still the slowest step
    in the loop.

    The sources are 40-60 word expansions rather than short training prompts, so the trial
    is off-distribution and is logged with `layered: true`.

    Raises:
        HttpError: 502 if the policy returns the wrong number of results or an empty one.
    """
    n = len(sources)
    items = [{"base_prompt": sources[i % n], "axis": axis} for i in range(k)]
    e = _STATE["policy"].expand_many(
        items,
        k=config.K_TRAINED,
        arm=arm,
        off_axis=off_axis,
        seed=seed,
        exp=config.off_axis_exp_for(s.get("model")),
        **(samp or {}),
    )
    res = e.get("results") or []
    if len(res) != k:
        raise HttpError(502, f"policy returned {len(res)} results for {k} sources")

    prompts, raws, degraded = [], [], False
    for i, r in enumerate(res):
        got = r.get("prompts") or []
        if not got:
            raise HttpError(502, f"policy returned nothing for source {i}")
        prompts.append(got[i % len(got)])
        raws.append(r.get("raw"))
        degraded = degraded or bool(r.get("degraded"))

    meta = dict(e)
    meta.pop("results", None)
    meta.update(
        {
            "prompts": prompts,
            "raw": "\n---\n".join(str(r) for r in raws),
            "degraded": degraded,
            "k": k,
            "axis_factor": (res[0] or {}).get("axis_factor"),
        }
    )
    return prompts, meta


def _expand_trial(
    s,
    axis,
    source,
    k,
    off_axis,
    arm,
    respin_of=None,
    sources=None,
    layer_of=None,
    samp=None,
    seed=None,
):
    """Ask the policy for K prompts, append the trial and return it, without rendering.

    Shared by /axis and /respin. Uses a matching speculative expansion when one exists.
    `sources` switches from a fresh expansion of the base prompt to a layered one over the
    previous round's prompts (_layer_expand).
    """
    index = len(s["trials"])
    t0 = time.time()
    spec = None if sources else _take_speculation(s, axis, k, off_axis, arm, seed)
    if spec:
        # Same arm/K/rung/seed as a live expansion; its age and cost are logged below.
        e = spec["e"]
    elif sources:
        _, e = _layer_expand(s, sources, axis, k, off_axis, arm, samp=samp, seed=seed)
    else:
        imgs = _policy_images(s)
        e = _STATE["policy"].expand(
            s["base_prompt"],
            axis,
            k=k,
            arm=arm,
            off_axis=off_axis,
            seed=seed,
            exp=config.off_axis_exp_for(s.get("model")),
            model=s.get("model"),
            **({"images": imgs} if imgs else {}),
            **(samp or {}),
        )
    # k_off_distribution marks trials with K != K_TRAINED so the analysis can separate them.
    _log(
        "expand",
        s,
        trial=index,
        arm=arm,
        axis=axis,
        axis_source=source,
        prompts=e["prompts"],
        degraded=e.get("degraded"),
        attempts=e.get("attempts"),
        raw=e.get("raw"),
        lora=e.get("lora"),
        k=k,
        k_off_distribution=(k != config.K_TRAINED),
        seed=seed,
        off_axis=e.get("off_axis"),
        step=e.get("step"),
        rung=e.get("rung"),
        band=e.get("band"),
        off_axis_note=e.get("off_axis_note"),
        respin_of=respin_of,
        layered=bool(sources),
        layer_of=layer_of,
        sources=list(sources or []),
        # `seconds` is what the participant waited (~0 on a speculative hit); spec_seconds
        # is what the policy took.
        speculative=bool(spec),
        spec_seconds=(spec or {}).get("seconds"),
        spec_age=(round(time.time() - spec["at"], 2) if spec else None),
        seconds=round(time.time() - t0, 2),
    )

    t = {
        "trial_id": S.new_id("t"),
        "index": index,
        "axis": axis,
        "axis_source": source,
        # The factor the audit conditions TPIPS on; for a study-vocabulary key, its
        # `tpips_factor`.
        "axis_factor": e.get("axis_factor") or axis,
        "arm": arm,
        "prompts": e["prompts"],
        "raw": e.get("raw"),
        "instruction": e.get("instruction"),
        "degraded": bool(e.get("degraded")),
        "k": k,
        "k_off_distribution": (k != config.K_TRAINED),
        # The rung the server resolved, not the value requested.
        "off_axis": e.get("off_axis"),
        "step": e.get("step"),
        "rung": e.get("rung"),
        "band": e.get("band"),
        "respin_of": respin_of,
        # Layered prompts descend from the previous round's, so `axis` alone does not
        # describe what varies across them.
        "layered": bool(sources),
        "layer_of": layer_of,
        "sources": list(sources or []),
        "speculative": bool(spec),
        "created": time.time(),
        "renders": [],
        "decision": None,
    }
    s["trials"].append(t)
    # Kept on the session so the next axis opens with the same controls.
    s["k"] = k
    s["off_axis"] = t["off_axis"]
    return t


IID_ID = "iid"


def _iid_as_round(s):
    """Return the session's i.i.d. set shaped like a round, for use as an edit source."""
    v = s.get("iid") or {}
    imgs = list(v.get("images") or [])
    g = _grid(s)
    return {
        "trial_id": IID_ID,
        "index": -1,
        "axis": "overall",
        "edit": False,
        "images": imgs,
        "prompts": [s["base_prompt"]] * len(imgs),
        "guidance_scale": g["guidance"],
        "init_diversity": config.INIT_DIVERSITY,
    }


def _edit_trial(s, axis, source, off_axis, arm, prev, samp=None, seed=None, focus=None):
    """Append and return an edit trial from the round-two policy, without rendering.

    Used instead of _layer_expand when the session's T2I model has a round-two ladder
    (config.layer_exp_for). The policy sees the previous round's prompts and images and writes
    one edit instruction per image; _render_trial applies instruction i to image i with the
    T2I model's editor, so image i keeps its content and gains axis B.

    K is the previous round's (an edit maps one image to one image). With *focus* (a grid
    index), the round edits config.K copies of that one picture.

    An edit of an edit is allowed and marked `off_distribution: true` with its `layer_depth`:
    the policy sees the current images, the original round-one prompts, and every axis varied
    so far joined as axis A.

    Raises:
        HttpError: 400 for an invalid *focus* or "overall" on an i.i.d. set with a non-tdig
            editor; 409 when there is nothing to edit; 502 when the policy server fails.
    """
    model = s.get("model") or _STATE["model"]
    exp = config.layer_exp_for(model)
    src_images = list(prev.get("images") or [])
    from_iid = prev.get("trial_id") == IID_ID
    # A described ("tdig") editor run was trained on i.i.d. sets with "overall" among its axes,
    # so for it the i.i.d. set is on-distribution and a policy-varied set is not; a round-one
    # editor is the reverse.
    iid_editor = (profiles.descriptor(exp) or {}).get("format") == "tdig"
    if from_iid and iid_editor:
        src_prompts = list(prev["prompts"])
        axes_before = ["overall"]
        depth = 1
    elif from_iid:
        # Legacy editor on the i.i.d. set: the images share the base prompt and differ only by
        # seed, so axis A is "overall". Off-distribution (that editor was trained on
        # policy-varied sets only), and flagged below.
        if (axis or "").strip().lower() == "overall":
            raise HttpError(
                400,
                "editing the original pictures along 'overall' is not a task "
                "the edit policy has seen -- use Generate for that",
            )
        src_prompts = list(prev["prompts"])
        axes_before = ["overall"]
        depth = 1
    elif prev.get("edit"):
        src_prompts = list(prev["source_prompts"])
        axes_before = list(prev.get("axes_before") or []) + [prev["axis"]]
        depth = int(prev.get("layer_depth") or 1) + 1
    else:
        src_prompts = list(prev.get("prompts") or [])
        axes_before = [prev["axis"]]
        depth = 1
    if not src_images or len(src_images) != len(src_prompts):
        raise HttpError(
            409,
            f"nothing to edit: {len(src_images)} images for "
            f"{len(src_prompts)} prompts on the previous round",
        )
    # "Vary this": `focus` is a grid index; the round edits K copies of that picture. The edit
    # policy was trained on K different images, so the round is marked off-distribution.
    focus_image = None
    if focus is not None:
        try:
            focus = int(focus)
        except (TypeError, ValueError):
            raise HttpError(400, "focus must be a picture index")
        if not 0 <= focus < len(src_images):
            raise HttpError(
                400, f"focus {focus} is not a picture on screen (0..{len(src_images) - 1})"
            )
        focus_image = src_images[focus]
        n = config.K
        src_images, src_prompts = [src_images[focus]] * n, [src_prompts[focus]] * n
    axis_a = ", then ".join(axes_before)

    index = len(s["trials"])
    t0 = time.time()
    try:
        e = _STATE["policy"].edit_plan(
            s["base_prompt"],
            axis_a,
            axis,
            src_prompts,
            source_images=src_images,
            arm=arm,
            off_axis=off_axis,
            exp=exp,
            model=model,
            seed=seed,
            **(samp or {}),
        )
    except HttpError:
        raise
    except Exception as err:
        # The policy server's 400s (e.g. "overall" layered on "overall") carry the reason.
        raise HttpError(502, f"round-two policy refused: {err}")
    ins = list(e["instructions"])
    # off_distribution carries forward from an edit of the i.i.d. set.
    off_dist = depth > 1 or from_iid or bool(prev.get("edit_of_iid"))
    if iid_editor:
        off_dist = not from_iid
    if focus is not None:
        off_dist = True
    _log(
        "expand",
        s,
        trial=index,
        arm=arm,
        axis=axis,
        axis_source=source,
        focus=focus,
        focus_image=focus_image,
        prompts=ins,
        degraded=e.get("degraded"),
        attempts=e.get("attempts"),
        raw=e.get("raw"),
        lora=e.get("lora"),
        k=len(ins),
        k_off_distribution=False,
        off_axis=e.get("off_axis"),
        step=e.get("step"),
        rung=e.get("rung"),
        band=e.get("band"),
        off_axis_note=e.get("off_axis_note"),
        respin_of=prev["trial_id"],
        layered=True,
        layer_of=prev["trial_id"],
        edit=True,
        layer_exp=exp,
        axis_a=axis_a,
        layer_depth=depth,
        edit_of_iid=from_iid or bool(prev.get("edit_of_iid")),
        off_distribution=off_dist,
        source_prompts=src_prompts,
        source_images=src_images,
        speculative=False,
        seconds=round(time.time() - t0, 2),
    )

    t = {
        "trial_id": S.new_id("t"),
        "index": index,
        "axis": axis,
        "axis_source": source,
        "axis_factor": axis,
        "arm": arm,
        # `prompts` holds the applied edit instructions, shown in the grid, download and log.
        "prompts": ins,
        "instructions": ins,
        "raw": e.get("raw"),
        "instruction": e.get("instruction"),
        "degraded": bool(e.get("degraded")),
        "k": len(ins),
        "k_off_distribution": False,
        "off_axis": e.get("off_axis"),
        "step": e.get("step"),
        "rung": e.get("rung"),
        "band": e.get("band"),
        "respin_of": prev["trial_id"],
        "layered": True,
        "layer_of": prev["trial_id"],
        "sources": src_prompts,
        "edit": True,
        "layer_exp": exp,
        "axis_a": axis_a,
        "axes_before": axes_before,
        "edit_of_iid": from_iid or bool(prev.get("edit_of_iid")),
        "layer_depth": depth,
        "off_distribution": off_dist,
        "source_prompts": src_prompts,
        "source_images": src_images,
        "focus": focus,
        "focus_image": focus_image,
        # The grid's settings, not the editor's: a later fresh respin starts from these.
        "guidance_scale": prev.get("guidance_scale"),
        "init_diversity": prev.get("init_diversity"),
        "speculative": False,
        "created": time.time(),
        "renders": [],
        "decision": None,
    }
    s["trials"].append(t)
    s["off_axis"] = t["off_axis"]
    return t


def _exclude_for(s, axis=None):
    """Return the axes the next suggestion must not offer: the rejected ones and *axis*."""
    out = list(s["rejected_axes"])
    if axis and axis not in out:
        out.append(axis)
    return out


def route_suggest(p):
    """Return suggested-axis chips, or start them in the background with `background`.

    `from=iid` asks what the model held fixed in the i.i.d. set (the backend.suggest framing).
    `from=current` asks it of the current grid, optionally narrowed to the `images` the
    participant selected. `images` must come from that set: the suggest server reads the
    paths from disk, so unchecked paths would allow arbitrary file reads.
    """
    s = _sess(p)
    if s["state"] not in (S.CHOOSE_AXIS, S.REVIEW):
        raise HttpError(409, f"suggest needs CHOOSE_AXIS or REVIEW (session is {s['state']})")
    if not s.get("iid"):
        raise HttpError(409, "no i.i.d. images yet")
    if s.get("suggestions") == "off":
        return {"axes": [], "disabled": True}

    src = p.get("from") or ("current" if s["state"] == S.REVIEW else "iid")
    if src not in ("iid", "current"):
        raise HttpError(400, "from must be 'iid' or 'current'")
    t = S.current_trial(s)
    if src == "current" and not (t and t.get("images")):
        raise HttpError(409, "no rendered grid yet -- ask from='iid'")
    pool = list(t["images"] if src == "current" else s["iid"]["images"])

    chosen = p.get("images")
    if chosen:
        stray = [i for i in chosen if i not in set(pool)]
        if stray:
            raise HttpError(
                400, f"images must be from this session's {src} set; " f"not in it: {stray[:3]}"
            )
        images = list(chosen)
    else:
        images = pool

    exclude = _exclude_for(s, t["axis"] if (src == "current" and t) else None)

    # The UI uses `background`: _start_suggest supersedes any automatic suggestion in flight,
    # and the chips arrive on the session as `suggested`. The synchronous form below is for API
    # callers and the smoke tests.
    if p.get("background"):
        _start_suggest(
            s, src, images, exclude, k=int(p.get("k", 4)), selected=(len(images) != len(pool))
        )
        return {"pending": True, "state": s["state"], "from": src, "n_images": len(images)}

    with _SUGGEST_LOCK:
        slot = _suggest_slot(s["session_id"])
        slot["gen"] += 1
        slot["pending"] = False

    t0 = time.time()
    r = _STATE["suggest"].suggest(s["base_prompt"], images, k=int(p.get("k", 4)), exclude=exclude)
    axes = _with_always(r["axes"], exclude)
    s["suggested"].append(
        {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "axes": axes,
            "excluded": exclude,
            "from": src,
            "n_images": len(images),
            "selected": len(images) != len(pool),
            "auto": False,
        }
    )
    _save(s)
    _log(
        "suggest",
        s,
        axes=axes,
        excluded=exclude,
        source=src,
        images=images,
        selected=(len(images) != len(pool)),
        seconds=round(time.time() - t0, 2),
    )
    return {"axes": axes, "state": s["state"], "from": src, "n_images": len(images)}


def _render_trial(s, t, guidance_scale, init_diversity, steps, resolution, why, suggest_after=None):
    """Render (or re-render) a trial's K prompts or edits, audit them, and return the render.

    `suggest_after` is the exclusion list for an automatic suggestion over the new images;
    None (e.g. an accept-time re-render) starts none.

    The render seed is `t["seed"]` when set and is stored back on the trial, so re-renders at
    new slider values reuse it.
    """
    tag = f"t{t['index']}_r{len(t['renders'])}"
    model = s.get("model") or _STATE["model"]
    if t.get("edit"):
        # The editor uses its own trained settings; the grid settings passed in do not apply.
        # Pictures marked to keep are not edited and pass through as the same file; their
        # instructions are logged, not applied.
        keep = set(t.get("keep") or [])
        srcs, ins = list(t["source_images"]), list(t["instructions"])
        todo = [i for i in range(len(srcs)) if i not in keep]
        seed = _seed_for(s["base_prompt"])
        if todo and len(todo) < len(srcs):
            r = _edit_grid(
                model, [srcs[i] for i in todo], [ins[i] for i in todo], _session_dir(s), tag, seed
            )
            got = iter(r["images"])
            r["images"] = [srcs[i] if i in keep else next(got) for i in range(len(srcs))]
            r["kept"] = sorted(keep)
        elif not todo:
            raise HttpError(400, "every picture is marked to keep -- unselect one to edit it")
        else:
            r = _edit_grid(model, srcs, ins, _session_dir(s), tag, seed)
        r["why"] = why
        t["renders"].append(r)
        t["images"] = r["images"]
        t["edit_settings"] = {
            k: r.get(k) for k in ("steps", "guidance_scale", "resolution", "seed")
        }
    else:
        r = _render_grid(
            model,
            prompts=t["prompts"],
            base_prompt=s["base_prompt"],
            out_dir=_session_dir(s),
            tag=tag,
            guidance_scale=guidance_scale,
            init_diversity=init_diversity,
            steps=steps,
            resolution=resolution,
            seed=t.get("seed"),
        )
        t["seed"] = r["seed"]
        r["why"] = why
        t["renders"].append(r)
        t["images"] = r["images"]
        t["guidance_scale"] = r["guidance_scale"]
        t["init_diversity"] = r["init_diversity"]
        t["steps"] = r["steps"]
        t["resolution"] = r["resolution"]
    _annotate_sidecar(s, t, tag, why)

    # Start the suggestion before the audit, which takes another second or two.
    if suggest_after:
        _start_suggest(s, "current", r["images"], suggest_after)

    audit = audit_v = None
    if _STATE["audit"]:
        try:
            # One audit server for every model: the audit reads PNGs from disk, and TPIPS
            # costs ~16 GiB per GPU (FLUX.1 + TPIPS does not fit on a 44 GiB GPU).
            rc = _STATE["audit_client"]
            # For an edit trial each image is scored against the round-one prompt of its
            # source; an edit instruction is not a description VQAScore can score.
            audit = rc.audit(
                r["images"],
                t["source_prompts"] if t.get("edit") else t["prompts"],
                s["base_prompt"],
                t["axis_factor"],
            )
            # Reference: the same axis measured on the i.i.d. images (variation from noise alone).
            audit_v = rc.audit(
                s["iid"]["images"],
                [s["base_prompt"]] * len(s["iid"]["images"]),
                s["base_prompt"],
                t["axis_factor"],
            )
        except Exception as e:  # a failed audit must not fail the trial
            print(f"[orch] audit failed: {type(e).__name__}: {e}", flush=True)
            audit = {"error": f"{type(e).__name__}: {e}"}
    t["audit"] = audit
    t["audit_iid"] = audit_v

    _log(
        "render",
        s,
        trial=t["index"],
        why=why,
        arm=t["arm"],
        axis=t["axis"],
        axis_source=t["axis_source"],
        k=t.get("k"),
        off_axis=t.get("off_axis"),
        step=t.get("step"),
        respin_of=t.get("respin_of"),
        prompts=t["prompts"],
        images=r["images"],
        seed=r.get("seed"),
        guidance_scale=r.get("guidance_scale"),
        init_diversity=r.get("init_diversity"),
        steps=r.get("steps"),
        resolution=r.get("resolution"),
        edit=bool(t.get("edit")),
        source_images=t.get("source_images"),
        degraded=t["degraded"],
        audit=audit,
        audit_iid=audit_v,
        seconds=r.get("seconds"),
    )
    return r


# Sampling overrides from the request, included only when sent. Validated here because vLLM
# raises on a bad range mid-round. Ranges: temperature (0, 2], top_p (0, 1], top_k -1 or > 0.
def _sampling_from(p):
    out = {}
    t = p.get("temperature")
    if t is not None:
        t = float(t)
        if not 0.0 < t <= 2.0:
            raise HttpError(400, f"temperature {t} out of range (0, 2]")
        out["temperature"] = t
    tp = p.get("top_p")
    if tp is not None:
        tp = float(tp)
        if not 0.0 < tp <= 1.0:
            raise HttpError(400, f"top_p {tp} out of range (0, 1]")
        out["top_p"] = tp
    tk = p.get("top_k")
    if tk is not None:
        tk = int(tk)
        if tk == 0 or tk < -1:
            raise HttpError(400, f"top_k {tk} must be -1 (off) or a positive count")
        out["top_k"] = tk
    return out


def route_axis(p):
    """CHOOSE_AXIS -> EXPAND_AND_RENDER -> REVIEW: create and render one trial."""
    s = _sess(p)
    if s["state"] != S.CHOOSE_AXIS:
        raise HttpError(409, f"axis needs CHOOSE_AXIS (session is {s['state']})")
    axis = (p.get("axis") or "").strip()
    if not axis:
        raise HttpError(400, "axis is required")
    source = p.get("source", "free")
    if source not in ("free", "suggested", "rephrased"):
        raise HttpError(
            400,
            "source must be 'free', 'suggested' or 'rephrased' "
            "('refined' belongs to /session/respin)",
        )
    rephrase = _rephrase_taken(s, p, axis, source)

    index = len(s["trials"])
    arm = S.assign_arm(s["arm_mode"], _STATE["arm_salt"], s["session_id"], index)
    g = _grid(s)
    guidance_scale = float(p.get("guidance_scale", g["guidance"]))
    init_diversity = _clamp_alpha(
        s.get("model") or _STATE["model"], float(p.get("init_diversity", config.INIT_DIVERSITY))
    )
    k = _validate_k(p.get("k", s.get("k") or config.K))
    off_axis = _off_axis_for(p, s.get("off_axis"))

    s.pop("retry_axis", None)
    s.pop("retry_off_axis", None)
    mode = p.get("mode", "generate")
    if mode not in ("generate", "edit"):
        raise HttpError(400, "mode must be 'generate' or 'edit'")
    if mode == "edit":
        # Edit the i.i.d. pictures instead of generating a set from the prompt.
        if not config.layer_exp_for(s.get("model") or _STATE["model"]):
            raise HttpError(400, "no edit policy is served for this T2I model; use Generate")
        kind = "edit0"
        t = _edit_trial(
            s,
            axis,
            source,
            off_axis,
            arm,
            _iid_as_round(s),
            samp=_sampling_from(p),
            seed=_policy_seed(s, axis, kind),
            focus=p.get("focus"),
        )
    else:
        kind = "gen"
        t = _expand_trial(
            s,
            axis,
            source,
            k,
            off_axis,
            arm,
            samp=_sampling_from(p),
            seed=_policy_seed(s, axis, kind),
        )
    t["seed"] = _policy_seed(s, axis, kind)
    if t.get("edit") and t.get("focus") is None:
        _apply_keep(t, p)
    if rephrase:
        t["rephrase"] = rephrase
    _render_trial(
        s,
        t,
        guidance_scale,
        init_diversity,
        g["steps"],
        g["resolution"],
        why="initial",
        suggest_after=_exclude_for(s, axis),
    )
    S.advance(s, S.REVIEW)
    _save(s)
    return _public(s)


def _rephrase_taken(s, p, axis, source):
    """Return the rephrase card's offer for this axis and whether it was taken, or None.

    The trial's axis is always what the participant submitted. With `source == "rephrased"`
    the axis is the card's phrase and `axis_raw` (required) is what was typed.
    """
    offer = p.get("rephrase")
    if source == "rephrased" and not (p.get("axis_raw") or "").strip():
        raise HttpError(400, "source 'rephrased' needs axis_raw, the text that was typed")
    if not offer and source != "rephrased":
        return None
    offer = offer if isinstance(offer, dict) else {}
    return {
        "axis_raw": (p.get("axis_raw") or axis).strip()[:500],
        "kind": str(offer.get("kind") or "")[:20],
        "offered": str(offer.get("axis") or "")[:200],
        "add": str(offer.get("add") or "")[:300],
        "taken": source == "rephrased",
    }


def route_rephrase(p):
    """Classify the typed axis and decide whether to offer a restatement (no state change).

    Valid in CHOOSE_AXIS or REVIEW. The suggest server's VLM (suggest_server REPHRASE_SYSTEM)
    returns a kind, a short axis and, when the request applies to every image, a prompt clause
    (`add`). `show` is decided here, not by the model. A failed call returns show=False.
    """
    s = _sess(p)
    if s["state"] not in (S.CHOOSE_AXIS, S.REVIEW):
        raise HttpError(409, f"rephrase needs CHOOSE_AXIS or REVIEW (session is {s['state']})")
    axis = (p.get("axis") or "").strip()
    if not axis:
        raise HttpError(400, "axis is required")
    t0 = time.time()
    try:
        r = _STATE["suggest"].rephrase(s.get("base_prompt") or "", axis)
    except Exception as e:
        _log(
            "rephrase",
            s,
            axis_raw=axis,
            error=f"{type(e).__name__}: {e}"[:200],
            seconds=round(time.time() - t0, 2),
            show=False,
        )
        return {"show": False, "kind": None, "axis": axis, "add": "", "axis_raw": axis}
    kind, sugg, add = r.get("kind"), (r.get("axis") or "").strip(), (r.get("add") or "").strip()
    norm = lambda x: " ".join(x.lower().replace(",", " ").rstrip(".").split())
    # Drop a clause whose content words (> 3 letters) are all already in the base prompt.
    base_words = set(norm(s.get("base_prompt") or "").split())
    add_words = [w for w in norm(add).split() if len(w) > 3]
    if add_words and all(w in base_words for w in add_words):
        add = ""
    # "Differs" ignores articles, filler such as "vary"/"different", and plural s.
    filler = {
        "the",
        "a",
        "an",
        "vary",
        "varied",
        "varying",
        "different",
        "various",
        "diversity",
        "variety",
        "variation",
        "variations",
        "range",
        "of",
        "in",
    }
    core = lambda x: [w.rstrip("s") for w in norm(x).split() if w not in filler]
    # Show the card only for an axis of more than two words that has something to act on (a
    # prompt clause, or an exclusion/complaint turned into an axis) and whose restatement differs
    # or carries a clause.
    actionable = bool(add) or kind in ("exclusion", "complaint")
    show = (
        len(axis.split()) > 2
        and actionable
        and ((bool(sugg) and core(sugg) != core(axis)) or bool(add))
    )
    out = {"show": show, "kind": kind, "axis": sugg, "add": add, "axis_raw": axis}
    _log("rephrase", s, **out, raw=r.get("raw"), seconds=round(time.time() - t0, 2))
    return out


def route_reprompt(p):
    """CHOOSE_AXIS or REVIEW -> CHOOSE_AXIS with a new base prompt and fresh i.i.d. images.

    Applies a constraint found by the rephrase card ("no bench" belongs in the prompt). The
    session keeps its id; earlier trials, rejected axes and suggestions are cleared from it
    (they stay in the event log). The gate is answered "yes" with reason "reprompt".
    """
    s = _sess(p)
    if s["state"] not in (S.CHOOSE_AXIS, S.REVIEW):
        raise HttpError(409, f"reprompt needs CHOOSE_AXIS or REVIEW (session is {s['state']})")
    new = (p.get("base_prompt") or "").strip()
    if not new:
        raise HttpError(400, "base_prompt is required")
    if len(new) > 2000:
        raise HttpError(400, "base_prompt is too long")
    old = s.get("base_prompt")
    _log(
        "reprompt",
        s,
        old_prompt=old,
        base_prompt=new,
        axis_raw=(p.get("axis_raw") or "")[:500],
        add=(p.get("add") or "")[:300],
        trials_dropped=len(s["trials"]),
    )
    _drop_speculation(s)
    s["reprompts"] = int(s.get("reprompts") or 0) + 1
    s["trials"], s["rejected_axes"], s["suggested"] = [], [], []
    s["iid"], s["needs_diversity"] = None, None
    s.pop("retry_axis", None)
    s.pop("retry_off_axis", None)
    s["state"] = S.START
    _save(s)  # route_prompt reads the session back from the store
    route_prompt({"session_id": s["session_id"], "base_prompt": new})
    s = _sess(p)
    s["needs_diversity"] = True
    s["gate_reason"] = "reprompt"
    S.advance(s, S.CHOOSE_AXIS)
    if p.get("axis"):
        s["retry_axis"] = str(p["axis"])[:200]
    _save(s)
    _log("gate", s, needs_diversity=True, reason="reprompt")
    return _public(s)


def route_respin(p):
    """REVIEW -> REVIEW: re-expand from the original prompt or on top of this set.

    Handles the controls that change what the policy writes: axis wording, off-axis diversity
    and K. Appends a new trial linked by `respin_of` instead of editing the current one, so
    earlier `render` events keep naming the prompts they drew. The arm is inherited from the
    current trial, so a respin chain compares within one arm.
    """
    s = _sess(p)
    if s["state"] != S.REVIEW:
        raise HttpError(409, f"respin needs REVIEW (session is {s['state']})")
    prev = S.current_trial(s)
    if not prev:
        raise HttpError(409, "no trial to respin")

    axis = (p.get("axis") or prev["axis"]).strip()
    if not axis:
        raise HttpError(400, "axis cannot be blank")
    k = _validate_k(p.get("k", prev.get("k") or config.K))
    off_axis = _off_axis_for(p, prev.get("off_axis"))

    #   fresh -- vary the original prompt again; the new round replaces this one.
    #   layer -- vary this round's prompts (or edit its images); the set keeps its variety
    #            and gains another kind.
    mode = p.get("mode", "fresh")
    if mode not in ("fresh", "layer"):
        raise HttpError(400, "mode must be 'fresh' or 'layer'")
    sources, layer_of = None, None
    if mode == "layer":
        # Layering takes the whole set: the policy was trained on sets of K. Grid selection
        # is for download and /session/suggest.
        if p.get("images"):
            raise HttpError(
                400,
                "layering does not take an image subset: the policy was trained on sets "
                "of K, and building on a hand-picked few is off-distribution for it. "
                "Selection is for downloading and for /session/suggest.",
            )
        sources = list(prev.get("prompts") or [])
        if not sources:
            raise HttpError(409, "nothing to build on -- this trial has no prompts")
        layer_of = prev["trial_id"]
    # 'refined' marks a re-worded axis, distinct from a typed ('free') or chip ('suggested') one.
    source = p.get("source") or ("refined" if axis != prev["axis"] else prev["axis_source"])
    if source not in ("free", "suggested", "refined", "rephrased"):
        raise HttpError(400, "source must be 'free', 'suggested', 'refined' or 'rephrased'")
    if source == "rephrased" and not p.get("source"):
        # The same rephrased axis again (a reroll or a slider change): the record is inherited.
        rephrase = prev.get("rephrase")
    else:
        rephrase = _rephrase_taken(s, p, axis, source)

    changed = [
        n
        for n, a, b in (
            ("axis", axis, prev["axis"]),
            ("k", k, prev.get("k")),
            ("off_axis", off_axis, prev.get("off_axis")),
        )
        if a != b
    ]
    if mode == "layer":
        # A layered round varies different text, so it is never a re-roll.
        changed.append("mode:layer")
    # Nothing changed is a re-roll: the policy seed advances and the respin is logged with
    # `changed: []`.
    edit_layer = mode == "layer" and bool(config.layer_exp_for(s.get("model") or _STATE["model"]))
    # The policy seed stays fixed while only the slider (or K) moves; a re-roll takes the next
    # draw. A layered round is keyed by the round it builds on.
    kind = (
        f"edit:{prev['trial_id']}"
        if edit_layer
        else f"layer:{prev['trial_id']}" if mode == "layer" else "gen"
    )
    new_draw = bool(p.get("new_draw"))
    if new_draw:
        changed.append("new_draw")
    if p.get("focus") is not None:
        if not edit_layer:
            raise HttpError(400, "varying one picture needs an edit policy for this T2I model")
        changed.append(f"focus:{p.get('focus')}")
    seed = _policy_seed(s, axis, kind, bump=(not changed) or new_draw)
    if edit_layer:
        if k != len(sources):
            changed.append("k:ignored_for_edit")
        t = _edit_trial(
            s,
            axis,
            source,
            off_axis,
            prev["arm"],
            prev,
            samp=_sampling_from(p),
            seed=seed,
            focus=p.get("focus"),
        )
        k = t["k"]
    else:
        t = _expand_trial(
            s,
            axis,
            source,
            k,
            off_axis,
            prev["arm"],
            respin_of=prev["trial_id"],
            sources=sources,
            layer_of=layer_of,
            samp=_sampling_from(p),
            seed=seed,
        )
    t["seed"] = seed
    if t.get("edit") and t.get("focus") is None:
        _apply_keep(t, p)
    _log(
        "respin",
        s,
        trial=t["index"],
        respin_of=prev["trial_id"],
        changed=changed,
        reroll=(not changed),
        axis=axis,
        k=k,
        off_axis=off_axis,
        step=t.get("step"),
        seed=seed,
        mode=mode,
        layered=(mode == "layer"),
        n_sources=len(sources or []),
        edit=bool(t.get("edit")),
        layer_depth=t.get("layer_depth"),
        **_sampling_from(p),
    )
    # Render-side settings are carried over unchanged, so only the prompts differ.
    g = _grid(s)
    if rephrase:
        t["rephrase"] = rephrase
    _render_trial(
        s,
        t,
        prev["guidance_scale"],
        prev["init_diversity"],
        g["steps"],
        g["resolution"],
        why="respin",
        suggest_after=_exclude_for(s, axis),
    )
    _save(s)
    return _public(s)


def route_restore(p):
    """REVIEW -> REVIEW: make an earlier round the current set again, without generating.

    The chosen round is copied to the end of the trial list as a new trial with
    `restored_from`; later rounds are kept. `trial_id == "iid"` returns to CHOOSE_AXIS with the
    original pictures. The abandoned attempt's axis and off_axis are stored as `retry_axis` /
    `retry_off_axis` for the UI to prefill.
    """
    s = _sess(p)
    if s["state"] != S.REVIEW:
        raise HttpError(409, f"restore needs REVIEW (session is {s['state']})")
    tid = p.get("trial_id")
    # Prefill the axis box and slider with the attempt being abandoned.
    leaving = S.current_trial(s) or {}
    if leaving.get("restored_from") and leaving.get("retry_axis") is not None:
        # Leaving a just-restored round: keep the attempt carried into it.
        retry = {
            "retry_axis": leaving["retry_axis"],
            "retry_off_axis": leaving.get("retry_off_axis"),
        }
    else:
        retry = {"retry_axis": leaving.get("axis"), "retry_off_axis": leaving.get("off_axis")}
    if tid == IID_ID:
        # Back to the original pictures; every round so far stays in the session.
        S.advance(s, S.CHOOSE_AXIS)
        s.update(retry)
        _log("restore", s, restored_from=IID_ID, n_trials=len(s["trials"]), **retry)
        _start_suggest(s, "iid", (s.get("iid") or {}).get("images"), s["rejected_axes"])
        _save(s)
        return _public(s)
    src = next((t for t in s["trials"] if t["trial_id"] == tid), None)
    if src is None:
        raise HttpError(404, f"no trial {tid!r} in this session")
    if src is S.current_trial(s):
        return _public(s)  # already the current set
    if not src.get("images"):
        raise HttpError(409, "that round has no images to go back to")
    import copy

    t = copy.deepcopy(src)
    t.update(
        {
            "trial_id": S.new_id("t"),
            "index": len(s["trials"]),
            "restored_from": src["trial_id"],
            "restored_from_index": src["index"],
            **retry,
            "decision": None,
            "speculative": False,
            "created": time.time(),
        }
    )
    s["trials"].append(t)
    s["k"] = t.get("k") or s.get("k")
    s["off_axis"] = t.get("off_axis", s.get("off_axis"))
    _log(
        "restore",
        s,
        trial=t["index"],
        restored_from=src["trial_id"],
        restored_from_index=src["index"],
        axis=t.get("axis"),
        edit=bool(t.get("edit")),
        **retry,
        layer_depth=t.get("layer_depth"),
        images=t.get("images"),
    )
    _start_suggest(s, "current", t["images"], _exclude_for(s, t.get("axis")))
    _save(s)
    return _public(s)


def route_fix(p):
    """CHOOSE_AXIS or REVIEW -> REVIEW: apply the user's own instruction to one picture.

    "Fix this picture": `image` is a grid index, `instruction` the change in the user's words
    (at most 400 characters). No policy is involved: the T2I model's editor applies the
    instruction to config.K copies of the picture, giving K attempts. The trial has
    `fix: true`.
    """
    s = _sess(p)
    if s["state"] not in (S.REVIEW, S.CHOOSE_AXIS):
        raise HttpError(409, f"fix needs REVIEW or CHOOSE_AXIS (session is {s['state']})")
    text = " ".join(str(p.get("instruction") or "").split())
    if not text:
        raise HttpError(400, "say what should change in the picture")
    if len(text) > 400:
        raise HttpError(400, "keep the instruction under 400 characters")
    if not config.layer_exp_for(s.get("model") or _STATE["model"]):  # same test as `can_edit`
        raise HttpError(400, "this T2I model has no image editor")
    review = s["state"] == S.REVIEW
    prev = S.current_trial(s) if review else _iid_as_round(s)
    imgs = list((prev or {}).get("images") or [])
    try:
        i = int(p.get("image"))
    except (TypeError, ValueError):
        raise HttpError(400, "image must be a picture index")
    if not 0 <= i < len(imgs):
        raise HttpError(400, f"image {i} is not a picture on screen (0..{len(imgs) - 1})")
    n = config.K
    index = len(s["trials"])
    arm = (
        prev.get("arm")
        if review
        else S.assign_arm(s["arm_mode"], _STATE["arm_salt"], s["session_id"], index)
    )
    g = _grid(s)
    t = {
        "trial_id": S.new_id("t"),
        "index": index,
        "axis": f"fix: {text}",
        "axis_source": "fix",
        "axis_factor": text,
        "arm": arm,
        "prompts": [text] * n,
        "instructions": [text] * n,
        "raw": None,
        "instruction": text,
        "degraded": False,
        "k": n,
        "k_off_distribution": False,
        "off_axis": prev.get("off_axis") if review else s.get("off_axis"),
        "step": None,
        "rung": None,
        "band": None,
        "respin_of": prev.get("trial_id"),
        "layered": True,
        "layer_of": prev.get("trial_id"),
        "sources": [(prev.get("prompts") or [""] * len(imgs))[i]] * n,
        "edit": True,
        "fix": True,
        "layer_exp": None,
        "axis_a": prev.get("axis"),
        "axes_before": [],
        "edit_of_iid": not review,
        "layer_depth": int(prev.get("layer_depth") or 0) + 1,
        "off_distribution": True,
        "source_prompts": [(prev.get("prompts") or [""] * len(imgs))[i]] * n,
        "source_images": [imgs[i]] * n,
        "focus": i,
        "focus_image": imgs[i],
        "guidance_scale": prev.get("guidance_scale", g["guidance"]),
        "init_diversity": prev.get("init_diversity", config.INIT_DIVERSITY),
        "speculative": False,
        "created": time.time(),
        "renders": [],
        "decision": None,
    }
    s["trials"].append(t)
    _log(
        "fix",
        s,
        trial=index,
        respin_of=prev.get("trial_id"),
        image=i,
        focus_image=imgs[i],
        instruction=text,
        k=n,
    )
    _render_trial(
        s,
        t,
        t["guidance_scale"],
        t["init_diversity"],
        g["steps"],
        g["resolution"],
        why="fix",
        suggest_after=_exclude_for(s),
    )
    if not review:
        S.advance(s, S.REVIEW)
    _save(s)
    return _public(s)


def route_merge(p):
    """CHOOSE_AXIS or REVIEW -> REVIEW: combine several pictures on screen into one.

    `images` are two or more grid indices; `instruction` optionally says how to combine them.
    The editor gets all chosen pictures as reference images for one output, config.K times. No
    policy is involved. Refused unless `_can_merge`: an editing run is served and the editor
    takes several references (FLUX.2-Klein, Qwen-Image).
    """
    s = _sess(p)
    if s["state"] not in (S.REVIEW, S.CHOOSE_AXIS):
        raise HttpError(409, f"combine needs REVIEW or CHOOSE_AXIS (session is {s['state']})")
    model = s.get("model") or _STATE["model"]
    if not _can_merge(model):
        raise HttpError(400, "this T2I model cannot combine pictures")
    review = s["state"] == S.REVIEW
    prev = S.current_trial(s) if review else _iid_as_round(s)
    imgs = list((prev or {}).get("images") or [])
    try:
        idx = sorted({int(i) for i in (p.get("images") or [])})
    except (TypeError, ValueError):
        raise HttpError(400, "images must be picture indices")
    if len(idx) < 2 or any(not 0 <= i < len(imgs) for i in idx):
        raise HttpError(400, f"choose at least two pictures on screen (0..{len(imgs) - 1})")
    text = " ".join(str(p.get("instruction") or "").split())[:400]
    shown = ", ".join(str(i + 1) for i in idx)
    instr = text or (
        "Combine these reference images into one coherent picture that keeps the "
        "main subject and brings in the distinctive elements of each image."
    )
    n = config.K
    index = len(s["trials"])
    arm = (
        prev.get("arm")
        if review
        else S.assign_arm(s["arm_mode"], _STATE["arm_salt"], s["session_id"], index)
    )
    g = _grid(s)
    refs = [imgs[i] for i in idx]
    t = {
        "trial_id": S.new_id("t"),
        "index": index,
        "axis": f"combine pictures {shown}" + (f": {text}" if text else ""),
        "axis_source": "merge",
        "axis_factor": text or "combine",
        "arm": arm,
        "prompts": [instr] * n,
        "instructions": [instr] * n,
        "raw": None,
        "instruction": instr,
        "degraded": False,
        "k": n,
        "k_off_distribution": False,
        "off_axis": prev.get("off_axis") if review else s.get("off_axis"),
        "step": None,
        "rung": None,
        "band": None,
        "respin_of": prev.get("trial_id"),
        "layered": True,
        "layer_of": prev.get("trial_id"),
        "sources": [s["base_prompt"]] * n,
        "edit": True,
        "merge": True,
        "merge_indices": idx,
        "merge_images": refs,
        "layer_exp": None,
        "axis_a": prev.get("axis"),
        "axes_before": [],
        "edit_of_iid": not review,
        "layer_depth": int(prev.get("layer_depth") or 0) + 1,
        "off_distribution": True,
        # A combined picture has no prompt; the base prompt stands in for later edit rounds.
        "source_prompts": [s["base_prompt"]] * n,
        "source_images": [list(refs)] * n,
        "guidance_scale": prev.get("guidance_scale", g["guidance"]),
        "init_diversity": prev.get("init_diversity", config.INIT_DIVERSITY),
        "speculative": False,
        "created": time.time(),
        "renders": [],
        "decision": None,
    }
    s["trials"].append(t)
    _log(
        "merge",
        s,
        trial=index,
        respin_of=prev.get("trial_id"),
        images=idx,
        merge_images=refs,
        instruction=instr,
        user_text=text,
        k=n,
    )
    _render_trial(
        s,
        t,
        t["guidance_scale"],
        t["init_diversity"],
        g["steps"],
        g["resolution"],
        why="merge",
        suggest_after=_exclude_for(s),
    )
    if not review:
        S.advance(s, S.REVIEW)
    _save(s)
    return _public(s)


def route_rerender(p):
    """REVIEW -> REVIEW: re-render the same K prompts at new slider values, without the policy."""
    s = _sess(p)
    if s["state"] != S.REVIEW:
        raise HttpError(409, f"rerender needs REVIEW (session is {s['state']})")
    t = S.current_trial(s)
    _render_trial(
        s,
        t,
        float(p.get("guidance_scale", t["guidance_scale"])),
        _clamp_alpha(
            s.get("model") or _STATE["model"], float(p.get("init_diversity", t["init_diversity"]))
        ),
        int(p.get("steps", _grid(s)["steps"])),
        int(p.get("resolution", _grid(s)["resolution"])),
        why="slider",
    )
    _save(s)
    return _public(s)


def route_review(p):
    """Record the decision on the current trial: accept -> DONE; reject -> CHOOSE_AXIS.

    Reject with `finish` goes to DONE and keeps decision="reject", for a participant who is
    done but not satisfied. On reject the axis joins `rejected_axes`, which suggestions
    exclude. On accept the set is re-rendered at the accept settings (request `hq`; default on
    unless --no-hq) unless it is an edit or already rendered at them.

    `ratings.varied` and `ratings.matched` (integers 1-5) are required in study mode.
    """
    s = _sess(p)
    if s["state"] != S.REVIEW:
        raise HttpError(409, f"review needs REVIEW (session is {s['state']})")
    decision = p.get("decision")
    if decision not in ("accept", "reject"):
        raise HttpError(400, "decision must be 'accept' or 'reject'")
    t = S.current_trial(s)
    t["decision"] = decision
    t["decided_at"] = time.time()
    t["decision_seconds"] = round(t["decided_at"] - t["created"], 2)
    t["comment"] = p.get("comment")
    # Per-set ratings (1-5): did it vary along the axis, and did it stay on the prompt.
    ratings = {}
    for key in ("varied", "matched"):
        v = (p.get("ratings") or {}).get(key)
        if v is None:
            continue
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise HttpError(400, f"ratings.{key} must be an integer")
        if not 1 <= v <= 5:
            raise HttpError(400, f"ratings.{key} must be 1-5")
        ratings[key] = v
    if (_STATE.get("study") or {}).get("on") and len(ratings) != 2:
        raise HttpError(400, "both ratings are required: ratings.varied and ratings.matched (1-5)")
    t["ratings"] = ratings or None

    # The review event repeats the trial's settings so the analysis needs no join.
    _log(
        "review",
        s,
        trial=t["index"],
        decision=decision,
        arm=t["arm"],
        axis=t["axis"],
        ratings=t["ratings"],
        rephrase=t.get("rephrase"),
        reprompts=s.get("reprompts") or 0,
        axis_source=t["axis_source"],
        guidance_scale=t["guidance_scale"],
        init_diversity=t["init_diversity"],
        degraded=t["degraded"],
        k=t.get("k"),
        off_axis=t.get("off_axis"),
        step=t.get("step"),
        k_off_distribution=t.get("k_off_distribution"),
        respin_of=t.get("respin_of"),
        n_renders=len(t["renders"]),
        comment=t["comment"],
        decision_seconds=t["decision_seconds"],
        audit=t.get("audit"),
        finish=bool(p.get("finish")),
    )

    if decision == "reject" and p.get("finish"):
        # Judged and done: no suggestion refill, since nobody is coming back to CHOOSE_AXIS.
        s["rejected_axes"].append(t["axis"])
        _drop_speculation(s)
        S.advance(s, S.DONE)
    elif decision == "reject":
        s["rejected_axes"].append(t["axis"])
        S.advance(s, S.CHOOSE_AXIS)
        # Pending draws were made for the previous trial index, whose arm may differ in an
        # A/B run. The suggestion below starts a fresh batch.
        _drop_speculation(s)
        # Re-ask over the i.i.d. images with the rejected axis excluded.
        _start_suggest(s, "iid", (s.get("iid") or {}).get("images"), s["rejected_axes"])
    else:
        # Skip the accept re-render when the grid already used the accept resolution and
        # steps (it would redraw identical images), and log the skip.
        acc = _grid(s, "accept")
        # An edit trial's images are the editor's; they have no accept-quality re-render.
        already_hq = bool(t.get("edit")) or (
            t.get("resolution") == acc["resolution"] and t.get("steps") == acc["steps"]
        )
        if already_hq:
            _log(
                "accept_hq_skipped",
                s,
                trial=t["index"],
                reason="grid already rendered at the accept settings",
                resolution=t.get("resolution"),
                steps=t.get("steps"),
            )
        if p.get("hq", _STATE["hq_on_accept"]) and not already_hq:
            # The accept is already logged; a failed re-render (e.g. OOM on a GPU shared with
            # the policy and suggest servers) keeps the interactive images and is logged.
            try:
                _render_trial(
                    s,
                    t,
                    t["guidance_scale"],
                    t["init_diversity"],
                    acc["steps"],
                    acc["resolution"],
                    why="accept_hq",
                )
            except Exception as e:
                t["hq_error"] = f"{type(e).__name__}: {e}"
                print(
                    f"[orch] accept re-render failed, keeping the interactive images: "
                    f"{t['hq_error']}",
                    flush=True,
                )
                _log("accept_hq_failed", s, trial=t["index"], error=t["hq_error"])
        S.advance(s, S.DONE)
    if s["state"] == S.DONE:
        _forget(s["session_id"])
    _save(s)
    return _public(s)


def route_get(p):
    return _public(_sess(p))


def route_media(p):
    """Serve a rendered image from the run directory or the pre-render cache.

    Paths outside those two roots get a 404. Cached i.i.d. sets are served from `cache/`
    directly, not copied into the session directory.
    """
    path = os.path.realpath(p.get("path") or "")
    roots = [os.path.realpath(r) for r in (_STATE["run_dir"], config.CACHE)]
    if not any(path.startswith(r + os.sep) for r in roots) or not os.path.isfile(path):
        raise HttpError(404, "not found")
    # `w` (clamped to 64-1024 px) requests a JPEG preview that wide, ~60-150 KB against ~1.5 MB
    # for a 1024px PNG. Previews are cached under cache/.previews/. Without `w` the original is
    # served.
    w = p.get("w")
    if w:
        try:
            w = max(64, min(1024, int(w)))
        except (TypeError, ValueError):
            raise HttpError(400, "w must be an integer")
        h = hashlib.md5(path.encode()).hexdigest()
        prev = os.path.join(config.CACHE, ".previews", h[:2], f"{h}_{w}.jpg")
        if not os.path.exists(prev):
            from PIL import Image

            os.makedirs(os.path.dirname(prev), exist_ok=True)
            im = Image.open(path).convert("RGB")
            im.thumbnail((w, w), Image.LANCZOS)
            tmp = f"{prev}.{os.getpid()}.{threading.get_ident()}.tmp"
            im.save(tmp, "JPEG", quality=85, optimize=True)
            os.replace(tmp, prev)
        with open(prev, "rb") as f:
            return Raw(f.read(), "image/jpeg", headers={"Cache-Control": "max-age=86400"})
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return Raw(f.read(), ctype, headers={"Cache-Control": "max-age=86400"})


def route_index(_p):
    # AXIS_FW_INDEX serves another page file instead of web/index.html (e.g. for a test
    # instance). The file is read per request, so edits go live.
    fp = os.environ.get("AXIS_FW_INDEX") or os.path.join(config.FRAMEWORK, "web", "index.html")
    if not os.path.exists(fp):
        return Raw(
            b"<h1>axis_framework</h1><p>No UI built yet (M2). The HTTP API is up; "
            b"see axis_framework/README.md.</p>",
            "text/html",
        )
    with open(fp, "rb") as f:
        return Raw(f.read(), "text/html")


def route_models(_p):
    """Return the T2I models and their capabilities, proxied from the T2I model servers.

    A union over servers, de-duplicated by key (first answer wins). A model whose server is
    down, or that no served run covers, is reported with `available: false` and a `why`.
    """
    seen, out = set(), []
    urls = {}
    for key, c in _STATE["renders"].items():
        urls.setdefault(c.url, []).append(key)
    if _STATE["render"].url not in urls:
        urls[_STATE["render"].url] = []

    errors = {}
    for url, keys in urls.items():
        try:
            j = client.T2IClient(url).models()
        except Exception as e:
            errors[url] = f"{type(e).__name__}: {e}"
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    spec = config.T2I_MODELS.get(k, {})
                    out.append(
                        {
                            "key": k,
                            "label": spec.get("label", k),
                            "available": False,
                            "loaded": False,
                            "why": f"its T2I model server is down ({errors[url]})",
                            "note": spec.get("note", ""),
                            "vram_gb": spec.get("vram_gb"),
                            "shared_latents": spec.get("shared_latents", True),
                            "guidance": spec.get("guidance", True),
                        }
                    )
            continue
        for m in j.get("models", []):
            if m["key"] in seen:
                continue
            # Report only the models a server was launched for, not those it could load by
            # eviction.
            if keys and m["key"] not in keys:
                continue
            seen.add(m["key"])
            out.append({**m, "url": url})
    served = set(_served_models())
    if served:
        for m in out:
            if m["key"] not in served and m.get("available", True):
                m["available"] = False
                m["why"] = "no run is served for this T2I model (AXIS_FW_SERVE)"
    if not out and errors:
        raise HttpError(502, f"no T2I model server reachable: {errors}")
    return {"models": sorted(out, key=lambda m: m["key"]), "default": _STATE["model"]}


def _all_images(s):
    """Return {image path: prompt} for every image this session produced.

    Built from the session, so a browser-supplied path can only name this session's images.
    """
    out = {}
    v = s.get("iid") or {}
    for fp in v.get("images") or []:
        out[fp] = s.get("base_prompt") or ""
    for t in s.get("trials") or []:
        for i, fp in enumerate(t.get("images") or []):
            ps = t.get("prompts") or []
            out[fp] = ps[i] if i < len(ps) else ""
    return out


def route_download(p):
    """Return the selected images as a zip, with a prompts.txt naming each file's prompt.

    Raises:
        HttpError: 400 for no images or images not from this session; 404 if a file is gone.
    """
    s = _sess(p)
    want = p.get("images") or []
    if not want:
        raise HttpError(400, "images is required")
    known = _all_images(s)
    stray = [i for i in want if i not in known]
    if stray:
        raise HttpError(400, f"images must be from this session; not in it: {stray[:3]}")
    missing = [i for i in want if not os.path.isfile(i)]
    if missing:
        raise HttpError(404, f"no longer on disk: {missing[:3]}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        lines = [f"base prompt: {s.get('base_prompt')}", ""]
        for n, fp in enumerate(want, 1):
            name = f"{n:02d}_{os.path.basename(fp)}"
            z.write(fp, name)
            lines += [name, known.get(fp) or "(no prompt recorded)", ""]
        z.writestr("prompts.txt", "\n".join(lines))
    _log("download", s, images=list(want), n=len(want))
    return Raw(buf.getvalue(), "application/zip")


def _client_info(p):
    """Return who a request came from, for the study record.

    participant_id: typed on the start screen or carried by the link (?pid=). client_id: a
    random id the browser keeps in localStorage. access_email: set only behind Cloudflare
    Access. No IP address is kept.
    """
    c = p.get("_client") or {}
    clip = lambda v, n: (str(v).strip()[:n] or None) if v is not None else None
    return {
        "participant_id": clip(p.get("user_id"), 120),
        "client_id": clip(p.get("client_id"), 64),
        "access_email": clip(c.get("access_email"), 200),
        "user_agent": clip(c.get("user_agent"), 300),
        "country": clip(c.get("country"), 8),
        "screen": clip(p.get("screen"), 40),
        # Prolific's own ids, when the study link carries them (STUDY_ID, SESSION_ID).
        "prolific_study_id": clip((p.get("prolific") or {}).get("study_id"), 64),
        "prolific_session_id": clip((p.get("prolific") or {}).get("session_id"), 64),
    }


def _suggestions_for(client):
    """Return "on" or "off" for this participant: the run's setting, or a stable half under 'split'.

    The split hashes the run's salt with the participant id (else the browser id), so a
    participant stays in one condition.
    """
    mode = (_STATE.get("study") or {}).get("suggestions", "on")
    if mode != "split":
        return mode
    who = client.get("participant_id") or client.get("client_id") or ""
    h = hashlib.sha256(f"{_STATE['arm_salt']}:suggestions:{who}".encode()).digest()[0]
    return "on" if h % 2 == 0 else "off"


# Caps per /client_event batch (events) and per event (bytes of JSON); any browser can post.
_UI_MAX_EVENTS, _UI_MAX_BYTES = 200, 4000


def route_client_event(p):
    """Log a batch of UI-only interactions as one "ui" event.

    These are interactions no route sees (chip clicks before editing, opening a picture,
    slider moves, tab visibility, leaving the page). The page flushes a batch every few
    seconds and on hide; each event carries its own client time.
    """
    events = p.get("events")
    if not isinstance(events, list):
        raise HttpError(400, "events must be a list")
    kept, dropped = [], 0
    for e in events[:_UI_MAX_EVENTS]:
        if not isinstance(e, dict) or len(json.dumps(e, default=str)) > _UI_MAX_BYTES:
            dropped += 1
            continue
        kept.append(e)
    dropped += max(0, len(events) - _UI_MAX_EVENTS)
    s = _STATE["store"].get(p.get("session_id") or "") or None
    _log("ui", s, client_id=str(p.get("client_id"))[:64], events=kept, dropped=dropped)
    return {"logged": len(kept)}


def route_study_consent(p):
    """Log the participant's consent to the instructions (no session needed)."""
    client = _client_info(p)
    _log(
        "consent",
        None,
        client=client,
        agreed=bool(p.get("agreed")),
        instructions_version=str(p.get("version") or "")[:40],
    )
    return {"saved": True}


def route_study_questionnaire(p):
    """Log the end-of-study answers and return the completion code.

    The code is returned only here, after `pictures_done` reaches --study-pictures and the
    answers are logged; it is never in /health or the page.
    """
    st = _STATE.get("study") or {}
    if not st.get("on"):
        raise HttpError(409, "this run is not in study mode")
    client = _client_info(p)
    if not client["participant_id"]:
        raise HttpError(400, "user_id (the participant id) is required")
    done = int(p.get("pictures_done") or 0)
    if done < st["pictures"]:
        raise HttpError(409, f"{done} of {st['pictures']} pictures done")
    ans = p.get("answers") or {}
    # Any short lower-case key is kept, so a new page question needs no restart. At most 40
    # keys; string values clipped to 3000 characters.
    answers = {
        k: (v[:3000] if isinstance(v, str) else v)
        for k, v in list(ans.items())[:40]
        if isinstance(k, str)
        and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", k)
        and isinstance(v, (str, int, float, bool, type(None)))
    }
    _log(
        "questionnaire",
        None,
        client=client,
        answers=answers,
        pictures_done=done,
        suggestions=_suggestions_for(client),
    )
    code = st.get("completion_code") or ""
    return {
        "saved": True,
        "completion_code": code,
        "completion_url": st["completion_url"].format(code=code) if code else "",
    }


def route_study_instructions(_p):
    """Serve web/study_instructions.html, the consent and instructions text."""
    fp = os.path.join(config.FRAMEWORK, "web", "study_instructions.html")
    with open(fp, "rb") as f:
        return Raw(f.read(), "text/html")


def route_client_error(p):
    """Log a browser-side exception as a "client_error" event.

    A screen that throws leaves the previous screen up, so later clicks get 409s and the event
    log alone would show only silence. Unauthenticated and session-optional; fields are
    truncated.
    """
    _log(
        "client_error",
        _STATE["store"].get(p.get("session_id") or "") or None,
        message=str(p.get("message"))[:2000],
        stack=str(p.get("stack"))[:4000],
        screen=str(p.get("screen"))[:200],
        url=str(p.get("url"))[:500],
        reported_state=str(p.get("client_state"))[:100],
    )
    print(f"[orch] client error on {p.get('screen')}: {str(p.get('message'))[:300]}", flush=True)
    return {"logged": True}


def route_loras(_p):
    """Return the off-axis ladder for `model` (default: the run's), proxied from the policy server.

    `enabled` is the orchestrator's decision (e.g. false in an A/B run), not the policy
    server's.
    """
    model = (_p or {}).get("model") or _STATE["model"]
    exp = config.off_axis_exp_for(model)
    lam, tau = config.off_axis_reward(exp)
    out = {
        "enabled": _STATE["off_axis"],
        "ladder": [],
        "model": model,
        "default_off_axis": config.OFF_AXIS_DEFAULT,
        "exp": exp,
        "lam": lam,
        "tau": tau,
        "why": None if _STATE["off_axis"] else _STATE["off_axis_why"],
    }
    try:
        out["ladder"] = _STATE["policy"].loras(exp).get("ladder", [])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def route_prompts(p):
    """Return the pre-rendered prompts for a model, offered as one-click start-screen cards.

    Lists only cache entries whose settings match this run's, ordered by
    config.GALLERY_PROMPTS (others after, alphabetically) and capped at `n` (default
    config.N_GALLERY).
    """
    model = p.get("model") or _STATE["model"]
    want = cache.settings(model)
    out = []
    for meta in cache.catalogue(model=model):
        if all(meta.get(k) == want[k] for k in cache.KEYED):
            out.append({"prompt": meta["prompt"], "images": meta["images"], "model": meta["model"]})
    # Curated prompts first, so typed prompts in the cache do not displace them.
    rank = {q: i for i, q in enumerate(config.GALLERY_PROMPTS)}
    out.sort(key=lambda m: (rank.get(m["prompt"], len(rank)), m["prompt"]))
    n = int(p.get("n") or config.N_GALLERY)
    return {
        "model": model,
        "prompts": out[:n],
        "n_cached": len(out),
        "n_curated": sum(1 for m in out if m["prompt"] in rank),
    }


def route_health(_p):
    out = {
        "status": "ok",
        "ready": True,
        "run": _STATE["run"],
        "arm_mode": _STATE["arm_mode"],
        "model": _STATE["model"],
        "cache": _STATE["cache"],
        "sessions": len(_STATE["store"].ids()),
        # Set sizes for the UI's text.
        "k": config.K,
        "n_iid": config.N_IID,
        "k_choices": list(config.K_CHOICES),
        "k_trained": config.K_TRAINED,
        "off_axis": _STATE["off_axis"],
        "off_axis_why": _STATE["off_axis_why"],
        "off_axis_default": config.OFF_AXIS_DEFAULT,
        "speculate": _STATE["speculate"],
        "ask_participant": _STATE.get("ask_participant", False),
        # The completion code is not here; only /study/questionnaire returns it.
        "study": {
            k: v
            for k, v in (_STATE.get("study") or {}).items()
            if k in ("on", "pictures", "suggestions")
        },
        "services": {},
    }
    out["render_pool"] = {k: [c.url for c in v] for k, v in _STATE["pool"].items()}
    svcs = [("policy", _STATE["policy"]), ("suggest", _STATE["suggest"])]
    pooled = [
        (f"render:{k}#{i}", c) for k, v in sorted(_STATE["pool"].items()) for i, c in enumerate(v)
    ]
    svcs += pooled or (
        [(f"render:{k}", c) for k, c in sorted(_STATE["renders"].items())]
        or [("render", _STATE["render"])]
    )
    for name, c in svcs:
        try:
            out["services"][name] = c.health()
        except Exception as e:
            out["services"][name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19100)
    ap.add_argument(
        "--run",
        default=time.strftime("run_%Y%m%d_%H%M%S"),
        help="names the output dir and the event log; one per data collection",
    )
    ap.add_argument("--policy-url", default=config.POLICY_URL)
    ap.add_argument(
        "--t2i-url",
        default=config.T2I_URL,
        help="fallback T2I model server, used for any model without its own",
    )
    ap.add_argument(
        "--audit-url",
        default=None,
        help="T2I model server that computes the audit metrics (needs TPIPS); default: "
        "--t2i-url. One server audits every model.",
    )
    ap.add_argument(
        "--t2i-urls",
        default=None,
        help="per-model T2I model servers, e.g. "
        "'sd35-medium=http://host:19102,flux2klein-9b=http://host:19112'. One server per "
        "model, each on its own GPU, so no model is evicted.",
    )
    ap.add_argument(
        "--t2i-pool",
        default=None,
        help="replicas of one model, to split a grid across GPUs, e.g. "
        "'sd35-large=http://h:19110,http://h:19111,http://h:19112'. Repeat the flag, or "
        "separate groups with ';', for several models. N replicas render a K-image grid "
        "K/N at a time; the images are unchanged.",
        action="append",
    )
    ap.add_argument("--suggest-url", default=config.SUGGEST_URL)
    ap.add_argument(
        "--arms",
        default="ours",
        choices=("ours", "baseline", "ab"),
        help="'ours': the policy only; 'baseline': the untuned base model only; 'ab': a "
        "random arm per trial, blinded to the participant",
    )
    ap.add_argument(
        "--arm-salt",
        default=None,
        help="per-run randomisation salt; default: --run, so the assignment is reproducible "
        "from the log",
    )
    ap.add_argument("--store", default="sqlite", choices=("memory", "sqlite"))
    ap.add_argument(
        "--model",
        default=None,
        help="T2I model a new session opens on (default: the first one "
        "AXIS_FW_SERVE serves); the UI can change it",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="always render the i.i.d. stage live, ignoring cache/",
    )
    ap.add_argument(
        "--no-audit",
        action="store_true",
        help="skip the server-side metrics (faster loop, no study data)",
    )
    ap.add_argument(
        "--no-hq", action="store_true", help="do not re-render at inference quality on accept"
    )
    ap.add_argument(
        "--speculate",
        type=int,
        default=config.SPECULATE,
        help="expand this many suggested axes in one batched call before the user picks one; "
        "0 disables. An axis the user types waits for a batch in flight.",
    )
    ap.add_argument(
        "--no-off-axis",
        action="store_true",
        help="withhold the off-axis diversity control even in a single-arm "
        "run, so every trial comes from the policy server's --lora",
    )
    ap.add_argument(
        "--base-path",
        default=os.environ.get("AXIS_FW_BASE_PATH", ""),
        help="also serve every route under this URL prefix (e.g. /axis), for a "
        "reverse proxy that forwards the path unchanged",
    )
    ap.add_argument(
        "--ask-participant",
        action="store_true",
        help="ask for a name or participant id before the first picture, "
        "unless the link already carries ?pid=",
    )
    ap.add_argument(
        "--study-pictures",
        type=int,
        default=0,
        help="study mode when > 0: a participant completes this many pictures, rates every "
        "finished set, answers a questionnaire and gets the completion code; the "
        "example gallery is hidden. 0: the open demo.",
    )
    ap.add_argument(
        "--study-suggestions",
        choices=("on", "off", "split"),
        default="on",
        help="whether the suggested-axis chips are shown. 'split' assigns on/off per "
        "participant id, deterministically, and logs it on session_new.",
    )
    ap.add_argument(
        "--completion-code",
        default="",
        help="shown after the questionnaire; never sent to the page before that",
    )
    ap.add_argument(
        "--completion-url",
        default="https://app.prolific.com/submissions/complete?cc={code}",
        help="link shown with the completion code; {code} is replaced",
    )
    a = ap.parse_args()
    # Default T2I model: the first one served.
    a.model = a.model or config.default_model()

    # Decide once whether the off-axis control is offered, and why not: /loras, /health, the
    # UI and the run log all report the same reason.
    if a.arms != "ours":
        off_axis, why = False, (
            f"--arms {a.arms}: arm B is the untuned base model and has no lam to move, so "
            f"a slider that did nothing for half the trials would unblind them"
        )
    elif a.no_off_axis:
        off_axis, why = False, "--no-off-axis"
    else:
        exp0 = config.off_axis_exp_for(a.model)
        rungs = config.off_axis_ladder(exp0)
        if not rungs:
            off_axis, why = False, (
                f"no {exp0 or '<unserved>'}_step* adapters under axis_framework/adapters "
                f"-- run axis_framework/scripts/build_alpha_ladders.sh (released checkpoints) or axis_framework/scripts/export_ladder.sh EXP={exp0}"
            )
        else:
            off_axis, why = True, None
            print(
                f"[orch] off-axis ladder for {a.model}: {len(rungs)} checkpoints of "
                f"{exp0} (steps {rungs[0]['step']}..{rungs[-1]['step']})",
                flush=True,
            )

    run_dir = os.path.join(config.RUNS, a.run)
    os.makedirs(run_dir, exist_ok=True)
    _STATE.update(
        {
            "run": a.run,
            "run_dir": run_dir,
            "ask_participant": a.ask_participant or a.study_pictures > 0,
            "study": {
                "on": a.study_pictures > 0,
                "pictures": a.study_pictures,
                "suggestions": a.study_suggestions,
                "completion_code": a.completion_code,
                "completion_url": a.completion_url,
            },
            "arm_mode": a.arms,
            "arm_salt": a.arm_salt or a.run,
            "model": a.model,
            "cache": not a.no_cache,
            "audit": not a.no_audit,
            "hq_on_accept": not a.no_hq,
            "off_axis": off_axis,
            "off_axis_why": why,
            "speculate": max(0, a.speculate),
            "store": S.make_store(a.store, os.path.join(run_dir, "sessions.sqlite")),
            "log": S.EventLog(os.path.join(run_dir, "events.jsonl")),
            "policy": client.PolicyClient(a.policy_url),
            "render": client.T2IClient(a.t2i_url),
            "renders": {},
            "pool": {},
            "caps": {},
            "suggest": client.SuggestClient(a.suggest_url),
        }
    )
    for pair in (a.t2i_urls or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--t2i-urls entry {pair!r} is not model=url")
        key, url = pair.split("=", 1)
        if key.strip() not in config.T2I_MODELS:
            raise SystemExit(f"--t2i-urls: unknown model {key.strip()!r}")
        _STATE["renders"][key.strip()] = client.T2IClient(url.strip())
    for group in a.t2i_pool or []:
        for entry in group.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise SystemExit(f"--t2i-pool entry {entry!r} is not model=url,url,...")
            key, urls = entry.split("=", 1)
            key = key.strip()
            if key not in config.T2I_MODELS:
                raise SystemExit(f"--t2i-pool: unknown model {key!r}")
            clients = [client.T2IClient(u.strip()) for u in urls.split(",") if u.strip()]
            if not clients:
                raise SystemExit(f"--t2i-pool {key}: no urls")
            _STATE["pool"][key] = clients
            # The first replica is also the model's owner (/models, /audit, unsharded renders).
            _STATE["renders"].setdefault(key, clients[0])
            print(
                f"[orch] render pool {key}: {len(clients)} replicas "
                f"({', '.join(c.url for c in clients)})",
                flush=True,
            )

    if _STATE["renders"] and a.model in _STATE["renders"]:
        _STATE["render"] = _STATE["renders"][a.model]
    _STATE["audit_client"] = client.T2IClient(a.audit_url) if a.audit_url else _STATE["render"]

    _STATE["log"].write(
        "run_start",
        run=a.run,
        arms=a.arms,
        arm_salt=_STATE["arm_salt"],
        policy_url=a.policy_url,
        render_url=a.t2i_url,
        suggest_url=a.suggest_url,
        k=config.K,
        init_diversity=config.INIT_DIVERSITY,
        render_settings={
            m: {st: config.render_settings(m, st) for st in ("grid", "iid", "accept")}
            for m in (_served_models() or [a.model])
        },
        serve=profiles.served(),
        model=a.model,
        cache=not a.no_cache,
        k_choices=list(config.K_CHOICES),
        k_trained=config.K_TRAINED,
        speculate=_STATE["speculate"],
        off_axis=off_axis,
        off_axis_why=why,
        off_axis_default=config.OFF_AXIS_DEFAULT,
        off_axis_exp=config.off_axis_exp_for(a.model),
        off_axis_reward=dict(
            zip(("lam", "tau"), config.off_axis_reward(config.off_axis_exp_for(a.model)))
        ),
        off_axis_ladder=[
            {k: v for k, v in r.items() if k != "path"}
            for r in config.off_axis_ladder(config.off_axis_exp_for(a.model))
        ],
        render_urls={k: c.url for k, c in _STATE["renders"].items()},
        render_pool={k: [c.url for c in v] for k, v in _STATE["pool"].items()},
    )

    serve(
        {
            ("GET", "/health"): route_health,
            ("GET", "/models"): route_models,
            ("GET", "/prompts"): route_prompts,
            ("GET", "/"): route_index,
            ("GET", "/media"): route_media,
            ("GET", "/session/get"): route_get,
            ("POST", "/session/new"): route_new,
            ("POST", "/session/prompt"): route_prompt,
            ("POST", "/session/gate"): route_gate,
            ("POST", "/session/suggest"): route_suggest,
            ("GET", "/loras"): route_loras,
            ("POST", "/session/axis"): route_axis,
            ("POST", "/session/rephrase"): route_rephrase,
            ("POST", "/session/reprompt"): route_reprompt,
            ("POST", "/session/respin"): route_respin,
            ("POST", "/client_error"): route_client_error,
            ("POST", "/client_event"): route_client_event,
            ("POST", "/study/consent"): route_study_consent,
            ("POST", "/study/questionnaire"): route_study_questionnaire,
            ("GET", "/study/instructions"): route_study_instructions,
            ("POST", "/download"): route_download,
            ("POST", "/session/rerender"): route_rerender,
            ("POST", "/session/restore"): route_restore,
            ("POST", "/session/fix"): route_fix,
            ("POST", "/session/merge"): route_merge,
            ("POST", "/session/review"): route_review,
        },
        a.host,
        a.port,
        "orch",
        ready_msg=f"run={a.run} arms={a.arms} model={a.model} "
        f"off_axis={'on' if off_axis else 'off'} "
        f"pool={len(_STATE['pool'].get(a.model, [])) or 1} "
        f"speculate={_STATE['speculate']} -> {run_dir}",
        base_path=a.base_path,
    )


if __name__ == "__main__":
    main()
