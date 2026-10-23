"""Transactional benchmark generation with immutable inputs and verifiable resume."""

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import time
import uuid

from benchmarks.coco import load_coco, NAME as COCO
from benchmarks.heldout import load_heldout, NAME as HELDOUT
from hpdv2.storage import atomic_json, digest, read_json, run_lock, sync_directory
from iid.runner import verify_png
from .assets import seed_for, verify_assets, file_hash
from .config import validate


def dataset(path):
    name = read_json(Path(path) / "manifest.json")["benchmark"]
    if name == COCO:
        return load_coco(Path(path))
    if name == HELDOUT:
        return load_heldout(Path(path))
    raise ValueError("Only COCO 2017 validation and held-out axis bundles are supported")


def environment():
    versions = {}
    for name in [
        "torch",
        "diffusers",
        "transformers",
        "numpy",
        "Pillow",
        "gurobipy",
        "safetensors",
        "clean-fid",
        "scipy",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def code_hash():
    root = Path(__file__).parents[1]
    paths = list((root / "baselines").glob("*.py")) + list((root / "benchmarks").glob("*.py"))
    paths += [root / "iid/runner.py", root / "hpdv2/storage.py"]
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(paths)})


def manifest_for(bundle, rows, cfg, assets, limit):
    return {
        "schema_version": 1,
        "kind": "benchmark_generation",
        "bundle": bundle,
        "rows_sha256": digest(rows),
        "config": cfg,
        "assets": assets,
        "limit": limit,
        "n_selected": len(rows),
        "full_benchmark": limit is None or len(rows) == bundle["count"],
        "code_sha256": code_hash(),
        "environment": environment(),
        "seed_contract": "sha256-prompt-seed-slot-stream-v1",
    }


def slots_path(run, row):
    return run / "work" / f'{row["index"]:05d}' / "slots"


def inspect(run, rows, manifest):
    if digest(rows) != manifest["rows_sha256"] or len(rows) != manifest["n_selected"]:
        raise ValueError("Saved run input identity mismatch")
    cfg = manifest["config"]
    verified = {}
    expected_dirs = {f'{r["index"]:05d}' for r in rows}
    if (run / "images").exists() and any(
        p.name not in expected_dirs for p in (run / "images").iterdir()
    ):
        raise ValueError("Foreign image-set directory")
    for row in rows:
        folder = slots_path(run, row)
        expected_slots = {f"{i:04d}" for i in range(cfg["k"])}
        if folder.exists() and (
            folder.is_symlink() or any(p.name not in expected_slots for p in folder.iterdir())
        ):
            raise ValueError("Foreign slot or symbolic slot directory")
        records = []
        adaptation_path = folder.parent / "adaptations.json"
        adaptation = read_json(adaptation_path) if adaptation_path.exists() else None
        if adaptation is not None:
            if (
                adaptation.get("run_hash") != digest(manifest)
                or adaptation.get("prompt_id") != row["id"]
            ):
                raise ValueError("Adaptation identity mismatch")
            prompts = adaptation.get("prompts")
            if (
                not isinstance(prompts, list)
                or len(prompts) != cfg["k"]
                or any(not isinstance(p, str) or not p.strip() for p in prompts)
            ):
                raise ValueError("Invalid saved adaptations")
            if cfg["method"] != "pag" and prompts != [row["prompt"]] * cfg["k"]:
                raise ValueError("Base prompt changed")
        for i in range(cfg["k"]):
            slot = folder / f"{i:04d}"
            if not slot.exists():
                records.append(None)
                continue
            if slot.is_symlink() or {p.name for p in slot.iterdir()} != {
                "image.png",
                "receipt.json",
            }:
                raise ValueError("Incomplete/corrupt committed slot")
            record = read_json(slot / "receipt.json")
            if (
                adaptation is None
                or record.get("adaptation_sha256") != digest(adaptation)
                or record.get("render_prompt") != adaptation["prompts"][i]
            ):
                raise ValueError("Saved T2I model prompt changed")
            if (
                record.get("run_hash") != digest(manifest)
                or record.get("prompt_id") != row["id"]
                or record.get("slot") != i
            ):
                raise ValueError("Receipt identity mismatch")
            candidate = record.get("candidate_id")
            if (
                type(candidate) is not int
                or candidate < 0
                or candidate
                >= (cfg["group"]["candidates"] if cfg["method"] == "group" else cfg["k"])
            ):
                raise ValueError("Invalid original candidate identity")
            if cfg["method"] != "group" and candidate != i:
                raise ValueError("Image seed-slot mismatch")
            if record.get("seed") != seed_for(row["id"], cfg["seed"], candidate):
                raise ValueError("Latent seed mismatch")
            sha, size = verify_png(slot / "image.png", cfg["resolution"])
            if record.get("sha256") != sha or record.get("bytes") != size:
                raise ValueError("Image checksum mismatch")
            records.append(record)
        if (
            cfg["method"] == "group"
            and any(records)
            and (not all(records) or len({r["candidate_id"] for r in records}) != cfg["k"])
        ):
            raise ValueError("Group transaction must have exactly K distinct candidates")
        if cfg["method"] == "group" and all(records):
            trace = read_json(folder.parent / "group_trace.json")
            if any(r.get("trace_sha256") != digest(trace) for r in records):
                raise ValueError("Group selection trace changed")
        view = run / "images" / f'{row["index"]:05d}'
        if view.exists():
            if (
                view.is_symlink()
                or not all(records)
                or read_json(view / "metadata.jsonl") != row["metadata"]
            ):
                raise ValueError("Published benchmark metadata mismatch")
            if {p.name for p in (view / "samples").iterdir()} != {
                f"{i:04d}.png" for i in range(cfg["k"])
            }:
                raise ValueError("Published image count mismatch")
            for i, record in enumerate(records):
                if file_hash(view / "samples" / f"{i:04d}.png") != record["sha256"]:
                    raise ValueError("Published image content mismatch")
        verified[row["id"]] = records
    return verified


def write_slot(target, image, record, resolution):
    target.mkdir(parents=True, exist_ok=False)
    image.convert("RGB").save(target / "image.png", format="PNG")
    with (target / "image.png").open("rb") as stream:
        os.fsync(stream.fileno())
    sha, size = verify_png(target / "image.png", resolution)
    record = {**record, "sha256": sha, "bytes": size}
    atomic_json(target / "receipt.json", record)
    sync_directory(target)
    return record


def publish(run, row, records):
    if not all(records):
        return
    dest = run / "images" / f'{row["index"]:05d}'
    if dest.exists():
        return
    stage = run / "staging" / uuid.uuid4().hex
    (stage / "samples").mkdir(parents=True)
    for i in range(len(records)):
        os.link(
            slots_path(run, row) / f"{i:04d}" / "image.png",
            stage / "samples" / f"{i:04d}.png",
        )
    atomic_json(stage / "metadata.jsonl", row["metadata"])
    sync_directory(stage / "samples")
    sync_directory(stage)
    dest.parent.mkdir(exist_ok=True)
    os.rename(stage, dest)
    sync_directory(dest.parent)


def generate(
    bundle_path,
    run,
    cfg,
    asset_path,
    *,
    execute=False,
    max_new_images=None,
    limit=None,
    indices=None,
    factory=None,
):
    validate(cfg)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be nonnegative")
    if execute and (type(max_new_images) is not int or max_new_images <= 0):
        raise ValueError("--execute requires positive --max-new-images")
    bundle, rows = dataset(bundle_path)
    if indices is not None:
        if limit is not None:
            raise ValueError("Use either --indices or --limit")
        if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("Invalid/duplicate selected indices")
        if not set(indices).issubset({r["index"] for r in rows}):
            raise ValueError("Foreign benchmark index")
        rows = [r for r in rows if r["index"] in set(indices)]
    if limit is not None:
        rows = rows[:limit]
    run = Path(run).resolve()
    assets = read_json(asset_path)
    manifest = manifest_for(bundle, rows, cfg, assets, limit)
    if indices is not None:
        manifest["selected_indices"] = sorted(indices)
        manifest["full_benchmark"] = len(rows) == bundle["count"]

    def check():
        if (run / "manifest.json").exists():
            if read_json(run / "manifest.json") != manifest:
                raise ValueError("Run manifest changed; choose a new run directory")
            if read_json(run / "input.json") != rows:
                raise ValueError("Saved benchmark inputs changed")
        elif run.exists() and any(p.name != ".lock" for p in run.iterdir()):
            raise ValueError("Nonempty run lacks manifest")
        return inspect(run, rows, manifest)

    records = check()
    count = lambda: sum(bool(v) for values in records.values() for v in values)
    if not execute:
        return {
            "execute": False,
            "method": cfg["method"],
            "prompts": len(rows),
            "k": cfg["k"],
            "full_benchmark": manifest["full_benchmark"],
            "verified_images": count(),
            "pending_images": len(rows) * cfg["k"] - count(),
            "max_candidate_steps_per_set": cfg["steps"]
            * (cfg["group"]["candidates"] if cfg["method"] == "group" else cfg["k"]),
        }

    def status(state, error=None):
        value = {
            "state": state,
            "updated": datetime.now(timezone.utc).isoformat(),
            "run_hash": digest(manifest),
            "expected_images": len(rows) * cfg["k"],
            "verified_images": count(),
            "complete": count() == len(rows) * cfg["k"],
            "full_benchmark": manifest["full_benchmark"],
            "error": error,
        }
        atomic_json(run / "status.json", value)
        return value

    with run_lock(run):
        records = check()
        if not (run / "manifest.json").exists():
            source = run / "source"
            root = Path(__file__).parents[1]
            source.mkdir()
            for package in ["baselines", "benchmarks", "hpdv2", "iid"]:
                shutil.copytree(
                    root / package,
                    source / package,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            atomic_json(run / "input.json", rows)
            atomic_json(run / "manifest.json", manifest)
        for row in rows:
            publish(run, row, records[row["id"]])
        if count() == len(rows) * cfg["k"]:
            return status("complete")
        names = (
            ["renderer"]
            + (["clip", "dino"] if cfg["method"] == "group" else [])
            + (["pag", "tokenizer", "pag_runtime"] if cfg["method"] == "pag" else [])
        )
        # Verify full content before loading weights, also on resume. The backend gets the verified
        # copy with expanded paths; the manifest keeps the lock as written.
        verified = verify_assets(asset_path, names)
        status("loading")
        backend = None
        used = 0
        try:
            if factory is None:
                from .backend import Backend

                factory = Backend
            backend = factory(verified, cfg, run)
            for row in rows:
                done = records[row["id"]]
                if all(done):
                    continue
                need = cfg["k"] - sum(bool(v) for v in done)
                if used >= max_new_images or (
                    cfg["method"] == "group" and used + need > max_new_images
                ):
                    break
                status("running")
                work = slots_path(run, row).parent
                work.mkdir(parents=True, exist_ok=True)
                adaptation_path = work / "adaptations.json"
                if adaptation_path.exists():
                    adaptation = read_json(adaptation_path)
                    if (
                        adaptation["run_hash"] != digest(manifest)
                        or adaptation["prompt_id"] != row["id"]
                    ):
                        raise ValueError("Adaptation identity mismatch")
                    prompts = adaptation["prompts"]
                else:
                    prompts = backend.adapt(row)
                    if len(prompts) != cfg["k"] or any(
                        not isinstance(p, str) or not p.strip() for p in prompts
                    ):
                        raise ValueError("Expected K nonempty T2I model prompts")
                    adaptation = {
                        "run_hash": digest(manifest),
                        "prompt_id": row["id"],
                        "prompts": prompts,
                    }
                    atomic_json(adaptation_path, adaptation)
                start = time.monotonic()
                if cfg["method"] == "group":
                    images, candidates, trace = backend.group(row)
                    if (
                        len(images) != cfg["k"]
                        or len(candidates) != cfg["k"]
                        or len(set(candidates)) != cfg["k"]
                    ):
                        raise ValueError("Invalid group output identities")
                    stage = run / "staging" / uuid.uuid4().hex
                    stage.mkdir(parents=True)
                    trace = {**trace, "elapsed_seconds": time.monotonic() - start}
                    group_records = []
                    for i, (image, candidate) in enumerate(zip(images, candidates)):
                        receipt = {
                            "run_hash": digest(manifest),
                            "prompt_id": row["id"],
                            "slot": i,
                            "candidate_id": candidate,
                            "seed": seed_for(row["id"], cfg["seed"], candidate),
                            "render_prompt": row["prompt"],
                            "adaptation_sha256": digest(adaptation),
                            "trace_sha256": digest(trace),
                        }
                        group_records.append(
                            write_slot(stage / f"{i:04d}", image, receipt, cfg["resolution"])
                        )
                    atomic_json(work / "group_trace.json", trace)
                    os.rename(stage, slots_path(run, row))
                    sync_directory(work)
                    records[row["id"]] = group_records
                    used += cfg["k"]
                else:
                    for i in range(cfg["k"]):
                        if done[i] or used >= max_new_images:
                            continue
                        latent_seed = seed_for(row["id"], cfg["seed"], i)
                        noise_seed = seed_for(row["id"], cfg["seed"], i, "cads-noise")
                        start = time.monotonic()
                        image, trace = backend.render(prompts[i], latent_seed, noise_seed)
                        receipt = {
                            "run_hash": digest(manifest),
                            "prompt_id": row["id"],
                            "slot": i,
                            "candidate_id": i,
                            "seed": latent_seed,
                            "cads_seed": noise_seed,
                            "render_prompt": prompts[i],
                            "adaptation_sha256": digest(adaptation),
                            "elapsed_seconds": time.monotonic() - start,
                            "trace": trace,
                        }
                        stage = run / "staging" / uuid.uuid4().hex
                        record = write_slot(stage, image, receipt, cfg["resolution"])
                        target = slots_path(run, row) / f"{i:04d}"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.rename(stage, target)
                        sync_directory(target.parent)
                        done[i] = record
                        used += 1
                        status("running")
                publish(run, row, records[row["id"]])
                status("running")
            return status("complete" if count() == len(rows) * cfg["k"] else "budget_exhausted")
        except Exception as error:
            status("failed", f"{type(error).__name__}: {error}")
            raise
        finally:
            if backend is not None:
                backend.close()
