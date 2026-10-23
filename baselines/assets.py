"""Asset locks: portable paths, full-file SHA-256 verification and seed derivation."""

import hashlib
import os
from pathlib import Path
import re

from hpdv2.storage import atomic_json, digest, read_json


# Asset locks store directories as "${T_DIG_MODELS}/<name>" or "${T_DIG_ROOT}/<path>", so a lock
# verifies on any machine. `_default` gives each variable's value when it is unset.
_ROOT = Path(__file__).resolve().parents[1]


def _default(name):
    root = Path(os.environ.get("T_DIG_ROOT", _ROOT))
    return {
        "T_DIG_ROOT": root,
        "T_DIG_DATA": root / "data",
        "T_DIG_RUNS": root / "runs",
        "T_DIG_MODELS": root / "models",
    }.get(name)


def expand_path(path):
    """Expand ${VAR} in an asset path, defaulting the T_DIG_* variables to the repository layout."""

    def value(m):
        v = os.environ.get(m.group(1)) or _default(m.group(1))
        if v is None:
            raise ValueError(f"Asset path uses unset variable {m.group(1)}: {path}")
        return str(v)

    return re.sub(r"\$\{(\w+)\}", value, str(path))


def contract_path(path):
    """Rewrite a path under T_DIG_MODELS or T_DIG_ROOT in ${VAR} form (inverse of expand_path)."""
    path = str(path)
    for name in ("T_DIG_MODELS", "T_DIG_ROOT"):
        base = str(Path(os.environ.get(name) or _default(name)).resolve())
        if path == base or path.startswith(base + os.sep):
            return "${" + name + "}" + path[len(base) :]
    return path


def local_checkpoint(checkpoint, cache=None):
    """Return a local directory for `checkpoint`.

    An existing directory is returned as given. `repo[@revision]` resolves through the Hugging Face
    cache and is downloaded when absent and the hub is reachable.
    """
    p = Path(str(checkpoint)).expanduser()
    if p.is_dir():
        return p
    repo, _, revision = str(checkpoint).partition("@")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    # The cache first: huggingface_hub 1.x refuses a cached snapshot under HF_HUB_OFFLINE=1
    # unless it is asked for local files only.
    try:
        return Path(
            snapshot_download(
                repo, revision=revision or None, cache_dir=cache, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return Path(snapshot_download(repo, revision=revision or None, cache_dir=cache))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(path):
    path = Path(path).resolve()
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and not any(
            part in {".cache", "__pycache__", "bin"} or part.endswith(".dist-info")
            for part in p.relative_to(path).parts
        )
        and p.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"Empty/missing model directory: {path}")
    return {
        "path": str(path),
        "files": {
            str(p.relative_to(path)): {
                "sha256": file_hash(p),
                "bytes": p.stat().st_size,
            }
            for p in files
        },
    }


def lock_assets(paths, out, provenance=None):
    if out.exists():
        raise ValueError("Asset lock exists; refusing overwrite")
    data = {
        "schema_version": 1,
        "assets": {
            k: dict(
                inventory(expand_path(v)), path=contract_path(inventory(expand_path(v))["path"])
            )
            for k, v in paths.items()
        },
        "provenance": provenance or {},
    }
    atomic_json(out, data)
    return data


def verify_assets(path, names):
    data = read_json(path)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported asset lock")
    # The returned data holds expanded paths. Content is compared file by file.
    for entry in data["assets"].values():
        entry["path"] = expand_path(entry["path"])
    for name in names:
        entry = data["assets"][name]
        if inventory(entry["path"])["files"] != entry["files"]:
            raise ValueError(f"Asset content changed: {name}")
    return data


def seed_for(prompt_id, global_seed, slot, stream="latent"):
    return int(
        digest(
            {
                "prompt_id": prompt_id,
                "seed": global_seed,
                "slot": slot,
                "stream": stream,
            }
        )[:15],
        16,
    )
