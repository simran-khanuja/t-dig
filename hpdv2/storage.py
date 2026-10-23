"""Canonical hashes, atomic files, durable JSONL and exclusive run ownership."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def scheduler_job_id(environ=None):
    """Return the batch job id: T_DIG_JOB_ID, else SLURM_JOB_ID ('' outside a job)."""
    environ = os.environ if environ is None else environ
    return environ.get("T_DIG_JOB_ID") or environ.get("SLURM_JOB_ID") or ""


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def prompt_id(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process owns this run directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def append_event(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path, repair_tail=False):
    """Read a JSONL journal, raising on any corrupt line.

    With `repair_tail`, an unterminated last line is copied verbatim to a `.tail-<sha256>.bak`
    file and truncated from the journal.
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not repair_tail:
            raise ValueError(
                "Unterminated journal tail; inspect it or use --repair-tail with --execute"
            )
        end = raw.rfind(b"\n") + 1
        # Exclusive creation never overwrites an earlier backup.
        backup = path.with_name(
            path.name + ".tail-" + hashlib.sha256(raw[end:]).hexdigest() + ".bak"
        )
        with backup.open("xb") as stream:
            stream.write(raw[end:])
            stream.flush()
            os.fsync(stream.fileno())
        with path.open("r+b") as stream:
            stream.truncate(end)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(path.parent)
        raw = raw[:end]
    result = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            raise ValueError(
                f"Corrupt journal record at line {number}; refusing to skip it"
            ) from None
        result.append(row)
    return result
