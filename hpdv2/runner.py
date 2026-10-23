"""Bounded execution, verified resumption, and auditable batch attempts."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import platform
import threading
import time

from .api import CallError
from .dataset import SCHEMA
from .policy import POLICY_VERSION, parse_decisions, system_prompt, validate_decisions
from .storage import append_event, atomic_json, digest, read_events, read_json, run_lock


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    versions = {"python": platform.python_version()}
    for name in (
        "litellm",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "huggingface-hub",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def make_manifest(data, rows, config):
    system = system_prompt(config["stage"])
    sources = {
        p.name: p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "schema_version": SCHEMA,
        "input_hash": digest(data),
        "pool_hash": digest(rows),
        "upstream_complete": (
            data.get("kind") == "prompt_pool"
            or data.get("metadata", {}).get("full_pipeline_complete") is True
        ),
        "n_input": len(rows),
        "policy_version": POLICY_VERSION,
        "system_prompt": system,
        "system_hash": digest(system),
        "code_hash": digest(sources),
        "config": config,
        "environment": environment(),
    }


def check_manifest(directory, manifest):
    path = Path(directory) / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(
                "Run manifest mismatch (input, policy, code, environment or configuration). Use a new run directory."
            )
    elif any((Path(directory) / name).exists() for name in ("checkpoint.jsonl", "summary.json")):
        raise ValueError(
            "Run artifacts exist without a manifest; refusing legacy/unverified resume"
        )


def recover(events, rows, manifest):
    known = {r["id"]: r for r in rows}
    done, errors = {}, {}
    identity = digest(manifest)
    for event in events:
        if event.get("run_hash") != identity:
            raise ValueError("Journal belongs to a different run")
        ids = event.get("ids")
        if (
            not isinstance(ids, list)
            or any(not isinstance(i, str) or i not in known for i in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("Invalid journal input IDs")
        batch = [known[i] for i in ids]
        if event.get("status") == "ok":
            records = validate_decisions(
                event.get("records"), batch, manifest["config"]["stage"], include_prompt=True
            )
            for row in records:
                if row["id"] in done:
                    raise ValueError("Duplicate successful decision in journal")
                done[row["id"]] = row
                errors.pop(row["id"], None)
        elif event.get("status") == "error":
            for row in batch:
                if row["id"] not in done:
                    errors[row["id"]] = {**row, "error": event.get("error", "unknown")}
        else:
            raise ValueError("Invalid journal event status")
    return done, errors


def summary(rows, manifest, events):
    done, errors = recover(events, rows, manifest)
    kept = [done[r["id"]] for r in rows if r["id"] in done and done[r["id"]]["keep"]]
    rejected = [done[r["id"]] for r in rows if r["id"] in done and not done[r["id"]]["keep"]]
    pending = [r for r in rows if r["id"] not in done]
    from collections import Counter

    return {
        "schema_version": SCHEMA,
        "kind": "filter_summary",
        "stage": manifest["config"]["stage"],
        "run_hash": digest(manifest),
        "metadata": {
            "n_input": len(rows),
            "n_kept": len(kept),
            "n_rejected": len(rejected),
            "n_pending": len(pending),
            "n_errors": len(errors),
            "complete": not pending,
            "upstream_complete": manifest["upstream_complete"],
            "full_pipeline_complete": not pending and manifest["upstream_complete"],
            "variation_axis_count_distribution": dict(
                Counter(str(len(r["variation_axes"])) for r in kept if "variation_axes" in r)
            ),
            "n_attempts_logged": sum(len(e.get("attempts", [])) for e in events),
        },
        "kept": kept,
        "rejected": rejected,
        "errors": [errors[r["id"]] for r in rows if r["id"] in errors],
        "pending": pending,
    }


class Budget:
    def __init__(self, maximum):
        self.maximum = maximum
        self.used = 0
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def take(self):
        with self.lock:
            if self.stopped.is_set() or self.used >= self.maximum:
                return False
            self.used += 1
            return True


def judge(batch, manifest, client, budget, attempts, sleep=time.sleep):
    # Short IDs spare the model from copying long hashes. The journal keeps raw responses
    # and the mapping.
    short_ids = manifest["config"].get("short_request_ids", False)
    wire_batch = (
        [{"id": f"p{i}", "prompt": row["prompt"]} for i, row in enumerate(batch)]
        if short_ids
        else batch
    )
    id_map = {wire["id"]: row["id"] for wire, row in zip(wire_batch, batch)}
    logs = []
    records = None
    last_error = "request_budget_exhausted"
    for attempt in range(attempts):
        if not budget.take():
            break
        log = {"time": timestamp()}
        try:
            reply = client(wire_batch, manifest["system_prompt"])
            log.update(
                raw_response=reply.text,
                usage=reply.usage,
                response_model=reply.model,
                finish_reason=reply.finish_reason,
            )
            if reply.finish_reason not in {"stop", "end_turn"}:
                raise ValueError("Non-final response (truncated, refused or tool call)")
            records = parse_decisions(reply.text, wire_batch, manifest["config"]["stage"])
            if short_ids:
                records = [{**row, "id": id_map[row["id"]]} for row in records]
                validate_decisions(records, batch, manifest["config"]["stage"], include_prompt=True)
        except (ValueError, TypeError) as error:
            last_error = f"Invalid response: {type(error).__name__}: {error}"
            retryable = True
        except CallError as error:
            last_error, retryable = str(error), error.retryable
        else:
            logs.append(log)
            break
        log["error"] = last_error
        logs.append(log)
        if not retryable:
            budget.stopped.set()  # Stop all workers on auth, model or config errors.
            break
        if attempt + 1 < attempts and not budget.stopped.is_set():
            sleep(min(2**attempt, 30))
    if not logs:
        return None
    event = {
        "run_hash": digest(manifest),
        "ids": [r["id"] for r in batch],
        "status": "ok" if records is not None else "error",
        "attempts": logs,
    }
    if short_ids:
        event["request_id_map"] = id_map
    if records is not None:
        event["records"] = records
    else:
        event["error"] = last_error
    return event


def plan(directory, rows, manifest, limit=None):
    check_manifest(directory, manifest)
    events = read_events(Path(directory) / "checkpoint.jsonl")
    done, errors = recover(events, rows, manifest)
    todo = [row for row in rows if row["id"] not in done]
    selected = todo if limit is None else todo[:limit]
    return {
        "n_input": len(rows),
        "n_success": len(done),
        "n_retryable_prompts": len(errors),
        "n_pending": len(todo),
        "n_selected": len(selected),
        "batches_without_retries": math.ceil(len(selected) / manifest["config"]["batch_size"]),
    }


def execute(
    directory,
    rows,
    manifest,
    client,
    *,
    limit=None,
    max_requests,
    workers=1,
    attempts=3,
    repair_tail=False,
    sleep=time.sleep,
):
    if max_requests <= 0 or workers <= 0 or attempts <= 0 or (limit is not None and limit < 0):
        raise ValueError("Invalid execution limits")
    directory = Path(directory)
    with run_lock(directory):
        check_manifest(directory, manifest)
        if not (directory / "manifest.json").exists():
            atomic_json(directory / "manifest.json", manifest)
        journal = directory / "checkpoint.jsonl"
        events = read_events(journal, repair_tail=repair_tail)
        done, _ = recover(events, rows, manifest)
        todo = [row for row in rows if row["id"] not in done]
        if limit is not None:
            todo = todo[:limit]
        size = manifest["config"]["batch_size"]
        chunks = iter(todo[i : i + size] for i in range(0, len(todo), size))
        budget = Budget(max_requests)

        def submit(pool):
            batch = next(chunks, None)
            return (
                None
                if batch is None
                else pool.submit(judge, batch, manifest, client, budget, attempts, sleep)
            )

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {f for _ in range(workers) if (f := submit(pool)) is not None}
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        event = future.result()
                        if event is not None:
                            append_event(journal, event)
                            events.append(event)
                            print(
                                f"Checkpointed {len(event['ids'])} prompts: {event['status']}",
                                flush=True,
                            )
                        if budget.used < max_requests and not budget.stopped.is_set():
                            new = submit(pool)
                            if new is not None:
                                pending.add(new)
        finally:
            result = summary(rows, manifest, events)
            atomic_json(directory / "summary.json", result)
        return result
