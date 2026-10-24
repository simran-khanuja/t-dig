"""The loop's state machine, its session store, and the study's event log.

The event log is the dataset: every reported number (arm win-rate, axis-aware/axis-blind
ratio per arm, fidelity distribution, grounding ablation) must be recomputable from
`runs/<run>/events.jsonl` alone, so each record carries the full payload of its transition.

    START
      -> SAMPLE_IID   K independent seeds, no axis
      -> GATE             "does this need diversity?" -- user yes/no
           no  -> DONE
           yes -> CHOOSE_AXIS
      -> CHOOSE_AXIS      free text, or a chip from /suggest
      -> REVIEW           the axis-diverse grid; sliders re-render it in place
           accept -> (optional high-quality re-render) -> DONE
           reject -> CHOOSE_AXIS with the rejected axis excluded from re-suggestion

The gate and the accept/reject step are study mode's; the open demo answers the gate itself
and stays in REVIEW, making further sets. The UI renders one screen per state, and the
analysis uses the state to know what a participant had seen when they answered. `advance` is
the only writer, so an illegal transition is a 409.
"""

import json
import os
import random
import sqlite3
import threading
import time
import uuid

START = "START"
GATE = "GATE"
CHOOSE_AXIS = "CHOOSE_AXIS"
REVIEW = "REVIEW"
DONE = "DONE"

# from -> allowed next. SAMPLE_IID and EXPAND_AND_RENDER are actions, not states: the
# request that performs one either lands on the next state or fails, and both are logged.
TRANSITIONS = {
    START: {GATE},
    GATE: {CHOOSE_AXIS, DONE},
    CHOOSE_AXIS: {REVIEW},
    REVIEW: {CHOOSE_AXIS, DONE},
    DONE: set(),
}


class IllegalTransition(Exception):
    pass


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- arms


def assign_arm(mode, salt, session_id, trial_index):
    """Return the arm ("ours" or "baseline") that produces this trial's grid.

    *mode* is "ours", "baseline", or "ab". Under "ab" the choice is deterministic in
    (salt, session, trial index), so a re-rendered trial lands on the same arm. The salt is
    per run, so two runs are randomised independently.

    Raises:
        ValueError: unknown *mode*.
    """
    if mode == "ours":
        return "ours"
    if mode == "baseline":
        return "baseline"
    if mode != "ab":
        raise ValueError(f"unknown arm mode {mode!r}")
    rng = random.Random(f"{salt}|{session_id}|{trial_index}")
    return rng.choice(["ours", "baseline"])


# --------------------------------------------------------------------------- log


class EventLog:
    """Append-only JSONL, one line per transition, flushed immediately.

    Flushing keeps the end of a session (where accept/reject lives) if the process is killed.
    One participant is ~50 lines.
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event, session_id=None, **payload):
        rec = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "session_id": session_id,
            **payload,
        }
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
        return rec


# --------------------------------------------------------------------------- store


class SessionStore:
    """Session store interface, implemented in memory (MemoryStore) or SQLite (SqliteStore).

    Sessions are whole JSON documents. This is live state for the loop; the analysis reads
    the event log.
    """

    def get(self, sid):
        raise NotImplementedError

    def put(self, s):
        raise NotImplementedError

    def ids(self):
        raise NotImplementedError


class MemoryStore(SessionStore):
    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def get(self, sid):
        return self._d.get(sid)

    def put(self, s):
        with self._lock:
            self._d[s["session_id"]] = s

    def ids(self):
        return list(self._d)


class SqliteStore(SessionStore):
    """Session store that survives an orchestrator restart.

    check_same_thread=False plus a lock: the HTTP server is threaded, and sqlite3 allows
    cross-thread use when access is serialised.
    """

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "session_id TEXT PRIMARY KEY, updated REAL, doc TEXT)"
        )
        self._db.commit()
        self._lock = threading.Lock()

    def get(self, sid):
        with self._lock:
            row = self._db.execute("SELECT doc FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, s):
        doc = json.dumps(s, ensure_ascii=False, default=str)
        with self._lock:
            self._db.execute(
                "INSERT INTO sessions(session_id, updated, doc) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET updated=excluded.updated, "
                "doc=excluded.doc",
                (s["session_id"], time.time(), doc),
            )
            self._db.commit()

    def ids(self):
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT session_id FROM sessions")]


def make_store(kind, path):
    return MemoryStore() if kind == "memory" else SqliteStore(path)


# --------------------------------------------------------------------------- session


def new_session(user_id=None, arm_mode="ours", meta=None):
    return {
        "session_id": new_id("s"),
        "user_id": user_id,
        "created": time.time(),
        "state": START,
        "arm_mode": arm_mode,
        "base_prompt": None,
        "iid": None,  # {images, seed, ...} from the T2I model server
        "needs_diversity": None,
        "suggested": [],  # the chips shown, in the order they were shown
        "rejected_axes": [],
        "trials": [],
        "meta": meta or {},
    }


def advance(s, to):
    if to not in TRANSITIONS[s["state"]]:
        raise IllegalTransition(
            f"cannot go {s['state']} -> {to}. Allowed: {sorted(TRANSITIONS[s['state']])}"
        )
    s["state"] = to
    return s


def current_trial(s):
    return s["trials"][-1] if s["trials"] else None
