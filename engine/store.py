"""Durable engine state in SQLite: current state, timeline, sleep pressure, and the work queue.

Exactly-once work, even through `kill -9`:
  * a work item is computed *outside* any transaction (it may take seconds and use the GPU);
  * its results are written, and the item flipped from 'pending' to 'done', in ONE short transaction;
  * if the process dies before that commit, nothing was written and the item is still pending,
    so it is simply redone on resume; if two runners ever race, only one flip succeeds.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from engine.policy import State

if TYPE_CHECKING:
    from engine.workload import WorkContext, Workload

SCHEMA = """
CREATE TABLE IF NOT EXISTS engine_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS timeline (
    id         INTEGER PRIMARY KEY,
    state      TEXT NOT NULL,          -- day / dream / deep_sleep / rest / off
    reason     TEXT NOT NULL,
    budget     REAL,
    started_at REAL NOT NULL,
    ended_at   REAL
);
CREATE TABLE IF NOT EXISTS work_queue (
    id         INTEGER PRIMARY KEY,
    state      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',   -- pending / done / failed
    attempts   INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    created_at REAL NOT NULL,
    done_at    REAL
);
CREATE INDEX IF NOT EXISTS work_queue_pending ON work_queue (state, status, id);
"""

OFF = "off"  # timeline marker for "the learner wasn't running"


@dataclass(frozen=True)
class WorkItem:
    id: int
    state: State
    kind: str
    payload: dict
    attempts: int


def reason_key(reason: str) -> str:
    """Reason with the numbers stripped, so '4.1 GB' -> '4.2 GB' doesn't start a new timeline row."""
    return re.sub(r"\d+(\.\d+)?", "#", reason)


class Store:
    def __init__(self, path: str | Path, max_attempts: int = 3) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=30.0, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # durable against process crashes, which is what kill -9 is
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ---- key/value --------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM engine_kv WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    @staticmethod
    def _set(conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute(
            "INSERT INTO engine_kv (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def set(self, key: str, value: Any) -> None:
        with self.transaction() as conn:
            self._set(conn, key, value)

    # ---- state and timeline -----------------------------------------------------------------

    def current_state(self) -> State:
        return State(self.get("current_state", State.REST.value))

    def entered_at(self, default: float) -> float:
        return float(self.get("entered_at", default))

    def current_reason(self) -> str:
        return self.get("current_reason", "")

    def enter_state(self, state: State, reason: str, budget: float, at: float) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE timeline SET ended_at = ? WHERE ended_at IS NULL", (at,))
            conn.execute(
                "INSERT INTO timeline (state, reason, budget, started_at) VALUES (?, ?, ?, ?)",
                (state.value, reason, budget, at),
            )
            if self.get("current_state") != state.value:
                self._set(conn, "entered_at", at)
            self._set(conn, "current_state", state.value)
            self._set(conn, "current_reason", reason)

    def heartbeat(self, at: float, state: State, dt: float) -> None:
        """Called every tick. Adds the tick to the state's active-compute total (REST doesn't count)."""
        with self.transaction() as conn:
            self._set(conn, "heartbeat", at)
            if state is not State.REST and dt > 0:
                key = f"active_seconds.{state.value}"
                self._set(conn, key, float(self.get(key, 0.0)) + dt)

    def recover(self, now: float) -> None:
        """On startup after a crash or kill: close the open timeline row at the last heartbeat
        and record the gap as 'off', so the history shows exactly when the learner wasn't running."""
        open_row = self.conn.execute("SELECT id FROM timeline WHERE ended_at IS NULL").fetchone()
        last_beat = self.get("heartbeat")
        if open_row is None or last_beat is None:
            return
        with self.transaction() as conn:
            conn.execute("UPDATE timeline SET ended_at = ? WHERE id = ?", (last_beat, open_row[0]))
            conn.execute(
                "INSERT INTO timeline (state, reason, budget, started_at, ended_at) VALUES (?, ?, 0, ?, ?)",
                (OFF, "learner was not running (stopped, killed or computer off)", last_beat, now),
            )
            self._set(conn, "current_state", State.REST.value)
            self._set(conn, "entered_at", now)
            self._set(conn, "current_reason", "starting up")

    def mark_off(self, at: float, reason: str = "learner stopped") -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE timeline SET ended_at = ? WHERE ended_at IS NULL", (at,))
            conn.execute(
                "INSERT INTO timeline (state, reason, budget, started_at, ended_at) VALUES (?, ?, 0, ?, ?)",
                (OFF, reason, at, at),
            )
            self._set(conn, "current_state", State.REST.value)
            self._set(conn, "current_reason", reason)
            self._set(conn, "heartbeat", at)

    def timeline(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT state, reason, budget, started_at, ended_at FROM timeline ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(zip(("state", "reason", "budget", "started_at", "ended_at"), r)) for r in rows][::-1]

    def active_seconds(self) -> dict[str, float]:
        return {s.value: float(self.get(f"active_seconds.{s.value}", 0.0)) for s in State if s is not State.REST}

    # ---- sleep pressure ---------------------------------------------------------------------

    def pressure(self) -> float:
        return float(self.get("sleep_pressure", 0.0))

    def _add_pressure(self, conn: sqlite3.Connection, delta: float) -> None:
        if delta:
            self._set(conn, "sleep_pressure", max(0.0, self.pressure() + delta))

    # ---- work queue -------------------------------------------------------------------------

    def enqueue(self, state: State, items: list[tuple[str, dict]]) -> list[int]:
        now = time.time()
        ids = []
        with self.transaction() as conn:
            for kind, payload in items:
                cur = conn.execute(
                    "INSERT INTO work_queue (state, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                    (state.value, kind, json.dumps(payload), now),
                )
                ids.append(cur.lastrowid)
        return ids

    def pending_count(self, state: State | None = None) -> int:
        if state is None:
            sql, args = "SELECT COUNT(*) FROM work_queue WHERE status = 'pending'", ()
        else:
            sql, args = "SELECT COUNT(*) FROM work_queue WHERE status = 'pending' AND state = ?", (state.value,)
        return int(self.conn.execute(sql, args).fetchone()[0])

    def queue_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for state, status, n in self.conn.execute(
            "SELECT state, status, COUNT(*) FROM work_queue GROUP BY state, status"
        ):
            out.setdefault(state, {})[status] = n
        return out

    def next_pending(self, state: State) -> WorkItem | None:
        row = self.conn.execute(
            "SELECT id, kind, payload, attempts FROM work_queue "
            "WHERE state = ? AND status = 'pending' ORDER BY id LIMIT 1",
            (state.value,),
        ).fetchone()
        if row is None:
            return None
        return WorkItem(row[0], state, row[1], json.loads(row[2]), row[3])

    def run_one(self, state: State, workload: "Workload", ctx: "WorkContext") -> str:
        """Run the next pending item for `state`. Returns 'done', 'empty', 'cancelled' or 'failed'."""
        from engine.workload import Cancelled

        item = self.next_pending(state)
        if item is None:
            return "empty"
        try:
            result = workload.run(item, ctx)  # compute only; no writes
        except Cancelled:
            return "cancelled"  # nothing written; the item stays pending
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the learner
            with self.transaction() as conn:
                attempts = item.attempts + 1
                status = "failed" if attempts >= self.max_attempts else "pending"
                conn.execute(
                    "UPDATE work_queue SET attempts = ?, status = ?, error = ? WHERE id = ?",
                    (attempts, status, f"{type(exc).__name__}: {exc}", item.id),
                )
            return "failed"
        try:
            with self.transaction() as conn:
                cur = conn.execute(
                    "UPDATE work_queue SET status = 'done', done_at = ? WHERE id = ? AND status = 'pending'",
                    (time.time(), item.id),
                )
                if cur.rowcount != 1:  # someone else completed it first; keep their results, drop ours
                    raise _AlreadyDone()
                self._add_pressure(conn, workload.save(item, result, conn))
        except _AlreadyDone:
            return "duplicate"
        return "done"


class _AlreadyDone(Exception):
    pass
