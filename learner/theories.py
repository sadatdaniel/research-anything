"""The learner's theories: every connection it has proposed, with its stage and the history of its tests.

Stages:
  proposed           found by the cheap search; not tested yet
  prediction-tested  passed the prediction test and the chance test (learner/verify.py)
  confirmed          also passed the source check: every record was re-read by the model and says what
                     the theory needs it to say, and every evidence quote is in its record word for word
  rejected           failed a test; the reason is kept, so the same wrong idea isn't believed again
  superseded         replaced by a better version of the same connection (or it vanished from the
                     knowledge after a correction)

Each change is logged with the learner's active dream time, which is what the H1 curves are drawn from.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time

from learner.patterns import Candidate

SCHEMA = """
CREATE TABLE IF NOT EXISTS theories (
    id          INTEGER PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,
    type        TEXT NOT NULL,
    summary     TEXT,
    record_ids  TEXT NOT NULL,
    params      TEXT,
    prediction  TEXT,
    stats       TEXT,
    groups      TEXT,
    evidence    TEXT,
    tests       TEXT,
    fingerprint TEXT,            -- hash of the record set: a changed theory is re-tested
    stage       TEXT NOT NULL,
    reason      TEXT,
    created_at  REAL,
    updated_at  REAL,
    dream_seconds REAL
);
CREATE TABLE IF NOT EXISTS theory_history (
    theory_id   INTEGER NOT NULL,
    stage       TEXT NOT NULL,
    reason      TEXT,
    at          REAL,
    dream_seconds REAL,
    claim       TEXT             -- the theory as a claim at that moment (kept for confirmations)
);
"""

LIVE = ("proposed", "prediction-tested", "confirmed")


def confidence(t: dict) -> tuple[int, dict]:
    """One 0-100 score summarising how sure the learner is, with its parts. Not a probability: a
    readable summary of the tests. Parts (each 0..1):
      chance      how far from coincidence (periodic: shuffle test, full marks at p <= 0.01;
                  cause/effect and chains: binomial chance, full marks at p <= 1e-6)
      prediction  share of its predictions that came true (returns predicted from earlier years, or
                  causes followed by their effect)
      support     how many times it happened (full marks at 6)
      source      re-read and confirmed (1), waiting for the re-reading (0.5), not checked or failed (0)"""
    tests = {x["test"]: x for x in (t.get("tests") or [])}
    stats = t.get("stats") or {}
    parts = {}
    ch = tests.get("chance")
    if ch and ch.get("p") is not None:
        full = 2.0 if t["type"] == "periodic" else 6.0
        parts["chance"] = round(min(1.0, -math.log10(max(ch["p"], 1e-12)) / full), 2)
    elif t["type"] == "identity":
        parts["chance"] = 1.0 if tests.get("lifetime", {}).get("passed") else 0.0
    else:
        parts["chance"] = 0.0
    pred = tests.get("prediction")
    if t["type"] == "periodic" and pred and pred.get("predicted"):
        parts["prediction"] = round(len(pred.get("came_true", [])) / len(pred["predicted"]), 2)
    elif t["type"] in ("cause_effect", "chain"):
        parts["prediction"] = round(float(stats.get("consistency", 0.0)), 2)
    else:
        parts["prediction"] = 1.0 if t["type"] == "identity" else 0.0
    parts["support"] = round(min(1.0, stats.get("hits", 0) / 6), 2)
    parts["source"] = {"confirmed": 1.0, "prediction-tested": 0.5}.get(t["stage"], 1.0 if (
        t["stage"] == "superseded" and t.get("evidence")) else 0.0)
    score = 100 * (0.35 * parts["chance"] + 0.25 * parts["prediction"] + 0.2 * parts["support"] + 0.2 * parts["source"])
    return round(score), parts


def ensure_schema(conn: sqlite3.Connection) -> None:
    # executescript() commits any open transaction, so only run it when the tables are missing.
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'theory_history'").fetchone():
        conn.executescript(SCHEMA)


def dream_seconds(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT value FROM engine_kv WHERE key = 'active_seconds.dream'").fetchone() \
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'engine_kv'").fetchone() else None
    return float(json.loads(row[0])) if row else 0.0


def record_fingerprint(record_ids) -> str:
    return hashlib.sha1(",".join(sorted(record_ids)).encode()).hexdigest()[:12]


def all_theories(conn: sqlite3.Connection, stages: tuple[str, ...] | None = None) -> list[dict]:
    ensure_schema(conn)
    cols = ("id", "key", "type", "summary", "record_ids", "params", "prediction", "stats", "groups", "evidence",
            "tests", "fingerprint", "stage", "reason", "created_at", "updated_at", "dream_seconds")
    rows = []
    for r in conn.execute(f"SELECT {', '.join(cols)} FROM theories ORDER BY id"):
        d = dict(zip(cols, r))
        for f in ("record_ids", "params", "prediction", "stats", "groups", "evidence", "tests"):
            d[f] = json.loads(d[f]) if d[f] else None
        d["confidence"], d["score_parts"] = confidence(d)
        if stages is None or d["stage"] in stages:
            rows.append(d)
    return rows


def set_stage(conn: sqlite3.Connection, theory_id: int, stage: str, reason: str = "") -> None:
    now, ds = time.time(), dream_seconds(conn)
    conn.execute("UPDATE theories SET stage = ?, reason = ?, updated_at = ?, dream_seconds = ? WHERE id = ?",
                 (stage, reason, now, ds, theory_id))
    claim = None
    if stage == "confirmed":
        t = next(t for t in all_theories(conn) if t["id"] == theory_id)
        claim = json.dumps(as_claim(t))
    conn.execute("INSERT INTO theory_history VALUES (?, ?, ?, ?, ?, ?)", (theory_id, stage, reason, now, ds, claim))


def upsert(conn: sqlite3.Connection, c: Candidate) -> tuple[int, bool]:
    """Insert a new candidate as 'proposed', or refresh an existing one. Returns (id, needs_testing)."""
    fp = record_fingerprint(c.record_ids)
    row = conn.execute("SELECT id, fingerprint, stage FROM theories WHERE key = ?", (c.key,)).fetchone()
    payload = (c.type, c.summary, json.dumps(c.record_ids), json.dumps(c.params), json.dumps(c.prediction),
               json.dumps(c.stats), json.dumps(c.groups), fp)
    now = time.time()
    if row is None:
        cur = conn.execute(
            "INSERT INTO theories (key, type, summary, record_ids, params, prediction, stats, groups, fingerprint, "
            "stage, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)",
            (c.key, *payload, now, now))
        set_stage(conn, cur.lastrowid, "proposed", "found by the search")
        return cur.lastrowid, True
    tid, old_fp, stage = row
    if old_fp == fp and stage != "superseded":
        return tid, False
    conn.execute("UPDATE theories SET type = ?, summary = ?, record_ids = ?, params = ?, prediction = ?, stats = ?, "
                 "groups = ?, fingerprint = ?, evidence = NULL, updated_at = ? WHERE id = ?", (*payload, now, tid))
    set_stage(conn, tid, "proposed", "its records changed after the knowledge improved; testing again")
    return tid, True


def record_tests(conn: sqlite3.Connection, theory_id: int, tests: list[dict]) -> None:
    conn.execute("UPDATE theories SET tests = ? WHERE id = ?", (json.dumps(tests), theory_id))


def confirmed_claims(conn: sqlite3.Connection) -> list[dict]:
    """Confirmed theories in the scorer's claim format."""
    return [as_claim(t) for t in all_theories(conn, ("confirmed",))]


def as_claim(t: dict) -> dict:
    return {"type": t["type"], "record_ids": t["record_ids"], "params": t["params"], "prediction": t["prediction"],
            "evidence": t["evidence"] or [], "summary": t["summary"]}


def claims_at(conn: sqlite3.Connection, seconds: float) -> list[dict]:
    """The claims the learner stood behind after this much dream time (for the H1 curves): every theory
    whose latest stage by then was 'confirmed', as it was when it was confirmed."""
    latest: dict[int, tuple[str, str | None]] = {}
    for tid, stage, ds, claim in conn.execute(
            "SELECT theory_id, stage, dream_seconds, claim FROM theory_history ORDER BY rowid"):
        if (ds or 0.0) <= seconds:
            latest[tid] = (stage, claim)
    return [json.loads(c) for s, c in latest.values() if s == "confirmed" and c]
