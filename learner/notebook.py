"""The notebook: the learner's explicit memory, in the same SQLite file as the cycle engine.

Sharing one database file means a note and its work item are committed in one transaction, which
keeps the engine's exactly-once guarantee.

Tables:
  sources  the original records exactly as read (the source check always goes back to these)
  notes    one row per record: event, places, people, details, summary, quiz question and answer
  links    what records say about the map and about names: same_as / located_in / on_river
           (each link keeps the record it came from, and whether that record states it word for word)

  python -m learner.notebook ingest --db learner-data/brain/brain.sqlite --world worlds/test/test-001
  python -m learner.notebook show   --db learner-data/brain/brain.sqlite [--limit 10]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    record_id  TEXT PRIMARY KEY,
    year       INTEGER NOT NULL,
    chronicler TEXT,
    text       TEXT NOT NULL,
    origin     TEXT NOT NULL,          -- file the record came from
    line       INTEGER NOT NULL         -- 1-based line in that file, for "open at this place"
);
CREATE TABLE IF NOT EXISTS notes (
    record_id  TEXT PRIMARY KEY REFERENCES sources(record_id),
    year       INTEGER NOT NULL,
    event      TEXT,
    places     TEXT,                    -- JSON lists
    people     TEXT,
    details    TEXT,
    summary    TEXT,
    question   TEXT,
    answer     TEXT,
    answer_verbatim INTEGER,            -- 1 if the quiz answer appears word for word in the record
    complete   INTEGER,
    raw        TEXT,
    loops      INTEGER,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS links (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,           -- same_as / located_in / on_river
    a          TEXT NOT NULL,
    b          TEXT NOT NULL,
    record_id  TEXT NOT NULL REFERENCES sources(record_id),
    stated     INTEGER NOT NULL DEFAULT 0  -- 1 if the record says it word for word; 0 = the model inferred it
);
CREATE INDEX IF NOT EXISTS links_ab ON links (kind, a, b);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(links)")}
    if "stated" not in cols:  # notebooks made before links were source-checked
        conn.execute("ALTER TABLE links ADD COLUMN stated INTEGER NOT NULL DEFAULT 0")


def ingest_world(conn: sqlite3.Connection, world_dir: str | Path) -> int:
    """Copy a world's visible records into `sources`. Idempotent. Returns the number of new records."""
    ensure_schema(conn)
    path = Path(world_dir) / "records.jsonl"
    before = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    conn.execute("BEGIN IMMEDIATE")
    with path.open() as fh:
        for i, line in enumerate(fh, 1):
            r = json.loads(line)
            conn.execute("INSERT OR IGNORE INTO sources VALUES (?, ?, ?, ?, ?, ?)",
                         (r["id"], r["year"], r.get("chronicler"), r["text"], str(path), i))
    conn.execute("COMMIT")
    return conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] - before


def unread(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    ensure_schema(conn)
    sql = ("SELECT s.record_id, s.year, s.chronicler, s.text FROM sources s "
           "LEFT JOIN notes n ON n.record_id = s.record_id WHERE n.record_id IS NULL ORDER BY s.year, s.record_id")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [{"id": r[0], "year": r[1], "chronicler": r[2], "text": r[3]} for r in conn.execute(sql)]


def unread_count(conn: sqlite3.Connection) -> int:
    ensure_schema(conn)
    return conn.execute("SELECT COUNT(*) FROM sources s LEFT JOIN notes n ON n.record_id = s.record_id "
                        "WHERE n.record_id IS NULL").fetchone()[0]


def save_note(conn: sqlite3.Connection, note, record_text: str, loops: int) -> None:
    """Write one note and its links. Call inside a transaction (the engine's run_one provides one)."""
    from learner.focused import check_answer
    from learner.sourcecheck import link_stated

    conn.execute(
        "INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (note.record_id, note.year, note.event, json.dumps(note.places), json.dumps(note.people),
         json.dumps(note.details), note.summary, note.question, note.answer,
         int(check_answer(note, record_text)), int(note.complete), note.raw, loops, time.time()))
    conn.execute("DELETE FROM links WHERE record_id = ?", (note.record_id,))
    for kind, pairs in (("same_as", note.same_as), ("located_in", note.located_in), ("on_river", note.on_river)):
        for a, b in pairs:
            conn.execute("INSERT INTO links (kind, a, b, record_id, stated) VALUES (?, ?, ?, ?, ?)",
                         (kind, a, b, note.record_id, int(link_stated(record_text, kind, a, b))))


def notes(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT record_id, year, event, places, people, details, summary, question, answer, "
                        "answer_verbatim, complete FROM notes ORDER BY year, record_id").fetchall()
    keys = ("record_id", "year", "event", "places", "people", "details", "summary", "question", "answer",
            "answer_verbatim", "complete")
    out = []
    for row in rows:
        d = dict(zip(keys, row))
        for k in ("places", "people", "details"):
            d[k] = json.loads(d[k] or "[]")
        out.append(d)
    return out


def links(conn: sqlite3.Connection, stated_only: bool = False) -> list[dict]:
    """Links between names and places. stated_only: just the ones a record says word for word
    (inferred links are kept as hypotheses for dream mode to test, never used as facts)."""
    where = " WHERE stated = 1" if stated_only else ""
    return [dict(zip(("kind", "a", "b", "record_id", "stated"), r))
            for r in conn.execute(f"SELECT kind, a, b, record_id, stated FROM links{where} ORDER BY id")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learner.notebook")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--db", required=True)
    ing.add_argument("--world", required=True)
    show = sub.add_parser("show")
    show.add_argument("--db", required=True)
    show.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db, isolation_level=None)
    if args.cmd == "ingest":
        print(f"ingested {ingest_world(conn, args.world)} new records; {unread_count(conn)} unread")
    else:
        ensure_schema(conn)
        for n in notes(conn)[: args.limit]:
            print(f"{n['record_id']} {n['year']} {n['event']!r} places={n['places']} details={n['details']}")
        print(f"... {len(notes(conn))} notes, {len(links(conn))} links, {unread_count(conn)} unread")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
