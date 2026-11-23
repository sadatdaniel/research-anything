"""Fusions: one new skill made from two or more skills, to look for connections *between* them.

A fusion copies each source skill's material (records) and notes into a fresh brain, with every record id
prefixed by the skill it came from ("val-000/r0014"), so nothing collides and every finding can still be
traced to its source skill. It doesn't copy theories: dreaming on the fusion finds and checks connections
across the combined material from scratch (a source skill's own findings stay in that skill). The fusion
remembers what it was made from (engine_kv 'pack.combined_from'), which is how the library shows it.

  python -m learner.combine --library results/runs/phase3/library val-000 val-001 --name "val-000 + val-001"
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

from learner import recipe as R
from learner.pack import slug


def _kv(conn: sqlite3.Connection) -> dict:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'engine_kv'").fetchone():
        return {}
    return {k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM engine_kv")}


def combine(library: Path, ids: list[str], name: str | None = None) -> dict:
    from engine.store import Store
    from learner import knowledge as kn
    from learner import notebook
    from learner import theories as th

    library = Path(library)
    if len(ids) < 2:
        raise ValueError("a fusion needs at least two skills")
    sources = []
    for sid in ids:
        db = library / f"{sid}.sqlite"
        if not db.exists():
            raise ValueError(f"no skill called {sid!r} in this library")
        sources.append((sid, db))
    name = name or " + ".join(ids)
    dest = library / f"{slug(name)}.sqlite"
    n = 2
    while dest.exists():
        dest = library / f"{slug(name)}-{n}.sqlite"
        n += 1
    fd, tmp_name = tempfile.mkstemp(suffix=".sqlite", dir=library)
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.unlink()
    made_from = []
    try:
        store = Store(tmp)
        conn = store.conn
        notebook.ensure_schema(conn)
        kn.ensure_schema(conn)
        th.ensure_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        for sid, db in sources:
            src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            kv = _kv(src)
            count = 0
            for rid, year, chron, text, origin, line in src.execute(
                    "SELECT record_id, year, chronicler, text, origin, line FROM sources"):
                conn.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
                             (f"{sid}/{rid}", year, chron, text, f"{sid}:{origin}", line))
                count += 1
            cols = [r[1] for r in src.execute("PRAGMA table_info(notes)")]
            for row in src.execute(f"SELECT {', '.join(cols)} FROM notes"):
                d = dict(zip(cols, row))
                d["record_id"] = f"{sid}/{d['record_id']}"
                conn.execute(f"INSERT INTO notes ({', '.join(d)}) VALUES ({', '.join('?' * len(d))})", tuple(d.values()))
            for kind, a, b, rid, stated in src.execute("SELECT kind, a, b, record_id, stated FROM links"):
                conn.execute("INSERT INTO links (kind, a, b, record_id, stated) VALUES (?, ?, ?, ?, ?)",
                             (kind, a, b, f"{sid}/{rid}", stated))
            made_from.append({"id": sid, "name": kv.get("pack.name") or sid, "version": kv.get("pack.version"),
                              "records": count})
            src.close()
        for k, v in (("pack.name", name), ("pack.combined_from", made_from), ("pack.created_at", time.time()),
                     ("recipe", R.active())):
            conn.execute("INSERT OR REPLACE INTO engine_kv (key, value) VALUES (?, ?)", (k, json.dumps(v)))
        conn.execute("COMMIT")
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
        store.close()
        if ok != "ok":
            raise ValueError(f"the new fusion didn't check out ({ok})")
        for suffix in ("-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
        os.replace(tmp, dest)
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
    return {"skill": dest.stem, "name": name, "combined_from": made_from, "records": sum(m["records"] for m in made_from)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learner.combine")
    ap.add_argument("--library", required=True)
    ap.add_argument("skills", nargs="+")
    ap.add_argument("--name")
    args = ap.parse_args(argv)
    print(json.dumps(combine(Path(args.library), args.skills, args.name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
