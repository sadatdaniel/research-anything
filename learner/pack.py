"""Mindpacks: a skill as one shareable, versioned, verifiable file (.mindpack).

A mindpack is a zip of data only (JSON, JSON Lines, Markdown; later safetensors), never code:

  mindpack.json      the manifest: format, skill id, name, version, parent hash, base model, recipe,
                     stats, whether the material is included, and every file with its sha256 and size
  theories.jsonl     every theory, with stage, tests, evidence, prediction
  history.jsonl      every stage change, with dream time
  notes.jsonl        day-reading notes, with their links
  judgments.jsonl    the model's re-reading answers and label screens
  sources.jsonl      the material itself (optional; on by default, for transparency)
  timeline.jsonl     the engine's states (reading, dreaming, resting) and why
  findings/          the same findings as Obsidian-ready Markdown

Failsafe:
  * reading checks every member name (no absolute paths, no '..'), caps sizes and compression ratio
    (zip bombs), requires the format version, and verifies every file against the manifest's sha256;
  * import builds the new brain in a temporary file, checks it, keeps the previous version of the same
    skill under .versions/, and only then moves the new one into place (one atomic rename);
  * a pack made with a different base model imports read-only (explore and ask, but no more training).

  python -m learner.pack export  --db results/runs/phase3/library/val-001.sqlite -o val-001.mindpack
  python -m learner.pack preview val-001.mindpack --library results/runs/phase3/library
  python -m learner.pack import  val-001.mindpack --library results/runs/phase3/library [--as-new]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from learner import recipe as R

FORMAT = "mindpack/1"
EXT = ".mindpack"
MAX_MEMBER = 1 << 30          # 1 GiB per file
MAX_TOTAL = 2 << 30           # 2 GiB per pack
MAX_RATIO = 200               # uncompressed / compressed, per file (zip bombs)
MODEL_DIR = Path(__file__).resolve().parents[1] / "learner-data" / "models" / "Ouro-1.4B"


class PackError(Exception):
    """A pack that can't be trusted or used. The message says why, in plain words."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def base_model() -> dict:
    """What the skill was learned on. The config's hash identifies the model without hashing gigabytes."""
    cfg = MODEL_DIR / "config.json"
    return {"name": "ByteDance/Ouro-1.4B", "config_sha256": sha256(cfg.read_bytes()) if cfg.exists() else None}


def _jsonl(rows) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


def _rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
        return []
    cur = conn.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur]


def _kv(conn: sqlite3.Connection) -> dict:
    return {r["key"]: json.loads(r["value"]) for r in _rows(conn, "engine_kv")}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "skill"


# ---- export -----------------------------------------------------------------------------------------

def export_pack(db: Path, out: Path, name: str | None = None, include_material: bool = True,
                include_findings: bool = True, upstream: dict | None = None) -> dict:
    db, out = Path(db), Path(out)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    kv = _kv(conn)
    links_by: dict[str, list] = {}
    for link in _rows(conn, "links"):
        links_by.setdefault(link["record_id"], []).append(
            {k: link[k] for k in ("kind", "a", "b", "stated")})
    notes = [{**n, "links": links_by.get(n["record_id"], [])} for n in _rows(conn, "notes")]
    files: dict[str, bytes] = {
        "theories.jsonl": _jsonl(_rows(conn, "theories")),
        "history.jsonl": _jsonl(_rows(conn, "theory_history")),
        "notes.jsonl": _jsonl(notes),
        "judgments.jsonl": _jsonl([{"table": "reread_judgments", **r} for r in _rows(conn, "reread_judgments")]
                                  + [{"table": "kind_judgments", **r} for r in _rows(conn, "kind_judgments")]
                                  + [{"table": "kinds", **r} for r in _rows(conn, "kinds")]),
        "timeline.jsonl": _jsonl(_rows(conn, "timeline")),
    }
    sources = _rows(conn, "sources")
    if include_material:
        files["sources.jsonl"] = _jsonl(sources)
    else:  # keep each record's hash, so a recipient with the files can check the quotes later
        files["sources.jsonl"] = _jsonl([{"record_id": s["record_id"], "year": s["year"],
                                          "sha256": sha256(s["text"].encode())} for s in sources])
    theories = _rows(conn, "theories")
    conn.close()
    if include_findings:
        from app.export_obsidian import export as obsidian

        with tempfile.TemporaryDirectory() as tmp:
            obsidian(db, Path(tmp) / "findings")
            for f in sorted((Path(tmp) / "findings").rglob("*.md")):
                files[f"findings/{f.relative_to(Path(tmp) / 'findings').as_posix()}"] = f.read_bytes()

    stages: dict[str, int] = {}
    for t in theories:
        stages[t["stage"]] = stages.get(t["stage"], 0) + 1
    contents = [{"path": p, "sha256": sha256(b), "bytes": len(b)} for p, b in sorted(files.items())]
    content_hash = sha256("".join(f"{c['path']}:{c['sha256']}\n" for c in contents).encode())
    origins = {}
    for s in sources:
        origins.setdefault(Path(s["origin"]).name, 0)
        origins[Path(s["origin"]).name] += 1
    manifest = {
        "format": FORMAT,
        "id": kv.get("pack.id") or str(uuid.uuid4()),
        "name": name or kv.get("pack.name") or db.stem,
        "version": int(kv.get("pack.version", 0)) + (0 if kv.get("pack.hash") == content_hash else 1),
        "parent": kv.get("pack.hash") if kv.get("pack.hash") != content_hash else kv.get("pack.parent"),
        "content_hash": content_hash,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": base_model(),
        "recipe": kv.get("recipe") or R.active(),
        "sources": [{"name": k, "records": v} for k, v in sorted(origins.items())],
        "stats": {"records": len(sources), "notes": len(notes), "theories": stages,
                  "dream_seconds": kv.get("active_seconds.dream", 0.0), "day_seconds": kv.get("active_seconds.day", 0.0)},
        "includes_material": include_material,
        "upstream": upstream or kv.get("pack.upstream"),
        "license": kv.get("pack.license", "unspecified"),
        "contents": contents,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mindpack.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for p, b in files.items():
            z.writestr(p, b)
    os.replace(tmp, out)  # never a half-written pack under the real name
    w = sqlite3.connect(db, timeout=30)
    with w:
        w.execute("CREATE TABLE IF NOT EXISTS engine_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        for k, v in (("pack.id", manifest["id"]), ("pack.name", manifest["name"]), ("pack.version", manifest["version"]),
                     ("pack.hash", content_hash), ("pack.parent", manifest["parent"])):
            w.execute("INSERT OR REPLACE INTO engine_kv (key, value) VALUES (?, ?)", (k, json.dumps(v)))
    w.close()
    return manifest


# ---- reading, safely ------------------------------------------------------------------------------------

def _safe_name(name: str) -> bool:
    """A member may only be a plain relative path: no root, no drive letter, no backslash, no '..'."""
    if not name or name.startswith("/") or "\\" in name or re.match(r"^[A-Za-z]:", name):
        return False
    parts = name.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def read_pack(data: bytes) -> tuple[dict, dict[str, bytes]]:
    """Open and fully verify a pack held in memory. Raises PackError with a plain reason."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise PackError("not a mindpack: the file isn't a zip archive")
    total = 0
    for info in z.infolist():
        if not _safe_name(info.filename):
            raise PackError(f"unsafe file name inside the pack: {info.filename!r}")
        if info.file_size > MAX_MEMBER:
            raise PackError(f"{info.filename} is too large ({info.file_size} bytes)")
        if info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
            raise PackError(f"{info.filename} expands suspiciously (possible zip bomb)")
        total += info.file_size
    if total > MAX_TOTAL:
        raise PackError("the pack is too large")
    if "mindpack.json" not in z.namelist():
        raise PackError("not a mindpack: no mindpack.json manifest")
    manifest = json.loads(z.read("mindpack.json"))
    if manifest.get("format") != FORMAT:
        raise PackError(f"unsupported format {manifest.get('format')!r} (this app reads {FORMAT})")
    listed = {c["path"]: c for c in manifest.get("contents", [])}
    present = set(z.namelist()) - {"mindpack.json"}
    if present - set(listed):
        raise PackError(f"files not listed in the manifest: {sorted(present - set(listed))[:5]}")
    if set(listed) - present:
        raise PackError(f"files missing from the pack: {sorted(set(listed) - present)[:5]}")
    files = {}
    for path, c in listed.items():
        b = z.read(path)
        if sha256(b) != c["sha256"] or len(b) != c["bytes"]:
            raise PackError(f"{path} doesn't match its checksum: the pack is damaged or was changed")
        files[path] = b
    want = sha256("".join(f"{p}:{listed[p]['sha256']}\n" for p in sorted(listed)).encode())
    if manifest.get("content_hash") != want:
        raise PackError("the manifest's content hash doesn't match its files")
    try:
        R.validate(manifest.get("recipe") or {})
    except ValueError as exc:
        raise PackError(f"its recipe uses something this app doesn't have: {exc}")
    return manifest, files


def _lines(b: bytes) -> list[dict]:
    return [json.loads(line) for line in b.decode().splitlines() if line.strip()]


# ---- preview and diff -----------------------------------------------------------------------------------

def find_skill(library: Path, pack_id: str) -> Path | None:
    for db in sorted(Path(library).glob("*.sqlite")):
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            if _kv(c).get("pack.id") == pack_id:
                return db
        except sqlite3.Error:
            pass
        finally:
            c.close()
    return None


def diff_theories(old: list[dict], new: list[dict]) -> dict:
    o = {t["key"]: t for t in old}
    n = {t["key"]: t for t in new}
    added = [t for k, t in n.items() if k not in o]
    gone = [t for k, t in o.items() if k not in n]
    changed = [(o[k], t) for k, t in n.items() if k in o and (o[k]["stage"] != t["stage"] or o[k]["fingerprint"] != t["fingerprint"])]

    def brief(t):
        return {"summary": t["summary"], "stage": t["stage"], "type": t["type"]}
    newly_confirmed = [t for t in added if t["stage"] == "confirmed"] + [t for a, t in changed if t["stage"] == "confirmed" and a["stage"] != "confirmed"]
    return {"added": [brief(t) for t in added], "removed": [brief(t) for t in gone],
            "changed": [{"summary": t["summary"], "from": a["stage"], "to": t["stage"]} for a, t in changed],
            "new_confirmed": len(newly_confirmed)}


def preview(data: bytes, library: Path | None = None) -> dict:
    manifest, files = read_pack(data)
    theories = _lines(files["theories.jsonl"])
    existing = find_skill(library, manifest["id"]) if library else None
    diff, current_version = None, None
    if existing:
        c = sqlite3.connect(f"file:{existing}?mode=ro", uri=True)
        old = _rows(c, "theories")
        current_version = _kv(c).get("pack.version")
        c.close()
        diff = diff_theories(old, theories)
    here = base_model()
    same_model = manifest["base_model"].get("config_sha256") in (None, here["config_sha256"]) or here["config_sha256"] is None
    return {
        "manifest": {k: manifest[k] for k in ("id", "name", "version", "parent", "created", "base_model", "sources",
                                              "stats", "includes_material", "license", "upstream", "content_hash")},
        "recipe": {"name": manifest["recipe"].get("name"), "version": manifest["recipe"].get("version"),
                   "fingerprint": R.fingerprint(R.validate(manifest["recipe"]))},
        "files": len(files), "bytes": sum(len(b) for b in files.values()),
        "same_base_model": same_model,
        "updates": existing.stem if existing else None, "current_version": current_version, "diff": diff,
        "confirmed": [t["summary"] for t in theories if t["stage"] == "confirmed"][:20],
    }


# ---- import ------------------------------------------------------------------------------------------

def _build_brain(path: Path, manifest: dict, files: dict[str, bytes]) -> None:
    from engine.store import Store
    from learner import knowledge as kn
    from learner import notebook
    from learner import theories as th

    store = Store(path)  # engine tables (engine_kv, timeline, work_queue)
    conn = store.conn
    notebook.ensure_schema(conn)
    kn.ensure_schema(conn)
    th.ensure_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    for s in _lines(files["sources.jsonl"]):
        if "text" in s:
            conn.execute("INSERT INTO sources (record_id, year, chronicler, text, origin, line) VALUES (?, ?, ?, ?, ?, ?)",
                         (s["record_id"], s["year"], s.get("chronicler"), s["text"], s.get("origin", ""), s.get("line", 0)))
    note_cols = ("record_id", "year", "event", "places", "people", "details", "summary", "question", "answer",
                 "answer_verbatim", "complete", "raw", "loops", "created_at")
    for n in _lines(files["notes.jsonl"]):
        conn.execute(f"INSERT INTO notes ({', '.join(note_cols)}) VALUES ({', '.join('?' * len(note_cols))})",
                     tuple(n.get(c) for c in note_cols))
        for link in n.get("links", []):
            conn.execute("INSERT INTO links (kind, a, b, record_id, stated) VALUES (?, ?, ?, ?, ?)",
                         (link["kind"], link["a"], link["b"], n["record_id"], link.get("stated", 0)))
    for j in _lines(files["judgments.jsonl"]):
        table = j.pop("table")
        if table not in ("reread_judgments", "kind_judgments", "kinds"):
            raise PackError(f"unknown judgment table {table!r}")
        cols = list(j)
        conn.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     tuple(j[c] for c in cols))
    for t in _lines(files["theories.jsonl"]):
        cols = list(t)
        conn.execute(f"INSERT INTO theories ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(t[c] for c in cols))
    for h in _lines(files["history.jsonl"]):
        cols = list(h)
        conn.execute(f"INSERT INTO theory_history ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(h[c] for c in cols))
    for row in _lines(files.get("timeline.jsonl", b"")):
        conn.execute("INSERT INTO timeline (state, reason, budget, started_at, ended_at) VALUES (?, ?, ?, ?, ?)",
                     (row["state"], row["reason"], row.get("budget"), row["started_at"], row.get("ended_at")))
    kv = {"pack.id": manifest["id"], "pack.name": manifest["name"], "pack.version": manifest["version"],
          "pack.hash": manifest["content_hash"], "pack.parent": manifest.get("parent"), "pack.imported_at": time.time(),
          "pack.upstream": manifest.get("upstream"), "pack.license": manifest.get("license"),
          "recipe": R.validate(manifest["recipe"]), "active_seconds.dream": manifest["stats"].get("dream_seconds", 0.0),
          "active_seconds.day": manifest["stats"].get("day_seconds", 0.0),
          "pack.read_only": not preview_model_ok(manifest)}
    for k, v in kv.items():
        conn.execute("INSERT OR REPLACE INTO engine_kv (key, value) VALUES (?, ?)", (k, json.dumps(v)))
    conn.execute("COMMIT")
    ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
    got = conn.execute("SELECT COUNT(*) FROM theories").fetchone()[0]
    want = sum(manifest["stats"]["theories"].values())
    store.close()
    if ok != "ok" or got != want:
        raise PackError(f"the rebuilt skill doesn't check out ({ok}; {got} of {want} theories)")


def preview_model_ok(manifest: dict) -> bool:
    here = base_model()["config_sha256"]
    return here is None or manifest["base_model"].get("config_sha256") in (None, here)


def import_pack(data: bytes, library: Path, as_new: bool = False) -> dict:
    """Import into a library folder. Returns where it went and what it replaced (kept as a version)."""
    library = Path(library)
    manifest, files = read_pack(data)
    existing = None if as_new else find_skill(library, manifest["id"])
    if as_new:
        manifest = {**manifest, "id": str(uuid.uuid4())}
    dest = existing or library / f"{slug(manifest['name'])}.sqlite"
    n = 2
    while not existing and dest.exists():
        dest = library / f"{slug(manifest['name'])}-{n}.sqlite"
        n += 1
    fd, tmp_name = tempfile.mkstemp(suffix=".sqlite", dir=library)
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.unlink()
    try:
        _build_brain(tmp, manifest, files)
        kept = None
        if existing:
            c = sqlite3.connect(f"file:{existing}?mode=ro", uri=True)
            old_version = _kv(c).get("pack.version", 0)
            c.close()
            vdir = library / ".versions" / existing.stem
            vdir.mkdir(parents=True, exist_ok=True)
            kept = vdir / f"v{old_version}-{int(time.time())}.sqlite"
            shutil.copy2(existing, kept)  # the previous version stays, for rollback
        for suffix in ("-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
        os.replace(tmp, dest)  # one atomic step: the skill is either the old one or the new one
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
    return {"skill": dest.stem, "path": str(dest), "version": manifest["version"],
            "replaced_version_kept_at": str(kept) if existing else None, "read_only": not preview_model_ok(manifest)}


def versions(library: Path, skill: str) -> list[dict]:
    vdir = Path(library) / ".versions" / skill
    return [{"file": p.name, "bytes": p.stat().st_size} for p in sorted(vdir.glob("*.sqlite"))] if vdir.exists() else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learner.pack")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--db", required=True)
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--name")
    e.add_argument("--without-material", action="store_true")
    pv = sub.add_parser("preview")
    pv.add_argument("pack")
    pv.add_argument("--library")
    im = sub.add_parser("import")
    im.add_argument("pack")
    im.add_argument("--library", required=True)
    im.add_argument("--as-new", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "export":
            m = export_pack(Path(args.db), Path(args.out), args.name, not args.without_material)
            print(f"wrote {args.out}: {m['name']} v{m['version']}, {len(m['contents'])} files, hash {m['content_hash'][:12]}")
        elif args.cmd == "preview":
            print(json.dumps(preview(Path(args.pack).read_bytes(), Path(args.library) if args.library else None), indent=2))
        else:
            print(json.dumps(import_pack(Path(args.pack).read_bytes(), Path(args.library), args.as_new), indent=2))
    except PackError as exc:
        print(f"refused: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
