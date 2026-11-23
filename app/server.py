"""The app's local server: one page plus a small read-only JSON API over the learner's brain.

  uv run python -m app.server --db results/runs/phase3/dream/val-000-dev1.sqlite --world worlds/val/val-000
  uv run python -m app.server --library results/runs/phase3/dream --worlds worlds     # every brain is a skill

The page talks to the learner only through this API (never to files), so the same page can later run
in the desktop window, the Windows build, and an Android WebView. The brain database is opened
read-only, so the viewer never gets in the engine's way. Standard library only.

  GET /                     the page
  GET /api/summary          counts, span, the engine's current state and why
  GET /api/records          every record with its year, text and the kind the learner gave it
  GET /api/theories         every theory: stage, tests, evidence, prediction, history
  GET /api/history          the engine's timeline (day / dream / rest ...) and every theory stage change
  GET /api/answer-key       fake worlds only (--world): the planted patterns and decoys, for comparison
  GET /api/export/obsidian  the findings as a zip of Markdown notes that Obsidian opens (graph view, backlinks)
  GET /api/skills           the library: every skill (brain) the server knows, with its counts
  Every /api/... call takes ?skill=<id> to pick a skill; without it, the first one.
  GET /api/settings         the learner's settings (config.toml beside the brain), with what each one does today
  POST /api/settings        save settings; engine limits apply when the learner next starts
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learner import theories as th  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"

# Settings the app shows. "engine" keys are PolicyConfig fields (engine/policy.py) and are used by the
# engine on its next start; "app" keys are saved now and wired to the operating system in Phase 7.
APP_SETTINGS = {
    "autostart": False,              # start learning when you log in
    "dream_when_away": True,         # dream by day too, once you've been away a while (the engine's "nap")
    "keep_awake_while_learning": False,  # stop the computer sleeping during dreams and training
    "check_for_updates": False,      # off by default: a sandbox makes no network calls unless you allow it
}
WIRED = {"autostart": "Phase 7", "keep_awake_while_learning": "Phase 7", "check_for_updates": "Phase 7",
         "dream_when_away": "next start"}


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def write_config(path: Path, engine: dict, app: dict) -> None:
    lines = ["# Sandbox Learner settings (written by the app). [engine] is read by the cycle engine.", "", "[engine]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in engine.items()]
    lines += ["", "[app]"] + [f"{k} = {_toml_value(v)}" for k, v in app.items()]
    path.write_text("\n".join(lines) + "\n")


class Brain:
    def __init__(self, db: Path, world: Path | None, config: Path | None = None) -> None:
        self.db = db
        self.world = world
        self.config = config or db.parent / "config.toml"

    def settings(self) -> dict:
        import tomllib
        from dataclasses import asdict

        from engine.policy import PolicyConfig

        data = tomllib.loads(self.config.read_text()) if self.config.exists() else {}
        engine = asdict(PolicyConfig.from_dict(data.get("engine", {})))
        app = {**APP_SETTINGS, **{k: v for k, v in data.get("app", {}).items() if k in APP_SETTINGS}}
        return {"engine": engine, "app": app, "wired": WIRED, "file": str(self.config)}

    def save_settings(self, body: dict) -> dict:
        from dataclasses import asdict

        from engine.policy import PolicyConfig

        current = self.settings()
        engine = {**current["engine"], **{k: v for k, v in (body.get("engine") or {}).items()
                                          if k in current["engine"]}}
        defaults = asdict(PolicyConfig())
        for k, v in engine.items():  # keep each value's type
            if isinstance(defaults[k], bool):
                engine[k] = bool(v)
            elif isinstance(defaults[k], float):
                engine[k] = float(v)
        PolicyConfig.from_dict(engine)  # validates the names
        app = {**current["app"], **{k: bool(v) for k, v in (body.get("app") or {}).items() if k in APP_SETTINGS}}
        never = 1e9  # "dream when away" off: the daytime nap never comes
        if not app["dream_when_away"]:
            engine["nap_after_idle_seconds"] = never
        elif engine["nap_after_idle_seconds"] >= never:
            engine["nap_after_idle_seconds"] = defaults["nap_after_idle_seconds"]
        changed = {k: v for k, v in engine.items() if v != defaults[k]}
        write_config(self.config, changed, app)
        return self.settings()

    def conn(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, isolation_level=None)

    def _has(self, conn, table: str) -> bool:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is not None

    def summary(self) -> dict:
        with self.conn() as c:
            n_src = c.execute("SELECT COUNT(*) FROM sources").fetchone()[0] if self._has(c, "sources") else 0
            n_notes = c.execute("SELECT COUNT(*) FROM notes").fetchone()[0] if self._has(c, "notes") else 0
            span = c.execute("SELECT MIN(year), MAX(year) FROM sources").fetchone() if n_src else (None, None)
            stages: dict[str, int] = {}
            if self._has(c, "theories"):
                for s, n in c.execute("SELECT stage, COUNT(*) FROM theories GROUP BY stage"):
                    stages[s] = n
            state, reason, active = "rest", "", {}
            if self._has(c, "engine_kv"):
                kv = {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM engine_kv")}
                active = {k.split(".", 1)[1]: v for k, v in kv.items() if k.startswith("active_seconds.")}
                row = c.execute("SELECT state, reason FROM timeline ORDER BY id DESC LIMIT 1").fetchone() \
                    if self._has(c, "timeline") else None
                if row:
                    state, reason = row
        name = self.world.name if self.world else self.db.stem
        return {"name": name, "records": n_src, "notes": n_notes, "span": list(span), "stages": stages,
                "state": state, "reason": reason, "active_seconds": active, "has_answer_key": bool(self.world)}

    def records(self) -> list[dict]:
        with self.conn() as c:
            if not self._has(c, "notes"):
                return []
            k = Knowledge(c)
        return [{"id": e.record_id, "year": e.year, "text": e.text, "kind": e.kind, "label": e.label,
                 "names": sorted(e.names)} for e in k.entries]

    def theories(self) -> list[dict]:
        with self.conn() as c:
            if not self._has(c, "theories"):
                return []
            ts = th.all_theories(c)
            hist: dict[int, list] = {}
            for tid, stage, reason, ds in c.execute(
                    "SELECT theory_id, stage, reason, dream_seconds FROM theory_history ORDER BY rowid"):
                hist.setdefault(tid, []).append({"stage": stage, "reason": reason, "dream_seconds": ds})
        for t in ts:
            t["history"] = hist.get(t["id"], [])
        return ts

    def history(self) -> dict:
        with self.conn() as c:
            timeline = [dict(zip(("state", "reason", "started_at", "ended_at"), r)) for r in
                        c.execute("SELECT state, reason, started_at, ended_at FROM timeline ORDER BY id")] \
                if self._has(c, "timeline") else []
            changes = [dict(zip(("theory_id", "stage", "reason", "dream_seconds", "summary", "type"), r)) for r in
                       c.execute("SELECT h.theory_id, h.stage, h.reason, h.dream_seconds, t.summary, t.type "
                                 "FROM theory_history h JOIN theories t ON t.id = h.theory_id ORDER BY h.rowid")] \
                if self._has(c, "theory_history") else []
        return {"timeline": timeline, "changes": changes}

    def obsidian_zip(self) -> bytes:
        from app.export_obsidian import export

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "findings"
            export(self.db, out)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(out.rglob("*.md")):
                    z.write(f, f.relative_to(out.parent))
            return buf.getvalue()

    def answer_key(self) -> dict:
        if not self.world:
            return {}
        truth = json.loads((self.world / "truth.json").read_text())
        return {"heldout_from": truth["heldout_from"],
                "patterns": [{k: p.get(k) for k in ("id", "type", "description", "record_ids", "heldout_years", "params")}
                             for p in truth["patterns"]],
                "decoys": [{k: d.get(k) for k in ("id", "type", "why", "record_ids")} for d in truth["decoys"]]}


def discover(library: Path, worlds: Path | None, config: Path | None) -> dict[str, "Brain"]:
    """Every brain in a folder is a skill. A fake world's answer key is found by name (val-001-dev4 ->
    worlds/*/val-001), so findings can be compared with what was planted."""
    world_dirs = [p for p in worlds.glob("*/*") if (p / "truth.json").exists()] if worlds else []
    out = {}
    for db in sorted(library.glob("*.sqlite")):
        match = max((w for w in world_dirs if db.stem.startswith(w.name)), key=lambda w: len(w.name), default=None)
        out[db.stem] = Brain(db.resolve(), match.resolve() if match else None, config or library / "config.toml")
    return out


def skill_card(sid: str, brain: "Brain") -> dict:
    s = brain.summary()
    return {"id": sid, "name": s["name"], "records": s["records"], "span": s["span"], "stages": s["stages"],
            "dream_seconds": s["active_seconds"].get("dream", 0.0), "state": s["state"], "reason": s["reason"],
            "has_answer_key": s["has_answer_key"], "combined_from": []}


def make_handler(brains):
    if isinstance(brains, Brain):
        brains = {brains.db.stem: brains}
    first = next(iter(brains))
    names = {"/api/summary": "summary", "/api/records": "records", "/api/theories": "theories",
             "/api/history": "history", "/api/answer-key": "answer_key", "/api/settings": "settings"}

    class Handler(BaseHTTPRequestHandler):
        def _brain(self) -> "Brain":
            from urllib.parse import parse_qs, urlparse

            sid = (parse_qs(urlparse(self.path).query).get("skill") or [first])[0]
            return brains.get(sid, brains[first])

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            static = (WEB / path.lstrip("/")).resolve()
            types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
            if static.suffix in types and static.is_file() and WEB.resolve() in static.parents:
                return self._send(200, static.read_bytes(), types[static.suffix])
            brain = self._brain()
            if path == "/api/skills":
                return self._send(200, json.dumps([skill_card(k, b) for k, b in brains.items()]).encode(),
                                  "application/json")
            if path == "/api/export/obsidian":
                body = brain.obsidian_zip()
                name = (brain.world.name if brain.world else brain.db.stem) + "-findings.zip"
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path in names:
                try:
                    data = getattr(brain, names[path])()
                except sqlite3.Error as exc:
                    return self._send(503, json.dumps({"error": str(exc)}).encode(), "application/json")
                return self._send(200, json.dumps(data).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/settings":
                return self._send(404, b"not found", "text/plain")
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self._send(403, b"settings can only be changed from this computer", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                data = self._brain().save_settings(body)
            except (ValueError, TypeError) as exc:
                return self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            self._send(200, json.dumps(data).encode(), "application/json")

        def log_message(self, *args) -> None:
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.server")
    ap.add_argument("--db", help="one learner's brain.sqlite")
    ap.add_argument("--world", help="a fake world's folder, to show its answer key next to the findings")
    ap.add_argument("--library", help="a folder of brains: each *.sqlite is a skill")
    ap.add_argument("--worlds", help="with --library: the worlds folder, to find each skill's answer key")
    ap.add_argument("--config", help="settings file (default: config.toml beside the brain)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    config = Path(args.config).resolve() if args.config else None
    if args.library:
        brains = discover(Path(args.library).resolve(), Path(args.worlds).resolve() if args.worlds else None, config)
        if not brains:
            ap.error(f"no *.sqlite brains in {args.library}")
    elif args.db:
        brains = Brain(Path(args.db).resolve(), Path(args.world).resolve() if args.world else None, config)
    else:
        ap.error("give --db or --library")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(brains))
    print(f"Sandbox Learner viewer on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
