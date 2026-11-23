"""The viewer's API and the Obsidian export, on a brain that dreamt with oracle notes and a fake model."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from app.export_obsidian import export
from app.server import Brain, make_handler
from chronicles.generator import WorldConfig, write_world
from learner import theories as th
from learner.knowledge import Knowledge
from learner.tests.test_dream import FakeLM, fresh_notebook, run_dream


@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("brain")
    world = tmp / "world"
    truth = write_world(world, "app-test:3", WorldConfig())
    store = fresh_notebook(tmp, world, truth)
    run_dream(store, FakeLM(truth, Knowledge(store.conn)))
    store.close()
    return tmp / "brain.sqlite", world


def test_api_serves_the_page_and_json(brain):
    db, world = brain
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Brain(db, world)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Learn Anything" in urllib.request.urlopen(base + "/").read()
        summary = json.loads(urllib.request.urlopen(base + "/api/summary").read())
        assert summary["records"] > 100 and summary["stages"].get("confirmed", 0) >= 2
        theories = json.loads(urllib.request.urlopen(base + "/api/theories").read())
        confirmed = [t for t in theories if t["stage"] == "confirmed"]
        assert confirmed and all(t["evidence"] and t["history"] for t in confirmed)
        key = json.loads(urllib.request.urlopen(base + "/api/answer-key").read())
        assert key["patterns"]
        records = json.loads(urllib.request.urlopen(base + "/api/records").read())
        assert {"id", "year", "text", "kind"} <= set(records[0])
        qs = json.loads(urllib.request.urlopen(base + "/api/questions").read())
        assert qs and {"text", "answer", "certainty", "status"} <= set(qs[0])
        assert any(q["status"] == "answered" for q in qs)
        g = confirmed[0]["groups"]
        a, b = (g[0][0], g[-1][0]) if len(g) > 1 else (g[0][0], g[0][0])
        r = json.loads(urllib.request.urlopen(base + f"/api/connection?a={a}&b={b}".replace(" ", "%20")).read())
        assert r["verdict"] in ("yes", "indirect", "related")
    finally:
        server.shutdown()


def test_confidence_ranks_confirmed_above_rejected(brain):
    import sqlite3
    db, _ = brain
    conn = sqlite3.connect(db)
    ts = th.all_theories(conn)
    conf = [t["confidence"] for t in ts if t["stage"] == "confirmed"]
    rej = [t["confidence"] for t in ts if t["stage"] == "rejected"]
    assert conf and rej and min(conf) > max(rej)
    assert all(0 <= t["confidence"] <= 100 for t in ts)


def test_obsidian_export_links_everything(brain, tmp_path):
    db, _ = brain
    out = tmp_path / "findings"
    stats = export(db, out)
    assert stats["connections"] >= 2
    readme = (out / "README.md").read_text()
    assert "[[Connections/" in readme
    note = next((out / "Connections").glob("*.md")).read_text()
    assert note.startswith("---") and "confidence:" in note and "[[Records/" in note
    rec = next((out / "Records").glob("*.md")).read_text()
    assert "[[Kinds/" in rec
    # every wikilink points at a note that exists
    import re
    for md in out.rglob("*.md"):
        for target in re.findall(r"\[\[([^\]|]+)", md.read_text()):
            assert (out / f"{target}.md").exists(), (md.name, target)


def test_library_lists_every_brain_and_picks_one_per_request(brain, tmp_path):
    import shutil

    from app.server import discover

    db, world = brain
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy(db, lib / "alpha.sqlite")
    shutil.copy(db, lib / "beta.sqlite")
    brains = discover(lib, None, None)
    assert set(brains) == {"alpha", "beta"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(brains))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        skills = json.loads(urllib.request.urlopen(base + "/api/skills").read())
        assert [s["id"] for s in skills] == ["alpha", "beta"] and skills[0]["stages"].get("confirmed")
        beta = json.loads(urllib.request.urlopen(base + "/api/summary?skill=beta").read())
        assert beta["name"] == "beta"
        assert b"text/css" in urllib.request.urlopen(base + "/themes.css").headers["Content-Type"].encode()
        req = urllib.request.Request(base + "/api/settings?skill=beta", method="POST",
                                     data=json.dumps({"engine": {"night_start": "22:00"}}).encode())
        saved = json.loads(urllib.request.urlopen(req).read())
        assert saved["engine"]["night_start"] == "22:00" and (lib / "config.toml").exists()
    finally:
        server.shutdown()


def test_pack_export_preview_and_import_over_the_api(brain, tmp_path):
    import shutil

    from app.server import discover

    db, _ = brain
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy(db, lib / "alpha.sqlite")
    brains = discover(lib, None, None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(brains, lib))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        r = urllib.request.urlopen(base + "/api/pack/export?skill=alpha")
        assert r.headers["Content-Disposition"].endswith('-v1.mindpack"')
        data = r.read()
        pv = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/api/pack/preview", data=data, method="POST")).read())
        assert pv["updates"] == "alpha" and pv["diff"]["added"] == [] and pv["can_import"]
        req = urllib.request.Request(base + "/api/pack/import", method="POST",
                                     data=json.dumps({"token": pv["token"], "as_new": True}).encode())
        done = json.loads(urllib.request.urlopen(req).read())
        skills = json.loads(urllib.request.urlopen(base + "/api/skills").read())
        assert done["skill"] in {s["id"] for s in skills} and len(skills) == 2
        bad = urllib.request.Request(base + "/api/pack/preview", data=b"not a pack", method="POST")
        try:
            urllib.request.urlopen(bad)
            raise AssertionError("a bad pack was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 422
    finally:
        server.shutdown()


def test_remove_moves_a_skill_to_the_trash_not_away(brain, tmp_path):
    import shutil

    from app.server import discover

    db, _ = brain
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy(db, lib / "alpha.sqlite")
    shutil.copy(db, lib / "beta.sqlite")
    brains = discover(lib, None, None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(brains, lib))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    post = lambda body: urllib.request.urlopen(urllib.request.Request(  # noqa: E731
        base + "/api/skills/remove", method="POST", data=json.dumps(body).encode()))
    try:
        done = json.loads(post({"id": "beta"}).read())
        assert done["removed"] == "beta" and not (lib / "beta.sqlite").exists()
        trashed = list((lib / ".trash").glob("beta-*.sqlite"))
        assert len(trashed) == 1 and trashed[0].stat().st_size > 0
        assert [s["id"] for s in json.loads(urllib.request.urlopen(base + "/api/skills").read())] == ["alpha"]
        try:
            post({"id": "../alpha"})
            raise AssertionError("an unknown skill was removed")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        assert (lib / "alpha.sqlite").exists()
    finally:
        server.shutdown()


def test_theme_values_are_not_overridden_by_page_defaults():
    """The page's fallback variables must have no specificity (:where), or they replace every theme's own
    values. That bug hid the whole Adwaita look (grey cards, white entries) until the Phase 3 quality check."""
    import re
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text()
    style = page[page.index("<style>"):page.index("</style>")]
    plain_root = [m.group(0) for m in re.finditer(r"(?m)^\s*:root\s*\{[^}]*\}", style)]
    assert not any("--card" in r or "--fill" in r for r in plain_root), plain_root
    assert ":where(:root)" in style
