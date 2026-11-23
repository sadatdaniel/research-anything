"""Mindpacks: export, verify, diff, import as an update (keeping the old version), and refusing bad packs."""

import io
import json
import sqlite3
import zipfile

import pytest

from chronicles.generator import WorldConfig, write_world
from learner import pack
from learner import theories as th
from learner.knowledge import Knowledge
from learner.tests.test_dream import FakeLM, fresh_notebook, run_dream


@pytest.fixture(scope="module")
def skill(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("skill")
    world = tmp / "world"
    truth = write_world(world, "pack-test:5", WorldConfig())
    store = fresh_notebook(tmp, world, truth)
    run_dream(store, FakeLM(truth, Knowledge(store.conn)))
    store.close()
    return tmp / "brain.sqlite"


def copy_db(src, dst):
    a, b = sqlite3.connect(src), sqlite3.connect(dst)
    a.backup(b)
    a.close()
    b.close()
    return dst


def test_export_is_verifiable_and_stable(skill, tmp_path):
    db = copy_db(skill, tmp_path / "a.sqlite")
    m1 = pack.export_pack(db, tmp_path / "a.mindpack")
    m2 = pack.export_pack(db, tmp_path / "a2.mindpack")
    assert m1["version"] == 1 and m2["version"] == 1 and m1["content_hash"] == m2["content_hash"]  # nothing changed
    manifest, files = pack.read_pack((tmp_path / "a.mindpack").read_bytes())
    assert manifest["format"] == pack.FORMAT and "theories.jsonl" in files and any(p.startswith("findings/") for p in files)
    assert manifest["recipe"]["name"] == "default"


def test_update_shows_a_diff_and_keeps_the_old_version(skill, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    db = copy_db(skill, tmp_path / "work.sqlite")
    data1 = (pack.export_pack(db, tmp_path / "v1.mindpack"), (tmp_path / "v1.mindpack").read_bytes())[1]
    first = pack.import_pack(data1, lib)
    assert first["replaced_version_kept_at"] is None
    # the learner keeps learning: one confirmed theory is rejected after more checking
    conn = sqlite3.connect(db)
    tid = next(t["id"] for t in th.all_theories(conn, ("confirmed",)))
    th.set_stage(conn, tid, "rejected", "test: a later check failed")
    conn.commit()
    conn.close()
    m2 = pack.export_pack(db, tmp_path / "v2.mindpack")
    assert m2["version"] == 2 and m2["parent"] == json.loads(zipfile.ZipFile(tmp_path / "v1.mindpack").read("mindpack.json"))["content_hash"]
    data2 = (tmp_path / "v2.mindpack").read_bytes()
    pv = pack.preview(data2, lib)
    assert pv["updates"] == first["skill"] and pv["current_version"] == 1
    assert [c["to"] for c in pv["diff"]["changed"]] == ["rejected"]
    second = pack.import_pack(data2, lib)
    assert second["skill"] == first["skill"] and second["replaced_version_kept_at"]
    assert pack.versions(lib, first["skill"])  # v1 kept for rollback
    conn = sqlite3.connect(lib / f"{first['skill']}.sqlite")
    assert conn.execute("SELECT stage FROM theories WHERE id = ?", (tid,)).fetchone()[0] == "rejected"


def _rezip(data: bytes, change) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in src.namelist():
            name2, body = change(name, src.read(name))
            if name2:
                z.writestr(name2, body)
    return out.getvalue()


def test_bad_packs_are_refused(skill, tmp_path):
    db = copy_db(skill, tmp_path / "b.sqlite")
    pack.export_pack(db, tmp_path / "b.mindpack")
    good = (tmp_path / "b.mindpack").read_bytes()
    tampered = _rezip(good, lambda n, b: (n, b.replace(b"confirmed", b"CONFIRMED", 1) if n == "theories.jsonl" else b))
    with pytest.raises(pack.PackError, match="checksum"):
        pack.read_pack(tampered)
    sneaky = _rezip(good, lambda n, b: ("../../evil.txt", b) if n == "notes.jsonl" else (n, b))
    with pytest.raises(pack.PackError, match="unsafe"):
        pack.read_pack(sneaky)
    with pytest.raises(pack.PackError, match="zip"):
        pack.read_pack(b"this is not a zip")
    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mindpack.json", json.dumps({"format": pack.FORMAT}))
        z.writestr("big.txt", b"\0" * (50 << 20))
    with pytest.raises(pack.PackError, match="bomb"):
        pack.read_pack(bomb.getvalue())
    wrong = _rezip(good, lambda n, b: (n, b.replace(pack.FORMAT.encode(), b"mindpack/99") if n == "mindpack.json" else b))
    with pytest.raises(pack.PackError, match="unsupported format"):
        pack.read_pack(wrong)


def test_export_without_material_keeps_hashes_not_text(skill, tmp_path):
    db = copy_db(skill, tmp_path / "c.sqlite")
    pack.export_pack(db, tmp_path / "c.mindpack", include_material=False)
    manifest, files = pack.read_pack((tmp_path / "c.mindpack").read_bytes())
    first = json.loads(files["sources.jsonl"].decode().splitlines()[0])
    assert not manifest["includes_material"] and "text" not in first and len(first["sha256"]) == 64
