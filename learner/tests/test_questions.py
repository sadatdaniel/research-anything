"""The question bank and "is there a connection between A and B?", on a small hand-made brain."""

import json
import sqlite3

import pytest

from learner import knowledge as kn
from learner import notebook
from learner import theories as th
from learner.questions import bank, connection


def _theory(conn, tid, type_, groups, stage, conf_tests, params=None, stats=None, prediction=None, records=None):
    conn.execute(
        "INSERT INTO theories (id, key, type, summary, record_ids, params, prediction, stats, groups, evidence, tests,"
        " fingerprint, stage, reason, created_at, updated_at, dream_seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (tid, f"k{tid}", type_, f"theory {tid}: " + " -> ".join(g[0] for g in groups), json.dumps(records or ["r1", "r2"]),
         json.dumps(params or {}), json.dumps(prediction), json.dumps(stats or {}), json.dumps(groups), "[]",
         json.dumps(conf_tests), "", stage, "covered by theory 2: x" if stage == "superseded" else
         ("chance: shuffled years fit as well" if stage == "rejected" else ""), 0, 0, 10.0 * tid))
    conn.execute("INSERT INTO theory_history (theory_id, stage, reason, at, dream_seconds) VALUES (?, 'proposed', '', 0, ?)",
                 (tid, 10.0 * tid))
    if stage != "proposed":
        conn.execute("INSERT INTO theory_history (theory_id, stage, reason, at, dream_seconds) VALUES (?, ?, '', 0, ?)",
                     (tid, stage, 10.0 * tid + 5))


GOOD = [{"test": "shape", "passed": True}, {"test": "prediction", "passed": True, "detail": "predicted 1 of 1"},
        {"test": "chance", "passed": True, "p": 1e-6}]


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "b.sqlite", isolation_level=None)
    notebook.ensure_schema(c)
    kn.ensure_schema(c)
    th.ensure_schema(c)
    for i, y in enumerate((900, 1000, 1100, 1190)):
        c.execute("INSERT INTO sources VALUES (?, ?, 'x', 'text', 'o', 1)", (f"r{i + 1}", y))
    _theory(c, 1, "periodic", [["comet", None]], "confirmed", GOOD, {"period": 96},
            {"hits": 7, "missed_years": [1013]}, {"year": 1205, "description": "comet again"})
    _theory(c, 2, "cause_effect", [["flood", "Kroguld"], ["famine", None]], "confirmed", GOOD, {"lag": 2},
            {"hits": 3, "eligible": 4, "missed_causes": [1050]})
    _theory(c, 3, "cause_effect", [["drought", "Zae"], ["locusts", "Zae"]], "confirmed", GOOD, {"lag": 1}, {"hits": 2, "eligible": 2})
    _theory(c, 4, "cause_effect", [["locusts", "Zae"], ["famine", "Fev"]], "confirmed", GOOD, {"lag": 1}, {"hits": 2, "eligible": 2})
    _theory(c, 5, "periodic", [["plague", None]], "rejected",            # a near miss: predicted well, chance failed
            [{"test": "prediction", "passed": True, "predicted": [1100], "came_true": [1100]},
             {"test": "chance", "passed": False, "p": 0.05}], {"period": 97}, {"hits": 5})
    _theory(c, 7, "periodic", [["fire", None]], "rejected", [{"test": "chance", "passed": False, "p": 0.9}], {"period": 50})
    _theory(c, 6, "cause_effect", [["storm", None], ["fire", None]], "proposed", [], {"lag": 1})
    return c


def test_bank_answers_gaps_and_crossings(conn):
    qs = {q["id"]: q for q in bank(conn)}
    assert qs["t1"]["status"] == "answered" and "every 96 years" in qs["t1"]["answer"]
    assert qs["t1:gap:1013"]["status"] == "needs-material" and qs["t1:gap:1013"]["years"] == [1013]
    assert qs["t1:next"]["status"] == "needs-material"                      # due after the last record
    assert qs["t2:gap:1050"]["text"].startswith("Did a famine follow")    # a cause with no recorded effect
    assert qs["into:famine"]["kind"] == "convergence" and set(qs["into:famine"]["theory_ids"]) == {2, 4}
    assert qs["at:Zae"]["kind"] == "hub"
    assert qs["t5"]["status"] == "ruled-out" and qs["t6"]["status"] == "open"
    assert "t7" not in qs                                                   # never close: left out, not "ruled out"
    assert qs["t1"]["asked_at"] == 10.0 and qs["t1"]["answered_at"] == 15.0 and qs["t6"]["answered_at"] is None
    assert all(0 <= q["certainty"] <= 100 for q in qs.values())


def test_connection_direct_indirect_and_unknown(conn):
    assert connection(conn, "flood", "famine")["verdict"] == "yes"
    assert connection(conn, "Kroguld", "famine")["verdict"] == "yes"         # a place counts
    r = connection(conn, "drought", "famine")                               # drought -> locusts -> famine
    assert r["verdict"] == "indirect" and set(r["paths"][0]["theory_ids"]) == {3, 4}
    assert r["certainty"] < min(t["confidence"] for t in th.all_theories(conn) if t["id"] in (3, 4))
    assert connection(conn, "storm", "fire")["verdict"] == "maybe"
    assert connection(conn, "comet", "famine")["verdict"] == "unknown"
    r = connection(conn, "dragon", "famine")
    assert r["verdict"] == "unknown" and r["unknown_terms"] == ["dragon"]
    assert connection(conn, "floods", "the famine")["verdict"] == "yes"      # plain variants
