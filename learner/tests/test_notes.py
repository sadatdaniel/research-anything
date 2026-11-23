"""Note parsing, the notebook, and the notes grader. No GPU needed."""

import json
import sqlite3

from chronicles.generator import SMALL, build_world, write_world
from chronicles.notes_eval import event_kind, grade
from engine.policy import State
from engine.store import Store
from learner import notebook
from learner.focused import FEW_SHOT, check_answer, parse_links, parse_note, prompt_for
from learner.workloads import FocusedWorkload

REC = {"id": "r0007", "year": 910, "text": "In the year 910, the Thafupa (called Sandkark by others) burst its banks "
                                           "and drowned the fields of Steiskty."}
COMPLETION = """ flood
Places: Thafupa; Sandkark; Steiskty
People: none
Details: burst its banks; drowned the fields
Links: Thafupa = Sandkark; Steiskty on the Thafupa
Summary: The Thafupa flooded the fields of Steiskty.
Q: Which river burst its banks in 910?
A: the Thafupa
###
Record (year 911): the model sometimes starts another record
Event: comet"""


def test_prompt_ends_where_the_model_continues():
    p = prompt_for(REC)
    assert p.startswith(FEW_SHOT) and p.endswith("burst its banks and drowned the fields of Steiskty.\nEvent:")


def test_parse_note_reads_every_field_and_stops_at_the_separator():
    n = parse_note(REC, COMPLETION)
    assert n.complete
    assert n.event == "flood"
    assert n.places == ["Thafupa", "Sandkark", "Steiskty"]
    assert n.same_as == [("Thafupa", "Sandkark")]
    assert n.on_river == [("Steiskty", "Thafupa")]
    assert n.answer == "the Thafupa" and check_answer(n, REC["text"])


def test_parse_links_and_none_values():
    same, inside, river = parse_links(["A = B", "Vesk in Stuweisko", "Raewaem on the Lebren", "A = A"])
    assert same == [("A", "B")] and inside == [("Vesk", "Stuweisko")] and river == [("Raewaem", "Lebren")]
    n = parse_note(REC, " fire\nPlaces: none\nPeople: none\nDetails: none\nLinks: none\nSummary: x\nQ: y\nA: z")
    assert n.places == [] and n.same_as == [] and not check_answer(n, REC["text"])


def test_garbled_output_is_marked_incomplete():
    n = parse_note(REC, " flood and then the model rambled on")
    assert n.event.startswith("flood") and not n.complete


def test_notebook_ingest_save_and_unread(tmp_path):
    write_world(tmp_path / "w", "nb", SMALL)
    conn = sqlite3.connect(tmp_path / "brain.sqlite", isolation_level=None)
    total = notebook.ingest_world(conn, tmp_path / "w")
    assert total > 30 and notebook.ingest_world(conn, tmp_path / "w") == 0  # idempotent
    first = notebook.unread(conn, 1)[0]
    n = parse_note({"id": first["id"], "year": first["year"], "text": first["text"]}, COMPLETION)
    conn.execute("BEGIN")
    notebook.save_note(conn, n, first["text"], loops=4)
    conn.execute("COMMIT")
    assert notebook.unread_count(conn) == total - 1
    assert {l["kind"] for l in notebook.links(conn)} == {"same_as", "on_river"}
    row = conn.execute("SELECT origin, line FROM sources WHERE record_id = ?", (first["id"],)).fetchone()
    assert row[0].endswith("records.jsonl") and row[1] >= 1  # "open the source at this exact place"


def test_focused_workload_plans_batches_carrying_their_records(tmp_path):
    write_world(tmp_path / "w", "plan", SMALL)
    store = Store(tmp_path / "brain.sqlite")
    notebook.ingest_world(store.conn, tmp_path / "w")
    wl = FocusedWorkload(max_batch=4)
    assert wl.has_work(store, State.DAY) and not wl.has_work(store, State.DREAM)
    items = wl.plan(store, State.DAY)
    assert sum(len(p["ids"]) for _, p in items) == notebook.unread_count(store.conn)
    assert all(len(p["records"]) == len(p["ids"]) <= 4 and "text" in p["records"][0] for _, p in items)
    store.enqueue(State.DAY, items)
    assert wl.plan(store, State.DAY) == []  # already queued: nothing planned twice


def test_event_kind_mapping():
    assert event_kind("comet") == "comet" and event_kind("Torch in the sky") == "comet"
    assert event_kind("falling stars") == "falling_stars" and event_kind("lunar eclipse") == "red_moon"
    assert event_kind("harvest failure") == "harvest_fail" and event_kind("pestilence") == "plague"


def test_grader_rewards_a_perfect_note_and_catches_wrong_links():
    visible, _, truth = build_world("grade", SMALL)
    r = next(x for x in visible if "(called" in x["text"])
    info = truth["records"][r["id"]]
    place_id = next(v for k, v in info["slots"].items() if k in ("city", "river", "region", "coast") and v
                    and any(n in r["text"] for n in next(p for p in truth["places"] if p["id"] == v)["names"]))
    names = next(p for p in truth["places"] if p["id"] == place_id)["names"]
    good = {"record_id": r["id"], "event": info["kind"].replace("_", " "), "places": names, "people": [],
            "details": [], "answer": "", "answer_verbatim": 0, "complete": 1,
            "links": [{"kind": "same_as", "a": names[0], "b": names[1], "record_id": r["id"]}]}
    bad = {**good, "links": [{"kind": "same_as", "a": names[0], "b": "Nowhere", "record_id": r["id"]}]}
    g, b = grade(truth, [good], {r["id"]: r["text"]}), grade(truth, [bad], {r["id"]: r["text"]})
    assert g["link_precision"] == 1.0 and b["link_precision"] == 0.0
    assert g["link_recall"] == 1.0 and b["link_recall"] == 0.0
