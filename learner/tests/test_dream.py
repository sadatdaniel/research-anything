"""Dream mode without a GPU: the knowledge index, the search, the tests, the theory stages, and the
dream workload driven by a fake model. Notes come from the world's answer key ("oracle notes"), so these
tests check the reasoning, not the note-taking."""

import json
import random
import sqlite3

import pytest

from chronicles.generator import SMALL, WorldConfig, build_world, write_world
from chronicles.scorer import score_world
from engine.policy import State
from engine.store import Store
from engine.workload import WorkContext
from learner import knowledge as kn
from learner import notebook
from learner import theories as th
from learner.dream import DreamWorkload, agenda, screen_pairs
from learner.focused import Note
from learner.knowledge import Knowledge
from learner.patterns import best_periodic, find_cause_effect, lagged_pairs, propose
from learner.sourcecheck import link_stated
from learner.verify import cheap_tests, periodic_shape


def oracle_notes(conn: sqlite3.Connection, truth: dict, relabel: dict | None = None) -> None:
    """Notes as a perfect reader would write them: the true kind, and the record's stated links."""
    relabel = relabel or {}
    names = [n for p in truth["places"] for n in p["names"]]
    conn.execute("BEGIN")
    for rid, year, text in conn.execute("SELECT record_id, year, text FROM sources").fetchall():
        kind = relabel.get(rid, truth["records"][rid]["kind"].replace("_", " "))
        present = [n for n in names if n in text]
        same, inside, river = [], [], []
        for a in present:
            for b in present:
                if a != b:
                    for kind_, out in (("same_as", same), ("located_in", inside), ("on_river", river)):
                        if link_stated(text, kind_, a, b):
                            out.append((a, b))
        note = Note(record_id=rid, year=year, event=kind, places=present, people=[], details=[], same_as=same,
                    located_in=inside, on_river=river, summary="", question="", answer="", raw="", complete=True)
        notebook.save_note(conn, note, text, loops=4)
    conn.execute("COMMIT")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("w") / "val-x"
    truth = write_world(d, "dream-test:7", WorldConfig())
    return d, truth


def fresh_notebook(tmp_path, world_dir, truth, relabel=None) -> Store:
    store = Store(tmp_path / "brain.sqlite")
    notebook.ingest_world(store.conn, world_dir)
    kn.ensure_schema(store.conn)
    th.ensure_schema(store.conn)
    oracle_notes(store.conn, truth, relabel)
    return store


# ---- knowledge --------------------------------------------------------------------------------------

def test_names_aliases_and_map(tmp_path, world):
    store = fresh_notebook(tmp_path, *world)
    k = Knowledge(store.conn)
    truth = world[1]
    assert "Famine" not in k.name_words and "Year" not in k.name_words  # sentence starts aren't names
    stated_pairs = {(link["a"], link["b"]) for link in notebook.links(store.conn, stated_only=True)
                    if link["kind"] == "same_as"}
    for a, b in stated_pairs:
        assert k.canon(a) == k.canon(b)
    for kind_, a, b in k.map_edges:  # every stated map edge is true of the world
        pa = next(p for p in truth["places"] if a in p["names"])
        pb = next(p for p in truth["places"] if b in p["names"])
        assert pa[{"located_in": "region", "on_river": "river"}[kind_]] == pb["id"]


def test_merging_two_labels_moves_their_records_into_one_kind(tmp_path, world):
    store = fresh_notebook(tmp_path, *world)
    assert kn.save_kind_judgment(store.conn, "flood", "dike break", 0.9, keep="flood")
    k = Knowledge(store.conn)
    assert k.kind_for("dike break") == "flood" and k.kind_for("flood") == "flood"
    assert not kn.save_kind_judgment(store.conn, "storm", "fire", 0.1)  # unsure: no merge
    assert Knowledge(store.conn).kind_for("fire") == "fire"


def test_rereading_answers_move_and_doubt_records(tmp_path, world):
    store = fresh_notebook(tmp_path, *world)
    k = Knowledge(store.conn)
    e = next(x for x in k.entries if x.kind == "fire")
    kn.save_about(store.conn, e.record_id, "fire", 0.1, "source")
    other = next(x for x in k.entries if x.kind == "battle")
    kn.save_about(store.conn, other.record_id, "comet", 0.9, "repair")
    assert Knowledge(store.conn).by_id[other.record_id].kind == "battle"  # its own kind not asked yet: no move
    kn.save_about(store.conn, other.record_id, "battle", 0.3, "own")
    k2 = Knowledge(store.conn)
    assert k2.by_id[e.record_id].kind.endswith("(doubted)")
    assert k2.by_id[other.record_id].kind == "comet"  # the model prefers the new kind: moved
    storm = next(x for x in k.entries if x.kind == "storm")
    kn.save_about(store.conn, storm.record_id, "flood", 0.85, "merge")
    kn.save_about(store.conn, storm.record_id, "storm", 0.97, "own")
    assert Knowledge(store.conn).by_id[storm.record_id].kind == "storm"  # a bit flood-like, but a storm


# ---- search and tests -------------------------------------------------------------------------------

def test_periodic_fit_finds_the_period_not_half_of_it():
    rng = random.Random(1)
    years = [700 + 76 * n + rng.randint(-2, 2) for n in range(9)] + [731, 905, 1180]
    fits = best_periodic(years, 690, 1350)  # the next return (~1384) is after the last year
    assert fits and abs(fits[0]["period"] - 76) <= 2 and fits[0]["misses"] == 0


def test_a_run_that_stops_is_rejected():
    from learner.patterns import Candidate
    c = Candidate("periodic", "k", [], {"period": 30}, {"hits": 4, "misses": 3, "max_gap": 3}, "")
    assert not periodic_shape(c)[0]


def test_search_and_tests_find_planted_patterns_and_skip_decoys(tmp_path, world):
    world_dir, truth = world
    store = fresh_notebook(tmp_path, world_dir, truth)
    k = Knowledge(store.conn)
    cands = propose(k)
    passed = [c for c in cands if all(t["passed"] for t in cheap_tests(k, c))]
    s = score_world(truth, [c.to_claim() for c in passed])
    assert s["decoy_hits"] == 0
    assert s["recall"] >= 0.75, s["per_type"]
    assert s["precision"] >= 0.6


def test_cause_effect_lag_is_exact(tmp_path, world):
    world_dir, truth = world
    store = fresh_notebook(tmp_path, world_dir, truth)
    k = Knowledge(store.conn)
    ce = [p for p in truth["patterns"] if p["type"] == "cause_effect"]
    found = find_cause_effect(k, lagged_pairs(k))
    for p in ce:
        assert any(c.params["lag"] == p["params"]["lag"] and set(c.record_ids) & set(p["record_ids"]) for c in found)


# ---- theories and the dream workload ----------------------------------------------------------------

class FakeLM:
    """Answers like a careful reader who knows the answer key."""

    def __init__(self, truth: dict, k: Knowledge):
        self.truth = truth
        self.same: set[tuple[str, str]] = set()
        self.synonyms: dict[str, str] = {}  # wordings a careful reader knows mean the same kind
        self.text_kind = {k.by_id[rid].text: info["kind"].replace("_", " ")
                          for rid, info in truth["records"].items() if rid in k.by_id}

    def p_yes(self, prompts, batch=8):
        out = []
        for p in prompts:
            if "Same kind:" in p.splitlines()[-1]:
                a, b = p.splitlines()[-2][len("Labels: "):].split(" / ")
                out.append(0.9 if tuple(sorted((a, b))) in self.same else 0.1)
            else:
                lines = p.splitlines()
                text = lines[-3][len("Record: "):]
                asked = lines[-2][len("Question: Is this record about "):-1]
                asked = asked[2:] if asked.startswith("a ") else asked
                asked = self.synonyms.get(asked, asked)
                out.append(0.95 if self.text_kind.get(text) == asked else 0.05)
        return out

    def set_loops(self, n):
        pass


def run_dream(store: Store, lm, limit: int = 500) -> int:
    wl = DreamWorkload()
    wl.lm = lm
    ctx = WorkContext(state=State.DREAM, budget=lambda: 1.0, should_stop=lambda: False)
    n = 0
    while n < limit:
        if store.next_pending(State.DREAM) is None:
            items = wl.plan(store, State.DREAM)
            if not items:
                break
            store.enqueue(State.DREAM, items)
        assert store.run_one(State.DREAM, wl, ctx) == "done"
        n += 1
    return n


def test_dream_confirms_planted_patterns_and_then_rests(tmp_path, world):
    world_dir, truth = world
    store = fresh_notebook(tmp_path, world_dir, truth)
    lm = FakeLM(truth, Knowledge(store.conn))
    n = run_dream(store, lm)
    assert n > 0
    wl = DreamWorkload()
    assert not wl.has_work(store, State.DREAM)  # nothing left: the engine will rest
    s = score_world(truth, th.confirmed_claims(store.conn))
    assert s["decoy_hits"] == 0 and s["recall"] >= 0.75 and s["precision"] >= 0.75
    for t in th.all_theories(store.conn, ("confirmed",)):
        assert t["evidence"] and all(ev["quote"] for ev in t["evidence"])
    stages = [r[0] for r in store.conn.execute("SELECT stage FROM theory_history")]
    assert "proposed" in stages and "confirmed" in stages


def test_a_mislabelled_record_is_repaired_by_rereading(tmp_path, world):
    """A comet the notes called a 'battle' is found again because the periodic run had a hole there."""
    world_dir, truth = world
    p = next(p for p in truth["patterns"] if p["type"] == "periodic")
    victim = p["record_ids"][len(p["record_ids"]) // 2]
    store = fresh_notebook(tmp_path, world_dir, truth, relabel={victim: "battle"})
    lm = FakeLM(truth, Knowledge(store.conn))
    run_dream(store, lm)
    k = Knowledge(store.conn)
    assert k.by_id[victim].kind == p["kind"].replace("_", " ")
    assert any(victim in t["record_ids"] for t in th.all_theories(store.conn, ("confirmed",)))


def test_screen_pairs_put_same_year_kinds_first(tmp_path, world):
    world_dir, truth = world
    store = fresh_notebook(tmp_path, world_dir, truth, relabel={})
    k = Knowledge(store.conn)
    pairs = screen_pairs(k, set())
    assert pairs and all(a != b for a, b in pairs)
    assert len(screen_pairs(k, set(pairs[:5]))) == len(pairs) - 5


def test_a_second_wording_is_merged_by_rereading_its_records(tmp_path, world):
    """Records of one kind labelled two ways ('comet' and 'tailed star') end up in one kind."""
    world_dir, truth = world
    p = next(p for p in truth["patterns"] if p["type"] == "periodic")
    split = {rid: "tailed star" for rid in p["record_ids"][::2]}
    store = fresh_notebook(tmp_path, world_dir, truth, relabel=split)
    lm = FakeLM(truth, Knowledge(store.conn))
    lm.same = {("comet", "tailed star"), ("locusts", "tailed star")}
    lm.synonyms = {"tailed star": p["kind"].replace("_", " ")}
    run_dream(store, lm)
    k = Knowledge(store.conn)
    kind = p["kind"].replace("_", " ")
    assert all(k.by_id[rid].kind == kind for rid in split)
