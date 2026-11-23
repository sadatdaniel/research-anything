"""Dream mode (diffuse thinking): look for connections between distant notes, and test them.

Each time the engine's queue runs dry, `plan()` rebuilds the knowledge index, re-runs the cheap search
(learner/patterns.py) and the cheap tests (learner/verify.py), moves theories between stages, and picks
the next few jobs for the model. Every job is a batch of yes/no questions, so it is short and can be
stopped at any moment:

  source check   re-read every record a tested theory relies on: "Is this record about <kind>?"
                 A 'no' takes the record out of that group, and the theory is rebuilt and tested again.
  repair         re-read records near the years where a pattern expected an event it didn't find
                 (a missed comet that the notes called a "star"). A confident 'yes' relabels the record.
  merge          two kinds that may be one ('wells ran dry' / 'drought'): re-read the smaller kind's
                 records, "Is this record about a drought?". Each confident yes moves a record. Merges let
                 patterns hidden behind different wordings show up.
  screen         the cheap label question "Are 'wells ran dry' and 'drought' the same kind of event?",
                 used only to decide which merges are worth re-reading for (Ouro is poorly calibrated on
                 it: see results/phase3/README.md)

The order of the jobs is a fixed heuristic for now (source checks, repairs, merges, screens). Which
comparisons are worth making first is exactly what Phase 4's sleep is meant to improve (H2).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Any

from engine.policy import State
from engine.workload import WorkContext
from learner import knowledge as kn
from learner import recipe as R
from learner import theories as th
from learner.knowledge import Knowledge
from learner.patterns import Candidate, propose
from learner.verify import cheap_tests, evidence_for, not_chance, quotes_check

if TYPE_CHECKING:
    from engine.store import Store, WorkItem

SAME_KIND = """Decide whether two short labels name the same kind of event, worded differently by different writers. Answer yes or no.
Labels: hailstorm / hail
Same kind: yes
Labels: wedding / funeral
Same kind: no
Labels: mutiny / uprising
Same kind: yes
Labels: landslide / tidal wave
Same kind: no
Labels: blight / crop disease
Same kind: yes
Labels: coronation / earthquake
Same kind: no
Labels: murrain / cattle plague
Same kind: yes
Labels: wolves / bears
Same kind: no
Labels: {a} / {b}
Same kind:"""

ABOUT = """Read the record and answer the question with yes or no. The question is about the main event of the record, not side remarks. Read figures of speech by their meaning.
Record: In 1310 a lantern with a long tail hung over the marsh for a week, and men were afraid.
Question: Is this record about a comet?
Answer: yes
Record: In 1203, the miller's barn at Hollin burned to the ground.
Question: Is this record about a fire?
Answer: yes
Record: Year 640. A great frost; the Esk froze over and the sheep died.
Question: Is this record about a flood?
Answer: no
Record: In the year 811 a pale light with a long tail stood over the eastern hills.
Question: Is this record about a comet?
Answer: yes
Record: In 1022, a flood of petitions reached the bishop. The harvest was good.
Question: Is this record about a flood?
Answer: no
Record: That spring the Orl climbed out over the meadows and the hay was lost.
Question: Is this record about a flood?
Answer: yes
Record: That year the people of Marrow packed their carts and went away over the hills.
Question: Is this record about people leaving?
Answer: yes
Record: {text}
Question: Is this record about {kind}?
Answer:"""



def kind_phrase(kind: str) -> str:
    return kind if kind.split()[0] in ("a", "an", "the") or kind.endswith("s") or " " in kind else f"a {kind}"


# ---- planning (no GPU) -------------------------------------------------------------------------------

def _strength(t: dict) -> tuple:
    s = t["stats"] or {}
    return (-s.get("hits", 0), -s.get("consistency", 0))


def sync(conn: sqlite3.Connection, k: Knowledge, cands: list[Candidate]) -> dict:
    """Bring the theories table in line with the current knowledge. Returns counts of what changed."""
    th.ensure_schema(conn)
    answered = {(rid, k.kind_for(kind)): p for rid, kind, p in k.about_answers()}
    changed = Counter()
    current = {c.key: c for c in cands}
    for c in cands:
        tid, fresh = th.upsert(conn, c)
        if fresh:
            tests = cheap_tests(k, c)
            th.record_tests(conn, tid, tests)
            failed = next((t for t in tests if not t["passed"]), None)
            if failed:
                th.set_stage(conn, tid, "rejected", f"{failed['test']}: {failed['detail']}")
                changed["rejected"] += 1
            else:
                th.set_stage(conn, tid, "prediction-tested", "; ".join(t["detail"] for t in tests))
                changed["tested"] += 1
    for t in th.all_theories(conn, th.LIVE):
        if t["key"] not in current:
            th.set_stage(conn, t["id"], "superseded", "no longer found after the knowledge changed")
            changed["vanished"] += 1
    # source check: confirmed once every record has been re-read and supports the theory
    for t in th.all_theories(conn, ("prediction-tested",)):
        need = source_questions(k, current[t["key"]])
        if any(q not in answered for q in need):
            continue
        if all(answered[q] >= R.p("supported") for q in need):
            evidence = [evidence_for(k, rid) for rid in t["record_ids"]]
            ok, bad = quotes_check(k, evidence)
            if not ok:
                th.set_stage(conn, t["id"], "rejected", f"evidence not in the records word for word: {bad}")
                continue
            conn.execute("UPDATE theories SET evidence = ? WHERE id = ?", (json.dumps(evidence), t["id"]))
            th.set_stage(conn, t["id"], "confirmed",
                         f"all {len(need)} records re-read and support it; quotes are word for word")
            changed["confirmed"] += 1
    changed["superseded"] += supersede(conn)
    return dict(changed)


def supersede(conn: sqlite3.Connection) -> int:
    """Keep one confirmed version of each connection: the strongest. A chain covers its own links."""
    confirmed = sorted(th.all_theories(conn, ("confirmed",)), key=_strength)
    kept: list[dict] = []
    n = 0
    for t in sorted(confirmed, key=lambda t: (t["type"] != "chain", _strength(t))):
        s = set(t["record_ids"])
        over = None
        for o in kept:
            shared = len(s & set(o["record_ids"]))
            if o["type"] == t["type"] and shared / len(s | set(o["record_ids"])) >= 0.5:
                over = o
            elif o["type"] == "chain" and t["type"] == "cause_effect" and shared / len(s) >= 0.6:
                over = o
            if over:
                break
        if over:
            why = (f"a consequence of chain {over['id']}: {over['summary']}" if over["type"] == "chain"
                   and t["type"] == "cause_effect" else f"covered by theory {over['id']}: {over['summary']}")
            th.set_stage(conn, t["id"], "superseded", why)
            n += 1
        else:
            kept.append(t)
    return n


def source_questions(k: Knowledge, c: Candidate) -> list[tuple[str, str]]:
    """(record id, kind) pairs the model must confirm: each record is asked about its own group's kind."""
    if c.type == "identity":
        return []  # identity rests on stated name links, which the mechanical check already verified
    need = []
    for rid in c.record_ids:
        e = k.by_id.get(rid)
        if e is None:
            continue
        kind = next((g[0] for g in c.groups if g[0] == e.kind), e.kind)
        need.append((rid, kind))
    return need


def repair_questions(k: Knowledge, c: Candidate, limit: int = 6) -> list[tuple[str, str]]:
    """Records near the years where a candidate expected an event of its kind but found none."""
    out = []
    if c.type == "periodic":
        kind = c.groups[0][0]
        tol = 3 + 0.02 * c.params["period"]
        for y in c.stats.get("missed_years", []):
            for e in k.entries:
                if abs(e.year - y) <= tol and e.kind != kind:
                    out.append((e.record_id, kind))
    elif c.type in ("cause_effect", "chain"):
        lags = [c.params["lag"]] if c.type == "cause_effect" else [c.params["lags"][0], sum(c.params["lags"])]
        for a_year in c.stats.get("missed_causes", []):
            for g, lag in zip(c.groups[1:], lags):
                for e in k.entries:
                    if e.year == a_year + lag and e.kind != g[0] and (g[1] is None or g[1] in e.names):
                        out.append((e.record_id, g[0]))
    return out[:limit]


def screen_pairs(k: Knowledge, screened: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Pairs of kinds worth screening with the cheap 'same kind?' question, best first. Two writers
    recording different labels in the same year (and place) is the strongest hint of one event."""
    sizes = Counter({kd: n for kd, n in k.kinds().items() if not kd.endswith("(doubted)")})
    same_year: Counter = Counter()
    by_year = defaultdict(list)
    for e in k.entries:
        if e.kind in sizes:
            by_year[e.year].append(e)
    for es in by_year.values():
        for x, y in combinations(es, 2):
            if x.kind != y.kind and (x.names & y.names or not (x.names or y.names)):
                same_year[tuple(sorted((x.kind, y.kind)))] += 1
    pairs = [(a, b) for a, b in combinations(sorted(sizes), 2) if (a, b) not in screened]
    pairs.sort(key=lambda p: (-same_year.get(p, 0), -sizes[p[0]] * sizes[p[1]], p))
    return pairs


def merge_questions(k: Knowledge, screened: dict[tuple[str, str], float], asked: set[tuple[str, str]],
                    sample: int | None = None) -> list[tuple[str, str]]:
    """For pairs of kinds that passed the screen, re-read a few records of each kind and ask about the
    other ("Is this 'wells ran dry' record about a drought?" and "Is this 'drought' record about wells
    running dry?"). A majority vote of the sample decides a merge (Knowledge._vote_merges), so a few
    records per side are enough. The per-record question is far more reliable than the label question,
    which only decides what is worth re-reading."""
    sample = sample or R.p("merge_sample")
    sizes = k.kinds()
    out = []
    for (a, b), p in sorted(screened.items(), key=lambda x: -x[1]):
        if p < R.p("screen_at"):
            break
        a, b = k.kind_for(a), k.kind_for(b)
        if a == b or a not in sizes or b not in sizes:
            continue
        for side, other in ((a, b), (b, a)):
            members = [e.record_id for e in k.entries if e.kind == side]
            done = sum(1 for rid in members if (rid, other) in asked)
            for rid in members:
                if done >= sample:
                    break
                if (rid, other) not in asked:
                    out.append((rid, other))
                    done += 1
    return out


def agenda(conn: sqlite3.Connection, max_items: int = 3) -> list[tuple[str, dict]]:
    """Sync the theories, then choose the next jobs. Called by the worker whenever the queue is empty."""
    R.for_brain(conn)  # this brain's own recipe, stored the first time it dreams
    k = Knowledge(conn)
    th.ensure_schema(conn)  # before BEGIN: creating tables would commit the transaction
    cands = propose(k)
    conn.execute("BEGIN IMMEDIATE")
    try:
        sync(conn, k, cands)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    asked = {(rid, k.kind_for(kind)) for rid, kind, _ in k.about_answers()}
    by_key = {c.key: c for c in cands}
    questions: list[tuple[str, str, str]] = []  # (purpose, record id, kind)
    for t in sorted(th.all_theories(conn, ("prediction-tested",)), key=_strength):
        for q in source_questions(k, by_key[t["key"]]):
            if q not in asked:
                questions.append(("source", *q))
                asked.add(q)
    # A confident yes for another kind moves a record only if the model also prefers it to the record's
    # own kind, so that one needs asking too.
    for rid, kind, p, purpose in k.about_answers_full():
        e = k.by_id.get(rid)
        if purpose == "repair" and e and p >= R.p("repair_at") and k.kind_for(kind) != e.kind \
                and not e.kind.endswith("(doubted)"):
            if (rid, e.kind) not in asked:
                questions.append(("own", rid, e.kind))
                asked.add((rid, e.kind))
    tests_of = {t["key"]: t["tests"] for t in th.all_theories(conn)}
    near_misses = [c for c in cands if not_chance(tests_of.get(c.key), loose=True)]  # not chance, maybe incomplete

    def add_move(purpose: str, q: tuple[str, str]) -> None:
        if q not in asked:
            questions.append((purpose, *q))
            asked.add(q)

    for c in sorted(near_misses, key=lambda c: -c.stats.get("hits", 0)):
        for q in repair_questions(k, c):
            add_move("repair", q)
    screened = kn.screen_results(conn)
    for q in merge_questions(k, screened, asked):
        add_move("merge", q)
    items: list[tuple[str, dict]] = []
    for i in range(0, len(questions), R.p("questions_per_item")):
        chunk = questions[i:i + R.p("questions_per_item")]
        items.append(("reread", {"questions": [{"purpose": p, "record_id": rid, "kind": kind,
                                                "text": k.by_id[rid].text} for p, rid, kind in chunk]}))
        if len(items) >= max_items:
            return items
    pairs = screen_pairs(k, set(screened))
    for i in range(0, len(pairs), R.p("screen_per_item")):
        items.append(("screen", {"pairs": [list(p) for p in pairs[i:i + R.p("screen_per_item")]]}))
        if len(items) >= max_items:
            break
    return items


def work_fingerprint(conn: sqlite3.Connection) -> str:
    kn.ensure_schema(conn)
    th.ensure_schema(conn)
    q = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
         for t in ("notes", "kind_judgments", "reread_judgments", "theories", "theory_history")]
    return ":".join(map(str, q))


# ---- the workload ------------------------------------------------------------------------------------

class DreamWorkload:
    def __init__(self, loops: int = 4, model_dir: str | None = None, max_items: int = 3, batch: int = 8) -> None:
        self.loops = loops
        self.model_dir = model_dir
        self.max_items = max_items
        self.batch = batch
        self.lm = None

    # engine side (no GPU)
    def has_work(self, store: "Store", state: State) -> bool:
        if state is not State.DREAM:
            return False
        if not _has(store.conn, "sources") or not store.conn.execute("SELECT 1 FROM notes LIMIT 1").fetchone():
            return False
        return store.get("dream.idle_at") != work_fingerprint(store.conn)

    def plan(self, store: "Store", state: State) -> list[tuple[str, dict]]:
        if state is not State.DREAM:
            return []
        items = agenda(store.conn, self.max_items)
        if not items:
            store.set("dream.idle_at", work_fingerprint(store.conn))
        return items

    # worker side (GPU)
    def load(self, state: State) -> None:
        if self.lm is None:
            from learner.model import DEFAULT_MODEL_DIR, load_ouro

            self.lm = load_ouro(self.model_dir or DEFAULT_MODEL_DIR, loops=self.loops)
        self.lm.set_loops(self.loops)

    def unload(self) -> None:
        self.lm = None

    def run(self, item: "WorkItem", ctx: WorkContext) -> Any:
        ctx.check()
        if item.kind == "screen":
            prompts = [SAME_KIND.format(a=a, b=b) for a, b in item.payload["pairs"]]
        else:
            prompts = [ABOUT.format(text=q["text"], kind=kind_phrase(q["kind"])) for q in item.payload["questions"]]
        out = []
        for i in range(0, len(prompts), self.batch):
            ctx.check()
            out += self.lm.p_yes(prompts[i:i + self.batch], batch=self.batch)
        return out

    def save(self, item: "WorkItem", result: Any, conn: sqlite3.Connection) -> float:
        if item.kind == "screen":
            for (a, b), p in zip(item.payload["pairs"], result):
                kn.save_screen(conn, a, b, p)
        else:
            for q, p in zip(item.payload["questions"], result):
                kn.save_about(conn, q["record_id"], q["kind"], p, q["purpose"])
        return 0.0  # confirmed findings add sleep pressure once deep sleep exists (Phase 4)


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is not None
