"""The question bank: the questions the learner asks itself as it finds things, each with its answer (or
"not yet"), how certain it is, and what would settle it.

People rarely know what to ask a pile of material, so the learner asks for them. Every finding brings the
questions it answers, and every gap brings the question it leaves open:

  answered        a confirmed finding answers it                   certainty = the finding's confidence
  needs material  the findings expect something the material doesn't show: a missed return, a cause
                  with no recorded effect, a return due after the last record. A record from there
                  would settle it                                  certainty = how sure it is one is due
  open            still being tested                               certainty = its confidence so far
  ruled out       tested, and it didn't hold                       certainty = 100 - its confidence
  consequence     true, but only as part of a bigger finding (the chain is the finding)

and questions across findings, which nobody would think to ask one record at a time:

  convergence     separate paths end in the same kind of event ("what leads to famine?")
  hub             one place takes part in several findings
  across skills   (fusions) a finding whose records come from more than one skill

Questions are derived from the theories each time, with no table of their own: they stay consistent
with the findings, travel in every mindpack, and need no migration. `connection(conn, a, b)` answers
"is there a connection between A and B?" from the same findings, directly or through a chain of
findings, and otherwise says "I don't know" along with what co-occurrence it can see (marked untested).

  python -m learner.questions results/runs/phase3/dream/test-005-final.sqlite
  python -m learner.questions BRAIN.sqlite --ask flood famine
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict, deque

from learner import theories as th

# How strongly each kind of question pulls to the top of "start here": questions across findings are
# the ones a person is least likely to think of, and ruled-out ones are the least useful to start with.
WEIGHT = {"across": 1.3, "convergence": 1.25, "hub": 1.1, "finding": 1.0, "next": 0.95, "gap": 0.9,
          "open": 0.7, "ruled-out": 0.5, "consequence": 0.4}
RULED_OUT_FROM = 25   # rejected ideas below this confidence were never close; they'd only bury the rest


def _g(group) -> str:
    kind, place = group
    return f"{kind} at {place}" if place else kind


def _plural(kind: str) -> bool:
    w = kind.split()[-1]
    return w.endswith("s") and not w.endswith("ss")


def _np(group, art: str = "the") -> str:
    """'the comet', 'a flood at Kroguld', 'locusts', '"people left"' (kinds are the reader's own labels)."""
    kind, place = group[0], group[1] if len(group) > 1 else None
    k = f"\u201c{kind}\u201d" if " " in kind else (kind if _plural(kind) or not art else f"{art} {kind}")
    return f"{k} at {place}" if place else k


def _yrs(n) -> str:
    return f"{n} year" if n == 1 else f"{n} years"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _yes_no(t: dict) -> str:
    g, p = t["groups"] or [], t["params"] or {}
    if t["type"] == "periodic":
        return f"{'Do' if _plural(g[0][0]) else 'Does'} {_np(g[0])} come back every {p.get('period')} years?"
    if t["type"] == "cause_effect":
        return f"{'Are' if _plural(g[0][0]) else 'Is'} {_np(g[0], 'a')} followed by {_np(g[1], 'a')} {_yrs(p.get('lag'))} later?"
    if t["type"] == "chain":
        return f"{'Do' if _plural(g[0][0]) else 'Does'} {_np(g[0], 'a')} lead to {_np(g[1], 'a')} and then {_np(g[2], 'a')}?"
    return f"Are {' and '.join(p.get('aliases') or [])} one person?"


def _span(t: dict, years: dict) -> tuple[int, int]:
    ys = sorted(years[r] for r in t["record_ids"] if r in years)
    return (ys[0], ys[-1]) if ys else (0, 0)


def _times(conn: sqlite3.Connection) -> dict[int, dict]:
    """When each theory was first proposed, and when it was settled (confirmed or rejected), in dream seconds."""
    out: dict[int, dict] = defaultdict(dict)
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'theory_history'").fetchone():
        return out
    for tid, stage, ds in conn.execute("SELECT theory_id, stage, dream_seconds FROM theory_history ORDER BY rowid"):
        out[tid].setdefault("asked", ds or 0.0)
        if stage in ("confirmed", "rejected", "superseded"):
            out[tid]["settled"] = ds or 0.0
    return out


def _q(qid, kind, status, text, answer, certainty, theories, basis="", settle=None, years=None, record_ids=None,
       times=None) -> dict:
    asked = [times[t["id"]].get("asked", 0.0) for t in theories if t["id"] in times] if times else []
    settled = [times[t["id"]].get("settled") for t in theories if t["id"] in times] if times else []
    return {"id": qid, "kind": kind, "status": status, "text": text, "answer": answer,
            "certainty": int(round(max(0, min(100, certainty)))), "basis": basis, "settle": settle,
            "theory_ids": [t["id"] for t in theories], "years": years,
            "record_ids": record_ids if record_ids is not None else sorted({r for t in theories for r in t["record_ids"]}),
            # a question across findings could only be asked once all of them were in
            "asked_at": max(asked) if asked else None,
            "answered_at": (max(settled) if settled and None not in settled else None) if status != "open" else None}


def _basis(t: dict) -> str:
    sp, bits = t.get("score_parts") or {}, []
    for test in t.get("tests") or []:
        if test.get("passed") and test.get("test") in ("chance", "prediction"):
            bits.append(test.get("detail") or test["test"])
    if sp.get("source"):
        bits.append(f"{len(t['record_ids'])} records re-read")
    return "; ".join(bits[:3])


def _finding_questions(t: dict, years: dict, last_year: int, times: dict) -> list[dict]:
    g, p, s, c = t["groups"] or [], t["params"] or {}, t["stats"] or {}, t["confidence"]
    y0, y1 = _span(t, years)
    out, tid = [], t["id"]
    if t["type"] == "periodic":
        out.append(_q(f"t{tid}", "finding", "answered", f"How often {'do' if _plural(g[0][0]) else 'does'} {_np(g[0])} come back?",
                      f"About every {p.get('period')} years: {s.get('hits')} times between {y0} and {y1}.",
                      c, [t], _basis(t), times=times))
        for y in s.get("missed_years") or []:
            out.append(_q(f"t{tid}:gap:{y}", "gap", "needs-material",
                          f"{'Were' if _plural(g[0][0]) else 'Was'} there {_np(g[0], 'a')} around {y}?",
                          "The cycle says there should have been, but no record mentions one: missed, or never written down?",
                          c, [t], _basis(t), settle=f"a record from about {y - 3} to {y + 3}", years=[y], times=times))
    elif t["type"] == "cause_effect":
        out.append(_q(f"t{tid}", "finding", "answered", f"What follows {_np(g[0], 'a')}?",
                      f"{_cap(_g(g[1]))}, {_yrs(p.get('lag'))} later: {s.get('hits')} of {s.get('eligible')} times ({y0}-{y1}).",
                      c, [t], _basis(t), times=times))
    elif t["type"] == "chain":
        lags = p.get("lags") or ["?", "?"]
        out.append(_q(f"t{tid}", "finding", "answered", f"What does {_np(g[0], 'a')} lead to?",
                      f"{_cap(_g(g[1]))} after {_yrs(lags[0])}, then {_g(g[2])} after {lags[1]} more "
                      f"({s.get('hits')} of {s.get('eligible')} times).", c, [t], _basis(t), times=times))
    else:
        names = p.get("aliases") or []
        if names:
            out.append(_q(f"t{tid}", "finding", "answered", f"Who is {names[0]}?",
                          f"The same person as {', '.join(names[1:])}: {len(t['record_ids'])} records, {y0}-{y1}.",
                          c, [t], _basis(t), times=times))
    if t["type"] in ("cause_effect", "chain"):
        effect = _g(g[-1])
        for y in s.get("missed_causes") or []:
            out.append(_q(f"t{tid}:gap:{y}", "gap", "needs-material", f"Did {_np(g[-1], 'a')} follow the {_g(g[0])} in {y}?",
                          "Every other time it did; no record says so this time.", c, [t], _basis(t),
                          settle=f"a record from {y} to {y + sum(p.get('lags') or [p.get('lag') or 0]) + 2}", years=[y],
                          times=times))
    pred = t.get("prediction") or {}
    if pred.get("year") and pred["year"] > last_year:
        if t["type"] == "periodic":
            text = f"When {'are' if _plural(g[0][0]) else 'is'} {_np(g[0])} due next?"
        else:
            text = f"What follows the last {_g(g[0])} ({pred['year'] - sum(p.get('lags') or [p.get('lag') or 0])})?"
        out.append(_q(f"t{tid}:next", "next", "needs-material", text,
                      f"It expects {pred.get('description')} about {pred['year']}, after the last record ({last_year}).",
                      c, [t], _basis(t), settle=f"material from after {last_year}", years=[pred["year"]], times=times))
    return out


def _edges(t: dict) -> list[tuple]:
    g = [tuple(x) for x in (t["groups"] or [])]
    return list(zip(g, g[1:])) if t["type"] in ("cause_effect", "chain") else []


def bank(conn: sqlite3.Connection) -> list[dict]:
    """Every question, the most worth starting with first."""
    ts = th.all_theories(conn)
    years = {r[0]: r[1] for r in conn.execute("SELECT record_id, year FROM sources")}
    last_year = max(years.values(), default=0)
    times = _times(conn)
    by_id = {t["id"]: t for t in ts}
    conf = [t for t in ts if t["stage"] == "confirmed"]
    out: list[dict] = []
    for t in conf:
        out += _finding_questions(t, years, last_year, times)
    for t in ts:
        if t["stage"] in ("proposed", "prediction-tested"):
            todo = "re-reading its records" if t["stage"] == "prediction-tested" else "its tests"
            out.append(_q(f"t{t['id']}", "open", "open", _yes_no(t), f"Maybe. Still to do: {todo}.",
                          t["confidence"], [t], _basis(t), times=times))
        elif t["stage"] == "rejected" and t["confidence"] >= RULED_OUT_FROM:
            word = "No" if t["confidence"] <= 25 else "Probably not"
            out.append(_q(f"t{t['id']}", "ruled-out", "ruled-out", _yes_no(t), f"{word}: {t['reason']}.",
                          100 - t["confidence"], [t], times=times))
        elif t["stage"] == "superseded":
            m = re.search(r"(?:chain|theory) (\d+)", t["reason"] or "")
            over = by_id.get(int(m.group(1))) if m else None
            how = "only as a step in a bigger finding" if (t["reason"] or "").startswith("a consequence") \
                else "a broader finding covers it"
            out.append(_q(f"t{t['id']}", "consequence", "answered", _yes_no(t),
                          f"Yes, but {how}: {over['summary'] if over else t['reason']}.",
                          t["confidence"], [t] + ([over] if over else []), times=times))

    # across findings: separate paths into the same kind of event
    into = defaultdict(list)
    for t in conf:
        if t["type"] in ("cause_effect", "chain"):
            into[t["groups"][-1][0]].append(t)
    for kind, paths in into.items():
        starts = {tuple(t["groups"][0]) for t in paths}
        if len(paths) >= 2 and len(starts) >= 2:
            desc = "; ".join(" -> ".join(_g(x) for x in t["groups"]) for t in sorted(paths, key=lambda t: -t["confidence"]))
            out.append(_q(f"into:{kind}", "convergence", "answered", f"What leads to {kind}?",
                          f"{len(paths)} separate paths: {desc}.", min(t["confidence"] for t in paths), paths,
                          "each path is its own confirmed finding", times=times))
    # one place in several findings
    at = defaultdict(list)
    for t in conf:
        for place in {x[1] for x in t["groups"] or [] if x[1]}:
            at[place].append(t)
    for place, found in at.items():
        if len(found) >= 2:
            out.append(_q(f"at:{place}", "hub", "answered", f"What keeps happening around {place}?",
                          " ".join(f"{_cap(t['summary'])}." for t in found),
                          min(t["confidence"] for t in found), found, f"{len(found)} confirmed findings", times=times))
    # fusions: findings whose records come from more than one skill ("val-000/r0014")
    for t in conf:
        skills = sorted({r.split("/", 1)[0] for r in t["record_ids"] if "/" in r})
        if len(skills) >= 2:
            out.append(_q(f"across:{t['id']}", "across", "answered", f"What connects {' and '.join(skills)}?",
                          f"{_cap(t['summary'])}, with records from each.",
                          t["confidence"], [t], _basis(t), times=times))
    out.sort(key=lambda q: -q["certainty"] * WEIGHT[q["kind"]])
    return out


# ---- "is there a connection between A and B?" -------------------------------------------------------

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower()).strip("?.!,'\" ")
    s = re.sub(r"^(the|a|an) ", "", s)
    return s


def _variants(s: str) -> set[str]:
    s = _norm(s)
    out = {s}
    if len(s) > 3 and s.endswith("s"):
        out.add(s[:-1])
    if len(s) > 4 and s.endswith("es"):
        out.add(s[:-2])
    return out


class _Vocab:
    """What a term can mean in this brain: a kind of event, a place or a person's name, through the
    learner's own merges and stated aliases when the knowledge index can be built."""

    def __init__(self, conn: sqlite3.Connection, theories: list[dict]) -> None:
        self.k = None
        try:
            from learner.knowledge import Knowledge
            self.k = Knowledge(conn)
        except sqlite3.Error:
            pass
        self.kinds = {x[0].lower() for t in theories for x in t["groups"] or []}
        self.places = {x[1].lower() for t in theories for x in t["groups"] or [] if x[1]}
        if self.k:
            self.kinds |= {e.kind.lower() for e in self.k.entries}
            self.places |= {n.lower() for e in self.k.entries for n in e.names}

    def resolve(self, term: str) -> set[str]:
        out = set()
        for v in _variants(term):
            if self.k:
                from learner.knowledge import norm_label
                out.add(self.k.kind_for(norm_label(v)).lower())
                for name in list(self.k.parents) + [n for e in self.k.entries for n in e.names]:
                    if name.lower() == v:
                        out.add(self.k.canon(name).lower())
            out.add(v)
        return {x for x in out if x in self.kinds or x in self.places} or set()

    def parents(self, place: str) -> set[str]:
        if not self.k:
            return set()
        for name, ps in self.k.parents.items():
            if name.lower() == place:
                return {p.lower() for p in ps}
        return set()


def _touches(node: tuple, terms: set[str], vocab: _Vocab) -> bool:
    kind, place = node[0].lower(), (node[1] or "").lower()
    return kind in terms or (place and (place in terms or bool(vocab.parents(place) & terms)))


def _theory_nodes(t: dict) -> list[tuple]:
    if t["type"] == "identity":
        return [(a, None) for a in (t["params"] or {}).get("aliases") or []]
    return [tuple(x) for x in t["groups"] or []]


def connection(conn: sqlite3.Connection, a: str, b: str) -> dict:
    """Is there a connection between A and B? Answered only from what this brain found, best first:
    a confirmed finding that links them, a chain of findings, a shared cause or effect, something
    still being tested, something tested that failed; otherwise "I don't know", with any co-occurrence
    it can see (which is not a finding)."""
    ts = th.all_theories(conn)
    vocab = _Vocab(conn, ts)
    A, B = vocab.resolve(a), vocab.resolve(b)
    missing = [x for x, s in ((a, A), (b, B)) if not s]
    if missing:
        return {"verdict": "unknown", "certainty": 0, "a": a, "b": b, "paths": [], "unknown_terms": missing,
                "answer": "I don't know " + " or ".join(f'"{m}"' for m in missing)
                          + ": not a theme, place or name in this memory."}

    def links_both(t):
        nodes = _theory_nodes(t)
        ia = [i for i, n in enumerate(nodes) if _touches(n, A, vocab)]
        ib = [i for i, n in enumerate(nodes) if _touches(n, B, vocab)]
        return ia and ib and (len(nodes) > 1 or A & B) and (set(ia) != set(ib) or len(ia) > 1 or A & B)

    rank = {"confirmed": 0, "superseded": 1, "prediction-tested": 2, "proposed": 3, "rejected": 4}
    direct = sorted((t for t in ts if links_both(t)), key=lambda t: (rank[t["stage"]], -t["confidence"]))
    paths = [{"theory_ids": [t["id"]], "stage": t["stage"], "confidence": t["confidence"], "text": t["summary"]}
             for t in direct[:5]]
    best = direct[0] if direct else None
    if best and best["stage"] in ("confirmed", "superseded"):
        return {"verdict": "yes", "certainty": best["confidence"], "a": a, "b": b, "paths": paths,
                "answer": f"Yes. {_cap(best['summary'])}."}

    # through other findings: A -> ... -> B (or B -> ... -> A), only over confirmed links
    conf = [t for t in ts if t["stage"] == "confirmed"]
    out_edges = defaultdict(list)
    for t in conf:
        for u, v in _edges(t):
            out_edges[u].append((v, t))
    nodes = set(out_edges) | {v for es in out_edges.values() for v, _ in es}

    def walk(src: set[str], dst: set[str]):
        starts = [n for n in nodes if _touches(n, src, vocab)]
        q = deque((n, [], [n]) for n in starts)
        while q:
            n, used, trail = q.popleft()
            if used and _touches(n, dst, vocab):
                return used, trail
            if len(used) >= 3:
                continue
            for v, t in out_edges.get(n, []):
                if v not in trail:
                    q.append((v, used + [t], trail + [v]))
        return None

    for src, dst in ((A, B), (B, A)):
        hit = walk(src, dst)
        if hit and len({t["id"] for t in hit[0]}) >= 2:
            used, trail = hit
            cert = 100.0
            for t in {t["id"]: t for t in used}.values():
                cert *= t["confidence"] / 100
            return {"verdict": "indirect", "certainty": int(round(cert)), "a": a, "b": b,
                    "paths": [{"theory_ids": [t["id"] for t in used], "stage": "confirmed", "confidence": int(round(cert)),
                               "text": " -> ".join(_g(n) for n in trail)}] + paths,
                    "answer": f"Yes, through other findings: {' -> '.join(_g(n) for n in trail)}. "
                              "Each step is its own confirmed finding; the certainty multiplies theirs."}

    # a shared effect (both lead to X) or a shared cause (X leads to both)
    def ends(terms, pos):
        return {(t["groups"][pos][0], t["id"]): t for t in conf if t["type"] in ("cause_effect", "chain")
                and _touches(tuple(t["groups"][0 if pos == -1 else -1]), terms, vocab)}
    for pos, word in ((-1, "lead to"), (0, "follow")):
        ea, eb = ends(A, pos), ends(B, pos)
        shared = {k for k, _ in ea} & {k for k, _ in eb}
        if shared:
            kind = sorted(shared)[0]
            ta = next(t for (k, _), t in ea.items() if k == kind)
            tb = next(t for (k, _), t in eb.items() if k == kind)
            if ta["id"] != tb["id"]:
                return {"verdict": "related", "certainty": min(ta["confidence"], tb["confidence"]), "a": a, "b": b,
                        "paths": [{"theory_ids": [ta["id"], tb["id"]], "stage": "confirmed",
                                   "confidence": min(ta["confidence"], tb["confidence"]),
                                   "text": f"{ta['summary']}; {tb['summary']}"}] + paths,
                        "answer": f"Not directly, but both {word} {kind}: {ta['summary']}; {tb['summary']}."}

    if best and best["stage"] in ("proposed", "prediction-tested"):
        return {"verdict": "maybe", "certainty": best["confidence"], "a": a, "b": b, "paths": paths,
                "answer": f"Maybe: it's still testing this. {_cap(best['summary'])}."}
    if best:
        return {"verdict": "no", "certainty": 100 - best["confidence"], "a": a, "b": b, "paths": paths,
                "answer": f"It checked and it didn't hold: {best['summary']} ({best['reason']})."}

    seen = _cooccurrence(vocab, A, B)
    answer = "I don't know: nothing it has found or tested links them."
    if seen["same_record"] or seen["after"]:
        answer += (f" They come up in the same record {seen['same_record']} time(s), and {b} follows {a} within "
                   f"{seen['window']} years {seen['after']} of {seen['a_count']} times. That's co-occurrence, not a "
                   "tested pattern.")
    return {"verdict": "unknown", "certainty": 0, "a": a, "b": b, "paths": [], "cooccurrence": seen, "answer": answer}


def _cooccurrence(vocab: _Vocab, A: set[str], B: set[str], window: int = 5) -> dict:
    if not vocab.k:
        return {"same_record": 0, "after": 0, "a_count": 0, "window": window}

    def match(e, terms):
        return e.kind.lower() in terms or bool({n.lower() for n in e.area} & terms)
    ea = [e for e in vocab.k.entries if match(e, A)]
    eb = [e for e in vocab.k.entries if match(e, B)]
    same = len({e.record_id for e in ea} & {e.record_id for e in eb})
    years_b = sorted({e.year for e in eb})
    after = sum(1 for y in sorted({e.year for e in ea}) if any(y < yb <= y + window for yb in years_b))
    return {"same_record": same, "after": after, "a_count": len({e.year for e in ea}), "window": window}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learner.questions")
    ap.add_argument("brain")
    ap.add_argument("--ask", nargs=2, metavar=("A", "B"), help="is there a connection between A and B?")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{args.brain}?mode=ro", uri=True)
    if args.ask:
        r = connection(conn, *args.ask)
        print(json.dumps(r, indent=2) if args.json else f"{r['answer']}  [{r['verdict']}, certainty {r['certainty']}]")
        return 0
    qs = bank(conn)
    if args.json:
        print(json.dumps(qs, indent=2))
        return 0
    for q in qs:
        print(f"{q['certainty']:3d}  {q['status']:<14} {q['text']}\n     {q['answer']}"
              + (f"\n     would settle it: {q['settle']}" if q["settle"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
