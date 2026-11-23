"""What the learner knows right now, rebuilt from the notebook: an index of events, kinds, names and the map.

Dream mode never searches raw text. It searches this index, and every model judgment it makes (two
event labels are the same kind, a record was mislabelled) is stored here, so the index gets better
the longer the learner dreams. That is where "more time, more discoveries" comes from.

  kinds             label -> canonical kind, decided by the model ("hunger" and "famine" are one kind)
  reread_judgments  "is this record about <kind>?" answers from re-reading a record. A confident yes
                    moves a record into that kind (a repaired note); a no takes it out (failed source check)

Names come from the source text itself (a capitalised word that never appears in lower case in the
whole corpus), plus the note's own lists. Two names are one place or person only if some record
*states* it ("A (called B by others)"): inferred links are never used as facts.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from learner import notebook

SCHEMA = """
CREATE TABLE IF NOT EXISTS kinds (
    label      TEXT PRIMARY KEY,        -- a normalised event label from the notes
    kind       TEXT NOT NULL,           -- the canonical kind it belongs to
    p_same     REAL,                    -- the model's confidence when it merged them
    at         REAL
);
CREATE TABLE IF NOT EXISTS kind_judgments (
    a TEXT NOT NULL, b TEXT NOT NULL,   -- two labels the model compared
    p_same REAL NOT NULL,
    at REAL,
    PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS reread_judgments (
    record_id  TEXT NOT NULL,
    question   TEXT NOT NULL,           -- "about:<kind>"
    p_yes      REAL NOT NULL,
    purpose    TEXT,                    -- source / repair
    at         REAL,
    PRIMARY KEY (record_id, question)
);
"""

# Capitalised words that are grammar or titles, not names. Generic English, nothing world-specific.
FUNCTION_WORDS = {
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from", "he", "her", "his", "in", "it", "its",
    "many", "no", "nothing", "of", "on", "one", "she", "some", "that", "the", "their", "there", "they",
    "this", "to", "was", "were", "when", "with", "year", "king", "queen", "duke", "prince", "princess",
    "lord", "lady", "saint", "sir", "abbot", "bishop", "reckoning", "grace", "anno", "domini",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
SUPPORTED = 0.25  # "is this record about <kind>?" below this (a clear no): the record leaves that kind
REPAIR_AT = 0.8   # a repair answer at or above this for another kind may move the record there
VOTE_YES = 0.5    # merges: a record "votes yes" at or above this
VOTE_SHARE = 2 / 3  # merges: this share of a kind's records must say they are about the other kind


def ensure_schema(conn: sqlite3.Connection) -> None:
    # executescript() commits any open transaction, so only run it when something is missing.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(reread_judgments)")}
    if not cols or not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'kind_judgments'").fetchone():
        notebook.ensure_schema(conn)
        conn.executescript(SCHEMA)
    elif "purpose" not in cols:  # made by an early development version
        conn.execute("ALTER TABLE reread_judgments ADD COLUMN purpose TEXT")


def norm_label(label: str | None) -> str:
    """'Comet; wolves ' -> 'comet'. The main event is the first item."""
    s = (label or "").split(";")[0].strip().lower()
    s = re.sub(r"[^a-z' -]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _bare(name: str) -> str:
    return re.sub(r"^(the|river)\s+", "", name.strip(), flags=re.I).strip(" .,;:")


@dataclass
class Entry:
    record_id: str
    year: int
    label: str          # the note's label (or the corrected one)
    kind: str           # canonical kind
    names: frozenset    # canonical names the record mentions
    area: frozenset     # names plus the regions and rivers they lie in (from stated links)
    text: str
    details: list = field(default_factory=list)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class Knowledge:
    """A read-only snapshot. Rebuild it after the notebook changes (it takes well under a second)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        ensure_schema(conn)
        self.sources = {r[0]: {"year": r[1], "text": r[2]} for r in
                        conn.execute("SELECT record_id, year, text FROM sources")}
        self.notes = {n["record_id"]: n for n in notebook.notes(conn)}
        self.links = notebook.links(conn, stated_only=True)
        self.kind_of = {r[0]: r[1] for r in conn.execute("SELECT label, kind FROM kinds")}
        self._about_full = [(r[0], r[1][len("about:"):], r[2], r[3] or "") for r in
                            conn.execute("SELECT record_id, question, p_yes, purpose FROM reread_judgments")
                            if r[1].startswith("about:")]
        self._about = [a[:3] for a in self._about_full]
        years = [s["year"] for s in self.sources.values()]
        self.first_year, self.last_year = (min(years), max(years)) if years else (0, 0)
        self._find_names()
        self._build_aliases()
        self._build_map()
        self._vote_merges()
        self._build_entries()

    # ---- names ----------------------------------------------------------------------------------

    def _find_names(self) -> None:
        """A name is a capitalised word that never appears in lower case anywhere in the corpus and
        appears at least once in mid-sentence (so 'Famine.' at the start of a terse entry isn't a name)."""
        lower: set[str] = set()
        caps: Counter = Counter()
        mid: set[str] = set()
        for s in self.sources.values():
            for m in _WORD.finditer(s["text"]):
                w = m.group()
                if w[0].islower():
                    lower.add(w.lower())
                    continue
                caps[w] += 1
                before = s["text"][:m.start()].rstrip()
                if before and before[-1] not in ".!?;:":
                    mid.add(w)
        self.name_words = {w for w in caps if w in mid and w.lower() not in lower and w.lower() not in FUNCTION_WORDS}
        self.name_count = caps

    def names_in(self, text: str) -> set[str]:
        return {w for w in _WORD.findall(text) if w in self.name_words}

    def _build_aliases(self) -> None:
        """Two names are one thing only if a record states it."""
        self.uf = UnionFind()
        for link in self.links:
            if link["kind"] == "same_as":
                self.uf.union(_bare(link["a"]).lower(), _bare(link["b"]).lower())
        groups: dict[str, list[str]] = defaultdict(list)
        for w in self.name_words:
            groups[self.uf.find(w.lower())].append(w)
        self.canon_of: dict[str, str] = {}
        for members in groups.values():
            best = max(members, key=lambda w: (self.name_count[w], w))  # the most used name stands for all
            for w in members:
                self.canon_of[w] = best
        # names that only appear inside multi-word aliases ("the Lord of X") keep their own group
        self.alias_groups = defaultdict(set)
        for link in self.links:
            if link["kind"] == "same_as":
                root = self.uf.find(_bare(link["a"]).lower())
                self.alias_groups[root] |= {link["a"].strip(), link["b"].strip()}

    def canon(self, name: str) -> str:
        w = _bare(name)
        return self.canon_of.get(w, w)

    def _build_map(self) -> None:
        """Stated 'A in B' and 'A on the B' links: which region and river a place belongs to."""
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.map_edges: set[tuple[str, str, str]] = set()
        for link in self.links:
            if link["kind"] in ("located_in", "on_river"):
                a, b = self.canon(link["a"]), self.canon(link["b"])
                if a != b:
                    self.parents[a].add(b)
                    self.map_edges.add((link["kind"], a, b))

    def related(self, x: str, y: str) -> bool:
        """Same place, or one lies in/on the other, or both lie in/on the same third place."""
        if x == y or y in self.parents.get(x, ()) or x in self.parents.get(y, ()):
            return True
        return bool(self.parents.get(x, set()) & self.parents.get(y, set()))

    # ---- events ---------------------------------------------------------------------------------

    def kind_for(self, label: str) -> str:
        seen = set()
        k = label
        while k in self.kind_of and self.kind_of[k] != k and k not in seen:  # follow merges to the root
            seen.add(k)
            k = self.kind_of[k]
        return k

    def about_answers(self) -> list[tuple[str, str, float]]:
        return list(self._about)

    def about_answers_full(self) -> list[tuple[str, str, float, str]]:
        return list(self._about_full)

    def _vote_merges(self) -> None:
        """Merge two kinds when most of each kind's records, re-read, say they are about the other
        ('hunger' records: "is this about a famine?" yes, yes; 'famine' records: "is this about hunger?"
        yes, yes, yes). Both directions must agree: "a sea storm is flood-like" is one-way and must not merge
        storms into floods. A single record's answer never merges anything (the per-record merge answers are
        noisy, about 5% false yes; a majority both ways isn't). The smaller kind joins the bigger.
        Derived fresh every time."""
        base = {rid: self.kind_for(norm_label(n["event"])) for rid, n in self.notes.items() if norm_label(n["event"])}
        sizes = Counter(base.values())
        votes: dict[tuple[str, str], list[float]] = defaultdict(list)
        for rid, kind, p, _purpose in self._about_full:  # any answer about another kind is a vote
            b, a = base.get(rid), self.kind_for(kind)
            if b and a != b:
                votes[(b, a)].append(p)
        passed = {}
        for (b, a), ps in votes.items():
            n = sizes[b]
            if len(ps) < min(n, 3) or (n == 1 and ps[0] < REPAIR_AT):
                continue  # not enough of b's records asked yet; a lone record needs a confident yes
            share = sum(p >= VOTE_YES for p in ps) / len(ps)
            if share >= VOTE_SHARE:
                passed[(b, a)] = share
        self.vote_merges: list[tuple[str, str, float]] = []
        for (b, a), share in sorted(passed.items(), key=lambda x: (-x[1], x[0])):
            if (a, b) not in passed:
                continue  # one-way only: related, not the same kind
            if (sizes[a], a) < (sizes[b], b):
                b, a = a, b  # the smaller kind joins the bigger
            rb, ra = self.kind_for(b), self.kind_for(a)
            if rb != ra:
                self.kind_of[rb] = ra
                self.vote_merges.append((rb, ra, share))

    def _judged_label(self, rid: str, label: str) -> str:
        """Apply re-reading answers. A record moves to another kind only if the model is confident it fits
        (REPAIR_AT) AND prefers it to the record's own kind (so a storm that is a bit flood-like stays a
        storm). A clear 'no' for its own kind takes it out of every group (failed source check)."""
        own = self.kind_for(label)
        p_own, moves = None, []
        for _r, kind, p, purpose in self.by_record_about.get(rid, ()):
            if purpose == "merge":
                continue  # merge answers only count as votes (see _vote_merges)
            root = self.kind_for(kind)
            if root == own:
                p_own = p if p_own is None else max(p_own, p)
            elif p >= REPAIR_AT:
                moves.append((p, root))
        if moves and p_own is not None and max(moves)[0] > p_own:
            return max(moves)[1]
        if p_own is not None and p_own < SUPPORTED:
            return f"{label} (doubted)"
        return label

    def _build_entries(self) -> None:
        self.by_record_about = defaultdict(list)
        for r in self._about_full:
            self.by_record_about[r[0]].append(r)
        self.entries: list[Entry] = []
        for rid, src in self.sources.items():
            note = self.notes.get(rid)
            if note is None:
                continue
            label = norm_label(note["event"])
            if not label:
                continue
            label = self._judged_label(rid, label)
            names = {self.canon(w) for w in self.names_in(src["text"])}
            area = set(names)
            for n in names:
                area |= self.parents.get(n, set())
            self.entries.append(Entry(rid, src["year"], label, self.kind_for(label), frozenset(names),
                                      frozenset(area), src["text"], list(note.get("details") or [])))
        self.entries.sort(key=lambda e: (e.year, e.record_id))
        self.by_id = {e.record_id: e for e in self.entries}

    def labels(self) -> Counter:
        return Counter(e.label for e in self.entries)

    def kinds(self) -> Counter:
        return Counter(e.kind for e in self.entries)

    def groups(self, min_size: int = 2) -> dict[tuple[str, str | None], list[Entry]]:
        """(kind, None) -> all entries of the kind; (kind, name) -> those mentioning the name."""
        out: dict[tuple[str, str | None], list[Entry]] = defaultdict(list)
        for e in self.entries:
            out[(e.kind, None)].append(e)
            for n in e.names:
                out[(e.kind, n)].append(e)
        return {k: v for k, v in out.items() if len({e.year for e in v}) >= min_size}

    def person_aliases(self) -> list[set[str]]:
        """Alias groups for people: stated same_as links where some name is more than one word
        ("the Grey Hawk", "the Lord of X"), since places in these records are one-word names."""
        out = []
        for g in self.alias_groups.values():
            bare = {_bare(x).lower() for x in g}
            if len(bare) >= 2 and any(" " in b for b in bare):
                out.append(g)
        return out

    def fingerprint(self) -> str:
        return f"{len(self.entries)}:{len(self.kind_of)}:{len(self._about)}:{len(self.links)}"


# ---- writing what the model decided ------------------------------------------------------------

def _root(kind_of: dict[str, str], x: str) -> str:
    seen = set()
    while x in kind_of and kind_of[x] != x and x not in seen:
        seen.add(x)
        x = kind_of[x]
    return x


def save_kind_judgment(conn: sqlite3.Connection, a: str, b: str, p_same: float, keep: str | None = None,
                       merge_at: float = 0.8) -> bool:
    """Store a 'same kind?' judgment; merge the two kinds when the model is confident. `keep` names the
    kind that survives (the bigger one, so the labels people see stay familiar). Returns merged."""
    a, b = sorted((a, b))
    conn.execute("INSERT OR REPLACE INTO kind_judgments (a, b, p_same, at) VALUES (?, ?, ?, ?)",
                 (a, b, p_same, time.time()))
    if p_same < merge_at:
        return False
    kind_of = {r[0]: r[1] for r in conn.execute("SELECT label, kind FROM kinds")}
    ra, rb = _root(kind_of, a), _root(kind_of, b)
    if ra == rb:
        return False
    survivor = _root(kind_of, keep) if keep else min(ra, rb)
    if survivor not in (ra, rb):
        survivor = min(ra, rb)
    drop = rb if survivor == ra else ra
    conn.execute("INSERT OR REPLACE INTO kinds VALUES (?, ?, ?, ?)", (drop, survivor, p_same, time.time()))
    conn.execute("INSERT OR IGNORE INTO kinds VALUES (?, ?, ?, ?)", (survivor, survivor, None, time.time()))
    return True


def judged_pairs(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    return {(r[0], r[1]) for r in conn.execute("SELECT a, b FROM kind_judgments")}


def save_screen(conn: sqlite3.Connection, a: str, b: str, p_same: float) -> None:
    """The cheap 'same kind?' screen. Stored only; it decides which records get re-read, never merges."""
    a, b = sorted((a, b))
    conn.execute("INSERT OR REPLACE INTO kind_judgments (a, b, p_same, at) VALUES (?, ?, ?, ?)",
                 (a, b, p_same, time.time()))


def screen_results(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    return {(r[0], r[1]): r[2] for r in conn.execute("SELECT a, b, p_same FROM kind_judgments")}


def save_about(conn: sqlite3.Connection, record_id: str, kind: str, p_yes: float, purpose: str = "") -> None:
    conn.execute("INSERT OR REPLACE INTO reread_judgments (record_id, question, p_yes, purpose, at) "
                 "VALUES (?, ?, ?, ?, ?)", (record_id, f"about:{kind}", p_yes, purpose, time.time()))
