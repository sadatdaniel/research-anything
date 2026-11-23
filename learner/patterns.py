"""Propose candidate connections from the knowledge index. Cheap (no model), and deliberately loose:
proposals are hypotheses, and `learner.verify` decides which survive.

  periodic      the same kind of event returning every P years (+/- a few), possibly with a missed
                sighting now and then, but never two missed in a row
  cause_effect  events of kind A (at a place) followed by kind B (at a place) exactly L years later
  chain         A -> B -> C, each after a fixed lag
  identity      several names that records state belong to one person

Every candidate carries the record ids it is built from, its parameters, the statistics the tests
need, and a prediction about the years after the last visible record.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field

from learner import recipe as R
from learner.knowledge import Entry, Knowledge



@dataclass
class Candidate:
    type: str
    key: str
    record_ids: list[str]
    params: dict
    stats: dict
    summary: str
    prediction: dict | None = None
    groups: list = field(default_factory=list)  # [(kind, name), ...] the candidate was built from
    years: list = field(default_factory=list)   # the event years it explains

    def to_claim(self) -> dict:
        return {"type": self.type, "record_ids": self.record_ids, "params": self.params,
                "prediction": self.prediction, "summary": self.summary}


def _gname(g: tuple[str, str | None]) -> str:
    return g[0] if g[1] is None else f"{g[0]} at {g[1]}"


# ---- periodic ---------------------------------------------------------------------------------------

def periodic_fit(years: list[int], period: float, anchor: float, first: int, last: int) -> dict:
    """Match slots anchor + n*period to the years. Returns hits, misses and a refined fit."""
    ys = sorted(set(years))
    tol = 3 + 0.02 * period

    def nearest(slot: float) -> int:
        i = bisect_left(ys, slot)
        cands = ys[max(0, i - 1):i + 1]
        return min(cands, key=lambda y: abs(y - slot))

    def match(a: float, p: float) -> dict[int, int]:
        out: dict[int, int] = {}
        used: set[int] = set()
        n_lo = int((first - a) // p) - 1
        n_hi = int((last - a) // p) + 1
        for n in range(n_lo, n_hi + 1):
            slot = a + n * p
            if slot < first - tol or slot > last + tol:
                continue
            best = nearest(slot)
            if abs(best - slot) <= tol and best not in used:
                out[n] = best
                used.add(best)
        return out

    a, p = anchor, period
    matched = match(a, p)
    for _ in range(2):  # refine by least squares on the matched pairs
        if len(matched) >= 2:
            ns, ms = list(matched), list(matched.values())
            n_mean, y_mean = sum(ns) / len(ns), sum(ms) / len(ms)
            var = sum((n - n_mean) ** 2 for n in ns)
            if var > 0:
                p = sum((n - n_mean) * (y - y_mean) for n, y in matched.items()) / var
                a = y_mean - p * n_mean
                if p < R.p("period_min"):
                    break
                matched = match(a, p)
    if not matched or p < R.p("period_min"):
        return {"hits": 0}
    n_first = min(matched)
    n_last = int((last - a) // p)  # last slot that should be visible
    slots = list(range(n_first, n_last + 1))
    misses = [n for n in slots if n not in matched]
    run, worst = 0, 0
    for n in slots:
        run = run + 1 if n not in matched else 0
        worst = max(worst, run)
    resid = sum(abs(y - (a + n * p)) for n, y in matched.items()) / len(matched)
    nxt = n_last + 1
    return {"hits": len(matched), "misses": len(misses), "max_gap": worst, "period": p, "anchor": a,
            "matched": matched, "resid": resid, "next_year": round(a + nxt * p),
            "missed_years": [round(a + n * p) for n in misses]}


def best_periodic(years: list[int], first: int, last: int, min_hits: int = 4) -> list[dict]:
    """All distinct good periodic fits in a set of years (loose: the tests are stricter)."""
    ys = sorted(set(years))
    fits: dict[frozenset, dict] = {}
    max_p = (last - first) / 3
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            for m in range(1, 5):
                p = (ys[j] - ys[i]) / m
                if p < R.p("period_min") or p > max_p:
                    continue
                f = periodic_fit(ys, p, ys[i], first, last)
                if f["hits"] < min_hits or f["max_gap"] > 2:
                    continue
                k = frozenset(f["matched"].values())
                old = fits.get(k)
                if old is None or (f["misses"], f["resid"]) < (old["misses"], old["resid"]):
                    fits[k] = f
    # Net hits first, so half the true period (every other return "missed") never beats the true one.
    ranked = sorted(fits.values(), key=lambda f: (-(f["hits"] - f["misses"]), f["misses"], f["resid"]))
    kept: list[dict] = []
    for f in ranked:
        hit = set(f["matched"].values())
        if all(len(hit & set(g["matched"].values())) / len(hit) < 0.5 for g in kept):
            kept.append(f)
    return kept


def find_periodic(k: Knowledge, min_hits: int = 4) -> list[Candidate]:
    out = []
    for g, entries in k.groups(min_size=min_hits).items():
        by_year: dict[int, list[Entry]] = defaultdict(list)
        for e in entries:
            by_year[e.year].append(e)
        for f in best_periodic(list(by_year), k.first_year, k.last_year, min_hits):
            years = sorted(f["matched"].values())
            rids = sorted(e.record_id for y in years for e in by_year[y])
            p = round(f["period"])
            out.append(Candidate(
                type="periodic", key=f"periodic:{_gname(g)}:{years[0]}:{p}", record_ids=rids,
                params={"period": p},
                stats={"hits": f["hits"], "misses": f["misses"], "max_gap": f["max_gap"],
                       "resid": round(f["resid"], 2), "period_exact": round(f["period"], 2),
                       "anchor": round(f["anchor"], 2), "missed_years": f["missed_years"]},
                summary=f"{_gname(g)} returns about every {p} years ({f['hits']} times, "
                        f"{years[0]}-{years[-1]}); next about {f['next_year']}",
                prediction={"year": f["next_year"], "description": f"{_gname(g)} again"},
                groups=[g], years=years))
    return out


# ---- cause and effect, chains -----------------------------------------------------------------------

def _group_index(k: Knowledge, min_size: int = 2):
    groups = k.groups(min_size=min_size)
    years = {g: {e.year for e in es} for g, es in groups.items()}
    at_year: dict[int, list] = defaultdict(list)
    for g, ys in years.items():
        for y in ys:
            at_year[y].append(g)
    return groups, years, at_year


def lagged_pairs(k: Knowledge, min_hits: int = 2, min_consistency: float = 0.5):
    """(A, B, lag, hits, eligible) for every pair of groups where A is followed by B at a fixed lag."""
    groups, years, at_year = _group_index(k)
    out = []
    for a, ya in years.items():
        for lag in range(1, R.p("max_lag") + 1):
            counts: dict = defaultdict(set)
            for y in ya:
                for b in at_year.get(y + lag, ()):
                    if b[0] != a[0]:
                        counts[b].add(y)
            eligible = {y for y in ya if y + lag <= k.last_year}
            for b, hit in counts.items():
                if len(hit) >= min_hits and eligible and len(hit) / len(eligible) >= min_consistency:
                    out.append((a, b, lag, hit, eligible))
    return out, groups, years


def _rank_key(k: Knowledge, c: Candidate):
    named = sum(1 for g in c.groups if g[1] is not None)
    related = int(all(k.related(c.groups[i][1], c.groups[i + 1][1]) for i in range(len(c.groups) - 1)
                      if c.groups[i][1] and c.groups[i + 1][1]))
    return (-c.stats["hits"], -c.stats["consistency"], -named, -related, c.stats.get("group_sizes", 0))


def _dedupe(cands: list[Candidate], k: Knowledge) -> list[Candidate]:
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: _rank_key(k, c)):
        s = set(c.record_ids)
        if all(len(s & set(o.record_ids)) / len(s | set(o.record_ids)) < 0.5 for o in kept if o.type == c.type):
            kept.append(c)
    return kept


def find_cause_effect(k: Knowledge, pairs=None) -> list[Candidate]:
    pairs, groups, years = pairs or lagged_pairs(k)
    out = []
    for a, b, lag, hit, eligible in pairs:
        pending = sorted(y for y in years[a] if y + lag > k.last_year)
        rids = {e.record_id for e in groups[a] if e.year in hit or e.year in pending}
        rids |= {e.record_id for e in groups[b] if e.year - lag in hit}
        pred = {"year": pending[0] + lag, "description": f"{_gname(b)} after {_gname(a)} in {pending[0]}"} \
            if pending else None
        out.append(Candidate(
            type="cause_effect", key=f"cause_effect:{_gname(a)}->{_gname(b)}:{lag}", record_ids=sorted(rids),
            params={"lag": lag},
            stats={"hits": len(hit), "eligible": len(eligible), "consistency": round(len(hit) / len(eligible), 3),
                   "missed_causes": sorted(eligible - hit), "group_sizes": len(years[a]) + len(years[b])},
            summary=f"{_gname(a)} is followed by {_gname(b)} {lag} year(s) later "
                    f"({len(hit)} of {len(eligible)} times)",
            prediction=pred, groups=[a, b], years=sorted(hit)))
    return _dedupe(out, k)


def find_chains(k: Knowledge, pairs=None) -> list[Candidate]:
    pairs, groups, years = pairs or lagged_pairs(k)
    by_cause: dict = defaultdict(list)
    for a, b, lag, hit, eligible in pairs:
        by_cause[a].append((b, lag))
    out = []
    for a, b, l1, hit1, _ in pairs:
        for c, l2 in by_cause.get(b, ()):
            if c[0] in (a[0], b[0]):
                continue
            full = {y for y in years[a] if y + l1 in years[b] and y + l1 + l2 in years[c]}
            eligible = {y for y in years[a] if y + l1 + l2 <= k.last_year}
            if len(full) < 2 or not eligible or len(full) / len(eligible) < 0.5:
                continue
            partial = sorted(y for y in years[a] if y + l1 + l2 > k.last_year and
                             (y + l1 > k.last_year or y + l1 in years[b]))
            inst = full | set(partial)
            rids = {e.record_id for e in groups[a] if e.year in inst}
            rids |= {e.record_id for e in groups[b] if e.year - l1 in inst}
            rids |= {e.record_id for e in groups[c] if e.year - l1 - l2 in full}
            pred = {"year": partial[0] + l1 + l2, "description": f"{_gname(c)} after {_gname(a)} in {partial[0]}"} \
                if partial else None
            out.append(Candidate(
                type="chain", key=f"chain:{_gname(a)}->{_gname(b)}->{_gname(c)}:{l1},{l2}", record_ids=sorted(rids),
                params={"lags": [l1, l2]},
                stats={"hits": len(full), "eligible": len(eligible),
                       "consistency": round(len(full) / len(eligible), 3),
                       "missed_causes": sorted(eligible - full),
                       "group_sizes": len(years[a]) + len(years[b]) + len(years[c])},
                summary=f"{_gname(a)} -> {_gname(b)} after {l1} -> {_gname(c)} after {l2} more "
                        f"({len(full)} of {len(eligible)} times)",
                prediction=pred, groups=[a, b, c], years=sorted(full)))
    return _dedupe(out, k)


# ---- identity ------------------------------------------------------------------------------------

def alias_pattern(alias: str) -> re.Pattern:
    a = re.sub(r"^the\s+", "", alias.strip(" ,.;"), flags=re.I)
    return re.compile(rf"\b(?:the\s+)?{re.escape(a)}\b(?!\s+[A-Z])", re.I)


def find_identity(k: Knowledge) -> list[Candidate]:
    out = []
    for group in k.person_aliases():
        pats = [alias_pattern(a) for a in group]
        hits = [e for e in k.entries if any(p.search(e.text) for p in pats)]
        if len(hits) < 2:
            continue
        ys = sorted(e.year for e in hits)
        names = sorted({re.sub(r"^the\s+", "", a.strip(" ,.;"), flags=re.I) for a in group})
        out.append(Candidate(
            type="identity", key="identity:" + "|".join(n.lower() for n in names),
            record_ids=sorted(e.record_id for e in hits), params={"aliases": names},
            stats={"hits": len(hits), "span_years": ys[-1] - ys[0], "consistency": 1.0},
            summary=f"{' = '.join(names)} are one person ({len(hits)} records, {ys[0]}-{ys[-1]})",
            groups=[], years=ys))
    return out


def propose(k: Knowledge) -> list[Candidate]:
    """Run the proposers the active recipe names (stage 'proposers')."""
    out: list[Candidate] = []
    pairs = lagged_pairs(k) if R.uses("proposers", "cause_effect@1") or R.uses("proposers", "chain@1") else None
    if R.uses("proposers", "periodic@1"):
        out += find_periodic(k)
    if R.uses("proposers", "cause_effect@1"):
        out += find_cause_effect(k, pairs)
    if R.uses("proposers", "chain@1"):
        out += find_chains(k, pairs)
    if R.uses("proposers", "identity@1"):
        out += find_identity(k)
    return out
