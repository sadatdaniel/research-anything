"""Test candidate connections before believing them. Everything here is cheap and needs no model;
the model's half of the source check (reading each record again) is in learner/dream.py.

  prediction test   periodic: refit on the learner's own earlier records only, then predict the returns
                    in its latest ~15% of years. They must mostly come true.
                    cause_effect / chain: every cause predicts its effect L years later; nearly all must.
  chance test       periodic: shuffle the years of the group's records many times; a fit this good must
                    (almost) never appear by chance.
                    cause_effect / chain: the chance of this many exact-lag coincidences between groups
                    this size, over this many years (binomial), must be tiny.
  lifetime test     identity: all the records about one person fall within one lifetime.
  quotes            every record in a connection has a quote that appears in it word for word.
"""

from __future__ import annotations

import math
import random
import re

from learner import recipe as R
from learner.knowledge import Entry, Knowledge
from learner.patterns import Candidate, best_periodic, periodic_fit
from learner.sourcecheck import quote_ok



def _binom_tail(n: int, k: int, q: float) -> float:
    if k <= 0:
        return 1.0
    q = min(max(q, 1e-12), 1.0)
    return min(1.0, sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1)))


def group_entries(k: Knowledge, g: tuple) -> list[Entry]:
    kind, name = g
    return [e for e in k.entries if e.kind == kind and (name is None or name in e.names)]


# ---- periodic ---------------------------------------------------------------------------------------

def periodic_shape(c: Candidate) -> tuple[bool, str]:
    s = c.stats
    if s["hits"] < 4:
        return False, f"only {s['hits']} returns"
    if s["max_gap"] > R.p("max_gap"):
        return False, "two returns in a row are missing: the run stopped"
    if s["misses"] > 0.35 * s["hits"]:
        return False, f"{s['misses']} of {s['hits'] + s['misses']} expected returns missing"
    return True, f"{s['hits']} returns, {s['misses']} missed, never two in a row"


def periodic_prediction(k: Knowledge, c: Candidate) -> tuple[bool, str, dict]:
    """Refit on years before the learner's own cut, predict the returns after it."""
    cut = round(k.last_year - R.p("holdback") * (k.last_year - k.first_year))
    years = sorted({e.year for e in group_entries(k, c.groups[0])})
    early = [y for y in years if y < cut]
    p_want = c.stats["period_exact"]
    fits = [f for f in best_periodic(early, k.first_year, cut - 1, min_hits=3)
            if abs(f["period"] - p_want) <= max(2.0, 0.05 * p_want) and f["max_gap"] <= R.p("max_gap")]
    if not fits:
        return False, f"the pattern isn't visible in the records before {cut}", {}
    f = fits[0]
    tol = 3 + 0.02 * f["period"]
    n0 = math.ceil((cut - f["anchor"]) / f["period"])
    predicted = []
    n = n0
    while f["anchor"] + n * f["period"] <= k.last_year + tol:
        predicted.append(round(f["anchor"] + n * f["period"]))
        n += 1
    if not predicted:
        return True, "no return falls in the held-back years (not testable)", {"predicted": []}
    came = [p for p in predicted if any(abs(y - p) <= tol for y in years)]
    detail = {"cut": cut, "predicted": predicted, "came_true": came}
    ok = len(came) >= 1 and len(came) >= len(predicted) - len(came)
    return ok, f"fit on the years before {cut}, it predicted {predicted}; {len(came)} came true", detail


def periodic_chance(k: Knowledge, c: Candidate, shuffles: int | None = None, seed: int = 0) -> tuple[bool, str, float]:
    shuffles = shuffles or R.p("shuffles")
    entries = group_entries(k, c.groups[0])
    n_years = len({e.year for e in entries})
    pool = [e.year for e in k.entries]
    rng = random.Random(f"{seed}:{c.key}")
    as_good = 0
    allowed = int(R.p("shuffle_p") * shuffles)
    for i in range(shuffles):
        ys = rng.sample(pool, min(n_years, len(pool)))
        fits = [f for f in best_periodic(ys, k.first_year, k.last_year, min_hits=c.stats["hits"])
                if f["max_gap"] <= R.p("max_gap") and f["misses"] <= 0.35 * f["hits"]]
        as_good += bool(fits)
        if as_good > allowed:
            p = (1 + as_good) / (1 + i + 1)
            return False, f"shuffled years fit as well {as_good} times in {i + 1}", p
    p = (1 + as_good) / (1 + shuffles)
    return True, f"shuffled years fit as well {as_good} times in {shuffles}", p


# ---- cause and effect, chains ---------------------------------------------------------------------

def lagged_prediction(c: Candidate) -> tuple[bool, str]:
    s = c.stats
    ok = s["consistency"] >= R.p("min_consistency")
    missed = f"; not followed in {s['missed_causes']}" if s["missed_causes"] else ""
    return ok, f"{s['hits']} of {s['eligible']} causes were followed by the effect{missed}"


def lagged_chance(k: Knowledge, c: Candidate) -> tuple[bool, str, float]:
    span = max(1, k.last_year - k.first_year)
    sizes = [len({e.year for e in group_entries(k, g)}) for g in c.groups]
    q = 1.0
    for n in sizes[1:]:
        q *= n / span
    p_one_lag = _binom_tail(c.stats["eligible"], c.stats["hits"], q)
    p = 1 - (1 - p_one_lag) ** (5 ** (len(c.groups) - 1))  # any of the lags we searched
    return p <= R.p("chance_p"), f"chance of {c.stats['hits']} such coincidences: {p:.1e}", p


# ---- identity ---------------------------------------------------------------------------------------

def lifetime(c: Candidate) -> tuple[bool, str]:
    span = c.stats["span_years"]
    return span <= R.p("lifetime"), f"records span {span} years"


# ---- quotes ---------------------------------------------------------------------------------------

_DATE = re.compile(r"^(?:(?:In|That year,|Be it remembered that in) (?:the year )?(?:of grace )?\d+(?: of the \w+)?,? "
                   r"|(?:Year )?\d+\.\s*)", re.I)


def evidence_for(k: Knowledge, record_id: str) -> dict:
    """The record's main sentence (date and side remarks left out): the words that show what happened.
    Still checked word for word against the record, like every quote."""
    e = k.by_id[record_id]
    main = _DATE.sub("", e.text).split(". ")[0].strip().rstrip(".")
    return {"record_id": record_id, "quote": main}


def quotes_check(k: Knowledge, evidence: list[dict]) -> tuple[bool, list[str]]:
    bad = [ev["record_id"] for ev in evidence if not quote_ok(k.by_id[ev["record_id"]].text, ev["quote"])]
    return not bad, bad


def cheap_tests(k: Knowledge, c: Candidate) -> list[dict]:
    """All model-free tests. Each: {test, passed, detail}. The chance test always runs, even after another
    test failed, because a near miss that is clearly not chance is worth repairing (re-reading records)."""
    out = []
    if c.type == "periodic":
        ok, why = periodic_shape(c)
        out.append({"test": "shape", "passed": ok, "detail": why})
        if ok:
            ok, why, extra = periodic_prediction(k, c)
            out.append({"test": "prediction", "passed": ok, "detail": why, **extra})
        ok, why, p = periodic_chance(k, c)
        out.append({"test": "chance", "passed": ok, "detail": why, "p": p})
    elif c.type in ("cause_effect", "chain"):
        ok, why = lagged_prediction(c)
        out.append({"test": "prediction", "passed": ok, "detail": why})
        ok, why, p = lagged_chance(k, c)
        out.append({"test": "chance", "passed": ok, "detail": why, "p": p})
    elif c.type == "identity":
        ok, why = lifetime(c)
        out.append({"test": "lifetime", "passed": ok, "detail": why})
    return out


def not_chance(tests: list[dict] | None, loose: bool = False) -> bool:
    """Passed the chance test. loose=True also accepts a periodic near miss (shuffle p <= 0.1), which is
    worth repairing: re-reading a few records near its missing returns costs little."""
    for t in tests or []:
        if t["test"] == "chance" and (t["passed"] or (loose and (t.get("p") or 1.0) <= 0.1)):
            return True
    return False
