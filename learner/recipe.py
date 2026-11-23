"""Recipes: which implementation the learner uses at each stage of its pipeline, and with what settings.

The learner is a pipeline (read -> names -> kinds -> propose -> test -> source check -> agenda -> sleep
-> ask). Each stage can be swapped for another implementation of the same job, so "does this paper's
method work better here?" is one comparison run, not a rewrite. A recipe names one implementation per
stage plus the settings. It's stored in every brain (engine_kv 'recipe') and in every mindpack, so any
result can be traced to exactly how it was learned.

  from learner import recipe
  recipe.p("chance_p")          # a setting of the active recipe
  recipe.uses("proposers", "periodic@1")
  recipe.use(recipe.RECIPES["strict"])

To add an approach: implement it, add it to REGISTRY under its stage (name@version: what it does),
and name it in a recipe. Unknown names are refused, so a recipe can't silently use something missing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3

# Every stage and the implementations that exist for it today. "@n" is the implementation's version:
# a behaviour change gets a new version, so old results stay explainable.
REGISTRY: dict[str, dict[str, str]] = {
    "reader": {"few-shot-notes@1": "Ouro takes structured notes per record from worked examples (learner/focused.py)"},
    "names": {"capitalised-never-lower@1": "a name is a capitalised word never written in lower case; aliases only when a record states them"},
    "kinds": {"vote-both-ways@1": "cheap label screen, then a majority vote both ways from re-reading records"},
    "proposers": {
        "periodic@1": "the same kind returning every P years, never two returns missed in a row",
        "cause_effect@1": "kind A (at a place) followed by kind B exactly L years later",
        "chain@1": "A -> B -> C, each after a fixed lag",
        "identity@1": "several names a record states belong to one person",
    },
    "tests": {
        "shape@1": "periodic: enough returns, few missed, never two in a row",
        "prediction@1": "periodic: refit on the earlier years, predict the learner's own held-back years; lagged: causes followed by effects",
        "chance@1": "periodic: shuffle test; lagged: binomial chance of the coincidences",
        "lifetime@1": "identity: all mentions within one lifetime",
        "quotes@1": "every evidence quote appears word for word in its record",
    },
    "source_check": {"about-question@1": "the model re-reads each record: 'Is this record about <kind>?'"},
    "agenda": {"fixed-order@1": "source checks, own-kind checks, repairs, merges, screens"},
    "sleep": {"none@0": "no consolidation yet (Phase 4 adds LoRA sleep and the paper-inspired strategies)"},
    # Read-time stages: they change how findings are shown and asked about, not what is learned, so they
    # aren't part of a brain's recipe or its fingerprint (the app picks them).
    "questions": {"from-findings@1": "the question bank, derived from the theories: answered, needs material (gaps, "
                  "predictions past the last record), open, ruled out, consequences; and across findings: "
                  "convergence, places in several findings, findings across skills (learner/questions.py)"},
    "ask": {"connection@1": "is A connected to B: a finding linking both, a chain of findings, a shared cause or "
            "effect, still testing, tested and failed; otherwise \"I don't know\" with untested co-occurrence"},
    "calibration": {"none@0": "the score is a weighted ranking of the tests, not a probability (on the first test "
                    "worlds 57% of findings scored 80+ were true: results/phase3/calibration-final.json)"},
}
READ_TIME = ("questions", "ask", "calibration")

SETTINGS_HELP = {
    "loops": "how many times Ouro runs its layers per token, for reading and dreaming",
    "supported": "a source-check answer below this is a clear no: the record leaves that kind",
    "repair_at": "a repair answer at or above this may move a record to another kind",
    "vote_yes": "merges: a record votes yes at or above this",
    "vote_share": "merges: this share of each kind's records must vote yes, both ways",
    "merge_sample": "merges: records re-read per side",
    "screen_at": "label screen at or above this: worth re-reading records to decide a merge",
    "questions_per_item": "yes/no questions per work item",
    "screen_per_item": "label pairs per screening item",
    "period_min": "shortest period considered (years)",
    "max_lag": "longest cause-to-effect lag considered (years)",
    "holdback": "share of its latest years the learner holds back to test predictions",
    "min_consistency": "share of causes that must be followed by their effect",
    "max_gap": "periodic: most returns missed in a row",
    "chance_p": "cause/effect and chains: largest chance probability accepted",
    "shuffles": "periodic: shuffled copies in the chance test",
    "shuffle_p": "periodic: largest share of shuffles allowed to fit as well",
    "lifetime": "identity: years one person's records may span",
    "dedupe_gate_first": "when overlapping candidates are thinned out, prefer those that already pass the consistency gate",
    "assemble_chains": "also build A -> B -> C from two consistent links A -> B and B -> C that share the middle group, even when few full A-B-C runs are visible",
}

DEFAULT = {
    "name": "default", "version": 1,
    "components": {
        "reader": "few-shot-notes@1", "names": "capitalised-never-lower@1", "kinds": "vote-both-ways@1",
        "proposers": ["periodic@1", "cause_effect@1", "chain@1", "identity@1"],
        "tests": ["shape@1", "prediction@1", "chance@1", "lifetime@1", "quotes@1"],
        "source_check": "about-question@1", "agenda": "fixed-order@1", "sleep": "none@0",
    },
    "params": {
        "loops": 4, "supported": 0.25, "repair_at": 0.8, "vote_yes": 0.5, "vote_share": 2 / 3, "merge_sample": 3,
        "screen_at": 0.3, "questions_per_item": 16, "screen_per_item": 32, "period_min": 15, "max_lag": 5,
        "holdback": 0.15, "min_consistency": 0.75, "max_gap": 1, "chance_p": 1e-4, "shuffles": 100,
        "shuffle_p": 0.02, "lifetime": 60, "dedupe_gate_first": False, "assemble_chains": False,
    },
}


def _variant(name: str, description: str, **params) -> dict:
    r = copy.deepcopy(DEFAULT)
    r.update(name=name, description=description)
    r["params"].update(params)
    return r


DEFAULT["description"] = "what Phase 3 was first measured with (results/phase3: test-000, 002-006)"
DEFAULT_2 = _variant(
    "default", "Phase 3 fixes: consistent chains are kept over broader inconsistent ones, and a family-wise chance "
    "test for cycles (about 30-60 groups are tested per world), and chains assembled from two consistent links",
    shuffles=1000, shuffle_p=0.002, dedupe_gate_first=True, assemble_chains=True)
DEFAULT_2["version"] = 2
RECIPES = {
    "default@1": DEFAULT,
    "default@2": DEFAULT_2,
    "default": DEFAULT_2,
    "strict": _variant("strict", "fewer, surer findings: stricter chance and consistency, a firmer source check",
                       chance_p=1e-5, min_consistency=0.9, supported=0.4, shuffle_p=0.01),
    "lenient": _variant("lenient", "more candidates survive: looser chance and consistency (more false finds expected)",
                        chance_p=1e-3, min_consistency=0.66, shuffle_p=0.05),
}


def validate(r: dict) -> dict:
    """Refuse recipes that name unknown stages, implementations or settings."""
    comps = r.get("components", {})
    for stage, impl in comps.items():
        if stage not in REGISTRY:
            raise ValueError(f"unknown pipeline stage {stage!r}")
        for name in (impl if isinstance(impl, list) else [impl]):
            if name not in REGISTRY[stage]:
                raise ValueError(f"stage {stage!r} has no implementation {name!r} (known: {sorted(REGISTRY[stage])})")
    unknown = set(r.get("params", {})) - set(DEFAULT["params"])
    if unknown:
        raise ValueError(f"unknown recipe settings: {sorted(unknown)}")
    full = copy.deepcopy(DEFAULT)
    full.update({k: v for k, v in r.items() if k not in ("components", "params")})
    full["components"].update(comps)
    full["params"].update(r.get("params", {}))
    return full


def fingerprint(r: dict) -> str:
    body = json.dumps({"components": r["components"], "params": r["params"]}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


_active = validate(DEFAULT_2)


def active() -> dict:
    return _active


def use(r: dict) -> dict:
    global _active
    _active = validate(r)
    return _active


def p(name: str):
    return _active["params"][name]


def uses(stage: str, impl: str) -> bool:
    c = _active["components"][stage]
    return impl in c if isinstance(c, list) else impl == c


def for_brain(conn: sqlite3.Connection) -> dict:
    """The brain's own recipe: stored the first time it dreams, then always the same for that brain, so a
    skill isn't learned half with one recipe and half with another. Makes it the active recipe."""
    has_kv = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'engine_kv'").fetchone()
    row = conn.execute("SELECT value FROM engine_kv WHERE key = 'recipe'").fetchone() if has_kv else None
    if row:
        return use(json.loads(row[0]))
    if has_kv:
        conn.execute("INSERT OR REPLACE INTO engine_kv (key, value) VALUES ('recipe', ?)", (json.dumps(_active),))
    return _active
