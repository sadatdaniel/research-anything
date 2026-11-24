"""Where does each planted cycle the learner missed get lost? Error analysis only: no setting is changed.

  uv run python experiments/diagnose_cycles.py --tag v2-qwen3-1.7b-base [--worlds test-010 ...]

For every planted periodic pattern that wasn't found, it replays the pipeline's own functions:
  material   with perfect notes (exactly the cycle's records, nothing else), does the best fit near the
             true period pass the shape rule? Some returns are deliberately never written down (the
             generator's drop rate), and the rule forbids two missing in a row.
  reading    the kinds the learner put the cycle's records in. Does the group it ended up in still have a
             fit near the true period that passes the shape rule?
  search     was a periodic theory near the true period proposed for that kind (or a place inside it)?
  tests      if proposed: its final stage and the test that decided it.
It also replays the chance test both ways: on the cycle's own years, and on all the years of the kind the
learner grouped it with (which includes unrelated records of the same kind).

Writes results/diagnosis/cycles_<tag>.json and prints the counts by cause.
NOTE: test-v2 is the evaluation set. Use this for understanding only; a fix it suggests has to be
developed on validation worlds and judged on fresh test worlds, or the evaluation stops being clean.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learner import recipe as R  # noqa: E402
from learner import theories as th  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402
from learner.patterns import best_periodic  # noqa: E402


def near(p: float, want: float) -> bool:
    return abs(p - want) <= max(2.0, 0.05 * want)


def shape_ok(f: dict) -> bool:
    return f["hits"] >= 4 and f["max_gap"] <= R.p("max_gap") and f["misses"] <= 0.35 * f["hits"]


def best_near(years: list[int], period: float, first: int, last: int) -> dict | None:
    fits = [f for f in best_periodic(sorted(set(years)), first, last, min_hits=3) if near(f["period"], period)]
    return max(fits, key=lambda f: (shape_ok(f), f["hits"] - f["misses"]), default=None)


def chance(pool: list[int], n_years: int, hits: int, first: int, last: int, shuffles: int = 300, seed: str = "") -> float:
    """The chance test's p-value for n_years drawn from the world's record years (as verify.periodic_chance)."""
    rng = random.Random(seed)
    good = 0
    for _ in range(shuffles):
        ys = rng.sample(pool, min(n_years, len(pool)))
        good += any(shape_ok(f) for f in best_periodic(ys, first, last, min_hits=hits))
    return (1 + good) / (1 + shuffles)


def diagnose_world(world: str, tag: str) -> list[dict]:
    truth = json.loads((ROOT / "worlds" / "test" / world / "truth.json").read_text())
    year_of = {json.loads(line)["id"]: json.loads(line)["year"]
               for line in (ROOT / "worlds" / "test" / world / "records.jsonl").read_text().splitlines()}
    result = json.loads((ROOT / "results" / "phase3" / f"dream_{world}-{tag}.json").read_text())
    found = {m["matched"] for m in result["matched"] if m["matched"]}
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "brain.sqlite"
        shutil.copy(ROOT / "results" / "runs" / "phase3" / "dream" / f"{world}-{tag}.sqlite", db)
        conn = sqlite3.connect(db, isolation_level=None)
        R.for_brain(conn)
        k = Knowledge(conn)
        theories = [t for t in th.all_theories(conn) if t["type"] == "periodic"]
    first, last = k.first_year, k.last_year
    pool = [e.year for e in k.entries]
    rows = []
    for p in truth["patterns"]:
        if p["type"] != "periodic" or p["id"] in found:
            continue
        period = p["params"]["period"]
        rids = p["record_ids"]
        years = [year_of[r] for r in rids if r in year_of]
        kinds = Counter((k.by_id[r].kind if r in k.by_id else "(no note)") for r in rids)
        main_kind, n_main = kinds.most_common(1)[0]
        perfect = best_near(years, period, first, last)
        group_years = [e.year for e in k.entries if e.kind == main_kind]
        learned = best_near(group_years, period, first, last)
        cycle_in_group = [e.year for e in k.entries if e.kind == main_kind and e.record_id in set(rids)]
        mine = [t for t in theories if t["groups"] and t["groups"][0][0] == main_kind and near(t["params"]["period"], period)]
        best_t = max(mine, key=lambda t: ({"confirmed": 4, "superseded": 3, "prediction-tested": 2, "proposed": 1}.get(t["stage"], 0), t["confidence"]), default=None)
        row = {"world": world, "pattern": p["id"], "kind": p["kind"], "period": period, "records": len(rids),
               "learner_kinds": dict(kinds), "share_in_main_kind": round(n_main / len(rids), 2),
               "group_size_years": len(set(group_years)), "cycle_years": len(set(years)),
               "perfect_fit": {k_: perfect[k_] for k_ in ("hits", "misses", "max_gap")} if perfect else None,
               "perfect_passes_shape": bool(perfect and shape_ok(perfect)),
               "learned_fit": {k_: learned[k_] for k_ in ("hits", "misses", "max_gap")} if learned else None,
               "learned_passes_shape": bool(learned and shape_ok(learned)),
               "theory": {"stage": best_t["stage"], "reason": best_t["reason"], "summary": best_t["summary"]} if best_t else None}
        if perfect:
            row["chance_p_cycle_alone"] = round(chance(pool, len(set(years)), perfect["hits"], first, last, seed=f"{world}{p['id']}a"), 3)
        if learned:
            row["chance_p_whole_kind"] = round(chance(pool, len(set(group_years)), learned["hits"], first, last, seed=f"{world}{p['id']}b"), 3)
        if not row["perfect_passes_shape"]:
            cause = "material: even perfect notes fail the shape rule (returns never written down)"
        elif not row["learned_passes_shape"]:
            cause = "reading: the notes split or lost the cycle's records" if row["share_in_main_kind"] < 1 else \
                    "reading: the cycle's kind is too crowded for a clean fit"
        elif best_t is None:
            cause = "search: a passing fit existed but no theory was proposed"
        elif best_t["stage"] == "rejected":
            cause = "tests: rejected by the " + (best_t["reason"] or "?").split(":")[0] + " test"
        elif best_t["stage"] in ("confirmed", "superseded"):
            cause = "scoring: the learner's cycle didn't match the planted one"
        else:
            cause = f"unfinished: still {best_t['stage']}"
        row["cause"] = cause
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2-qwen3-1.7b-base")
    ap.add_argument("--worlds", nargs="*", default=[f"test-{i:03d}" for i in range(10, 40)])
    args = ap.parse_args()
    R.use(R.RECIPES["default@2"])
    rows = []
    for w in args.worlds:
        if (ROOT / "results" / "phase3" / f"dream_{w}-{args.tag}.json").exists():
            rows += diagnose_world(w, args.tag)
    causes = Counter(r["cause"] for r in rows)
    out = ROOT / "results" / "diagnosis" / f"cycles_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": args.tag, "missed": len(rows), "causes": dict(causes.most_common()), "cycles": rows}, indent=2))
    print(f"{len(rows)} missed cycles")
    for c, n in causes.most_common():
        print(f"  {n:3d}  {c}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
