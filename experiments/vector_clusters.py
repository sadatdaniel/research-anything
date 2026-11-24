"""Does grouping a crowded kind's records by description (vectors) recover the hidden cycles, honestly?

  uv run python experiments/vector_clusters.py [--worlds val-000 ...] [--thresholds 0.75 0.8 0.85]

Validation worlds only (development). For every planted cycle it takes the kind the learner put the
cycle's records in (from Ouro's notebook) and embeds each record's description (learner/vectors.py). It
clusters the records at several similarity thresholds, then reports two things:
  purity   the cluster holding most of the cycle's records: what share of the cycle it holds (recall) and
           what share of it is the cycle (purity). Without clustering, purity is cycle / whole kind.
  replay   the pipeline's own cycle search and tests (shape, chance with 1000 shuffles at the recipe's
           p <= 0.002, prediction on the learner's held-back years) on every cluster, exactly as they run
           on a kind today, nothing loosened. A pass that matches the planted cycle is a find; a pass
           that matches nothing planted is a false find; one matching a planted decoy is a decoy.
The threshold "none" is today's behaviour: the whole kind is one group.
Writes results/vectors/clusters_<embedder>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learner import recipe as R  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402
from learner.patterns import best_periodic  # noqa: E402
from learner.vectors import BgeEmbedder, cluster, description  # noqa: E402


def shape_ok(f: dict) -> bool:
    return f["hits"] >= 4 and f["max_gap"] <= R.p("max_gap") and f["misses"] <= 0.35 * f["hits"]


def chance_ok(pool: list[int], n_years: int, hits: int, first: int, last: int, seed: str) -> tuple[bool, float]:
    """verify.periodic_chance on a year list: shuffled sets of the same size must rarely fit as well."""
    shuffles, allowed = R.p("shuffles"), int(R.p("shuffle_p") * R.p("shuffles"))
    rng, good = random.Random(seed), 0
    for i in range(shuffles):
        ys = rng.sample(pool, min(n_years, len(pool)))
        good += any(shape_ok(f) for f in best_periodic(ys, first, last, min_hits=hits))
        if good > allowed:
            return False, (1 + good) / (2 + i)
    return True, (1 + good) / (1 + shuffles)


def prediction_ok(years: list[int], period: float, first: int, last: int) -> bool:
    """verify.periodic_prediction on a year list: refit before the learner's own cut, predict after it."""
    cut = round(last - R.p("holdback") * (last - first))
    early = [y for y in years if y < cut]
    fits = [f for f in best_periodic(early, first, cut - 1, min_hits=3)
            if abs(f["period"] - period) <= max(2.0, 0.05 * period) and f["max_gap"] <= R.p("max_gap")]
    if not fits:
        return False
    f = fits[0]
    tol = 3 + 0.02 * f["period"]
    n = math.ceil((cut - f["anchor"]) / f["period"])
    predicted = []
    while f["anchor"] + n * f["period"] <= last + tol:
        predicted.append(round(f["anchor"] + n * f["period"]))
        n += 1
    if not predicted:
        return True
    came = [p for p in predicted if any(abs(y - p) <= tol for y in years)]
    return len(came) >= 1 and len(came) >= len(predicted) - len(came)


def what_it_matches(fit_years: set[int], period: float, truth: dict, year_of: dict) -> str:
    for kind, items in (("find", truth["patterns"]), ("decoy", truth["decoys"])):
        for p in items:
            if p["type"] != "periodic":
                continue
            ys = {year_of[r] for r in p["record_ids"] if r in year_of}
            if abs(p["params"]["period"] - period) <= max(3.0, 0.06 * p["params"]["period"]) and len(fit_years & ys) >= 0.5 * len(fit_years):
                return f"{kind}:{p['id']}"
    return "false"


def replay(groups: list[list], pool: list[int], first: int, last: int, truth: dict, year_of: dict, tag: str) -> list[str]:
    """Run the cycle search and tests on each group of entries; return what each passing cycle matches."""
    out = []
    for gi, entries in enumerate(groups):
        years = sorted({e.year for e in entries})
        if len(years) < 4:
            continue
        for f in best_periodic(years, first, last, min_hits=4):
            if not shape_ok(f):
                continue
            ok, _ = chance_ok(pool, len(years), f["hits"], first, last, seed=f"{tag}:{gi}:{round(f['period'])}")
            if ok and prediction_ok(years, f["period"], first, last):
                out.append(what_it_matches(set(f["matched"].values()), f["period"], truth, year_of))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="*", default=[f"val-{i:03d}" for i in range(5)])
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.75, 0.8, 0.85, 0.9])
    args = ap.parse_args()
    R.use(R.RECIPES["default@2"])
    emb = BgeEmbedder()
    report = {"embedder": "bge-small@1", "recipe": "default@2", "worlds": args.worlds, "cycles": [], "by_threshold": {}}
    tally = {t: {"finds": set(), "false": 0, "decoys": 0, "groups": 0} for t in ["none"] + args.thresholds}
    for w in args.worlds:
        wd = ROOT / "worlds" / "val" / w
        truth = json.loads((wd / "truth.json").read_text())
        year_of = {json.loads(line)["id"]: json.loads(line)["year"] for line in (wd / "records.jsonl").read_text().splitlines()}
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "nb.sqlite"
            shutil.copy(ROOT / "results" / "runs" / "phase3" / f"{w}.sqlite", db)
            k = Knowledge(sqlite3.connect(db, isolation_level=None))
        pool = [e.year for e in k.entries]
        seen_kinds = set()
        for p in truth["patterns"]:
            if p["type"] != "periodic":
                continue
            cyc = set(p["record_ids"])
            kinds = [k.by_id[r].kind for r in cyc if r in k.by_id]
            if not kinds:
                continue
            kind = max(set(kinds), key=kinds.count)
            entries = [e for e in k.entries if e.kind == kind]
            vecs = emb.embed([description(e) for e in entries])
            row = {"world": w, "pattern": p["id"], "kind": kind, "period": p["params"]["period"],
                   "cycle_in_kind": sum(e.record_id in cyc for e in entries), "kind_size": len(entries), "by_threshold": {}}
            for t in ["none"] + args.thresholds:
                labels = [0] * len(entries) if t == "none" else cluster(vecs, t)
                groups: dict[int, list] = {}
                for e, lab in zip(entries, labels):
                    groups.setdefault(lab, []).append(e)
                best = max(groups.values(), key=lambda g: sum(e.record_id in cyc for e in g))
                inside = sum(e.record_id in cyc for e in best)
                row["by_threshold"][str(t)] = {"clusters": len(groups), "best_cluster_size": len(best),
                                               "cycle_recall": round(inside / max(1, row["cycle_in_kind"]), 2),
                                               "purity": round(inside / len(best), 2)}
                if kind not in seen_kinds:   # replay each kind once per threshold (two cycles can share a kind)
                    res = replay(list(groups.values()), pool, k.first_year, k.last_year, truth, year_of, f"{w}:{kind}:{t}")
                    tally[t]["finds"] |= {f"{w}:{r}" for r in res if r.startswith("find")}
                    tally[t]["false"] += sum(r == "false" for r in res)
                    tally[t]["decoys"] += sum(r.startswith("decoy") for r in res)
                    tally[t]["groups"] += sum(len({e.year for e in g}) >= 4 for g in groups.values())
            seen_kinds.add(kind)
            report["cycles"].append(row)
            print(f"{w} {p['id']} {kind:10} cycle {row['cycle_in_kind']:2}/{row['kind_size']:2} | " +
                  "  ".join(f"{t}: {v['clusters']}cl recall {v['cycle_recall']} purity {v['purity']}" for t, v in row["by_threshold"].items()), flush=True)
    planted = sum(1 for w in args.worlds for p in json.loads((ROOT / "worlds" / "val" / w / "truth.json").read_text())["patterns"] if p["type"] == "periodic")
    for t, v in tally.items():
        report["by_threshold"][str(t)] = {"cycles_found": len(v["finds"]), "of": planted, "false_cycles": v["false"],
                                          "decoys": v["decoys"], "groups_tested": v["groups"], "found": sorted(v["finds"])}
    out = ROOT / "results" / "vectors" / "clusters_bge-small.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\nreplay (cycle search + shape, chance and prediction tests, unchanged):")
    for t, v in report["by_threshold"].items():
        print(f"  threshold {t:>5}: {v['cycles_found']}/{v['of']} planted cycles pass | false cycles {v['false_cycles']} | decoys {v['decoys']} | groups tested {v['groups_tested']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
