"""A second, independent test for cycles: do the cycle's records describe the same thing?

  uv run python experiments/coherence_test.py [--notebooks ouro qwen3]

Validation worlds only. Hard clustering by description shattered cycles (results/vectors/clusters_bge-small.json),
so this keeps the whole kind and adds evidence instead:
  1. the cycle search runs on the whole kind, as today;
  2. for each candidate cycle that passes the shape rule, two p-values:
       chance     the existing test on the years (shuffled year sets, 1000 shuffles)
       coherence  the mean similarity of its records' description vectors, against random subsets of the
                  same size from the same kind (2000 draws)
     Under "nothing is there" descriptions carry no information about years, so the two are independent and
     Fisher's method combines them: X = -2(ln p1 + ln p2), chi-square with 4 degrees of freedom;
  3. it passes if the combined p <= the recipe's threshold (0.002), AND the prediction test still passes,
     which is what stops the planted decoys.
Vectors are either raw or centred per chronicler (each writer's average vector subtracted: the cheapest
way to take style out). Writes results/vectors/coherence.json.
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
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from learner import recipe as R  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402
from learner.patterns import best_periodic  # noqa: E402
from learner.vectors import BgeEmbedder, description  # noqa: E402
from vector_clusters import prediction_ok, shape_ok, what_it_matches  # noqa: E402

NOTEBOOK = {"ouro": "{w}.sqlite", "qwen3": "{w}.qwen3-1.7b-base.sqlite"}


def chance_p(pool: list[int], n_years: int, hits: int, first: int, last: int, seed: str, shuffles: int = 1000,
             stop_above: float = 1.0) -> float:
    """The chance test's p-value. Stops early once it's certain to exceed stop_above (the value can only
    grow), returning that lower bound: enough to decide pass or fail, without the remaining shuffles."""
    rng, good = random.Random(seed), 0
    for _ in range(shuffles):
        ys = rng.sample(pool, min(n_years, len(pool)))
        good += any(shape_ok(f) for f in best_periodic(ys, first, last, min_hits=hits))
        if (1 + good) / (1 + shuffles) > stop_above:
            break
    return (1 + good) / (1 + shuffles)


def product_needed(threshold: float) -> float:
    """The largest p1 * p2 that Fisher's method (4 degrees of freedom) still accepts at this threshold."""
    lo, hi = 0.0, 200.0
    for _ in range(100):
        x = (lo + hi) / 2
        if math.exp(-x / 2) * (1 + x / 2) > threshold:
            lo = x
        else:
            hi = x
    return math.exp(-hi / 2)


def coherence_p(vecs: np.ndarray, members: list[int], seed: str, draws: int = 2000) -> float:
    sim = vecs @ vecs.T
    def mean_pair(idx):
        sub = sim[np.ix_(idx, idx)]
        n = len(idx)
        return (sub.sum() - np.trace(sub)) / (n * (n - 1))
    obs = mean_pair(members)
    rng = random.Random(seed)
    n_all = len(vecs)
    ge = sum(mean_pair(rng.sample(range(n_all), len(members))) >= obs for _ in range(draws))
    return (1 + ge) / (1 + draws)


def fisher(p1: float, p2: float) -> float:
    x = -2 * (math.log(p1) + math.log(p2))
    return math.exp(-x / 2) * (1 + x / 2)   # chi-square survival function, 4 degrees of freedom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebooks", nargs="*", default=["ouro", "qwen3"])
    args = ap.parse_args()
    R.use(R.RECIPES["default@2"])
    threshold = R.p("shuffle_p")
    emb = BgeEmbedder()
    rows, tally = [], defaultdict(lambda: {"finds": set(), "false": 0, "decoys": 0})
    for nb in args.notebooks:
        worlds = [f"val-{i:03d}" for i in range(5)] if nb == "ouro" else ["val-003", "val-004"]
        for w in worlds:
            wd = ROOT / "worlds" / "val" / w
            truth = json.loads((wd / "truth.json").read_text())
            year_of = {json.loads(l)["id"]: json.loads(l)["year"] for l in (wd / "records.jsonl").read_text().splitlines()}
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "nb.sqlite"
                shutil.copy(ROOT / "results" / "runs" / "phase3" / NOTEBOOK[nb].format(w=w), db)
                conn = sqlite3.connect(db, isolation_level=None)
                k = Knowledge(conn)
                chron = {r[0]: r[1] for r in conn.execute("SELECT record_id, chronicler FROM sources")}
            pool, first, last = [e.year for e in k.entries], k.first_year, k.last_year
            kinds = {k.by_id[r].kind for p in truth["patterns"] if p["type"] == "periodic" for r in p["record_ids"] if r in k.by_id}
            # every kind big enough to hide a cycle in, not just the planted ones, so false finds can show
            sizes = defaultdict(int)
            for e in k.entries:
                sizes[e.kind] += 1
            for kind in sorted(kd for kd in sizes if sizes[kd] >= 8):
                entries = [e for e in k.entries if e.kind == kind]
                raw = emb.embed([description(e) for e in entries])
                by_c = defaultdict(list)
                for i, e in enumerate(entries):
                    by_c[chron.get(e.record_id)].append(i)
                centred = raw.copy()
                for idx in by_c.values():
                    centred[idx] -= raw[idx].mean(axis=0)
                centred /= np.linalg.norm(centred, axis=1, keepdims=True).clip(1e-9)
                years = sorted({e.year for e in entries})
                for f in best_periodic(years, first, last, min_hits=4):
                    if not shape_ok(f):
                        continue
                    fy = set(f["matched"].values())
                    members = [i for i, e in enumerate(entries) if e.year in fy]
                    if len(members) < 3:
                        continue
                    tag = f"{nb}:{w}:{kind}:{round(f['period'])}"
                    p_co = {name: coherence_p(vecs, members, tag + name) for name, vecs in (("raw", raw), ("centred", centred))}
                    # the chance p-value only needs to be exact up to the largest value any method could accept
                    need = max([threshold] + [min(1.0, product_needed(threshold) / p) for p in p_co.values()])
                    p_ch = chance_p(pool, len(years), f["hits"], first, last, tag, stop_above=need)
                    pred = prediction_ok(years, f["period"], first, last)
                    match = what_it_matches(fy, f["period"], truth, year_of)
                    row = {"notebook": nb, "world": w, "kind": kind, "period": round(f["period"], 1), "hits": f["hits"],
                           "kind_years": len(years), "matches": match, "prediction": pred, "p_chance": round(p_ch, 4)}
                    for name in ("raw", "centred"):
                        row[f"p_coherence_{name}"] = round(p_co[name], 4)
                        row[f"p_combined_{name}"] = round(fisher(p_ch, p_co[name]), 5)
                    rows.append(row)
                    for method, p in (("today (chance only)", p_ch), ("+ coherence, raw", row["p_combined_raw"]),
                                      ("+ coherence, centred", row["p_combined_centred"])):
                        if p <= threshold and pred:
                            t = tally[f"{nb}: {method}"]
                            if match.startswith("find"):
                                t["finds"].add(f"{w}:{match}")
                            elif match.startswith("decoy"):
                                t["decoys"] += 1
                            else:
                                t["false"] += 1
            print(f"  {nb} {w}: {sum(1 for r in rows if r['notebook'] == nb and r['world'] == w)} candidate cycles tested", flush=True)
    planted = {"ouro": 9, "qwen3": 4}
    summary = {m: {"cycles_found": len(v["finds"]), "of": planted[m.split(":")[0]], "false": v["false"], "decoys": v["decoys"],
                   "found": sorted(v["finds"])} for m, v in sorted(tally.items())}
    out = ROOT / "results" / "vectors" / "coherence.json"
    out.write_text(json.dumps({"threshold": threshold, "summary": summary, "candidates": rows}, indent=2))
    print("\nplanted cycles that pass (and anything false that passes):")
    for m, v in summary.items():
        print(f"  {m:32} {v['cycles_found']}/{v['of']} found | false {v['false']} | decoys {v['decoys']}")
    print("\ncandidates matching a planted cycle:")
    for r in rows:
        if r["matches"].startswith("find"):
            print(f"  {r['notebook']} {r['world']} {r['kind']:9} every {r['period']:5}y | p chance {r['p_chance']:.4f} | coherence raw {r['p_coherence_raw']:.4f} centred {r['p_coherence_centred']:.4f} | combined raw {r['p_combined_raw']:.5f} centred {r['p_combined_centred']:.5f} | prediction {r['prediction']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
