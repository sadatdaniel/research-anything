"""Phase 3: let the learner dream on a world it has already read, and score what it confirms over time.

  uv run python experiments/phase3_dream.py worlds/val/val-000 [--max-minutes 60] [--tag run1]

It copies the world's notebook (results/runs/phase3/<world>.sqlite, made by phase3_read_worlds.py) to
results/runs/phase3/dream/<world>-<tag>.sqlite and runs dream mode exactly as the engine's worker does
(plan when the queue is empty, run one item, save it in one transaction), without the scheduler. Every
second of planning and model work counts as active dream time, like the engine's heartbeat.

Writes results/phase3/dream_<world>-<tag>.json: recall and precision of the confirmed theories against
active dream time (the H1 curve), the B1 baseline (every proposal, untested) at the end, and timings.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronicles.scorer import score_world  # noqa: E402
from engine.policy import State  # noqa: E402
from engine.store import Store  # noqa: E402
from engine.workload import WorkContext  # noqa: E402
from learner import theories as th  # noqa: E402
from learner.dream import DreamWorkload  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402
from learner.patterns import propose  # noqa: E402

RUNS = ROOT / "results" / "runs" / "phase3"


def summary(score: dict) -> dict:
    return {k: score[k] for k in ("claims", "found", "planted", "recall", "precision", "decoy_hits",
                                  "false_claims", "predictions", "predictions_correct")} | \
        {"per_type": score["per_type"]}


def curve(conn, truth: dict, end: float, step: float) -> list[dict]:
    points, t = [], 0.0
    while True:
        s = score_world(truth, th.claims_at(conn, t))
        points.append({"dream_seconds": round(t, 1), **summary(s)})
        if t >= end:
            return points
        t = min(end, t + step)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="start from the plain notebook even if a run exists")
    args = ap.parse_args()
    world = Path(args.world)
    src = RUNS / f"{world.name}.sqlite"
    db = RUNS / "dream" / f"{world.name}-{args.tag}.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh or not db.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        shutil.copy(src, db)
    store = Store(db)
    wl = DreamWorkload(loops=args.loops)
    t_load = time.perf_counter()
    wl.load(State.DREAM)
    load_s = time.perf_counter() - t_load
    ctx = WorkContext(state=State.DREAM, budget=lambda: 1.0, should_stop=lambda: False)
    active = float(store.get("active_seconds.dream", 0.0))
    budget = args.max_minutes * 60
    counts = {"plans": 0, "items": 0, "questions": 0, "plan_seconds": 0.0, "model_seconds": 0.0}
    log_every, last_log = 60.0, -1e9
    while active < budget:
        t0 = time.perf_counter()
        item = store.next_pending(State.DREAM)
        if item is None:
            items = wl.plan(store, State.DREAM)
            counts["plans"] += 1
            dt = time.perf_counter() - t0
            counts["plan_seconds"] += dt
            active += dt
            store.set("active_seconds.dream", active)
            if not items:
                break
            store.enqueue(State.DREAM, items)
            continue
        n_q = len(item.payload.get("pairs") or item.payload.get("questions") or [])
        outcome = store.run_one(State.DREAM, wl, ctx)
        dt = time.perf_counter() - t0
        counts["model_seconds"] += dt
        counts["items"] += outcome == "done"
        counts["questions"] += n_q
        active += dt
        store.set("active_seconds.dream", active)
        if active - last_log >= log_every:
            last_log = active
            stages = {}
            for t in th.all_theories(store.conn):
                stages[t["stage"]] = stages.get(t["stage"], 0) + 1
            print(f"  {active:7.0f}s dream: {counts['items']} items, {counts['questions']} questions; theories {stages}",
                  flush=True)
    truth = json.loads((world / "truth.json").read_text())
    final = score_world(truth, th.confirmed_claims(store.conn))
    k = Knowledge(store.conn)
    b1 = score_world(truth, [c.to_claim() for c in propose(k)])
    report = {
        "world": str(world), "tag": args.tag, "finished": active < budget, "dream_seconds": round(active, 1),
        "model_load_seconds": round(load_s, 1), "counts": {k_: round(v, 1) for k_, v in counts.items()},
        "confirmed": summary(final), "b1_all_proposals": summary(b1),
        "curve": curve(store.conn, truth, active, step=max(10.0, active / 40)),
        "theories": [{k_: t[k_] for k_ in ("id", "type", "summary", "stage", "reason", "dream_seconds")}
                     for t in th.all_theories(store.conn) if t["stage"] != "rejected" or t["type"] != "periodic"],
        "matched": [{"claim": r["claim"], "matched": r["matched"], "decoy": r["decoy"]} for r in final["rows"]],
    }
    out = ROOT / "results" / "phase3" / f"dream_{world.name}-{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k_: report[k_] for k_ in ("dream_seconds", "finished", "counts", "confirmed", "b1_all_proposals")},
                     indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
