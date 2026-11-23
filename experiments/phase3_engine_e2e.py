"""Phase 3 end to end: the whole learner (day reading + dreaming) under the cycle engine, real model,
real sensors.

1. Load the 41-record sample world into a fresh brain database.
2. Start the engine with the `learner` workload, with "night" starting now: it reads every record
   first (you can't dream about what you haven't read), then dreams.
3. Once it is dreaming, a "game" grabs 1 GB of GPU memory for 25 s: the learner must rest and free the
   GPU, then come back and carry on dreaming.
4. When there's nothing left to dream about, it must rest.
5. Check: one note per record, no work item done twice, theories confirmed and scored against the
   answer key, and the timeline explains every switch.

  uv run python experiments/phase3_engine_e2e.py
"""

from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pynvml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chronicles.scorer import score_world  # noqa: E402
from experiments.phase2_engine_e2e import MB, count_notes, gpu_mb, tree, wait  # noqa: E402
from learner import notebook  # noqa: E402
from learner import theories as th  # noqa: E402

WORLD = ROOT / "results" / "phase1" / "blind" / "world_A"
RUN = ROOT / "results" / "runs" / "phase3_e2e"


def q(db: Path, sql: str, default=0):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql).fetchone()[0]
    except sqlite3.OperationalError:
        return default
    finally:
        conn.close()


def state(db: Path) -> tuple[str, str]:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT state, reason FROM timeline ORDER BY id DESC LIMIT 1").fetchone()
        return row or ("", "")
    except sqlite3.OperationalError:
        return ("", "")
    finally:
        conn.close()


def main() -> int:
    pynvml.nvmlInit()
    RUN.mkdir(parents=True, exist_ok=True)
    db = RUN / "brain.sqlite"
    for f in RUN.glob("brain.*"):
        f.unlink()
    conn = sqlite3.connect(db, isolation_level=None)
    total = notebook.ingest_world(conn, WORLD)
    conn.close()
    log = open(RUN / "engine.log", "w")
    engine = subprocess.Popen([sys.executable, "-m", "engine", "run", "--db", str(db), "--workload", "learner",
                               "--night-now", "--min-state", "0"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    result: dict = {"records": total}
    started = time.monotonic()
    try:
        if wait(lambda: count_notes(db) >= total, timeout=900, step=1.0) is None:
            print("FAIL: not every record read after 15 min. See", RUN / "engine.log")
            return 1
        result["all_read_after_s"] = round(time.monotonic() - started, 1)
        print(f"all {total} records read after {result['all_read_after_s']} s")
        if wait(lambda: q(db, "SELECT COUNT(*) FROM work_queue WHERE state='dream' AND status='done'") >= 3,
                timeout=600, step=0.5) is None:
            print("FAIL: no dreaming after reading. See", RUN / "engine.log")
            return 1
        result["dreaming_after_s"] = round(time.monotonic() - started, 1)
        print(f"dreaming after {result['dreaming_after_s']} s ({state(db)[1]})")

        hog = subprocess.Popen([sys.executable, "-m", "engine", "hog", "--mb", "1000", "--seconds", "25"],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        wait(lambda: any(p.pid == hog.pid and (p.usedGpuMemory or 0) / MB > 500
                         for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)), timeout=60)
        t0 = time.monotonic()
        freed = wait(lambda: gpu_mb(tree(engine.pid)) == 0, timeout=40, step=0.05)
        result["gpu_freed_after_s"] = None if freed is None else round(time.monotonic() - t0, 1)
        print(f"game started -> learner freed the GPU after {result['gpu_freed_after_s']} s")
        hog.wait(timeout=90)
        t1 = time.monotonic()
        back = wait(lambda: gpu_mb(tree(engine.pid)) > 1000, timeout=180)
        result["back_after_game_s"] = None if back is None else round(time.monotonic() - t1, 1)
        print(f"game ended   -> learner back after {result['back_after_game_s']} s ({state(db)[1]})")

        rested = wait(lambda: state(db)[0] == "rest" and gpu_mb(tree(engine.pid)) == 0, timeout=1800, step=1.0)
        result["rests_when_done"] = rested is not None
        result["finished_after_s"] = round(time.monotonic() - started, 1)
        print(f"resting after {result['finished_after_s']} s: {state(db)[1]}")
    finally:
        engine.send_signal(signal.SIGTERM)
        try:
            engine.wait(timeout=30)
        except subprocess.TimeoutExpired:
            engine.kill()
        log.close()

    conn = sqlite3.connect(db)
    result["notes"] = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    result["work_items"] = {f"{s}/{st}": n for s, st, n in conn.execute(
        "SELECT state, status, COUNT(*) FROM work_queue GROUP BY state, status")}
    planned = [i for row in conn.execute("SELECT payload FROM work_queue WHERE state='day' AND status='done'")
               for i in json.loads(row[0])["ids"]]
    result["records_planned_twice"] = len(planned) - len(set(planned))
    asked = [(r[0], r[1]) for r in conn.execute("SELECT record_id, question FROM reread_judgments")]
    result["questions_answered"] = len(asked)
    result["active_seconds"] = {k.split(".", 1)[1]: round(json.loads(v), 1) for k, v in conn.execute(
        "SELECT key, value FROM engine_kv WHERE key LIKE 'active_seconds.%'")}
    stages: dict[str, int] = {}
    for t in th.all_theories(conn):
        stages[t["stage"]] = stages.get(t["stage"], 0) + 1
    result["theories"] = stages
    truth = json.loads((WORLD / "truth.json").read_text())
    s = score_world(truth, th.confirmed_claims(conn))
    result["score"] = {k: s[k] for k in ("claims", "found", "planted", "recall", "precision", "decoy_hits",
                                         "predictions", "predictions_correct")}
    result["confirmed"] = [t["summary"] for t in th.all_theories(conn, ("confirmed",))]
    result["timeline"] = [dict(zip(("state", "reason", "start", "end"), r)) for r in conn.execute(
        "SELECT state, reason, started_at, ended_at FROM timeline ORDER BY id")]
    conn.close()
    ok = (result["notes"] == total and result["records_planned_twice"] == 0
          and (result.get("gpu_freed_after_s") or 99) <= 10 and result.get("back_after_game_s") is not None
          and result.get("rests_when_done") and stages.get("confirmed", 0) >= 1)
    result["pass"] = bool(ok)
    out = ROOT / "results" / "phase3" / "engine_e2e.json"
    out.write_text(json.dumps(result, indent=2))
    for row in result["timeline"]:
        span = f"{(row['end'] or row['start']) - row['start']:6.1f}s"
        print(f"  {row['state']:<10} {span}  {row['reason']}")
    print(json.dumps({k: v for k, v in result.items() if k != "timeline"}, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
