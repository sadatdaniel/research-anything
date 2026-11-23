"""Phase 2 end to end: the cycle engine runs day mode with the real model on real hardware.

1. Load the 41-record sample world into a fresh brain database.
2. Start the engine (real sensors, FocusedWorkload: Ouro takes notes at 4 loops).
3. Once some notes exist, a "game" grabs 1 GB of GPU memory for 25 s: the learner must rest and free the GPU.
4. When the game ends, the learner must resume and finish every record.
5. Check: exactly one note per record, no work item done twice, and the timeline explains every switch.

  uv run python experiments/phase2_engine_e2e.py
"""

from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pynvml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learner import notebook  # noqa: E402

WORLD = ROOT / "results" / "phase1" / "blind" / "world_A"
RUN = ROOT / "results" / "runs" / "phase2_e2e"
MB = 1024 * 1024


def gpu_mb(pids: set[int]) -> float:
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return sum((p.usedGpuMemory or 0) / MB for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid in pids)


def tree(pid: int) -> set[int]:
    try:
        return {pid} | {c.pid for c in psutil.Process(pid).children(recursive=True)}
    except psutil.Error:
        return set()


def count_notes(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def wait(pred, timeout: float, step: float = 0.2) -> float | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return time.monotonic() - t0
        time.sleep(step)
    return None


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
    engine = subprocess.Popen([sys.executable, "-m", "engine", "run", "--db", str(db), "--workload", "focused",
                               "--min-state", "0"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    result: dict = {"records": total}
    started = time.monotonic()
    try:
        t = wait(lambda: count_notes(db) >= 6, timeout=600)
        if t is None:
            print("FAIL: no notes after 10 min. See", RUN / "engine.log")
            return 1
        result["first_notes_after_s"] = round(time.monotonic() - started, 1)
        held = gpu_mb(tree(engine.pid))
        print(f"{count_notes(db)}/{total} notes after {result['first_notes_after_s']} s; learner holds {held:.0f} MB")

        hog = subprocess.Popen([sys.executable, "-m", "engine", "hog", "--mb", "1000", "--seconds", "25"],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        wait(lambda: any(p.pid == hog.pid and (p.usedGpuMemory or 0) / MB > 500
                         for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)), timeout=60)
        t0 = time.monotonic()
        freed = wait(lambda: gpu_mb(tree(engine.pid)) == 0, timeout=40, step=0.05)
        result["gpu_freed_after_s"] = None if freed is None else round(time.monotonic() - t0, 1)
        notes_at_rest = count_notes(db)
        print(f"game started -> learner freed the GPU after {result['gpu_freed_after_s']} s "
              f"({notes_at_rest} notes so far)")
        hog.wait(timeout=90)
        t1 = time.monotonic()
        back = wait(lambda: gpu_mb(tree(engine.pid)) > 1000, timeout=180)
        result["back_after_game_s"] = None if back is None else round(time.monotonic() - t1, 1)
        print(f"game ended   -> learner back at work after {result['back_after_game_s']} s")

        done = wait(lambda: count_notes(db) >= total, timeout=1200, step=1.0)
        result["all_notes_after_s"] = None if done is None else round(time.monotonic() - started, 1)
        rested = wait(lambda: gpu_mb(tree(engine.pid)) == 0, timeout=60)
        result["rests_when_done"] = rested is not None
    finally:
        engine.send_signal(signal.SIGTERM)
        try:
            engine.wait(timeout=30)
        except subprocess.TimeoutExpired:
            engine.kill()
        log.close()

    conn = sqlite3.connect(db)
    result["notes"] = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    result["distinct_records_noted"] = conn.execute("SELECT COUNT(DISTINCT record_id) FROM notes").fetchone()[0]
    items = conn.execute("SELECT status, COUNT(*) FROM work_queue GROUP BY status").fetchall()
    result["work_items"] = dict(items)
    planned = [i for row in conn.execute("SELECT payload FROM work_queue WHERE status='done'")
               for i in json.loads(row[0])["ids"]]
    result["records_planned_twice"] = len(planned) - len(set(planned))
    result["timeline"] = [dict(zip(("state", "reason", "start", "end"), r)) for r in conn.execute(
        "SELECT state, reason, started_at, ended_at FROM timeline ORDER BY id")]
    conn.close()
    ok = (result["notes"] == total and result["records_planned_twice"] == 0
          and (result.get("gpu_freed_after_s") or 99) <= 10 and result.get("back_after_game_s") is not None)
    result["pass"] = ok
    out = ROOT / "results" / "phase2" / "engine_e2e.json"
    out.write_text(json.dumps(result, indent=2))
    for row in result["timeline"]:
        span = f"{(row['end'] or row['start']) - row['start']:6.1f}s"
        print(f"  {row['state']:<10} {span}  {row['reason']}")
    print(json.dumps({k: v for k, v in result.items() if k != "timeline"}, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
