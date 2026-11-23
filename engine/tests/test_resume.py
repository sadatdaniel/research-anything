"""kill -9 the whole engine in the middle of work, restart it, and check nothing was lost or done twice."""

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = 60


def start_engine(db: Path, max_seconds: float) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "engine", "run", "--db", str(db), "--fake-sensors",
           "--min-state", "0", "--tick", "0.2", "--max-seconds", str(max_seconds),
           "--workload-args", f'{{"day_items": {ITEMS}, "seconds_per_item": 0.05}}']
    return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def done_rows(db: Path) -> list[int]:
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute("SELECT item_id FROM dummy_results ORDER BY item_id")]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def worker_pids(engine_pid: int) -> list[int]:
    out = subprocess.run(["pgrep", "-P", str(engine_pid)], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def test_kill_minus_9_then_resume_loses_and_duplicates_nothing(tmp_path):
    db = tmp_path / "brain.sqlite"
    first = start_engine(db, max_seconds=60)
    deadline = time.monotonic() + 30
    while len(done_rows(db)) < 10 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(done_rows(db)) >= 10, first.stdout.read() if first.poll() is not None else "engine too slow"
    workers = worker_pids(first.pid)
    os.kill(first.pid, signal.SIGKILL)  # no cleanup, no goodbye
    first.wait()
    # The worker must die with the engine (PR_SET_PDEATHSIG), not keep running as an orphan.
    time.sleep(0.5)
    for pid in workers:
        assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split()[2]
    before = done_rows(db)
    assert 10 <= len(before) < ITEMS

    second = start_engine(db, max_seconds=30)
    deadline = time.monotonic() + 30
    while len(done_rows(db)) < ITEMS and time.monotonic() < deadline:
        time.sleep(0.1)
    second.send_signal(signal.SIGTERM)
    second.wait(timeout=15)

    rows = done_rows(db)
    assert sorted(rows) == list(range(1, ITEMS + 1)), "every item done exactly once"
    conn = sqlite3.connect(db)
    states = [r[0] for r in conn.execute("SELECT state FROM timeline ORDER BY id")]
    conn.close()
    assert "off" in states  # the gap while it was dead is recorded in the history
