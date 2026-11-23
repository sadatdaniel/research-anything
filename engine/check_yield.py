"""Go/no-go check on the real GPU: does the engine hand the GPU back when another program needs it?

1. start the engine with a workload that holds real GPU memory (like a model would);
2. start a "game" (another process grabbing 2 GB of GPU memory);
3. measure how long until the learner holds no GPU memory at all;
4. stop the game and measure how long until the learner is back at work.

Pass: the GPU is handed back within 10 s.   Run:  uv run python -m engine.check_yield
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pynvml

ROOT = Path(__file__).resolve().parents[1]
MB = 1024 * 1024


def gpu_mb_by_pid() -> dict[int, float]:
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    out: dict[int, float] = {}
    for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
        for p in getter(h):
            out[p.pid] = (p.usedGpuMemory or 0) / MB
    return out


def tree(pid: int) -> set[int]:
    try:
        proc = psutil.Process(pid)
        return {pid} | {c.pid for c in proc.children(recursive=True)}
    except psutil.Error:
        return set()


def held_by(pids: set[int]) -> float:
    return sum(mb for pid, mb in gpu_mb_by_pid().items() if pid in pids)


def wait_for(predicate, timeout: float, step: float = 0.1) -> float | None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return time.monotonic() - start
        time.sleep(step)
    return None


def main() -> int:
    pynvml.nvmlInit()
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        print("The laptop is on battery, so the engine would (correctly) rest. Plug it in and rerun.")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="yield-check-"))
    db = tmp / "brain.sqlite"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    engine = subprocess.Popen(
        [sys.executable, "-m", "engine", "run", "--db", str(db), "--workload", "torch-hold",
         "--hold-mb", "1500", "--min-state", "0",
         "--workload-args", '{"day_items": 100000, "seconds_per_item": 0.2}'],
        cwd=ROOT, env=env, stdout=open(tmp / "engine.log", "w"), stderr=subprocess.STDOUT,
    )
    hog = None
    try:
        loaded = wait_for(lambda: held_by(tree(engine.pid)) >= 1000, timeout=120)
        if loaded is None:
            print("FAIL: the learner never loaded onto the GPU. Log:\n" + (tmp / "engine.log").read_text())
            return 1
        print(f"learner loaded on the GPU after {loaded:.1f} s, holding {held_by(tree(engine.pid)):.0f} MB")

        hog = subprocess.Popen([sys.executable, "-m", "engine", "hog", "--mb", "2000", "--seconds", "25"],
                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if wait_for(lambda: gpu_mb_by_pid().get(hog.pid, 0) >= 1500, timeout=60) is None:
            print("FAIL: the test 'game' could not grab GPU memory")
            return 1
        t0 = time.monotonic()
        handed_back = wait_for(lambda: held_by(tree(engine.pid)) == 0, timeout=30, step=0.05)
        if handed_back is None:
            print("FAIL: the learner kept its GPU memory for 30 s after the game started")
            return 1
        handed_back = time.monotonic() - t0
        print(f"game started -> learner held 0 MB after {handed_back:.1f} s")

        hog.wait(timeout=60)
        t1 = time.monotonic()
        back = wait_for(lambda: held_by(tree(engine.pid)) >= 1000, timeout=120)
        print(f"game ended   -> learner back at work after "
              f"{'never' if back is None else f'{time.monotonic() - t1:.1f} s'}")
    finally:
        if hog is not None and hog.poll() is None:
            hog.kill()
        engine.send_signal(signal.SIGTERM)
        try:
            engine.wait(timeout=20)
        except subprocess.TimeoutExpired:
            engine.kill()

    status = subprocess.run([sys.executable, "-m", "engine", "status", "--db", str(db)],
                            cwd=ROOT, capture_output=True, text=True).stdout
    print("\n" + status)
    ok = handed_back <= 10.0 and back is not None
    print(f"{'PASS' if ok else 'FAIL'}: GPU handed back in {handed_back:.1f} s (target <= 10 s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
