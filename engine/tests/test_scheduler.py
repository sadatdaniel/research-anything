"""Scheduler with a real worker process (CPU-only dummy workload) and fake sensors."""

import time
from dataclasses import replace
from datetime import datetime

from engine.policy import Policy, PolicyConfig, State
from engine.scheduler import Scheduler
from engine.sensors import FakeSensors, GpuProcess, Snapshot
from engine.store import Store
from engine.workload import DummyWorkload
from engine.worker import WorkerHandle

REF = "engine.workload:DummyWorkload"


class Clock:
    def __init__(self, hour: int):
        self.hour = hour

    def __call__(self) -> datetime:
        return datetime(2026, 11, 23, self.hour, 0)


class LiveFake(FakeSensors):
    def read(self):
        return replace(self.snapshot, taken_at=time.time())


def make(tmp_path, hour=14, **wl_kwargs):
    kwargs = {"day_items": 5, "seconds_per_item": 0.05, **wl_kwargs}
    store = Store(tmp_path / "brain.sqlite")
    sensors = LiveFake(Snapshot(taken_at=0, gpu_present=True, vram_total_mb=6141, vram_free_mb=6000,
                                gpu_temp_c=55, ram_available_mb=8000, cpu_others_pct=5, on_ac=True))
    clock = Clock(hour)
    policy = Policy(PolicyConfig(min_state_seconds=0, night_idle_seconds=0, sleep_pressure_threshold=1.0))
    sched = Scheduler(store, sensors, policy, DummyWorkload(**kwargs),
                      WorkerHandle(str(store.path), REF, kwargs, grace_seconds=3), now=clock)
    return sched, store, sensors, clock


def tick_until(sched, predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        decision = sched.tick()
        if predicate(decision):
            return decision
        time.sleep(0.1)
    raise AssertionError("condition not reached")


def test_reads_by_day_then_rests_when_done(tmp_path):
    sched, store, *_ = make(tmp_path)
    assert sched.tick().state is State.DAY
    assert sched.worker.alive()
    tick_until(sched, lambda d: d.state is State.REST and "nothing new" in d.reason)
    assert not sched.worker.alive()
    assert store.queue_counts()["day"] == {"done": 5}


def test_hard_stop_hands_the_gpu_back_fast(tmp_path):
    sched, store, sensors, _ = make(tmp_path, day_items=10_000)
    assert sched.tick().state is State.DAY
    time.sleep(1.0)  # worker is busy
    sensors.set(others_on_gpu=(GpuProcess(999, "game", 3000.0),))
    d = sched.tick()
    assert d.state is State.REST and "game" in d.reason
    assert not sched.worker.alive()
    assert sched.last_stop_seconds < 3.0  # stopped politely, within the grace period
    sensors.set(others_on_gpu=())
    assert sched.tick().state is State.DAY  # back to work as soon as the game is gone


def test_night_cycle_dream_sleep_dream(tmp_path):
    sched, store, _, clock = make(tmp_path, hour=2, pressure_per_dream=0.25, pressure_per_sleep=0.5)
    assert sched.tick().state is State.DREAM
    tick_until(sched, lambda d: d.state is State.DEEP_SLEEP)  # pressure reached 1.0
    tick_until(sched, lambda d: d.state is State.DREAM)  # consolidated back to 0
    states = [r["state"] for r in store.timeline()]
    assert states[:3] == ["dream", "deep_sleep", "dream"]
    assert store.active_seconds()["dream"] > 0 and store.active_seconds()["deep_sleep"] > 0


def test_crashing_worker_is_restarted_then_benched(tmp_path):
    sched, store, *_ = make(tmp_path, day_items=10_000)
    sched.max_crashes = 2
    assert sched.tick().state is State.DAY
    for _ in range(2):
        time.sleep(0.5)
        sched.worker._proc.kill()
        sched.worker._proc.join()
        d = sched.tick()
    assert d.state is State.REST and "crashed 2 times" in d.reason
