"""The cycle engine's main loop: read the sensors, decide, start/switch/stop the worker, log it all.

Ticks every second, so a hard stop (someone else needs the GPU, too hot, low RAM, battery) reaches
the worker within about a second, and the worker is gone within the stop grace period after that.
"""

from __future__ import annotations

import signal
import threading
import time
from datetime import datetime
from typing import Callable

from engine.lock import FolderLock
from engine.policy import Decision, Policy, State
from engine.store import Store, reason_key
from engine.worker import WorkerHandle


class Scheduler:
    def __init__(self, store: Store, sensors, policy: Policy, planner, worker: WorkerHandle,
                 clock: Callable[[], float] = time.time,
                 now: Callable[[], datetime] = datetime.now,
                 max_crashes: int = 3, crash_window_seconds: float = 600.0) -> None:
        self.store = store
        self.sensors = sensors
        self.policy = policy
        self.planner = planner  # workload instance used only for has_work(); never touches the GPU
        self.worker = worker
        self.clock = clock
        self.now = now
        self.max_crashes = max_crashes
        self.crash_window_seconds = crash_window_seconds
        self._crashes: list[float] = []
        self._last_exit_code: int | None = None
        self._last_tick: float | None = None
        self.last_stop_seconds: float | None = None  # how long the last hand-back of the GPU took

    # ---- one tick ---------------------------------------------------------------------------

    def tick(self) -> Decision:
        t = self.clock()
        snap = self.sensors.read()
        current = self.store.current_state()
        decision = self.policy.decide(
            snap, self.now(), self.store.pressure(), current,
            t - self.store.entered_at(t), self.planner.has_work(self.store, State.DAY),
            self.planner.has_work(self.store, State.DREAM),
        )
        decision = self._check_crashes(current, decision, t)

        if decision.state is not current:
            self._apply(decision, t)
        elif decision.hard_stop and reason_key(decision.reason) != reason_key(self.store.current_reason()):
            self.store.enter_state(decision.state, decision.reason, decision.budget, t)  # same state, new cause
        elif decision.state is not State.REST:
            if self.worker.alive():
                self.worker.set_budget(decision.budget)
            else:
                self._start_worker(decision)

        dt = 0.0 if self._last_tick is None else t - self._last_tick
        self.store.heartbeat(t, decision.state, dt)
        self._last_tick = t
        return decision

    def _check_crashes(self, current: State, decision: Decision, t: float) -> Decision:
        """A worker that exits on its own while it should be working has crashed.
        Too many crashes in the window keep the learner resting until the window has passed."""
        if current is not State.REST and self.worker.pid is not None and not self.worker.alive():
            self._last_exit_code = self.worker.exit_code()
            self.worker.stop()  # reap it; it's restarted below if allowed
            self._crashes.append(t)
        self._crashes = [c for c in self._crashes if t - c < self.crash_window_seconds]
        if len(self._crashes) >= self.max_crashes:
            minutes = self.crash_window_seconds / 60
            return Decision(State.REST, f"worker crashed {len(self._crashes)} times in {minutes:.0f} min "
                                        f"(last exit code {self._last_exit_code}); see the logs",
                            0.0, hard_stop=True)
        return decision

    def _start_worker(self, decision: Decision) -> None:
        self.worker.start(decision.state, decision.budget)
        if self.worker.pid is not None:
            self.sensors.register_own_pid(self.worker.pid)

    def _apply(self, decision: Decision, t: float) -> None:
        if decision.state is State.REST:
            pid = self.worker.pid
            self.last_stop_seconds = self.worker.stop()
            if pid is not None:
                self.sensors.unregister_own_pid(pid)
        elif self.worker.alive():
            self.worker.switch(decision.state, decision.budget)
        else:
            self._start_worker(decision)
        self.store.enter_state(decision.state, decision.reason, decision.budget, t)

    # ---- the loop ---------------------------------------------------------------------------

    def run(self, stop_event: threading.Event | None = None, tick_seconds: float = 1.0,
            max_seconds: float | None = None, on_tick: Callable[[Decision], None] | None = None) -> None:
        stop_event = stop_event or threading.Event()
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stop_event.set())
        with FolderLock(self.store.path.with_suffix(".lock")):
            self.store.recover(self.clock())
            started = time.monotonic()
            try:
                while not stop_event.is_set():
                    decision = self.tick()
                    if on_tick is not None:
                        on_tick(decision)
                    if max_seconds is not None and time.monotonic() - started >= max_seconds:
                        break
                    stop_event.wait(tick_seconds)
            finally:
                self.worker.stop()
                self.store.mark_off(self.clock())
