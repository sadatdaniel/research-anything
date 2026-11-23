"""The worker process: the only process that touches the GPU.

Why a separate process: PyTorch keeps a few hundred MB of GPU memory (its CUDA context) until the
process exits, so the only way to truly hand the GPU back is to end the process. REST therefore means
"stop the worker": politely first (it finishes or abandons its current item), then terminate, then kill.

The worker also dies with the engine: on Linux the kernel kills it the moment the engine process dies
(PR_SET_PDEATHSIG), and on every platform it checks that its parent is still alive between items.
An orphaned worker could otherwise keep running next to a restarted engine.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import signal
import sys
import time
from dataclasses import dataclass, field

from engine.policy import State

STATES = list(State)


def _die_with_parent() -> None:
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except OSError:
            pass


def _worker_main(db_path: str, workload_ref: str, workload_kwargs: dict,
                 state_index, budget, stop) -> None:
    _die_with_parent()
    parent = mp.parent_process()

    from engine.store import Store
    from engine.workload import WorkContext, make_workload

    workload = make_workload(workload_ref, workload_kwargs)
    store = Store(db_path)
    loaded: State | None = None

    def should_stop() -> bool:
        return stop.is_set() or (parent is not None and not parent.is_alive())

    try:
        while not should_stop():
            state = STATES[state_index.value]
            if state is State.REST:
                break
            if state is not loaded:
                workload.load(state)
                loaded = state
            ctx = WorkContext(state=state, budget=lambda: budget.value, should_stop=should_stop)
            outcome = store.run_one(state, workload, ctx)
            if outcome == "empty":
                new_items = workload.plan(store, state)
                if new_items:
                    store.enqueue(state, new_items)
                else:
                    stop.wait(0.5)
    finally:
        workload.unload()
        store.close()


@dataclass
class WorkerHandle:
    """Engine-side handle: start, switch state, and stop the worker."""

    db_path: str
    workload_ref: str
    workload_kwargs: dict = field(default_factory=dict)
    grace_seconds: float = 6.0

    def __post_init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._state_index = self._ctx.Value("i", STATES.index(State.REST))
        self._budget = self._ctx.Value("d", 0.0)
        self._stop = self._ctx.Event()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def exit_code(self) -> int | None:
        return None if self._proc is None else self._proc.exitcode

    def start(self, state: State, budget: float) -> None:
        if self.alive():
            self.switch(state, budget)
            return
        self._stop.clear()
        self.switch(state, budget)
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self.db_path, self.workload_ref, self.workload_kwargs,
                  self._state_index, self._budget, self._stop),
            name="learner-worker",
            daemon=True,
        )
        self._proc.start()

    def switch(self, state: State, budget: float) -> None:
        self._state_index.value = STATES.index(state)
        self._budget.value = budget

    def set_budget(self, budget: float) -> None:
        self._budget.value = budget

    def stop(self) -> float:
        """Stop the worker and return how long it took (seconds). Always ends with the process gone."""
        if self._proc is None:
            return 0.0
        started = time.monotonic()
        self._stop.set()
        self._proc.join(self.grace_seconds)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.kill()  # SIGKILL on Linux, TerminateProcess on Windows
            self._proc.join()
        self._proc = None
        return time.monotonic() - started
