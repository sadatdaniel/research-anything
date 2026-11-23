"""What the engine runs: the contract a workload implements, plus dummy workloads for testing.

A workload is constructed in two places:
  * in the engine process, to answer has_work() and plan() - the constructor must NOT touch the GPU;
  * in the worker process, where load()/run()/unload() do the real (GPU) work.

run() computes a result without writing anything. save() writes the result inside the
engine's transaction and returns the change in sleep pressure (positive = more to consolidate,
negative = consolidated). That split is what makes every item happen exactly once.
"""

from __future__ import annotations

import importlib
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from engine.policy import State

if TYPE_CHECKING:
    from engine.store import Store, WorkItem


class Cancelled(Exception):
    """Raised inside run() when the engine asks the worker to stop."""


@dataclass
class WorkContext:
    state: State
    budget: Callable[[], float]  # 0..1, read it often: it changes as the machine gets busier or freer
    should_stop: Callable[[], bool]

    def check(self) -> None:
        if self.should_stop():
            raise Cancelled()


class Workload(Protocol):
    def has_work(self, store: "Store", state: State) -> bool: ...
    def plan(self, store: "Store", state: State) -> list[tuple[str, dict]]: ...
    def load(self, state: State) -> None: ...
    def unload(self) -> None: ...
    def run(self, item: "WorkItem", ctx: WorkContext) -> Any: ...
    def save(self, item: "WorkItem", result: Any, conn: sqlite3.Connection) -> float: ...


def make_workload(ref: str, kwargs: dict | None = None) -> Workload:
    """'package.module:ClassName' -> instance. Used to rebuild the workload inside the worker."""
    module_name, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module_name), attr)(**(kwargs or {}))


class DummyWorkload:
    """CPU-only stand-in for the learner, used by tests and dry runs.

    Day: a fixed number of "reading" items. Dream: endless "connection" items, each adding sleep
    pressure. Deep sleep: "consolidation" items that relieve pressure. Every item writes one row to
    dummy_results, which has no uniqueness constraint, so a duplicated item would show up as a duplicate row.
    """

    def __init__(self, day_items: int = 20, seconds_per_item: float = 0.2,
                 pressure_per_dream: float = 0.25, pressure_per_sleep: float = 0.5) -> None:
        self.day_items = day_items
        self.seconds_per_item = seconds_per_item
        self.pressure_per_dream = pressure_per_dream
        self.pressure_per_sleep = pressure_per_sleep
        self.loaded_for: State | None = None

    @staticmethod
    def _ensure_table(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS dummy_results (item_id INTEGER, state TEXT, at REAL)")

    @staticmethod
    def _day_planned(store: "Store") -> bool:
        # Derived from the queue itself, so "planned" and "enqueued" can never disagree after a crash.
        return store.conn.execute("SELECT 1 FROM work_queue WHERE state = 'day' LIMIT 1").fetchone() is not None

    def has_work(self, store: "Store", state: State) -> bool:
        if state is State.DAY:
            return store.pending_count(State.DAY) > 0 or not self._day_planned(store)
        return True

    def plan(self, store: "Store", state: State) -> list[tuple[str, dict]]:
        if state is State.DAY:
            if self._day_planned(store):
                return []
            return [("read", {"n": i}) for i in range(self.day_items)]
        if state is State.DREAM:
            return [("dream", {})]
        if state is State.DEEP_SLEEP:
            return [("consolidate", {})] if store.pressure() > 0 else []
        return []

    def load(self, state: State) -> None:
        self.loaded_for = state

    def unload(self) -> None:
        self.loaded_for = None

    def run(self, item: "WorkItem", ctx: WorkContext) -> Any:
        deadline = time.monotonic() + self.seconds_per_item
        while time.monotonic() < deadline:
            ctx.check()
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        return {"kind": item.kind}

    def save(self, item: "WorkItem", result: Any, conn: sqlite3.Connection) -> float:
        self._ensure_table(conn)
        conn.execute("INSERT INTO dummy_results (item_id, state, at) VALUES (?, ?, ?)",
                     (item.id, item.state.value, time.time()))
        if item.kind == "dream":
            return self.pressure_per_dream
        if item.kind == "consolidate":
            return -self.pressure_per_sleep
        return 0.0


class TorchHoldWorkload(DummyWorkload):
    """Like DummyWorkload, but holds real GPU memory while loaded, like a model would.

    Used by the yield check: does the engine give the GPU back when another program needs it?
    """

    def __init__(self, hold_mb: int = 1500, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hold_mb = hold_mb
        self._buffer = None

    def load(self, state: State) -> None:
        super().load(state)
        if self._buffer is None:
            import torch

            self._buffer = torch.empty(self.hold_mb * 1024 * 1024, dtype=torch.uint8, device="cuda")
            self._buffer.fill_(1)

    def unload(self) -> None:
        super().unload()
        if self._buffer is not None:
            import torch

            self._buffer = None
            torch.cuda.empty_cache()
