"""Learner workloads that plug into the cycle engine (engine/workload.py describes the contract).

FocusedWorkload is day mode: read unread records and write notes. Batches are sized from free GPU
memory inside the worker, and generation stops early when the engine asks the worker to stop, so a
hard stop doesn't wait for a whole batch to finish.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from engine.policy import State
from engine.workload import WorkContext
from learner import notebook

if TYPE_CHECKING:
    from engine.store import Store, WorkItem


class FocusedWorkload:
    def __init__(self, loops: int = 4, max_batch: int = 4, model_dir: str | None = None) -> None:
        self.loops = loops
        self.max_batch = max_batch
        self.model_dir = model_dir
        self.lm = None  # loaded in the worker only

    # engine side (no GPU)
    def has_work(self, store: "Store", state: State) -> bool:
        return state is State.DAY and notebook.unread_count(store.conn) > 0

    def plan(self, store: "Store", state: State) -> list[tuple[str, dict]]:
        if state is not State.DAY:
            return []
        queued = {row[0] for row in store.conn.execute(
            "SELECT json_each.value FROM work_queue, json_each(json_extract(payload, '$.ids')) "
            "WHERE state = 'day' AND status = 'pending'")}
        todo = [r for r in notebook.unread(store.conn) if r["id"] not in queued]
        size = self.max_batch
        return [("read", {"ids": [r["id"] for r in todo[i:i + size]],
                          "records": [{"id": r["id"], "year": r["year"], "text": r["text"]} for r in todo[i:i + size]]})
                for i in range(0, len(todo), size)]

    # worker side (GPU)
    def load(self, state: State) -> None:
        if self.lm is None:
            from learner.model import DEFAULT_MODEL_DIR, load_ouro

            self.lm = load_ouro(self.model_dir or DEFAULT_MODEL_DIR, loops=self.loops)
        self.lm.set_loops(self.loops)

    def unload(self) -> None:
        self.lm = None  # the worker process exits right after, which frees the GPU completely

    def run(self, item: "WorkItem", ctx: WorkContext) -> Any:
        from learner.focused import read_records

        ctx.check()
        records = item.payload["records"]  # items carry their records, so the worker never needs the DB here
        notes = read_records(self.lm, records, should_stop=ctx.should_stop)
        ctx.check()  # stopped early: discard the half-written notes; the item stays pending
        return notes

    def save(self, item: "WorkItem", result: Any, conn: sqlite3.Connection) -> float:
        texts = {r["id"]: r["text"] for r in item.payload["records"]}
        for note in result:
            notebook.save_note(conn, note, texts[note.record_id], self.loops)
        return 0.0  # reading alone adds no sleep pressure; confirmed findings from dreams do (Phase 3)
