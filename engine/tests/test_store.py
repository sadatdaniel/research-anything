from engine.policy import State
from engine.store import OFF, Store
from engine.workload import Cancelled, DummyWorkload, WorkContext


def ctx(stop=lambda: False):
    return WorkContext(state=State.DAY, budget=lambda: 1.0, should_stop=stop)


def results(store):
    return store.conn.execute("SELECT item_id FROM dummy_results ORDER BY item_id").fetchall()


def test_run_one_writes_once_and_marks_done(tmp_path):
    store = Store(tmp_path / "brain.sqlite")
    wl = DummyWorkload(seconds_per_item=0)
    store.enqueue(State.DAY, wl.plan(store, State.DAY)[:3])
    assert [store.run_one(State.DAY, wl, ctx()) for _ in range(4)] == ["done", "done", "done", "empty"]
    assert [r[0] for r in results(store)] == [1, 2, 3]


def test_cancelled_item_writes_nothing_and_stays_pending(tmp_path):
    store = Store(tmp_path / "brain.sqlite")
    wl = DummyWorkload(seconds_per_item=0.5)
    store.enqueue(State.DAY, [("read", {})])
    assert store.run_one(State.DAY, wl, ctx(stop=lambda: True)) == "cancelled"
    assert store.pending_count(State.DAY) == 1
    assert store.conn.execute("SELECT name FROM sqlite_master WHERE name='dummy_results'").fetchone() is None


def test_racing_runner_cannot_complete_an_item_twice(tmp_path):
    path = tmp_path / "brain.sqlite"
    store_a, store_b = Store(path), Store(path)
    store_a.enqueue(State.DAY, [("read", {})])

    class SlowA(DummyWorkload):
        def run(self, item, c):
            # While A is still computing, B finishes the very same item.
            assert store_b.run_one(State.DAY, DummyWorkload(seconds_per_item=0), ctx()) == "done"
            return {}

    assert store_a.run_one(State.DAY, SlowA(seconds_per_item=0), ctx()) == "duplicate"
    assert len(results(store_a)) == 1  # only B's result was kept


def test_failing_item_is_retried_then_marked_failed(tmp_path):
    store = Store(tmp_path / "brain.sqlite", max_attempts=2)

    class Broken(DummyWorkload):
        def run(self, item, c):
            raise ValueError("bad page")

    store.enqueue(State.DAY, [("read", {})])
    assert store.run_one(State.DAY, Broken(), ctx()) == "failed"
    assert store.pending_count(State.DAY) == 1
    assert store.run_one(State.DAY, Broken(), ctx()) == "failed"
    assert store.pending_count(State.DAY) == 0
    assert store.queue_counts()["day"] == {"failed": 1}


def test_pressure_rises_with_dreams_and_falls_with_sleep(tmp_path):
    store = Store(tmp_path / "brain.sqlite")
    wl = DummyWorkload(seconds_per_item=0, pressure_per_dream=0.25, pressure_per_sleep=0.5)
    store.enqueue(State.DREAM, [("dream", {})] * 4)
    for _ in range(4):
        store.run_one(State.DREAM, wl, WorkContext(State.DREAM, lambda: 1.0, lambda: False))
    assert store.pressure() == 1.0
    store.enqueue(State.DEEP_SLEEP, [("consolidate", {})] * 3)
    for _ in range(3):
        store.run_one(State.DEEP_SLEEP, wl, WorkContext(State.DEEP_SLEEP, lambda: 1.0, lambda: False))
    assert store.pressure() == 0.0  # never negative


def test_timeline_and_recovery_after_a_kill(tmp_path):
    store = Store(tmp_path / "brain.sqlite")
    store.enter_state(State.DAY, "new material to read", 0.8, at=100.0)
    store.heartbeat(150.0, State.DAY, 50.0)
    # ... process killed here; restarted at t=400 ...
    store.recover(400.0)
    rows = store.timeline()
    assert [(r["state"], r["started_at"], r["ended_at"]) for r in rows] == [
        ("day", 100.0, 150.0),
        (OFF, 150.0, 400.0),
    ]
    assert store.current_state() is State.REST
    assert store.active_seconds()["day"] == 50.0
