"""Command line for the cycle engine.

  python -m engine run    --db PATH [--workload dummy|torch-hold] [--night-now] [--fake-sensors] ...
  python -m engine status --db PATH
  python -m engine hog    --mb 2000 --seconds 30      # pretend to be a game that needs the GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta

from engine.policy import Policy, PolicyConfig, State
from engine.sensors import FakeSensors, Snapshot, SystemSensors
from engine.store import Store
from engine.workload import make_workload
from engine.worker import WorkerHandle

WORKLOADS = {
    "dummy": "engine.workload:DummyWorkload",
    "torch-hold": "engine.workload:TorchHoldWorkload",
    "focused": "learner.workloads:FocusedWorkload",  # day mode: Ouro reads records and takes notes
}


def _fmt_time(ts: float | None) -> str:
    return "now" if ts is None else datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def healthy_snapshot() -> Snapshot:
    return Snapshot(taken_at=time.time(), gpu_present=True, vram_total_mb=6141, vram_free_mb=6000,
                    gpu_temp_c=50, ram_available_mb=8000, cpu_others_pct=5, on_ac=True)


class _LiveFake(FakeSensors):
    def read(self) -> Snapshot:
        from dataclasses import replace

        return replace(self.snapshot, taken_at=time.time())


def cmd_run(args: argparse.Namespace) -> int:
    config = PolicyConfig.from_toml(args.config) if args.config else PolicyConfig()
    if args.night_now:  # make "now" fall inside the night window
        start = datetime.now() - timedelta(minutes=1)
        config.night_start = start.strftime("%H:%M")
        config.night_end = (start + timedelta(hours=12)).strftime("%H:%M")
        config.night_idle_seconds = 0.0
    if args.min_state is not None:
        config.min_state_seconds = args.min_state
    if args.pressure_threshold is not None:
        config.sleep_pressure_threshold = args.pressure_threshold
    if args.vram_need is not None:
        config.vram_need_day_mb = config.vram_need_dream_mb = args.vram_need
        config.vram_need_deep_sleep_mb = args.vram_need
    kwargs = json.loads(args.workload_args) if args.workload_args else {}
    if args.workload == "torch-hold":
        kwargs.setdefault("hold_mb", args.hold_mb)

    from engine.scheduler import Scheduler

    store = Store(args.db)
    sensors = _LiveFake(healthy_snapshot()) if args.fake_sensors else SystemSensors(
        ignore_process_names=tuple(config.ignore_gpu_process_names))
    ref = WORKLOADS[args.workload]
    scheduler = Scheduler(
        store=store, sensors=sensors, policy=Policy(config), planner=make_workload(ref, kwargs),
        worker=WorkerHandle(str(args.db), ref, kwargs, grace_seconds=args.grace),
    )
    last = {"state": None}

    def report(decision) -> None:
        if decision.state is not last["state"]:
            print(f"{_fmt_time(time.time())}  {decision.state.value:<10}  {decision.reason}", flush=True)
            last["state"] = decision.state

    scheduler.run(tick_seconds=args.tick, max_seconds=args.max_seconds, on_tick=report)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = Store(args.db)
    state = store.current_state()
    print(f"state:          {state.value}  ({store.current_reason()})")
    print(f"since:          {_fmt_time(store.get('entered_at'))}")
    print(f"sleep pressure: {store.pressure():.2f}")
    hours = {k: round(v / 3600, 3) for k, v in store.active_seconds().items()}
    print(f"active hours:   {hours}")
    print(f"work queue:     {store.queue_counts()}")
    print("timeline:")
    for row in store.timeline(args.limit):
        print(f"  {_fmt_time(row['started_at'])}-{_fmt_time(row['ended_at'])}  "
              f"{row['state']:<10}  {row['reason']}")
    return 0


def cmd_hog(args: argparse.Namespace) -> int:
    import torch

    buf = torch.empty(args.mb * 1024 * 1024, dtype=torch.uint8, device="cuda")
    buf.fill_(1)
    torch.cuda.synchronize()
    print(f"holding {args.mb} MB of GPU memory for {args.seconds} s", flush=True)
    time.sleep(args.seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m engine")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the cycle engine")
    run.add_argument("--db", required=True)
    run.add_argument("--config", help="config.toml with an [engine] table")
    run.add_argument("--workload", choices=sorted(WORKLOADS), default="dummy")
    run.add_argument("--workload-args", help="JSON kwargs for the workload")
    run.add_argument("--hold-mb", type=int, default=1500)
    run.add_argument("--night-now", action="store_true", help="treat now as night (for testing)")
    run.add_argument("--fake-sensors", action="store_true", help="pretend the machine is idle and healthy")
    run.add_argument("--min-state", type=float)
    run.add_argument("--pressure-threshold", type=float)
    run.add_argument("--vram-need", type=float, help="override the GPU memory each state needs (MB)")
    run.add_argument("--tick", type=float, default=1.0)
    run.add_argument("--grace", type=float, default=6.0)
    run.add_argument("--max-seconds", type=float)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="show state, pressure, active hours and the timeline")
    status.add_argument("--db", required=True)
    status.add_argument("--limit", type=int, default=20)
    status.set_defaults(func=cmd_status)

    hog = sub.add_parser("hog", help="hold GPU memory, like a game would")
    hog.add_argument("--mb", type=int, default=2000)
    hog.add_argument("--seconds", type=float, default=30)
    hog.set_defaults(func=cmd_hog)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
