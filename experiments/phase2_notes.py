"""Phase 2: how well does Ouro take notes? Read a whole world, then grade the notes against the answer key.

  uv run python experiments/phase2_notes.py --world worlds/test/test-001 [--loops 4] [--limit 60]

Writes results/phase2/notes_<world>_loops<N>.json and prints a summary. The notebook itself goes to
results/runs/phase2/<world>_loops<N>.sqlite (gitignored), so it can be inspected afterwards.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from chronicles.notes_eval import grade  # noqa: E402
from learner import notebook  # noqa: E402
from learner.focused import FEW_SHOT, read_batch  # noqa: E402
from learner.model import load_ouro  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="worlds/test/test-001")
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N records")
    ap.add_argument("--max-batch", type=int, default=8)
    args = ap.parse_args()
    world = Path(args.world)
    name = f"{world.name}_loops{args.loops}" + (f"_first{args.limit}" if args.limit else "")
    db = ROOT / "results" / "runs" / "phase2" / f"{name}.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    conn = sqlite3.connect(db, isolation_level=None)
    notebook.ingest_world(conn, world)
    todo = notebook.unread(conn, args.limit)

    lm = load_ouro(loops=args.loops)
    tokens = len(lm.tokenizer(FEW_SHOT)["input_ids"]) + 60 + 110
    batch = lm.safe_batch(tokens, cap=args.max_batch)
    print(f"{len(todo)} records, loops={args.loops}, batch={batch}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        notes_ = read_batch(lm, chunk)
        conn.execute("BEGIN IMMEDIATE")
        for r, n in zip(chunk, notes_):
            notebook.save_note(conn, n, r["text"], args.loops)
        conn.execute("COMMIT")
        done = i + len(chunk)
        if done % (batch * 10) < batch or done == len(todo):
            rate = done / (time.perf_counter() - start)
            print(f"  {done}/{len(todo)} records, {rate * 60:.1f} per minute", flush=True)
    seconds = time.perf_counter() - start

    truth = json.loads((world / "truth.json").read_text())
    sources = {r[0]: r[1] for r in conn.execute("SELECT record_id, text FROM sources")}
    by_record: dict[str, list] = {}
    for link in notebook.links(conn):
        by_record.setdefault(link["record_id"], []).append(link)
    rows = [{**n, "links": by_record.get(n["record_id"], [])} for n in notebook.notes(conn)]
    report = grade(truth, rows, sources)
    report["run"] = {"world": str(world), "records": len(todo), "loops": args.loops, "batch": batch,
                     "seconds": round(seconds, 1), "seconds_per_record": round(seconds / max(1, len(todo)), 2),
                     "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
    out = ROOT / "results" / "phase2" / f"notes_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("counts", "wrong_events")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
