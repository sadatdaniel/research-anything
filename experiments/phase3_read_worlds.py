"""Phase 3 prep: day-read whole worlds into their own notebooks, so dream mode has notes to work on.

  uv run python experiments/phase3_read_worlds.py worlds/val/val-001 worlds/val/val-002 ...

Each world gets results/runs/phase3/<world>.sqlite (gitignored; with --model other than ouro-1.4b,
<world>.<model>.sqlite). Resumable: records that already have
a note are skipped, so an interrupted run just continues. A short note-quality summary is written next
to each notebook (<world>.notes.json) using the Phase 2 grader.
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

from chronicles.notes_eval import grade  # noqa: E402
from learner import notebook  # noqa: E402
from learner.focused import FEW_SHOT, read_batch  # noqa: E402

RUNS = ROOT / "results" / "runs" / "phase3"


def notebook_path(world: Path, model: str = "ouro-1.4b") -> Path:
    return RUNS / (f"{world.name}.sqlite" if model == "ouro-1.4b" else f"{world.name}.{model}.sqlite")


def grade_notebook(conn: sqlite3.Connection, world: Path) -> dict:
    truth = json.loads((world / "truth.json").read_text())
    sources = {r[0]: r[1] for r in conn.execute("SELECT record_id, text FROM sources")}
    by_record: dict[str, list] = {}
    for link in notebook.links(conn):
        by_record.setdefault(link["record_id"], []).append(link)
    rows = [{**n, "links": by_record.get(n["record_id"], [])} for n in notebook.notes(conn)]
    report = grade(truth, rows, sources)
    return {k: v for k, v in report.items() if k not in ("counts", "wrong_events")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--model", default="ouro-1.4b", help="a name in learner.model.MODELS")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    lm = None
    for w in map(Path, args.worlds):
        conn = sqlite3.connect(notebook_path(w, args.model), isolation_level=None)
        notebook.ingest_world(conn, w)
        todo = notebook.unread(conn)
        if todo:
            if lm is None:
                from learner.model import load_model

                lm = load_model(args.model, loops=args.loops)
            batch = lm.safe_batch(len(lm.tokenizer(FEW_SHOT)["input_ids"]) + 170, cap=args.max_batch)
            print(f"{w.name}: {len(todo)} to read, batch {batch}", flush=True)
            start = time.perf_counter()
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                notes_ = read_batch(lm, chunk, max_new_tokens=110)
                conn.execute("BEGIN IMMEDIATE")
                for r, n in zip(chunk, notes_):
                    notebook.save_note(conn, n, r["text"], lm.loops)
                conn.execute("COMMIT")
            print(f"{w.name}: read in {time.perf_counter() - start:.0f} s", flush=True)
        summary = grade_notebook(conn, w)
        (notebook_path(w, args.model).with_suffix(".notes.json")).write_text(json.dumps(summary, indent=2))
        print(f"{w.name}: events {summary['event_accuracy']}, link recall {summary['link_recall']}, "
              f"kept links true {summary['link_precision']}", flush=True)
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
