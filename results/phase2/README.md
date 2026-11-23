# Phase 2 results: notebook and focused reading (day mode)

**Verdict: GO.** Ouro-1.4B, reading at 4 loops, takes notes that are accurate enough to build on. On an untouched test world (333 records):
- event labels are 95% right;
- places are 98% correct, and 90% of them are found;
- **every link kept as fact is true**, because inferred ones are filtered out by the source check.

The cycle engine runs day mode end to end with the real model. When another program needs the GPU it hands it back in about 1 s, picks the work up again, and finishes with exactly one note per record.

Reproduce:
```bash
uv run python experiments/phase2_notes.py --world worlds/val/val-000     # read a whole world, grade the notes
uv run python experiments/phase2_engine_e2e.py                          # engine + real model + a GPU interruption
```

## What was built

| File | What it does |
|---|---|
| `learner/notebook.py` | The learner's explicit memory, in the same SQLite file as the engine, so a note and its work item commit together. Tables: `sources` (the original records, with file and line, so a quote can be opened at its exact place), `notes`, `links` (each flagged *stated* or *inferred*) |
| `learner/focused.py` | The few-shot note-taker. Ouro is a base model, so it gets worked examples whose names appear in no world. For each record it writes: event, places, people, details, links, and a self-quiz question with its answer. Batches are sized from free GPU memory and split if memory runs out |
| `learner/sourcecheck.py` | Mechanical source checks: is a quote word for word in the record, and does the record itself *state* a link ("A (called B by others)", "A in B", "A on the B", "A, whom men called B")? |
| `learner/workloads.py` | `FocusedWorkload`: day mode for the cycle engine. Generation stops mid-batch when the engine says rest |
| `chronicles/notes_eval.py` | Grades notes against the per-record answer key |

## Note quality

All runs at 4 loops, batch 2 (sized from free memory), RTX 4050 laptop.

| | `test-001`, prompt v1 (used for tuning) | `val-000`, prompt v2 (not tuned on) | `test-002`, prompt v2 (**final, untouched**) |
|---|---|---|---|
| Records | 237 | 227 | 333 |
| Notes in the right format | 99.2% | 100% | 100% |
| Event label correct | 90.7% | **97.8%** | **95.2%** |
| Places: found / correct | 95.2% / 95.2% | 93.6% / 97.4% | 89.5% / 97.5% |
| Stated links captured | 95.0% | 86.4% | 85.1% |
| **Kept links that are true** | 75.0% (no source check yet) | **100%** | **100%** (42/42) |
| Comet/locust colour | 96.9% | 93.3% | 93.3% |
| Comet direction | 94.7% | 77.8% (7 of 9) | 90.0% |
| Comet duration | 100% | 100% | 100% |
| Quiz answer word for word | 75.7% | **88.5%** | 85.0% |
| Seconds per record | 4.45 | **3.46** | 3.36 |

### What the errors taught us (prompt v1 → v2)
- **Category instead of the thing:** locusts were called "plague" 10 times, and "Krobrousk emptied" was called "flood". The prompt now says to name the thing itself and gives two new worked examples.
- **Figures of speech and side remarks:** "a flood of pilgrims" became a flood (that's the planted decoy wording doing its job), and "Nothing else of note" became the event "nothing". The prompt now says to read figures of speech by meaning and ignore side remarks.
- **Inferred links:** all 13 wrong links were inferences the record never states ("fought the men of X at Y" → "Y in X"; "flood on R drowned the fields of C" → "C on the R"). Now only links the record states word for word are kept as fact (`links.stated = 1`). Inferred ones stay as hypotheses for dream mode to test. Replaying the source check on the v1 run: 39/39 kept links are true, none of the true ones were lost, and all 13 wrong ones were dropped.
- **Prompt length matters on a looped model** (see below), so the Summary line was dropped: 911 → 794 prompt tokens, and about 20 fewer tokens generated per record.

### 4 loops vs 2 loops (14 hand-picked records)
4 loops is clearly better for note-taking:
- name links caught: 5/5 at 4 loops vs 3/5 at 2;
- map links caught: 3/3 vs 1/3;
- at 2 loops the model invents details ("the water level was 10 feet above the ground"), echoes the instructions, and treats "Torch in the sky" as a place.

2 loops is 2.6× faster (1.75 vs 4.6 s per record). **Day reading uses 4 loops**, because note errors compound later. This is an early data point for H3: thinking deeper helps.

## The looped model's memory cost
Ouro stores a key/value cache entry for **every layer on every loop**: 24 layers × 4 loops = 96 entries per token, about **0.8 MB per token per prompt**, four times a normal model of the same size. With a roughly 800-token few-shot prompt, only about 2 prompts fit at once in 6 GB next to the 2.7 GB model. `LoadedModel.safe_batch()` sizes batches from the memory free right now, and `read_batch()` halves a batch that doesn't fit. The retry must happen *outside* the `except` block, because the traceback otherwise keeps the failed attempt's tensors alive: we hit exactly that and it cascaded into more out-of-memory errors.

**Possible speed-up:** compute the shared few-shot prefix's cache once and reuse it. Not done yet.

## Engine end to end (`engine_e2e.json`)

41-record world, real sensors, `FocusedWorkload`. After 8 notes, a "game" grabs 1 GB of GPU memory for 25 s.

| Check | First run | **After the fix** |
|---|---|---|
| GPU freed after the game started | 1.8 s | **1.1 s** |
| Back at work after the game ended | 6.7 s | 6.1 s |
| All 41 records noted | ✗ (20/41) | **✓ in 225 s** |
| Exactly one note per record, no item done twice | ✓ | ✓ |
| Rests when there's nothing left to do | ✗ | **✓** ("night; nothing to dream about yet") |

**The first run found a real policy bug.** It was 2 a.m., so after a minute of reading the engine switched to "night: dream", with half the records unread and no dream work. It then sat on the GPU doing nothing. Fixed: **unread material comes first, even at night** (you can't dream about what you haven't read), and **no dream work means REST**. Both are covered by new engine tests.

## Development hygiene
The v1 prompt was tuned on `test-001`. From now on prompt work uses `val` worlds only, and reported numbers come from test worlds that were never tuned on (`test-002` above).
