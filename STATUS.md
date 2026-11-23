# STATUS: where Sandbox Learner stands

> **Read this first when picking the work back up.** It's updated at every milestone. The full design and success criteria live in [PLAN.md](PLAN.md); this file says what's done, what's in progress, what was decided, and what comes next.

_Last updated: 2026-11-23 (after the related-work check)_

## In one paragraph
A small looped language model ([Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B)) that lives in a local sandbox and keeps learning from a folder of documents:
- **By day** it reads and takes notes.
- **At night** it "dreams": it hunts for connections between distant records and checks each one against the source, against held-back records, and against shuffled dates.
- **In deep sleep** it trains its confirmed findings into a LoRA adapter.

A resource-aware cycle engine decides when it may use the GPU. The core claim to prove (H2) is that it **learns how to learn**: after sleeping, it finds connections faster in material it has never seen. All scoring uses generated fake chronicles with planted patterns, because the model already knows real history.

## Phase status

| Phase | What | Status | Evidence |
|---|---|---|---|
| 0 | Setup and feasibility | ✅ GO | [results/phase0/README.md](results/phase0/README.md) |
| 1 | Fake-chronicle generator, scorer, keyword baseline (B2) | ✅ GO: B2 9–12% (< 30%); blind readers found 7/7 patterns, 0 false, 0 decoys | [results/phase1/README.md](results/phase1/README.md) |
| 1b | Resource-aware cycle engine | ✅ GPU handed back in 0.9 s | [results/phase0/README.md](results/phase0/README.md#cycle-engine-phase-1b-real-gpu-check) |
| 2 | Notebook and focused reading (day mode) | ✅ GO: untouched test world: events 95%, places 98% correct, kept links 100% true; engine end-to-end passes | [results/phase2/README.md](results/phase2/README.md) |
| 3 | Dream mode and verification; first app window | ⏳ next (outline below) | n/a |
| 4 | Sleep (LoRA consolidation, review, memory audit); H2 and H4, with baseline B4 and the H2 ablations | not started | n/a |
| 5 | Depth vs breadth (loops vs hours); H3 | not started | n/a |
| 6 | Disguised real comet test; ask interface | not started | n/a |
| 7 | Public release: AppImage and Windows zip | not started | n/a |
| 8 | Android companion | later | n/a |

## Key numbers so far
- **Ouro-1.4B on the RTX 4050 (6 GB):**
  - 2.7 GB in GPU memory;
  - 13.9 tokens/s at 4 loops, batch 1 (7.1 at 8 loops), and **41.7 tokens/s at batch 4**;
  - LoRA training fits: 4.1 GB at 512 tokens, 5.0 GB at 1024.
- **Cycle engine:** hands the GPU back in **0.9 s** when another program grabs it, and is back at work 3.1 s after it closes.
- **Keyword baseline B2:** **9%** recall on the test split, 6% on transfer, with 3–4% precision (16% and 10% before the location mentions).
- **Blind readers** (fresh agents that saw only the records): A, on the 41-record sample, scored 3/3; B, on a 218-record world, scored 4/4. No false claims, no decoys; predictions 6/7.
- **Note-taking at 4 loops vs 2** (14 records): 4 loops is far better (name links 5/5 vs 3/5, map links 3/3 vs 1/3; 2 loops invents details) but slower (4.6 vs 1.75 s/record). **Day reading uses 4 loops.**

## Decisions and gotchas (don't re-learn these)
- **transformers 4.57.1, not the 4.54.1 the model card recommends.** Ouro's bundled code now needs ≥ 4.56 and fails on 4.54.1 with "key_cache has no setter".
- **Two Ouro bugs are fixed at load time in [learner/model.py](learner/model.py)** (the model files stay untouched):
  1. The cache never fills `layers`, so transformers built a decode mask covering only the newest token, and padded batches produced garbage. We patch `get_mask_sizes`.
  2. PyTorch's fast attention kernel rejects the head-broadcast mask, so we expand it to a contiguous tensor.
- **Loop count** is set with `LoadedModel.set_loops(n)`, which changes both `config.total_ut_steps` and `model.total_ut_steps`. Ouro always runs every loop; its early-exit gate only picks which loop's output to use.
- **The model runs in a separate worker process.** PyTorch keeps its CUDA context until the process exits, so REST means ending the worker. The worker dies with the engine (`PR_SET_PDEATHSIG` on Linux, plus a parent check).
- **Exactly-once work:** an item is computed outside any transaction, and its results are written and the item marked done in one short transaction. A `kill -9` therefore never loses or repeats work.
- **STARS has no license file**, so we write our own regularizer from the paper. The 730M looped fallback (`ifm-ai/xllm-loop`) is Apache-2.0.
- **Engine GPU-memory budget** (measured): 4.0 GB to read or dream, 5.2 GB to sleep.
- **Night rule:** unread material comes first even at night (you can't dream about what you haven't read), and no dream work means REST. Hard stops and the minimum time in a state still apply.
- **Day reading uses 4 loops.** 2 loops is 2.6× faster but invents details and misses links.
- **Only *stated* links are facts** (`links.stated = 1`, from `learner/sourcecheck.py`). Inferred links are hypotheses for dream mode to test.
- **Development hygiene:** prompt and heuristic work uses `val` worlds; reported numbers come from test worlds never tuned on (`test-001` is spent, `test-002` was used for the Phase 2 final).
- **Looped models have a 4× key/value cache:** Ouro stores one entry per layer *per loop*, about 0.8 MB per token per prompt at 4 loops. A 700-token few-shot prompt therefore allows a batch of only about 2–3. `LoadedModel.safe_batch()` sizes batches from free memory, and `read_batch` splits a batch in half when it runs out of memory. Retry **outside** the `except` block, because the traceback keeps the failed attempt's tensors alive.
- **What's new and what isn't** (see [RELATED_WORK.md](RELATED_WORK.md), checked 2026-11-23):
  - LLM "sleep" with consolidation and dreaming is published (Google 2026, Auto-Dreamer, Lee et al.), and so is "learning how to learn" from self-written training data (SEAL). We cite these; we don't claim them.
  - H2 is still open only in its narrow form: sleeping on *checked* findings and *search outcomes* makes a small model faster at discovering unstated patterns in new material.
  - To claim it, Phase 4 must beat **B4** (the same lessons given as written notes in the prompt, no weight update) and must show that sleeping on search outcomes beats sleeping on facts only.
- **Periodic prediction tolerance** is `max(2 × jitter, 5% of the period)`. Each occurrence is shifted by up to ±jitter, so a correct "last + period" can be off by twice the jitter. Found by blind reader A.

## Next: Phase 3 (dream mode and verification)
Nothing is in progress. Phase 2 is closed: see [results/phase2/README.md](results/phase2/README.md).

Outline to start from:
1. **Group the notes** into canonical events and places.
   - Places: union-find over *stated* `same_as` links, plus stated `located_in`/`on_river` for the map.
   - Events: normalise the free-text labels. The model judges "are these the same kind of event?" on short lists.
2. **Propose candidates** (cheap, no model):
   - periodic runs within an event group with matching telltale details;
   - lagged pairs between groups, using the map so that "flood on river R" is linked to "famine in city C on R";
   - chains from two lagged links;
   - identity from stated person aliases.
3. **Test each candidate** (this is the verification design in PLAN.md):
   - **source check:** quotes verbatim (`learner/sourcecheck.py`), plus the model reading the original passages;
   - **prediction test:** the learner holds back its own latest about 15% of visible years, and the pattern must predict them;
   - **shuffle test:** the pattern must vanish when the years are shuffled.
4. **Store theories** with their stage (proposed → source-checked → prediction-tested → confirmed / rejected (reason)), and emit claims in the scorer's format.
5. **H1 curves:** recall and precision against active hours on test worlds, with B0, B1 and B2 as baselines.
6. **First app window:** timeline plus connections, run on fake worlds, so planted and found patterns can be seen together.

## Open questions for you
- none right now

## Git
- Commits are made as `sadatdaniel <sadatdaniel@gmail.com>`, set in this repository only. **Never add a Claude co-author line.**

## How to pick up
```bash
cd ~/Documents/claude_projects/sandbox-learner
uv sync                                    # Python environment (PyTorch cu128, transformers 4.57.1)
uvx --from "huggingface_hub[cli]>=0.34,<1.0" hf download ByteDance/Ouro-1.4B --local-dir learner-data/models/Ouro-1.4B   # if missing
uv run pytest chronicles/tests engine/tests -q                     # all tests (about 10 s)
uv run python -m chronicles.generator splits --out worlds           # regenerate the 45 worlds (gitignored)
uv run python -m chronicles.baseline_b2 --split worlds/test         # baseline score
uv run python -m engine.check_yield                                 # real-GPU yield check
uv run python experiments/phase0_feasibility.py                     # model speed, memory and LoRA fit
```

## Layout
| Folder | What |
|---|---|
| `chronicles/` | world generator, phrasing banks, scorer, keyword baseline B2, sample world (`samples/small_world.md`) |
| `engine/` | cycle engine: sensors, policy (day/dream/deep sleep/rest), SQLite store, worker process, scheduler, CLI |
| `learner/` | `model.py` (Ouro loading and fixes); notebook, focused, diffuse, verify, sleep to come |
| `experiments/` | measurement scripts |
| `results/` | phase reports and raw numbers |
| `learner-data/` (gitignored) | models; later the learner's brain (notebook, adapters, history) |
| `worlds/` (gitignored) | generated worlds |

## Log
- **2026-11-23:**
  - Plan approved. Phase 0 GO; cycle engine built (yield 0.9 s).
  - Chronicle generator, scorer and B2 built; B2 started at 33%, and after widening the phrasing it's at 16%.
  - Blind reader A scored 3/3, which exposed the scorer's prediction-tolerance bug (fixed) and the geography gap (in progress).
  - Git repository initialized.
  - Blind reader B scored 4/4, which exposed the "two drops in a row" flaw (fixed). Phase 1 GO.
  - Phase 2 notebook, reader and grader built; 4 loops beats 2 for note-taking.
  - First commit `c1b4262` (as sadatdaniel, no co-author).
  - Full note-taking run on `test-001`; links source-checked; night policy fixed (read first, rest when there's nothing to dream about); prompt sharpened.
  - End-to-end engine run passes (GPU freed in 1.1 s, 41/41 notes, rests when done). Phase 2 GO on untouched `test-002`.
  - Related-work check ([RELATED_WORK.md](RELATED_WORK.md)): the sleep framing is published; the narrow H2 test isn't, as far as the check went. Added baseline B4 (written notes instead of sleep) and the H2 ablations to PLAN.md.
