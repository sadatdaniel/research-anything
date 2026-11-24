# OBJECTIVE: what Learn Anything is for, and what we've learned

> **Read this first, then [STATUS.md](STATUS.md) (where things stand) and [PLAN.md](PLAN.md) (the full design).** This file holds the *why* and the real understanding: the things that must survive when a long session is summarized. It's updated at every milestone. When a finding changes our understanding, it changes here first.

_Last updated: 2026-11-24_

## The objective

**Build a learning machine, not a bigger model.** It's a small model on an ordinary laptop (RTX 4050, 6 GB) that you point at a folder of material: chronicles, a writer's books and letters, an archive. It:
- studies the material by itself: it reads by day and **dreams** at night;
- finds patterns and connections **nobody stated**;
- **checks every one** against the sources and against chance before believing it;
- **gets better at learning**: after "sleeping" on what it confirmed, it discovers faster in material it has never seen.

People rarely know what to ask, so it brings the questions too. It answers only from what it learned, and says "I don't know" otherwise.

**The selling point:** a private research assistant that studies your archive overnight and hands you checked discoveries you didn't know to ask for. Each comes with its evidence, the odds it's chance, and a prediction you can test. It isn't the model's size that does this, but search, verification and learning. The same small model asked directly found 0% of hidden patterns; inside the pipeline it finds about 50–60%.

**Name:** Learn Anything. The folder and package are still `sandbox-learner`. What it learns from one theme is a **memory**, and it travels as a `.mindpack`.

## What we're trying to prove

| | Hypothesis | Status |
|---|---|---|
| **H1** | More time produces more true, unstated discoveries, while false ones stay low | Partly. It works, but on worlds of this size dreaming runs out of work in minutes, so "more time" can't yet show. That needs harder worlds |
| **H2** | **It learns how to learn:** after sleeping on its checked findings, it discovers faster in unseen material, and faster than the same lessons given as notes (baseline B4). **This is the novel claim** | Not tested (Phase 4) |
| H3 | Thinking deeper (more loops) helps | Supported for *judgment*: Ouro at 4 loops vs 2 found 94% vs 59% of links, with 1% vs 8% false "yes" |
| H4 | It remembers what it slept on without losing general ability | Not tested (Phase 4) |

## What we've learned so far (the real understanding)

1. **Verification is what works, not the model.** The same model asked directly found 0% (B0). Its untested proposals were 5% true (B1). Inside the pipeline it reaches about 50–60% recall with about 70–80% precision and almost no decoys believed. (results/phase3/)
2. **Small evaluations lie.** Two recipes on 12 patterns tied (9/12 vs 8/12, ±25 points). We now evaluate on **test-v2**: 30 fresh worlds, 122 patterns, about ±8 points, paired, with intervals over whole worlds. (results/evalset_test-v2.json)
3. **A careful checker matters less than expected.** Ouro is by far the most careful yes/no checker (1% false yes against 4–15% for others). End to end, though, a faster model tied it (Qwen3-1.7B 7/10 vs Ouro 6/10), because the statistical tests catch most of what a looser checker lets through. (results/bakeoff/)
4. **Don't mix models within a memory.** A fast reader with Ouro as the checker did worst: the checker re-sorted records into the reader's vocabulary and a cycle vanished.
5. **Consistent notes save work.** Readers with less consistent labels raise 2.5–2.8× as many questions while dreaming.
6. **Why cycles are missed** (Qwen3 on test-v2, 27 missed of 48):
   - 11 hide in a crowded kind and rightly fail the chance test, because by years alone they could be luck;
   - 6 are unfindable in the material itself;
   - 5 fail the prediction test because the predicted return was never written down;
   - 5 are reading errors.

   Fix 1, the current work: **group by description before the cycle test** (the Halley idea). The records carry a signature, paraphrased differently by each chronicler. Fix 2: **a silent prediction counts as untested, not wrong.** (results/diagnosis/)
7. **The score ranks well but isn't a probability yet:** 57% of findings scored 80+ were true (ECE 0.225). The app calls it a "score" until it's calibrated on validation worlds.
8. **Newer isn't better here.** The newest model tried (MiniCPM5-2B) was the most easily fooled checker, and LFM3 couldn't follow the note format at all. The model is a swappable part chosen by measurement, not by release date.

## The research program (in order)

1. **Vector dreaming**, in two modes as in sleep research: deep-sleep-like checking and confirming, then REM-like association in vector space.
   - First step (**in progress**): cluster a crowded kind's records by the meaning of their descriptions, then test each cluster for cycles. The aim is to recover the hidden cycles honestly, without loosening the chance test.
2. **Silent predictions count as untested**, with the decoy rate measured.
3. **Phase 4, verified sleep (H2).**
   - Train LoRA only on findings that passed every test, plus which searches paid off.
   - Quickest readout: after sleep, are a cycle's records closer together in the model's *own* vector space?
   - Must beat B4 (notes in the prompt) and sleeping on facts only.
4. **Papers as swappable parts**, one at a time: false-discovery control (Benjamini–Hochberg, online LORD), Hawkes and Granger lags, TLogic rules, AlignScore checking, calibration.
5. **New pattern families** (Phase 6c), each tested on planted fake worlds first:
   - **contextual patterns:** a measured quality of the text tracks another source's timeline, e.g. a writer's life and her characters' mood;
   - **hidden-design patterns:** events that enact one scheme, as in *Se7en* or Langdon. This type needs the strictest checks against seeing designs in noise.
6. Real material: Gutenberg books, then rediscovery tests (Halley, the Bengal famines).

**Running now:** the test-v2 comparison, Ouro-1.4B against Qwen3-1.7B. Qwen3 found 50% [42–58%] at 71% precision [63–79%]; Ouro's half finishes around midnight on 2026-11-24. That decides which model the learner uses.

## How we work (rules that must not be lost)

- **Clean evaluation:**
  - tune only on validation worlds;
  - report only on untouched test worlds;
  - a fix suggested by test-world errors gets judged on *fresh* test worlds.

  test-v2 has now been examined, for the cycle diagnosis.
- **Every stage is swappable** (`learner/recipe.py`, `learner/model.py`), and every result names its recipe and model.
- **Every phase ends with a quality check:** the full tests, the UI audit at 5 widths in light and dark with 0 issues, review by eye, and `results/phaseN/QA.md`.
- **Honesty in the app:** "tends to coincide", never "causes". Untested ideas are labelled. It says "I don't know" when it doesn't.
- **Repository code from a model** only runs after it's been read, and only for models on a reviewed list.
- **Commits** go out as sadatdaniel, **never with a Claude co-author line**.

## What you care about (product)

- **Memories:** learned from a theme folder. You can combine them into **fusions**, export and import them as versioned `.mindpack` files with their material, update them from a git upstream, and delete them (to the trash).
- **Asking:** NotebookLM-style chat that answers only from the loaded memories. The **question bank** holds questions it asked itself, with scores; you can also ask "is A connected to B?"
- **Exploring:** a timeline, an Obsidian-style graph, an Obsidian-compatible findings folder, and contextual and hidden-design patterns.
- **The app:** a modern GNOME (libadwaita) look:
  - one place per navigation level (the sidebar has Library, Settings and About);
  - five sections inside a memory, at the top on wide screens and at the bottom on narrow ones;
  - card grids that reflow;
  - fixed-size Preferences.
- **Running:** resource-aware (it gives the GPU back immediately), private, and portable on Linux and Windows. Online lookup is opt-in.

## Changelog of this file
- 2026-11-24: created at your request, so understanding survives when the session is summarized.
