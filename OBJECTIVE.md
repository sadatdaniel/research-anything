# OBJECTIVE: the research question, and what we've learned

> **Read this first, then [STATUS.md](STATUS.md) (where things stand) and [PLAN.md](PLAN.md) (the full design).** This file holds the *why* and the real understanding: the things that must survive when a long session is summarized. It's updated at every milestone. When a finding changes our understanding, it changes here first.

_Last updated: 2026-11-24_

## The objective

**This is a research project about a learning pipeline, not a product built around a model** (framing corrected at your request on 2026-11-24: "you're thinking in models, it's a pipeline... think like a research project").

**The research question:** can a learning *pipeline* (read → dream → verify → sleep), rather than a larger model:
- discover patterns that a text collection never states;
- and get better at discovering (**learning how to learn**) by training itself on what it has verified?

We optimise the pipeline component by component (every stage swappable, every change measured on fresh worlds with paired intervals). We present it the way research is presented: question, hypotheses, method, benchmark, protocol, results (negative ones included), open problems. **Self-learning is the centre**: the system's own tests (held-back predictions, shuffle tests, re-reading) are its teacher, and what passes is what it trains on.

**The audience is researchers first:**
- people who study self-learning, verification and discovery;
- people who want a benchmark or testbed;
- scholars who could later run it on their own corpora.

The product side (an app, memories, mindpacks) is how results are inspected and shared, not the headline. Evidence so far that the method does the work: the same model asked directly found 0% of the hidden patterns; inside the pipeline it finds 50% at 71% precision on 30 fresh test worlds.

**The method is the contribution, not the model** (agreed 2026-11-24). Any capable model should find deeper patterns inside the pipeline than on its own.
- A model is chosen only when it's measurably better at a job (Ouro is the most careful checker).
- Where the model runs is a setting:
  - **this computer**;
  - **another computer** through a standard model server (vLLM, llama.cpp, Ollama), for example a rented GPU;
  - **an online API**: opt-in, recorded per memory, and it can't sleep, because we can't train its weights.
- The pipeline needs only two things from a model: *write a note*, and *the probability of "yes"*.
- **The proof experiment** (fake worlds, invented names, so nothing can be memorised):
  1. a small model inside the pipeline;
  2. a much bigger model asked directly;
  3. the bigger model inside the pipeline.

  If 1 beats 2, the method does the work. If 3 beats 1, it scales. It's measured hardest on the deep types: chains, convergence, contextual and hidden-design patterns.
- "Deeper questions" are scored too: the question bank's "needs material" questions are checked against each world's hidden records.

**Name:** **Research Anything**: using an intelligent pipeline to discover hidden patterns (chosen 2026-11-24; earlier working names were Learn Anything, then Sandbox Learner). The folder and package are still `sandbox-learner`. What it learns from one theme is a **memory**, and it travels as a `.mindpack`.

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
6b. **Hard clustering by description shatters cycles** (validation worlds, small embedder). The clusters were pure but followed each chronicler's *style*: the best cluster held only 18–54% of a cycle, and 0 of 9 cycles were found, against 7 of 9 on the whole kind. Lesson, also from the literature: don't cut the kind apart; add the description as a *second, independent test* and combine it with Fisher's method. Tested: **description coherence is a real signal** (median p 0.014 for true cycles against 0.52 for other candidates). It recovered nothing on the validation worlds, though, because they barely have the crowded-kind problem, and their one crowded cycle also fails prediction (a silent return). Next: coherence as a rescue (each test at half the threshold) together with the silent-prediction fix, on more validation worlds, then a fresh test set. Removing style per chronicler destroys the signal. (results/vectors/README.md)
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
- **Every stage starts with a look at the literature** (at your request, 2026-11-24): published approaches that might do the stage better, checked again after any negative result. They're recorded in REFERENCES.md and tried as recipe options.
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
