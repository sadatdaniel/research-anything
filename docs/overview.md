# Research Anything: using an intelligent pipeline to discover hidden patterns

_The full research write-up. The short version is the [README](../README.md)._

**Can an artificial learner find patterns that a collection of documents never states, tell them apart from coincidence, and become better at finding them, the way people improve at noticing by sleeping on what they've learned?**

> Research in progress. Discovery with verification is built and measured. Learning how to learn (sleep) is the next experiment. Every number below links to its data.

---

## Motivation

Many discoveries came from noticing a pattern scattered across records nobody had read together:
- Halley, in centuries of comet sightings;
- Snow, in the addresses of cholera deaths;
- Semmelweis, in which wards mothers died in.

The pattern was in the material, but no single record stated it.

Language models read such records easily. Asked directly, though, they retell what's written. They rarely notice what isn't. They can't tell a pattern from a coincidence. And they don't get better at noticing.

People learn this skill by alternating study with sleep. Sleep is when the brain replays what it learned, consolidates it, and recombines distant memories into new associations (McClelland et al. 1995; Lewis, Knoblich and Poe 2018). We ask whether an artificial learner built on the same rhythm can do the same.

## The idea

The learner alternates three states:
1. **Reading:** it takes notes on every record.
2. **Dreaming:** it proposes connections across everything it has read, and **tests each one against the material itself**:
   - the evidence must exist word for word;
   - every record it relies on is re-read;
   - the pattern must vanish when the dates are shuffled;
   - it must predict records it held back from itself.

   What survives is a finding. What fails is kept, with the reason.
3. **Sleeping:** it trains on its **verified** findings, and on which of its searches paid off.

**No one labels anything.** The learner's teachers are its own tests against the material. The central claim is that this loop makes it **better at discovering**, not just more knowledgeable: after sleep, it should find patterns faster in material it has never seen.

## Hypotheses

| | Hypothesis | Status |
|---|---|---|
| **H1** | Given fixed material, verified discoveries keep growing with time spent dreaming, while false ones stay rare | Partly supported. Discovery works, but our test worlds are exhausted within minutes. Harder worlds are needed to see the effect of time |
| **H2** | **Learning how to learn:** after sleeping on its verified discoveries, the learner discovers faster in unseen material. It must also beat the same lessons handed to it as text, and sleep on facts alone | **Next experiment; the central claim** |
| H3 | Thinking longer per step (more recurrent depth) improves the judgments verification depends on | Supported for checking: at 4 loops vs 2, 94% vs 59% of stated links found, and false "yes" answers 1% vs 8% of the time |
| H4 | What was learned in sleep is kept, without losing general ability | Not yet tested |

## How we test it

- **Worlds that can't be memorised.** We generate fake chronicles: several invented chroniclers describe one invented history, each in their own words. Planted in them are patterns no record states:
  - phenomena that return on a schedule;
  - causes followed by effects years later;
  - chains of events;
  - one person under several names.

  The worlds also contain:
  - **decoys:** coincidences that look like patterns but predict nothing;
  - occurrences that **nobody wrote down**;
  - **final years held back** to test predictions.

  Every name is invented, so success measures discovery, not something remembered from training.
- **Measures:** how many planted patterns are found; how many claims are true; how many decoys are believed; how many predictions come true; time.
- **Protocol:**
  - methods are developed on validation worlds and reported on untouched test worlds;
  - a fix motivated by test-world errors is judged on *fresh* worlds;
  - every comparison is paired (the same worlds), with 95% intervals over whole worlds.
- **Baselines:** asking the model directly; the learner's ideas without testing; a keyword script. For H2: the same lessons given as text, with no sleep.
- **The current evaluation set:** 30 worlds, 122 planted patterns, 8,957 records.

## Results so far

**Verification, not the model, does the work:**

| | Patterns found | Claims that are true |
|---|---|---|
| The model asked directly ¹ | 0% | – |
| Its ideas, untested ¹ | 65% | 4.6% |
| A keyword script | 11.5% | 4.7% |
| **The learner (dreaming with verification)**: 30 worlds, 122 patterns | **50%** [42–58] | **71%** [63–79] |

It believed 2 of 60 decoys, and 86% of its predictions came true. A second reader model on the same worlds: _in progress_.
¹ Measured on the first test set (6 worlds, 26 patterns), not yet on the current one.

**What we've learned:**
1. **Testing is what turns ideas into findings.** Untested, the learner's ideas are 5% true.
2. **Small evaluations mislead.** Two variants compared on 12 patterns tied within ±25 points, which is why the evaluation set was enlarged.
3. **A more careful judge didn't produce more findings.** Across seven models, the most careful yes/no judge (1% false "yes", against 4–15%) found no more end to end, because the statistical tests catch most of what a looser judge lets through.
4. **Why patterns are missed** (27 of 48 cycles):
   - 11 hide in a crowded category and rightly fail the chance test on dates alone;
   - 6 are unfindable in the material;
   - 5 fail because the predicted return was never written down;
   - 5 are misreadings.
5. **A negative result:** grouping records by the similarity of their descriptions tore cycles apart, because the similarity followed each writer's style. As a *second, independent* test instead, it is a strong signal (median p 0.014 for real cycles against 0.52 for other candidates). It's now being combined with the date test.
6. **The learner's confidence ranks findings well, but isn't yet a calibrated probability:** findings scored 80 or more were true 57% of the time.

Data and analysis for each: `results/phase3/`, `results/bakeoff/`, `results/diagnosis/`, `results/vectors/`, `results/eval_v2/`.

## Open questions

- **H2 itself:** what should a learner sleep on so that it gets better at *discovering*, not just at remembering? How is that measured cleanly against simply being told the lessons?
- **The statistics of discovery:**
  - controlling false discoveries when hypotheses are generated and tested one after another;
  - combining many weak clues;
  - confidence that means what it says.
- **Recognising the same thing across writers,** when every writer describes it in different words.
- **Deeper patterns,** each tested against planted truth and decoys:
  - a quality of one source following the timeline of another (an author's life and her characters' mood);
  - one scheme explaining many separate events;
  - rules that reverse over time;
  - unreliable witnesses.
- **Real collections,** with rediscovery tests on known results: periodic comets, eclipse cycles, epidemiology.

## For researchers

We'd welcome collaboration and critique, especially from people working on learning and memory, statistics, computational linguistics, the digital humanities and the history of science.
- **Evaluate your own discovery or hypothesis-generation approach** on the generated worlds. Each comes with its answer key, decoys and held-back years.
- **Test a consolidation or self-training method** against the H2 protocol.
- **Propose a new kind of hidden pattern,** with a way to plant and score it.
- **Run a rediscovery test** on a collection from your field.
- **Replicate or challenge any result here:** each records exactly how it was produced.

## Reproducing

You'll need Linux, an NVIDIA GPU with at least 6 GB of memory (a laptop GPU is enough), Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uvx --from "huggingface_hub[cli]>=0.34,<1.0" hf download ByteDance/Ouro-1.4B --local-dir learner-data/models/Ouro-1.4B
uv run python -m chronicles.generator splits --out worlds          # the test worlds
uv run python experiments/phase3_read_worlds.py worlds/val/val-000  # reading
uv run python experiments/phase3_dream.py worlds/val/val-000        # dreaming and verification, scored against the answer key
experiments/eval_v2.sh && uv run python experiments/eval_v2_report.py   # the full evaluation (about a day)
```

## Documents

- [OBJECTIVE.md](../OBJECTIVE.md): the question, what we've learned, the research program.
- [PLAN.md](../PLAN.md): the full design.
- [STATUS.md](../STATUS.md): progress, with a dated log.
- [REFERENCES.md](../REFERENCES.md) and [RELATED_WORK.md](../RELATED_WORK.md): the ideas this builds on, and how it differs from published work on sleep and learning to learn.

## License

The code is licensed under the **GNU Affero General Public License, version 3 or later** ([LICENSE](../LICENSE)). If you change it and share it, *or run it as an online service*, your changes must be shared under the same licence. What you learn with it (memories, findings, mindpacks) is yours. Models are downloaded separately under their own licences, and none are included here.

## Citation

To be added with the first preprint.
