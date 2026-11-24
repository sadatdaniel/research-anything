# Why the learner misses cycles

_2026-11-24, at your request. Script: [experiments/diagnose_cycles.py](../../experiments/diagnose_cycles.py). Data: Qwen3-1.7B's test-v2 run, which found 21 of 48 planted cycles. This is error analysis only: nothing was tuned. test-v2 is the evaluation set, so any fix below is developed on validation worlds and judged on fresh test worlds._

## Where the 27 missed cycles were lost

| Cause | Missed cycles | What happened |
|---|---|---|
| **Rejected by the chance test: a crowded kind** | **11** | All 11 are comets. A cycle of 6–9 returns sits inside a "comet" kind of 18–34 records. On its own years it's clearly not chance (p = 0.003), but judged as part of the whole kind it looks like chance (p = 0.03–0.68). With 30 comets over a thousand years, random year sets really do contain cycles that often, so **by years alone the test is right not to trust it**. |
| The material makes it unfindable | 6 | Even with perfect notes, two returns in a row were never written down (the generator drops 15% on purpose), and the shape rule forbids that. Recall on the *findable* cycles is 21 of 42 (50%). |
| **Rejected by the prediction test: silence counted as failure** | **5** | In 4 of the 5, the predicted return happened but no record of it exists. The test counted the missing record as a wrong prediction. In the 5th, the period drifted (1 of 3 predictions within tolerance). |
| Reading split the cycle's records | 5 | All are locusts, partly labelled "plague" or "sickness" by Qwen3. Ouro's notes on the same worlds will show whether this is the reader or the material. |

## What would fix them (to try as recipe options, on fresh worlds)

1. **Group cycles by description, not just by kind.** This is the Halley idea: the same comet is recognised by what it looks like, not only by when it came. The generator gives each planted cycle a signature (for example "a white tailed star in the north, about nine nights"). The unrelated comets get other colours, directions and durations.
   - The notes keep it, but in each chronicler's own words: "chalky light", "bone-coloured", "beneath the Wain", "eight nights".
   - Exact word matching finds the signature in 4/7, 0/7 and 5/7 of three cycles' comets, against about 1 of 16–29 other comets.
   - So: ask the model "do these two records describe the same thing (colour, direction, duration)?" for records of a crowded kind, cluster by the answers, and look for cycles inside each cluster.
   - Smaller, cleaner groups should pass the chance test honestly, without loosening it. This could recover most of the 11.
2. **Count a silent prediction as untested, not wrong.** When no record at all falls near a predicted return, the material is silent there. That's no evidence either way, so the prediction should count as untested.
   - Risk: the prediction test is also what stops the planted decoy cycles, which are built to predict nothing. So a single silent miss would be *inconclusive*, and the decoy rate must be measured.
   - This could recover up to 4 of the 5.
3. **Reading:** wait for Ouro's numbers on the same worlds before changing anything.

If fixes 1 and 2 work, cycle recall could rise from about 50% to well above 70% of the findable cycles, without giving up the statistical guard. "Group by description before testing for a cycle" is the first change here that's an algorithmic improvement and not just tuning. Whether it's novel depends on how it compares with published event-clustering methods, which still needs checking.
