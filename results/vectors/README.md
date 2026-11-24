# Grouping cycles by description: what was tried and what it showed

_2026-11-24. Validation worlds only (val-000…004 with Ouro's notes; val-003 and val-004 also with Qwen3's). Embedder: bge-small@1 (BAAI/bge-small-en-v1.5, CPU). The reason for this work is in [results/diagnosis/README.md](../diagnosis/README.md): 11 of the 27 cycles Qwen3 missed on test-v2 hid in crowded kinds._

## 1. Hard clustering: it shatters cycles (`experiments/vector_clusters.py`)

The records of a crowded kind were clustered by description, and the cycle search ran per cluster.
- The clusters were pure, up to 1.0, but the best cluster held only 18–54% of a cycle.
- Result: **0 of 9** cycles found, against 7 of 9 on the whole kind.
- The clusters followed each chronicler's writing style, which matches the literature: embeddings mix style and content.

## 2. Description coherence as a second test, combined with Fisher's method (`experiments/coherence_test.py`)

The kind stays whole. For each candidate cycle: does its records' mean similarity beat random subsets of the same kind? That p-value is combined with the chance test's p-value (Fisher, chi-square with 4 degrees of freedom) at the recipe's threshold, and the prediction test is still required.

| Notes from | Today (chance only) | + coherence (raw vectors) | + coherence (style removed per chronicler) |
|---|---|---|---|
| Ouro, 9 planted cycles | 6 | 6 | 4 |
| Qwen3, 4 planted cycles | 3 | 3 | 1 |

No false cycles and no decoys in any column.

**What it shows:**
- **Coherence is a real signal.** Median coherence p-value: **0.014** for candidates that are planted cycles, **0.52** for the other 163.
- **These validation worlds hardly have the crowded-kind problem.** Most of their cycles pass the chance test outright. The one crowded case, val-003's comet, reaches a passing combined p (0.0002–0.0003), but then fails the prediction test (likely a return that was never written down). It needs both fixes.
- **Fisher's method can dilute strong temporal evidence** when a true cycle's descriptions vary: val-003's locusts go from 0.003 to 0.006 combined. Better: use coherence as a **rescue**. Pass if the chance test passes *or* the combination passes, each at half the threshold, so the overall error rate is unchanged.
- **Removing style per chronicler destroys the signal.** It isn't used.

## Next

1. As a recipe option, coherence as a rescue (raw vectors, each test at half the threshold) **together with** "a silent prediction counts as untested".
2. These validation worlds have too few crowded-kind failures to develop on, so more validation worlds need reading once the GPU is free. Then judge on a **fresh** test set, not test-v2 (already examined).
3. Compare embedders: bge-small against the learner model's own hidden states. The second is also the planned readout for sleep.
