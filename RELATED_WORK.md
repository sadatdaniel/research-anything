# Related work: is our core claim (H2) new?

_Checked 2026-11-23 through web searches, plus reading the abstracts of the closest papers and Google's blog post. This is not a full literature review: the full papers weren't read, and new work in this area appears every month. Check again before any public write-up._

## Verdict

**The sleep idea and "learning how to learn" are not ours.** In 2025–2026 several groups published:
- language-model "sleep" with a consolidation stage and a dreaming stage;
- models that write their own training data and learn to do it better;
- learned memory consolidators that carry over to new environments.

We should cite these papers and not claim the framing.

**What I couldn't find is the specific test H2 makes,** which combines three things:
1. **The outcome is discovery speed on brand-new material.** That means time to find patterns that no record states, scored against planted answers, with decoys present. The prior work measures something else:
   - how much is retained;
   - task success;
   - question answering on stated facts;
   - accuracy of hypotheses about labels.
2. **What gets consolidated is the search skill** (which comparisons paid off), trained into the weights of a small model. The prior work stores strategies as text memory, or trains on facts.
3. **Everything trained on first passed a check** against the source, held-back records and shuffled dates. Prior "dreaming" trains on self-generated data that is filtered by usefulness, not checked against the source.

So **H2 is still open, as far as this check goes**, but only in that narrower form. Every part of H2 exists somewhere; the open question is whether the combination does something the parts don't.

**The benchmark:** I found no benchmark for discovering unstated periodic, cause-and-effect or identity patterns in dated text records, with decoys and held-out predictions. The closest are:
- HypoBench's synthetic datasets;
- DiscoveryWorld's simulated experiments;
- HistoryBankQA's RecurrenceQA, which asks about known recurring events rather than spotting them unprompted.

That makes the benchmark a modest contribution in its own right.

## Closest work

| Work | What it does | Overlap with us | Difference |
|---|---|---|---|
| Behrouz, Hashemi, Javanmard, Mirrokni (Google), [Language Models Need Sleep](https://arxiv.org/abs/2606.03979) (2026, on OpenReview since Sep 2025); [blog post](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/) | Sleep = **consolidation** (short-term memory is distilled into newly added parameters) + **dreaming** (an RL-generated curriculum of synthetic practice; random expert activation to "uncover latent patterns") | Same name and the same two stages as ours | Measures retention and capability (continual learning, unseen languages, 10M-token context, fact incorporation, math), not discovery. Dreams aren't checked against sources. Needs MoE/Hope architecture changes |
| Ye et al., [Auto-Dreamer](https://arxiv.org/abs/2605.20616) (May 2026) | A learned offline consolidator (GRPO) rewrites an agent's memory bank to find recurring patterns and shared procedures. Trained on ScienceWorld only, still ahead on held-out ALFWorld and WebArena | **Closest to H2:** learned consolidation that carries over to new environments | Text memory, not weights. Measures agent task success. The patterns are about the agent's own experience, not the material |
| Zweiger et al., [SEAL](https://arxiv.org/abs/2506.10943) (NeurIPS 2025) | The model writes its own fine-tuning data ("self-edits"), and RL teaches it to write better ones. Evaluated on knowledge incorporation and few-shot ARC | Learning how to learn new material, through its own generations and weight updates | Learns facts that a passage states, not unstated links across many records. The reward is downstream QA |
| Lee, McLeish, Goldstein, Fanti, [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference](https://arxiv.org/abs/2605.26099) (2026) | N offline recurrent passes write the context into fast weights. Longer sleep (larger N) helps most on deeper reasoning | Supports **H3**: more recurrence helps | SSM fast weights, synthetic tasks and math; not discovery |
| Lin et al., [Sleep-time Compute](https://arxiv.org/abs/2504.13171) (2025) | Thinks about a context offline before questions arrive, using about 5× less compute at question time | H1's premise: offline compute on fixed material pays off | Pre-computes answers to likely questions. No weight updates, no discovery scoring |
| Yang et al., [Synthetic continued pretraining (EntiGraph)](https://arxiv.org/abs/2409.07431) (ICLR 2025) | Takes the entities in a small corpus, has a model write about how pairs of them relate, then trains on that text. Closed-book QA rises log-linearly with synthetic tokens | "More compute on the same data gives more knowledge", and it connects entities | Scores QA on stated facts. The relations it writes aren't checked. Uses an 8B model with large compute. **Useful for H4** (getting knowledge into weights) |
| Yang et al., [Synthetic Bootstrapped Pretraining](https://arxiv.org/abs/2509.15248) (ICLR 2026) | Learns how related documents relate, then synthesizes new documents | Relations between documents | Produces pretraining data; it isn't discovery |
| Zhou et al., [HypoGeniC](https://arxiv.org/abs/2404.04326) (2024); [HypoBench](https://arxiv.org/abs/2504.11524) (2025) | Proposes hypotheses from labelled examples, updates them bandit-style, and scores them on held-out examples. HypoBench includes synthetic datasets | Propose then test on held-out data; synthetic ground truth | Hypotheses about labels (classification). No time order, no learning across tasks in the weights |
| Qian et al., [HypoForge](https://arxiv.org/abs/2608.25770) (Aug 2026) | Learns reusable skills for generating and testing hypotheses from earlier runs, without fine-tuning | An accumulated discovery skill | Skills live in prompts, not weights; different benchmarks. **The model for baseline B4 below** |
| Ouyang et al., [ReasoningBank](https://arxiv.org/abs/2509.25140) (ICLR 2026) | Distils strategies from successes *and* failures into a memory the agent retrieves from | "Which strategies paid off", and keeping failures | In-context memory, agent tasks |
| [Organon](https://arxiv.org/abs/2610.33046) (2026), search summary only | A persistent agent that revises its own search strategies and memory from experience | Learning a search strategy | Improves within a task; transfer speed not reported |
| Jansen et al., [DiscoveryWorld](https://arxiv.org/abs/2406.06769) (NeurIPS 2024) | Simulated discovery tasks whose parameters change every run | A benchmark with fresh discoveries each time | Running experiments, not reading records |
| [HistoryBankQA](https://arxiv.org/abs/2509.12720) (2025), RecurrenceQA | Asks when earlier rounds of known recurring events happened | Periodicity in historical events | Recall of known events, not discovery |
| Older: STaR (Zelikman et al. 2022); Voyager (Wang et al. 2023) | STaR trains on its own successful reasoning. Voyager reuses a learned skill library in a new Minecraft world | Training on what worked; a skill that carries to a new world | Not about discovering patterns in material |

## What this changes in the plan

1. **Narrow the H2 claim:** *sleeping on verified findings and on which comparisons paid off makes a small model faster at discovering unstated patterns in material it has never seen.* We won't claim the sleep framing, training on its own output, or "learning to learn" in general.
2. **Add controls, so the claim holds up against the closest work** (now in [PLAN.md](PLAN.md#baselines)):
   - **B4, written notes instead of sleep** (in the style of HypoForge, ReasoningBank and Auto-Dreamer). Before a new world, the learner gets a written list of what kinds of comparisons paid off, in its prompt, with no weight update.
     - If B4 does as well as sleep, the speed-up is real but the weights add nothing. That is still a useful result, but not the one we want.
   - **Sleep on facts only vs facts plus search outcomes.**
     - Facts from training worlds can't help on a new world, because every world has fresh names.
     - So any speed-up from facts-only sleep is a general skill, and the gap between the two arms shows what the search outcomes add.
   - **Sleep on unchecked findings** (B1's output). This shows whether checking before training matters.
3. **H3:** Lee et al. support betting that more recurrence helps. Our version is about discovery, on an already-trained looped model.
4. **H4:** EntiGraph-style "write about how these two things relate" text is a candidate way to get facts into the weights.
