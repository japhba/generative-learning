# Reader–source specificity experiment

Frozen before experimental inference on 12 September 2026. This is a targeted
test following the earlier task search, not a new broad search or a preregistration
with an external registry. All episodes are freshly seeded.

## Question

Does the preferred rehearsal-answer source change with the reader? A universally
better source is insufficient evidence of model specificity. Identical saved text
has no memory of its generator in a fresh inference call.

Readers and generators are local Qwen3.5-9B (the existing pinned BF16 checkpoint)
and Gemma 4 26B A4B IT via OpenRouter, pinned to DekaLLM's BF16 endpoint, with
provider fallback disabled. Both run with thinking disabled. The remote serving
revision and kernel stack are not under our control; complete response metadata
are retained. An unrelated JSON-echo infrastructure probe precedes this protocol.

## Fixed design

- Three settings: three-factor product decomposition with K=0 and K=4 initial
  exact teacher pairs, and Bézout with K=0, coprime coefficients from 3 through 19.
  **Two or more factors equal to 1 are forbidden in every factorization question**,
  including support, rehearsal, generation, and evaluation in every source arm.
  The factorization settings share question episodes across K. Selection follows
  the strongest existing constrained-factorization result and the recent Bézout
  candidate; no settings are selected using this experiment's results.
- 16 episodes per setting, 32 rehearsal questions and 12 test questions per
  episode. Within each episode all support, rehearsal and test questions differ.
  Finite task domains can repeat questions across independent fresh calls.
- Rehearsal counts N = 0, 2, 8, 32. N=32 is the sole confirmatory endpoint.
- Every reader sees exact teacher answers, Qwen answers, and Gemma answers to
  **the same ordered rehearsal questions**. Initial support, test question,
  final prompt template and decoding are identical across sources within reader.
  There are no source/author labels. N=0 is one shared initial-only baseline.
- Each generated answer is sampled independently offline, conditioned only on
  the fixed initial support and its assigned rehearsal question, at temperature
  0.7 and top-p 0.9. Evaluation uses temperature 0 and top-p 1. Maximum output is
  256 tokens for both roles and both models. The saved stripped content, including
  an incomplete prefix if truncated, is replayed without validation, filtering,
  repair, selection, or added advice. Empty content occupies its original slot.
  This differs from the earlier 1024-token/empty-truncation protocol, so earlier
  numerical accuracies are context, not pooled data.
- Generation seed is the same for the two generators; evaluation seed is the
  same across sources for a given episode/query. Seeds are not a guarantee of
  deterministic execution across servers. Complete requests are cached for
  resumption. No training, verifier feedback, or cross-call model state.
- Randomize request scheduling reproducibly to interleave settings and sources;
  do not read partial accuracy curves. Finish the entire fixed cohort before
  primary analysis. No adaptive endpoint, model or task changes based on outcomes.
- Check exact teacher validity, within-episode disjointness, source-blind request
  reconstruction, actual provider, absence of returned reasoning, completion
  statuses and context budgets. Context overflow is a failed design check, never
  grounds for selectively omitting an episode. Strict JSON evaluation accepts any
  mathematically valid answer, with truncated final completions incorrect.
- Remote billed-cost ceiling $5, checked before additional calls. Expected cost
  substantially below this ceiling; no second model download is needed.

## Analysis

Let a[r,s] be test accuracy for reader r with source s, where Q and G denote
Qwen and Gemma. The primary endpoint is the mean own-source preference:

    H = ((a[Q,Q] - a[Q,G]) + (a[G,G] - a[G,Q])) / 2.

Twice H is the reader-by-source interaction. A source main effect cancels from H.
Report the entire 2×2 matrix, both reader-specific preferences, both initial-only
baselines, and both teacher comparisons. H>0 alone need not mean *both* readers
prefer their own source, or that either improves over its baseline.

Use paired bootstrap confidence intervals over the 16 complete episodes (20,000
resamples) and an exact one-sided sign-flip test of episode-level H (65,536 sign
assignments). The sign test relies on symmetry/exchangeability of episode
contrasts; it is not a randomized-treatment permutation guarantee. Holm-correct
the three primary H tests. Individual own preferences, gain versus initial and
teacher, and all N<32 results are descriptive, with paired intervals. Plot all
learning curves; do not fit a power-law exponent to four points.

Post-inference diagnostics describe answer validity, length, duplicate content,
and factorization strategies for each generator. They never change the replayed
samples. A positive interaction can also reflect differences in difficulty,
correctness sensitivity, verbosity or formatting; it does not establish shared
neural pathways. Additional matched-quality and representation interventions
would be needed to isolate that account.
