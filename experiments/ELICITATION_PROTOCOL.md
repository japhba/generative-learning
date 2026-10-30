# Self-generated practice and the elicitation gap

This document records the initial plan. Two dated amendments are retained in
`results/elicitation_pilot/amendment_1.json` and `amendment_2.json`: require short-mode
completion after the math model ignored that instruction, then accept unambiguous
explicit final integers without LaTeX boxes. Both changes preceded the general
model's fresh confirmation cohort, seed 63000. The original strict scores and the
earlier math-model confirmation at seed 62000 remain available. The latter does not
establish a meaningful short-versus-reasoning gap because short responses truncated.

The target is an improvement in access to existing information through a model's
own generated computation. The external question distribution is fixed. This
replaces the earlier coverage experiment as the candidate mechanism; it does not
relabel IID/ranked acquisition as recognition/generation.

## First gate: an implementable elicitation advantage

The pilot considers arithmetic programs and explicitly specified six-state
permutation machines. All transition rules are visible in each question. No
unknown dictionary labels are acquired. Programs have four, six or eight steps.
The exact simulator is used only by a separate analyzer. The generator, answer
parser and plurality selector never inspect the correct answer.

Two small open models are considered in a fixed order: Qwen2.5-Math-1.5B-Instruct,
then Qwen2.5-1.5B-Instruct only if the first model supplies no passing task family.
Their exact Hugging Face revisions are recorded in the executable protocol.

Calibration uses 16 questions per family/depth (96 total), seed 61000. Each question
has a short greedy answer (96 new tokens), a concise greedy reasoning answer
(512 tokens), and four independent sampled reasoning answers (temperature 1,
512 tokens each, no top-k/top-p truncation or repetition penalty). The final answer
is an integer in a box; a standalone integer is also parsed. Truncations and parse
failures count as wrong. Plurality uses completed parsed answers, breaks ties by
first occurrence, and never filters by correctness. An entirely invalid group
is an abstention and remains in all accuracy denominators.

Choose the first model/family in the declared order for which plurality accuracy
is at least 70%, the gain over short greedy answers is at least 15 percentage points,
and at least 95% of sampled responses complete. Run a single independent
confirmation using 48 questions per depth (144 total), seed 62000, with the frozen
same conditions. Report paired whole-problem bootstrap intervals and per-depth
outcomes. The pilot does not fit a scaling exponent. Oracle pass@4 is explicitly
an evaluation diagnostic, never an implementable teacher or training selector.

## Subsequent learning comparison

A demonstrated gap licenses a prospective test of whether self-generated practice
can consolidate the gain. The planned primary contrast is fresh current-model
rollouts versus rollouts from a frozen initial model, on the same fixed questions.
Both use the same internally computed target rule. Ground-truth selection is a
separate condition if studied, and cannot be called reward-free self-training.

Before weight training, freeze the cohort, optimizer, budgets and policy refresh
schedule on new training-only problems. Retain the initial-model no-training
baseline, a direct self-imitation control, and a supplied-answer reference if
needed. Evaluate at fixed response budgets; account separately for all generated
tokens, optimizer work, accepted targets and exogenous observations. Learning
curves should first establish a repeatable mechanism; different fitted exponents
are not a selection criterion.

An ICL analogue retains selected self-generated practice in context. A linear
Gaussian wake–sleep analogue trains recognition weights on samples produced by
its own generator. These are candidate implementations of the same elicitation
mechanism, not an assertion of identical gradient dynamics or existing evidence
for faster on-policy learning.

## Scientific limits and provenance

Identically distributed, unfiltered self-imitation has zero expected likelihood
gradient at the sampling parameters. Here the candidate asymmetry is extra
reasoning and aggregation in the target-producing procedure. A positive
elicitation gap alone does not demonstrate that consolidation or on-policy
refresh improves learning. A generator can also propagate its own errors.

All source hashes and data are written before the first response. Existing Zipf
results remain frozen. Computation uses only institutional gpu_lowp; no paid API
requests or Vast rentals are needed for the pilot. Models and large caches stay
in the configured fast cache. SLURM operational records stay under ignored results.

Relevant background: [usable information](https://arxiv.org/abs/2002.10689),
[wake–sleep](https://www.cs.toronto.edu/~hinton/absps/ws.pdf), and
[reward-free online self-training](https://arxiv.org/abs/2510.18814).
