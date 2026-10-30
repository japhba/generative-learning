# Self-generated practice and elicitation: results

The full-trace recipe received a fresh-data confirmation with three optimization
seeds. Final reasoning accuracy averaged **97.5% with a current source**
versus **92.6% with a frozen source**, from a common
85.4% baseline. The paired difference is +4.9 percentage
points, with a 95% paired seed-and-question bootstrap interval of
[+1.2, +9.7] points. This is one model and a small arithmetic task;
it does not establish distinct scaling exponents or successful short-answer compression.

The [standalone report](../reports/elicitation-pilot.html) includes the full curves,
failed recipes, an interactive confirmation checkpoint table, and downloadable
analyzed predictions and provenance.

## Model and training

The learner is **Qwen/Qwen2.5-1.5B-Instruct**, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. All weights are fine-tuned with AdamW;
this is not adapter training or an added classifier. The successful recipe uses
FP32 weights/state, BF16 autocast, learning rate 1e-6, no decay, clipping at 1, one
update per batch of 16, and completion-token NLL summed then divided by the fixed
value `128 * batch_size`.

Every arm receives the same 96 fixed arithmetic programs in four shuffled passes.
Each program specifies a starting integer and 4, 6, or 8 arithmetic operations.
Both main arms obtain four fresh reasoning samples per presentation. Internal
plurality over completed parsed final answers selects an actual full reasoning
trace. Incorrect targets remain; no exact answer or reward selects them. The only
main-pair intervention is whether the source retains its initial weights or uses
the current learner after each batch. This is on-policy self-training with a
likelihood loss, not RLVR. Exact answers are used only for offline evaluation.

## Three distinct observations

1. **Elicitation without weight updates:** on 144 fresh confirmation questions,
   the original weights achieved 0.7% with short answers,
   80.6% with greedy reasoning, and 88.2%
   with four-sample consensus. The response procedures use different compute.
2. **ICL with self-produced examples:** retaining 4 or 16 full traces raised
   reasoning accuracy from 84.0% to 94.4% on a separate cohort. This is one
   exploratory demonstration sequence. Answer-only context did not help.
   Full-trace context caused almost all short-mode responses to truncate.
3. **Weight learning:** the initial one-seed full-trace pilot reached 100.0%
   current-source versus 87.5% frozen-source reasoning accuracy, from 84.0%.
   It motivated the unchanged fresh-data, three-seed confirmation above.

Confirmation seed accuracies, in order 71000, 71001, 71002:

| Source | Reasoning accuracy by seed | Mean short-answer accuracy |
| --- | --- | --- |
| Frozen | 95.8%, 90.3%, 91.7% | 0.5% |
| Current | 97.2%, 97.2%, 97.9% | 1.6% |

The confirmation uses new practice seed 71100 and evaluation seed 71200. Its three
training seeds share 144 test questions; these are not 432 independent questions.
All six runs finished before correctness scoring. Intervals are limited by three
seeds and one fresh cohort. Perfect observed scores can yield degenerate bootstrap
bands without proving perfect population accuracy. The confirmation repeats the
same architecture, not another model family.

All three optimization seeds favor the current source, but the mean gap is smaller
than the exploratory pilot's 12.5 points. At 192 presentations the current arm
already achieved 96.5% using about 246,000 generation-plus-training input tokens;
the frozen arm reached 92.6% at 384 presentations and about 452,000 tokens. At equal
final presentations the current arm used about 9.2% more such tokens. These are
measured work proxies, not FLOPs or a fitted scaling law.

## Retained negative results

Answer-only consolidation used three seeds, full-weight AdamW at 1e-5, and four
updates per batch. It did not improve answers on new programs. Reasoning-prompt
outputs shrank from about 144 tokens to 7–9 tokens after the first batch. The
frozen-source learner partially memorized practice answers; the current-source
learner also lost the ability to produce useful practice targets.

In the full-trace pilot, unfiltered one-rollout self-imitation fell to 16.7%
reasoning accuracy. It is a mechanism control with fewer rollouts, not an equal-cost
competitor. A zero expected raw score gradient does not imply zero finite-sample
parameter motion, especially with clipping, AdamW, and finite precision.

The full-trace follow-up changed the prompt, target representation, learning rate,
normalization, and update count together. Its difference from the failed recipe
cannot be attributed solely to retaining reasoning traces. The fresh-versus-frozen
comparison within each recipe does hold these choices fixed.

The screened math-specialized model ignored the short-answer instruction; its
96-token responses truncated. A parser amendment later admitted unambiguous final
integers without LaTeX boxes, before the general model's fresh confirmation. Raw
responses, original strict scores, and both dated amendments remain available.

## Linear probabilistic mechanisms

The [original toy](LINEAR_ELICITATION_TOY.md) supplies an already useful generator
`x = Az + noise` and an imperfect recognition map `Bx`. Internally generated pairs
align B with `B* = Aᵀ(AAᵀ + sigma²I)⁻¹`. Mean error falls from 8 to 1.8566, near
the optimal 1.8422. The weighted error is exactly the squared distance from B to
B*. This toy alone does not compare refreshing a sampler.

The [posterior-refresh extension](POSTERIOR_REFRESH_TOY.md) initializes one latent
posterior step with current versus frozen recognition weights. It uses the same
8×8 generator and an exact-invariant linear Gaussian posterior transition. At the
same internal sampling budget, current initialization gives error 2.1024 versus
5.3060 for frozen initialization. Normalized weight overlap is 0.9300 versus
0.2969. Stable population dynamics approach the optimal inverse in the current
arm and a biased one-step teacher in the frozen arm.

The extra posterior computation is an explicit asymmetry. No new facts or oracle
latents enter these updates. This supplies a possible refresh mechanism within a
linear model; it does not reduce LLM arithmetic to those same weight dynamics.
The toy's geometric modes and biased frozen limit do not establish different
power-law exponents. The LLM curves are too short and saturate too quickly for a
credible exponent comparison.

## Reproduction and resources

Methods: [elicitation](ELICITATION_PROTOCOL.md),
[answer-only consolidation](ELICITATION_CONSOLIDATION_PROTOCOL.md),
[full-trace pilot](ELICITATION_TRACES_PROTOCOL.md), and
[fresh-data confirmation](ELICITATION_TRACE_CONFIRMATION_PROTOCOL.md).

Audits cover 18 learning cases and 90 checkpoints: source and data hashes, exact
sampled target tokens, causal schedules, counters, paired initial teacher samples,
and baseline predictions. Eight task/parser checks pass. The toy additionally
verifies posterior normal equations, risk decomposition, transition invariance,
and population-update stability. No paid APIs or Vast rentals were used; new paid
spend is $0. Training uses institutional H100s through `gpu_lowp` only.
All completed allocations together used 103.5 H100 GPU-minutes, including
screening, evaluation, and archival; optimizer work alone is recorded separately.

Final models retained for answer-only seed 68000, full-trace pilot seed 69000, and
confirmation seed 71000 were archived to the configured shared cache with SHA-256
verification. All training allocations completed and released their GPUs. Full
token and timing records remain in the run outputs. Counts of generation tokens
plus training input tokens are work counters, not FLOPs; the main pair matches
question, rollout, and optimizer counts but allows actual trace lengths to differ.

Rebuild the report after the independent analyzers have completed:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/build_elicitation_report.py
```
