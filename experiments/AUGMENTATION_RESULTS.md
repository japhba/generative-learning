# Teacher supervision with supplemental generated targets

This follow-up preserves every canonical teacher example and adds either a verified
model-generated solution or a repeated teacher solution on the same accepted
problems. It addresses the reduced teacher access in the original replacement
experiment. The task, three training seeds, warm starts, learning rate and core
training implementation are unchanged. The exact design is in
[AUGMENTATION_PROTOCOL.md](AUGMENTATION_PROTOCOL.md); the original negative result
remains in [TRAINING_RESULTS.md](TRAINING_RESULTS.md).

All six new runs completed on Qwen3-0.6B-Base. Keeping teacher supervision improved
the generated-target arm relative to the earlier rollout-only experiment, but its
advantage over matched teacher replay is small and inconsistent across training
seeds. The original evaluation's primary difference is +2.08 percentage points,
with a three-seed bootstrap interval spanning zero. The learning curves are
irregular, so these results do not establish distinct fine-tuning scaling exponents.

## Endpoint results on the original evaluation identities

Each cell is mean exact complete-trajectory accuracy across the three training
seeds, with 128 held-out problems per stratum per seed. The endpoint is 4,096 new
prompts after the common warm start.

| Training method | Easy | In distribution | Long horizon |
| --- | ---: | ---: | ---: |
| Historical ordinary SFT | 47.40% | 35.68% | 20.57% |
| Teacher + matched teacher replay | 54.43% | 40.10% | 23.96% |
| Teacher + verified generated targets | 53.65% | 42.19% | 27.60% |

The primary generated-minus-replay ID differences are -3.125, +0.781 and +8.594
percentage points for seeds 6200, 6201 and 6202. Their mean is +2.083 points, with
95% paired seed-bootstrap interval [-3.125, +8.594]. The secondary long-horizon
difference is +3.646 points, interval [-2.344, +9.375]. With only three seeds these
intervals have very limited resolution; neither establishes a reliable generated
target advantage. Intervals are pointwise and unadjusted for multiple comparisons.

Both augmented methods beat historical ordinary SFT at this endpoint in every seed
and stratum. They also receive more examples and optimizer updates. This does not
establish better compute efficiency or an effect specific to generated solutions.

## Learning curves and resource accounting

| Mean per seed, after warm start | Ordinary SFT | Teacher replay | Teacher + rollout |
| --- | ---: | ---: | ---: |
| New prompt identities | 4,096 | 4,096 | 4,096 |
| Training example presentations | 4,096 | 6,843 | 6,843 |
| Optimizer updates | 256 | 428.33 | 428.33 |
| Reported train + rollout tokens | 469,009 | 782,453 | 3,249,246 |
| Training + rollout phase seconds | 23.71 | 43.61 | 379.91 |

Generation therefore costs 8.71 times the matched replay arm's own measured phase
time. Replay inherits the generating arm's selected curriculum, so its standalone
time excludes the cost of acquiring that curriculum. Reported tokens count the
training input/completion stream and rollout input/output; they are not hardware
FLOPs. Phase times exclude evaluation, loading, checkpointing and the shared warm
start. Historical SFT timing is not a simultaneous controlled benchmark.

The generated arm supervises about 120,287 completion tokens per seed, versus
95,871 for replay. Matching examples and updates therefore leaves different target
token counts and token weighting. Its accepted supplements average 4.8–4.9 moves,
while the corresponding shortest teacher paths average 2.9–3.0 moves.

The ID mean curves dip repeatedly before the last checkpoint: generated accuracy
is 22.66%, 20.31%, then 42.19% at 1,024, 2,048 and 4,096 prompts; replay is 19.27%,
17.97%, then 40.10%. Fitting a smooth power law over a selected favorable segment
would conceal this structure. All prescribed checkpoints are reported on prompt,
example, token and phase-time axes.

## Mechanistic observations

The new generating runs accept supplements for 66.09%, 63.50% and 71.61% of prompts.
Only 27.45%, 29.18% and 28.20% of accepted plans match the canonical teacher plan;
the rest are verified alternatives. This measures target diversity, not proof that
each alternative supplies new task information.

Across the first versus last 128 prompts, the mean number of distinct completions
among four attempts changes from 3.836 to 3.273, 3.859 to 3.664, and 3.945 to 3.477.
The earlier rollout-only runs narrowed much more strongly (see
[STRATEGY_DYNAMICS.md](STRATEGY_DYNAMICS.md)). Continued teacher access is associated
with retaining more sampled diversity in this intervention; teacher access,
additional updates and mixed target weighting change together.

This retained sampling diversity does not mean fully input-dependent greedy
planning: all six final models begin with action A on every one of their 384
original evaluation prompts. All three operations occur later in their plans.
The initial-action bias therefore survives this intervention.
Of the generated arm's 678 failures pooled across the three original endpoint
test sets, 626 execute a correct eight-move horizon without reaching the goal.
This diagnostic suggests path selection remains a major limitation, even where
transition arithmetic is correct; it does not change the strict success score.

All 63 prescribed comparisons of weight-displacement directions are available:
three seeds, seven nonzero budgets, and query-projection matrices in layers 0, 14
and 27. At the endpoint, generated-versus-replay displacement cosines range from
0.230 to 0.458. The arms move in different parameter directions despite their
shared prompt stream. These are directions relative to the common warm start,
not overlap with known true task weights, and do not establish which direction
is mechanistically better.

## Comparisons and limitations

The primary contrast is teacher-plus-rollout versus teacher-plus-matched-replay
in-distribution accuracy after 4,096 new prompts. Both methods receive all 4,096
canonical teacher targets. Every accepted supplemental prompt occurs at exactly
the same stream position and optimizer minibatch boundary in the two arms. Their
supplemental targets differ: the actual accepted generated completion versus the
canonical shortest teacher completion. No teacher solution from the current batch
is trained before that batch's four proposals per problem are sampled.

The comparison includes differences in solution lengths and token weighting.
Token-mean cross-entropy gives longer generated targets more supervised tokens
within a mixed minibatch. Equal prompt identities, example counts and optimizer
updates do not equalize compute or isolate semantic content. The matched replay
control inherits the generating learner's adaptive acceptance curriculum; it does
not generate or choose its own supplements.

The original SFT curve is historical context with the same prompt stream and warm
weights. It sees fewer example presentations and optimizer updates. A gain over
that baseline can therefore reflect additional practice, even without new targets.
Its old phase time is contextual rather than a simultaneous hardware benchmark.

All three seeds and all prescribed checkpoints are retained. The intervention was
chosen after inspecting the previous evaluation results, so the repeated-identity
comparison is exploratory. We do not select a new exponent or favorable checkpoint
after looking at these curves.

## Additional fresh-identity validation

A separate set was frozen at 2026-09-05 21:29:18 UTC, using sampling seed 93201 and
256 problems in each shortest-distance stratum. It excludes all 1,413 identities
from previous evaluation datasets. The eligible easy, in-distribution and long
pools contain 5,122, 9,694 and 662 identities, respectively. Sampling is uniform
without replacement within each stratum, and the same 768 questions are used for
all nine final checkpoints.

This secondary check was designed after earlier results and partial follow-up
results were available, but before any fresh-test predictions were inspected.
It is not the original preregistration or a new set of independent training seeds.
The original primary comparison remains unchanged, and no training, hyperparameter
adjustment or intermediate-checkpoint selection follows the fresh evaluation.

Evaluation uses the original greedy decoder, strict trajectory verifier and EOS
requirement, with model gradients disabled. Source versions, model-file hashes,
full predictions and independently checked input/output token totals are saved.
Uncertainty resamples the three training seeds, conditional on the common test
set; it does not treat reused questions as additional independent model runs or
include uncertainty from sampling the test identities.

All nine final checkpoints completed this evaluation, yielding 6,912 independently
regraded trajectories. Mean accuracies are:

| Final checkpoint | Easy | In distribution | Long horizon |
| --- | ---: | ---: | ---: |
| Historical ordinary SFT | 45.05% | 34.64% | 19.92% |
| Teacher + matched teacher replay | 50.26% | 38.41% | 23.05% |
| Teacher + verified generated targets | 49.09% | 41.54% | 25.65% |

The generated-minus-replay ID difference is +3.125 percentage points, paired
seed-bootstrap interval [-3.516, +11.719]. Its seed differences are +1.172,
-3.516 and +11.719 points. The long-horizon difference is +2.604 points, interval
[-1.172, +5.859]. Both intervals span zero. The small mean advantage persists,
but these data do not establish a reliable benefit specific to generated targets.

The fresh results also weaken the consistency of the comparison with historical
ordinary SFT: seed 6201's generated arm is worse than SFT in all three strata, and
its replay arm is worse in ID and long-horizon accuracy. Average improvements
therefore should not be described as universal across seeds. Fresh evaluation
used 693,198 prompt tokens, 305,779 output tokens, and 143.73 seconds of recorded
generation time; no parameter updates or paid API calls occurred.

## A mechanistic link to the linear toy

The separate [weight-coverage toy](WEIGHT_COVERAGE_TOY.md) now supplies explicit true
weights, one-layer SGD updates, a Bayesian interpretation, and a precise categorical
Boltzmann representation of the data distribution. Its dense-weight simulation
checks the formulas numerically across 128 paired dictionaries.

At unit learning rate, normalized weight overlap equals observed probability mass,
and squared weight error equals posterior Bayes classification error. Active
sampling improves coverage, while generating guesses adds no information about
independent unknown labels. At a smaller learning rate, repeated supervision is
still useful for fitting already observed weights, and those metric identities
separate. At learning rate 0.25 and 128 presentations, ranked sampling has about
4.85% classification error versus 13.79% for IID, yet its weight MSE is worse:
0.443 versus 0.182.

This makes the distinction between acquiring new information and fitting known
targets concrete. It does not prove a cause for the transformer results: the toy
has independent lookup rows, while the LLM shares parameters across problems and
learns sequence predictions. Its coverage exponents do not automatically predict
the LLM's fine-tuning rates.

## Verification, artifacts and reproduction

The 50 offline unit tests passed across the existing and new suites. The main
analysis additionally reverified all 49,152 rollout candidates, the first accepted
target, teacher coverage, complete mixed training streams, minibatch carry-over,
counters, source/data hashes, and identical warm-start predictions. A separate
review regraded all 2,304 new-arm original endpoint trajectories. The fresh-test
audit independently reconstructed its sample using separate BFS/split logic and
regraded all 6,912 final predictions; its record is
`results/training_augmentation_fresh_eval/independent_audit.json`. The dense linear
toy passed 4,096 matrix/metric checks. These checks validate implementation and
accounting, not statistical certainty about the scientific effect.

Shared archival completed after evaluation: preselected seed 6200's two new final
models, all three parent warm starts, and every new weight probe were verified by
file list, byte count and SHA-256. There are 98 archived files totalling 12.82 GB,
including 48 probe files. Seeds 6201 and 6202's new final models remain only in the
node-local fast cache. Exact storage details and reproduction commands are in
[AUGMENTATION_PROTOCOL.md](AUGMENTATION_PROTOCOL.md). The permitted H100 allocation
was released after 48 minutes; training, evaluation and archival steps completed
successfully before the holding job was cancelled. No paid API calls were made;
the conservative cumulative OpenRouter accounting remains $0.37499892.

Machine-readable reports are `results/training_augmentation_analysis/summary.json`
and `results/training_augmentation_fresh_eval/summary.json`. Regenerate analyses
from retained small artifacts and the shared weight probes:

```bash
OPENBLAS_NUM_THREADS=4 "$VENV_LOCAL/generative-learning/bin/python" \
  experiments/analyze_augmentation.py \
  results/training_augmentation_seed6200 results/training_augmentation_seed6201 \
  results/training_augmentation_seed6202 --output results/augmentation_reanalysis \
  --weight-cosines --weight-probe-root /ceph/scratch/jbauer/generative-learning/training
"$VENV_LOCAL/generative-learning/bin/python" experiments/evaluate_augmentation_fresh.py \
  summarize --output results/training_augmentation_fresh_eval
"$VENV_LOCAL/generative-learning/bin/python" experiments/plot_augmentation_fresh.py \
  results/training_augmentation_fresh_eval/summary.json
```

The final command draws a readable single-panel overview from the frozen summary;
it does not modify the original evaluator, its archives, predictions or statistics.
Plot dots denote training seeds and lines join means across the three test strata.
Selected figures:

- `results/training_augmentation_fresh_eval/fresh_accuracy.png`
- `results/training_augmentation_analysis/in_distribution_problems_presented.png`
- `results/training_augmentation_analysis/in_distribution_phase_seconds.png`
- `results/training_augmentation_analysis/rollout_diversity.png`
- `results/weight_coverage_toy/weighted_mse.png`
- `results/weight_coverage_toy/classification_error.png`
