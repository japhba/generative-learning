# What generation can add: corrective state coverage

This investigation tests the claim that generation can improve learning by exploring
parts of the possibility space that demonstrations miss. The useful distinction
is which states acquire informative labels. Every LLM arm uses supervised next-action
cross-entropy; exploratory arms execute actions and obtain teacher corrections at
the states those actions create.

The result is positive for this specific mechanism: exploratory actions plus
corrective feedback remove a persistent recovery-error floor in both Qwen3-0.6B-Base
and SmolLM2-135M, across three training seeds each. Random and frozen-model exploration
reach the same endpoint. This supports generation as a way to obtain useful training
states, without establishing an advantage unique to adaptive self-generation.
The linear experiment exposes the corresponding identifiable weight direction exactly.

## Why the initial planning experiments were insufficient

The earlier rollout-distillation runs retained successful solutions. They omitted
most states reached after mistakes, which are precisely where corrective feedback
can be useful. The literature already develops this approach:
[DAgger](https://arxiv.org/abs/1011.0686) trains with labels on learner-induced states,
and [on-policy distillation](https://arxiv.org/abs/2306.13649) applies teacher feedback
to student-generated language-model sequences. Our experiments isolate a small
version of this mechanism rather than claiming a new general method.

A further modular-planning pilot used stepwise actions and exact oracle corrections.
All four final models repeated a single action across every validation trajectory.
The expert, current-policy, random and frozen-policy conditions differed in success
rate, but those differences were explained by which constant action they selected.
All four runs and their independently regraded trajectories are retained in
`results/feedback_pilot_seed7310` and `results/feedback_pilot_analysis`.

## A controlled task with an explicit missing direction

The tracking state is error `e`, the requested motion `u` is an integer from -2 to 2,
and action `a` is an integer from -3 to 3. The simulator applies

```text
e_next = e + a - u.
```

The oracle action is `clip(u-e,-3,3)`. Expert training episodes begin at zero error,
so an expert stays at zero and only demonstrates five `(u,e)` inputs. Exploratory
actions create nonzero errors. The teacher labels those states with corrective
actions, including states reached by mistaken actions. The task prompt supplies
the dynamics and objective, so pretrained knowledge can also contribute.

All conditions use the same requested-motion stream, new-label count, replay count,
target-token count and optimizer updates. They differ in the errors they visit:
expert actions, current-model exploration, random actions, or a frozen initial
model's exploration. Model-based exploration explicitly mixes 90% model sampling
with 10% uniform random actions. Generated actions are not assumed correct.

The model chooses among seven legal next-action tokens; the simulator performs
state arithmetic. This is full-weight LM fine-tuning used as a constrained control
policy. It does not measure unrestricted text generation, policy-gradient RL, or
a recognition architecture versus a generative architecture.

## Calibration and retained unfavorable results

The first tracking pilot, seed 7510 with learning rate 1e-4, did not reliably master
or retain nominal tracking. The next pilot, seed 7511 at 3e-5, mastered the five
nominal training inputs but could lose that behavior during continued expert-only
training. These are optimizer/task-calibration failures, not confirmation of the
scientific claim. Their complete results remain in `results/tracking_pilot_analysis`
and `results/tracking_calibration_analysis`.

A stability pilot, seed 7512, reduced the learning rate to 1e-5, added label smoothing
0.1 and preserved warm optimizer moments. This combination retained perfect nominal
tracking under continued training. The changes were not individually ablated, so
the investigation does not identify which stabilization was necessary.

At 4,096 labels in the stability pilot, mean recovery error was 2.568 for expert
demonstrations and 0.01758 for each exploratory condition, equal to the exact oracle
on those episodes. Nominal error remained zero for all four arms. This motivated
the unchanged three-seed confirmation and separate model-family replication, with
selection records saved before their predictions. Pilots are not pooled with them.

## Frozen confirmation design

Each model uses training seeds 7520–7522 and 8,192 post-warm oracle labels. The warm
start trains on 2,048 expert examples per epoch until two consecutive epochs predict
all five nominal training inputs correctly, capped at 20 epochs. This calibration
uses training inputs only. Warm examples and optimizer moments are then shared
across the four arms within that seed.

Each round acquires 512 labels, adds 512 uniform replay examples from accumulated
data, shuffles, and takes 16 optimizer steps of batch size 64. AdamW updates all
float32 model parameters with bfloat16 forward autocast, learning rate 1e-5, no
weight decay, and gradient clipping at norm 1. Conditional cross-entropy uses
90% oracle one-hot plus 10% uniform target smoothing. The corresponding categorical
teacher is a Boltzmann distribution with an optimal-action logit bonus `log(64)`;
its normalization is 70. This is an interpretation of the fixed training target,
not an RBM learning experiment.

The common test set contains 256 fresh 16-step command sequences per condition:
zero initial error, initial error uniformly in +/-1 through +/-3, and larger initial
errors uniformly in +/-4 through +/-8. The primary metric is mean absolute error
over recovery trajectories at 8,192 labels. The exact oracle has a small positive
error because action limits can prevent immediate correction. The secondary
stability metric requires zero error on every one of the final eight steps.

Recovery explicitly intervenes on the initial state outside expert demonstrations.
Individual Markov inputs may repeat between acquisition and fresh evaluation
episodes. The task tests learning a controller and robustness to initial deviations,
not generalization to an entirely unseen set of numeric inputs. No fresh test
trajectory or feedback enters training.

## Confirmation results

All 24 main arms completed. Each endpoint below averages three training seeds on
the same 256 fresh recovery episodes, after 8,192 acquired state labels. Lower is
better. The oracle's positive error reflects bounded actions during initial recovery.

| Acquisition policy | Qwen3-0.6B recovery error | SmolLM2-135M recovery error |
| --- | ---: | ---: |
| Expert demonstrations | 2.21248 | 2.60710 |
| Current-model exploration + feedback | 0.01831 | 0.01831 |
| Random exploration + feedback | 0.01831 | 0.01831 |
| Frozen-model exploration + feedback | 0.01831 | 0.01831 |
| Exact oracle on these episodes | 0.01831 | 0.01831 |

The primary on-policy-minus-expert differences are -2.19417 for Qwen, with paired
seed-bootstrap interval [-2.48242, -1.93872], and -2.58879 for SmolLM2, interval
[-2.62939, -2.52124]. All six seed-level differences favor exploratory feedback.
These intervals resample only the three training seeds per model and condition on
the common episodes; their small-seed resolution is limited.

Nominal tracking is perfect at every prescribed checkpoint in every main arm.
At the endpoint, all exploratory conditions also achieve zero error on all final
eight steps of every recovery and larger-shift episode. Expert-only conditions
do not recover on these perturbed episodes. Larger-shift mean error is 5.96875
for expert-only training and 0.26733 for every exploratory condition, in both models.

The advantage is not present at every budget. At 512 acquired labels, Qwen's mean
recovery error temporarily rises from 2.32 to 13.01 for current-model exploration;
SmolLM2 rises from 2.63 to 3.68. By 1,024 labels, those errors are 0.0736 and 1.8070;
by 2,048 they are 0.0212 and 0.1198. Every exploratory condition reaches the oracle
endpoint by 8,192. The curves show acquisition costs, transient incorrect recovery
behavior, and saturation; they do not justify fitting distinct power-law exponents.

Expert-only training visits exactly five distinct inputs in every run. Current-model
exploration visits 109–115 inputs in Qwen and 124–142 in SmolLM2; random exploration
visits 237–238. The important change is coverage of nonzero-error states, not merely
additional presentations of the original five examples.

## Behavioral control coefficients

The 29-input probe fits decoded action to `intercept + b*u + c*e` on the region
where the ideal action is unsaturated. Its ideal coefficients are `(0,1,-1)`.
Both models' warm and expert-only policies get only 5/29 grid inputs correct.
Their fitted corrective coefficient is close to zero or has the wrong sign:
Qwen expert coefficients are 0.0482, 0.0602 and 0.0000; SmolLM2's are 0.1506,
0.1867 and 0.0964. Every exploratory final policy gets all 29 inputs correct,
with fitted coefficients `(0,1,-1)` up to numerical regression precision.

These are effective input-response coefficients, not elements of the transformer
weight matrix. They show acquisition of the corrective behavior predicted by the
linear model, without identifying a unique internal circuit or a true transformer
parameter vector. Actual query-projection displacement probes are also retained.
The measured model sizes are 596,049,920 and 134,515,008 parameters.

## Supplemental exact policy evaluation and numerical check

A separate lookup evaluation integrates over the entire IID command distribution
using dynamic programming. The initial error domain -8 through 8 is verified closed
under every final lookup policy, so 85 `(u,e)` inputs suffice. Lookup actions match
all 294,912 recorded final test actions across the 24 models.

Under this checked numerical lookup convention, expected recovery error is 2.26379
for Qwen expert-only training and 2.69925 for SmolLM2 expert-only training. Every
exploratory policy achieves the exact oracle values: 0.01750 recovery error,
0.26593136 larger-shift error, and probability 1 of stable tracking. Thus the
endpoint improvement persists beyond the particular sampled command sequences.

The first, unnecessarily wide grid used errors from -88 through 88. It added a
padding token to a SmolLM2 batch and flipped one near-tied expert action (`u=0,e=-5`);
the strict agreement check failed. Matching inference initialization alone did not
resolve it. Replaying the original batches matched their recorded actions. The
natural domain, padded to the original 128-row batch size and explicitly checked
for closure, passed all action-agreement checks. Both failures, original scripts,
diagnosis and correction records are retained. Exact integration is claimed for
the verified lookup policy under this numerical convention; the original batched
test scores remain the primary results.

## The explicit-weight mechanism

The [linear Gaussian model](LINEAR_FEEDBACK.md) has true weights `(1,-1)` acting on
`(u,e)`. With expert inputs `(u,0)`, the gradient in the corrective direction is
exactly zero. Its Gaussian posterior leaves that weight at its prior regardless
of the number of demonstrations. Exploration makes the input Gram matrix full rank
and permits informative teacher feedback to identify that direction.

Across 256 paired repetitions and 1,024 labels, squared weight error is 1.000061 for
expert demonstrations, 0.000340 for learner exploration and 0.000068 for random
exploration. The corrective posterior mean is respectively 0, -1.001265 and
-1.000087. Every saved sequential posterior matches an independent batch calculation.

Adding small process noise exposes the corrective direction weakly even under
expert control. Expert weight error then decreases to 0.024403 at 1,024 labels,
versus 0.000336 for learner exploration. The obstruction becomes low information
per sample rather than strict non-identifiability. This supplies a connection
between state covariance, parameter learning and test error with no hidden layer.

It also limits the scaling claim. Removing an error floor is different from showing
two asymptotic power-law exponents. With persistent excitation in every direction,
both Gaussian learners can have inverse-sample variance rates with different
constants. The transformer experiments do not yet quantitatively inherit those
rates or expose a known true transformer parameter vector.

## Scope of the conclusion

The mechanism requires informative feedback and an acquisition distribution that
exposes relevant missing behavior. Guesses alone supply no new independent labels.
Random or frozen-policy exploration can supply the same benefit, and matching the
acquired observations would remove this information difference. The effect is
therefore compatible with generation acting as data augmentation; it does not
establish an intrinsic advantage of producing an answer over studying the same
information, or a corresponding human learning effect.

The exact protocols are in [TRACKING_PROTOCOL.md](TRACKING_PROTOCOL.md) and
[FEEDBACK_PROTOCOL.md](FEEDBACK_PROTOCOL.md). Model runs, all negative pilots,
source snapshots and selection records are retained. No paid API calls or Vast
rentals have been made during this investigation.

## Compute, verification and reproduction

Each main arm receives 8,192 acquired labels and 16,384 training presentations,
with 256 post-warm optimizer updates. Measured training/generation phase seconds
average as follows:

| Acquisition | Qwen3-0.6B | SmolLM2-135M |
| --- | ---: | ---: |
| Expert | 39.27 | 19.18 |
| Current model | 44.47 | 21.78 |
| Random | 41.30 | 20.61 |
| Frozen model | 45.15 | 20.60 |

These are recorded phases, not total experiment wall time. Generation timing excludes
prompt tokenization, CPU oracle/sampling work, evaluation, loading and checkpointing;
training timing includes its minibatch preparation. Shared warm-up is accounted
separately and calibration can require different numbers of passes between models.
The two GPU jobs ran on a shared institutional node. No isolated-hardware throughput
claim follows from these figures. On-policy forward sampling also increases counted
training/acquisition tokens by about 50% relative to expert-only training, while
backpropagation occurs only on the matched training presentations.

The offline suite passed 58 tests before the exact evaluator was added; its three
additional dynamic-programming tests also passed. Main analysis independently
regraded 110,592 trajectories, rechecked all acquired states and oracle labels,
verified replay indices and matched optimizer counts, and established zero overlap
between fresh test command sequences and training sequences. The exact lookup
check additionally matched all 294,912 final recorded actions. The two Gaussian
simulations passed 16,896 posterior mean/covariance checks in total.

Shared archival verified 420 files by file list, byte count and SHA-256: all 24 main
final models, six warm models with their optimizer states, and all 144 weight-probe
files. Total size is 62,959,231,391 bytes. The archive root is
`/ceph/scratch/jbauer/generative-learning/tracking`; per-file manifests are
`results/feedback_experiment/qwen_archive_manifest.json` and
`results/feedback_experiment/smollm_archive_manifest.json`. Final checkpoints store
model weights; optimizer state is preserved for the common warm starts.

Both permitted `gpu_lowp` allocations were released: Qwen/pilots after 1:03:54 and
SmolLM2 after 0:27:41. The allocation times include setup, diagnostics and archival.
Vast spending is **$0 of the authorized $10**. No new paid API calls were made;
the prior conservative OpenRouter accounting remains $0.37499892.

Primary summaries are in `results/tracking_main_analysis/summary.json` and
`results/tracking_smollm_analysis/summary.json`. The combined endpoint, exact-policy
and behavioral summaries are in `results/tracking_comparison/summary.json`.
Regenerate the small-artifact analyses without GPU inference:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_tracking.py \
  results/tracking_main_seed7520 results/tracking_main_seed7521 \
  results/tracking_main_seed7522 --output results/tracking_main_analysis
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_tracking.py \
  results/tracking_smollm_seed7520 results/tracking_smollm_seed7521 \
  results/tracking_smollm_seed7522 --output results/tracking_smollm_analysis
"$VENV_LOCAL/generative-learning/bin/python" experiments/compare_tracking_models.py \
  --output results/tracking_comparison
```

For a fresh training reproduction, use a new output folder and run `prepare`,
`warm`, then each of `expert`, `on_policy`, `random`, `frozen_policy` with
`train_tracking.py` or `train_tracking_smollm.py`. Pass the same `--output` each time;
at preparation set `--seed`, `--labels 8192`, `--eval-seed 94400` and
`--eval-per-stratum 256`. Source hashes and exact settings are saved automatically.
The experiment environment remains outside the NFS project and models use the
configured cache directories.

Selected figures:

- `results/tracking_comparison/cross_model_recovery.png`
- `results/tracking_comparison/corrective_response.png`
- `results/linear_feedback/weight_mse.png`
- `results/linear_feedback/corrective_weight.png`
- `results/linear_feedback_process_noise/weight_mse.png`

The comparison plot uses a log error axis and a symmetric-log label axis with a
512-label linear region to include the shared warm-start point. All six checkpoints
are plotted; no exponent is fitted to these curves.
