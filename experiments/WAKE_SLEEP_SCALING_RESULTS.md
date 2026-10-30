# Current update: finite real data and approximate recognition

Updated 09 September 2026, 20:15 BST.
Current finite-data report: http://127.0.0.1:8095/wake-sleep-finite-data.html

We returned to loss versus real observation count at fixed task and model size. The generator is learned from the same finite observations; all dreams are actual unfiltered draws. The full recognizer shows no benefit. A standard factorized recognizer improves with sleep: at N=8,192 and matched executed dense compute, observable excess loss is 0.010578 for wake versus 0.004370 for sleep, 58.7% lower with improvements in all 12 seeds. Posterior teaching on existing real examples reaches 0.004111. A paired capacity check on the exact same data and generator removes the sleep advantage when recognition is fully expressive.

This supports internal knowledge transfer under approximate recognition. It does not establish that novel generated inputs are essential or that asymptotic data exponents differ; the restricted model has approximation floors. The earlier million-state exponents below concern compute versus latent-state count, a different scaling axis.

Full results and qualifications: experiments/FINITE_DATA_WAKE_SLEEP_RESULTS.md. New derivation: experiments/FINITE_DATA_WAKE_SLEEP_THEORY.md. All work completed on local CPU, paid spend $0; no new LLM experiment.

---

The following is the preserved historical population size-scaling study.

# Wake–sleep scaling: the million-state extension

Updated 09 September 2026, 18:38 BST.
Canonical report: http://127.0.0.1:8095/wake-sleep-scaling.html

## Current conclusion

The toy now spans 16–1,048,576 latent configurations, 256 times the previous
maximum and a factor of 65,536 across the complete range. There is a clear
finite-range size-scaling advantage for ELBO + sleep in this inference model.
The magnitude and fitted exponents depend on optimizer settings, fitting window
and evaluation criterion. This is cost to fixed posterior error versus problem
size, using exact population gradients; it is not a new LLM or sample-scaling
measurement.

All 216 study runs (9 sizes × 6 paired seeds × 2 methods × 2 Adam stabilizers)
and 12 CPU-to-GPU bridge runs attained posterior KL <= 0.1. All Slurm jobs have
ended, and full final checkpoints are retained. Paid APIs and compute rental: $0.

## Main numerical control

Same fixed, correctly specified two-layer sigmoid belief generator; six binary
observations; affine full-state recognition softmax; initial random bias standard
deviation 3. Learning rates stay at the previously selected 0.1 for ELBO and 0.3
for ELBO + sleep. Large sizes were not separately retuned. The six seeds are the
prespecified original seeds 9540–9545, reused across every size and method.

The following uses Adam epsilon = 1e-12 and counts executed recognition matrix
products after caching the fixed generator. Brackets are 95% paired-seed bootstrap
intervals, excluding model and extrapolation uncertainty.

| Fitted state-count range | ELBO cost exponent | ELBO + sleep |
|---|---:|---:|
| 16–4,096 | 1.747 [1.663, 1.833] | 1.081 [1.066, 1.097] |
| 16–1,048,576 | 1.484 [1.467, 1.520] | 1.064 [1.056, 1.073] |
| 4,096–1,048,576 | 1.185 [1.080, 1.388] | 1.041 [1.029, 1.053] |
| 65,536–1,048,576 | 1.359 [1.190, 1.599] | 1.057 [1.041, 1.070] |

At 1,048,576 states:

- ELBO: 32,351.5 mean updates.
- ELBO + sleep: 93.0 mean updates.
- Charging sleep's extra matrix product gives a mean dense-compute ratio of
  231.91, median ratio 239.35, and mean-ratio
  bootstrap interval [197.73, 266.33].
- The earlier uncached reference FLOP convention gives a 224.91 ratio.

The paired exponent gap over the newly added 4,096–1,048,576 range is
0.145, with seed interval
[0.04794613211405507, 0.3376939743171865].
The rolling-window figure also shows substantial variation, including a middle
window with no clear exponent gap. One fixed baseline exponent is not established.

## Why the stabilizer and metric matter

The default epsilon = 1e-8 gives full-range fitted powers
1.657 and
1.064,
and a larger million-state mean compute ratio of 1720.1.
As per-state gradients shrink, the stabilizer changes effective updates; some
of the stronger apparent baseline scaling therefore depends on that numerical
scale. The benefit remains large with epsilon reduced by four orders of magnitude.
The theory document gives the exact loss-rescaling identity for Adam.

For posterior-KL targets 0.4, 0.2 and 0.1, the smaller-stabilizer full-range
baseline powers are 1.357, 1.435 and 1.484, versus approximately 1.06 for sleep.
The corresponding million-state dense-compute gains are about 34, 109 and 232.
The stricter posterior criterion makes the recovery advantage more pronounced.

A direct coefficient test is more limited. To reach centered weight cosine
overlap 0.8, sleep uses 9.32 times less dense compute at a million states.
Over 4,096–1,048,576, however, those cost exponents are 0.956 and 1.019: both
approximately linear. At overlap 0.5 the million-state gain is 2.41 and full-range
powers are almost identical. Thus the main larger scaling gap concerns recovering
severely underestimated posterior explanations, rather than every measure of
weight alignment. These overlap thresholds were added as descriptive mechanism
checks; the original primary criterion remains posterior KL <= 0.1.

## Implementation and validation

`deep_wake_sleep_gpu.py` executes float64 population updates with CUDA graphs.
The fixed generative joint is cached; updates, optimizer equations and random
initialization procedure match the NumPy experiment. Nine gradient/trajectory
audit cases cover three depths and three objectives. Maximum 100-step weight
difference is below 7e-14; restoring optimizer tensors gives identical continuation.
Full on-disk optimizer/RNG checkpoints were also exercised when rescheduling
parts of the long sweep, with posterior KL reproduced on restoration.

The 12 bridge runs showed that first-hit compute can depend on evaluation spacing:
a brief error dip may be missed by a coarse grid. Therefore all five smaller sizes
were rerun on the same GPU implementation and dense checkpoint schedule as the
four new sizes. The earlier CPU curve is preserved separately and was not stitched
to the new larger-size measurements.

The cached dense count is 4 n_x d K per wake update and 6 n_x d K per combined
update. It excludes elementwise kernels, one-time generator setup and evaluations;
the old uncached reference count is retained separately. Update-count plots remove
this accounting convention from the comparison. Recorded active training time is
2.77 GPU-hours, excluding setup, evaluations,
checkpoint transfer and interrupted work that was replayed. No artificial data
ranking, correctness checker, rejection or temperature adjustment was introduced.

This recognizer has one output per full latent configuration. Its posterior
coefficients can also be compiled directly from the generator. The result compares
specified inference-learning rules; it is not an algorithm-independent lower bound.
The sleep gradient here is its exact generative expectation, not a finite sampled
batch. The LLM comparison below uses actual generated examples and a different
scaling axis.

Key artifacts: `results/deep_wake_sleep/million_analysis.json`, `million_size/`,
`million_anchors/`, `gpu_audit.json`, and `million_execution.json`.
The report contains the cost, update-count, rolling-slope and weight-overlap plots,
with links to raw data and reproducible code. No work was pushed or published.

## Historical 4,096-state and LLM-budget extension

The following earlier results remain for comparison.

### Earlier wider-range results

Updated 09 September 2026, 11:44 BST.
Canonical report: http://127.0.0.1:8095/wake-sleep-scaling.html

## Current conclusion

The larger toy cohort preserves a finite-size compute advantage: the original
16 seeds now span 16–4096 latent configurations, and all 160 arm/seed/size cases
reach posterior KL <= 0.1. This is cost-versus-problem-size scaling.

Extending the fixed-task LLM comparison eightfold removes the early external-task
accuracy advantage of sleep. A lasting LLM error-versus-compute exponent advantage
is not established. These are different scaling axes and should not be conflated.

## Full sixteen-seed size extension

Same two-layer fixed, correctly specified sigmoid belief generator; same affine
full-state recognizer, moderate initial neglect 3, population updates and rates
(ELBO 0.1, sleep 0.3). The new largest size was not separately retuned.

- ELBO cost exponent: 1.6421, seed-bootstrap interval [1.4982743697143603, 1.7846185559556826].
- ELBO + sleep: 1.0658, interval [1.0423457326909937, 1.0908216717553096].
- Paired exponent-gap interval: [0.4471519575338099, 0.7036656160047925].
- At 4096 states: 1.516e+11 versus 2.014e+09 estimated FLOPs.
- Cost ratio: 75.27 using means; 41.30 using medians.
- Mean-ratio bootstrap interval: [34.83798060326849, 133.25106034152367].

The baseline's adjacent slopes remain variable. These are descriptive finite-range
fits, not universal asymptotic laws. They closely preserve the earlier 16–1024
sixteen-seed fit of 1.66 versus 1.07. The new 4096 cases use seeds 9540–9555;
the earlier three-seed forecast experiment used 9530–9532 and retains its censored
case. Those are distinct cohorts; the older censored result was not overwritten.

## Actual larger-budget LLM weights

Qwen2.5-1.5B-Instruct, two rank-four q/v LoRA adapters, frozen backbone, both
conditional adapters trained. Same task, 480 real training observations, batch 8,
recognition LR 0.0003, generator LR 0.000001 and temperature 1. All self-generated
cause–observation pairs are retained, without a checker or external rule labels
in updates. Same three seeds 9610–9612 and all three arms.

All nine continuations finish at 4096 updates without triggering the stationarity
rule. Optimizer/RNG state was restored from update 512; old checkpoint q probabilities
are reproduced. Evaluation cadence is denser. Base model revision recorded during
final evaluation: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

The final common allowance is 2.41187e+17 estimated FLOPs,
eight times the earlier baseline allowance. Actual evaluated checkpoints are
ELBO/forward 4096 versus sleep 3648, giving sleep 0.78% less compute. This is the
latest stored sleep checkpoint within the allowance, selected independently of
its evaluation scores. The plot includes these actual measurements in addition
to the regular evaluation cadence.

| Method | Two-example accuracy | True-posterior KL | Model-posterior inference KL | Three-example accuracy |
|---|---:|---:|---:|---:|
| ELBO | 97.78% | 0.1257 | 0.0800 | 99.17% |
| ELBO + sleep | 97.64% | 0.1483 | 0.0520 | 97.66% |
| Posterior teaching | 97.78% | 0.0673 | 0.0465 | 98.65% |

The old matched-budget two-example accuracies were 72.36% versus 82.22%; the
advantage is absent at the larger allowance. ELBO also wins on the three-example
cohort at this allowance. Both arms average 97.78% at exactly 4096 updates, where
sleep has used 11.41% more compute.

Sleep still has lower two-example inference KL from its own model posterior.
That is a different metric: the generators differ and continue learning, so their
posteriors are moving targets. It is not a bound on external task error. At equal
4096 updates, generator correct-digit probabilities are 94.64% for ELBO and
95.19% for sleep; at the matched compute allowance they are 94.64% and 92.06%.
The later phase is consistent with a growing role for generator learning; this
was not an isolated causal test of that explanation.

Effective probability-error exponents (descriptive, fully observed windows):

| Baseline-update-equivalent compute | ELBO | Sleep |
|---|---:|---:|
| 64-256 | 0.609 | 0.885 |
| 128-512 | 0.410 | 0.460 |
| 256-1024 | 0.510 | 0.248 |
| 512-2048 | 0.832 | 0.474 |
| 1024-4096 | 1.313 | 1.219 |

The favorable early exponent ordering reverses in later windows. Rule-specific
curves and the complete per-seed trajectories are retained. No extrapolation
beyond the observed LLM budgets was fitted.

## Wider toy error range and mechanism controls

54 runs increase the fixed-task budget to 262144 wake-equivalent updates and
lower the stopping target from KL .03 to 1e-9: three latent sizes (64,256,1024),
six fixed seeds and three methods. Eighteen further runs at 256 states reduce
all rates tenfold. The curves now cover several orders of magnitude in error,
but show delayed recovery, steep transitions and optimizer floors rather than
one sustained error power law. Most cases do not attain the requested 1e-9;
all failures/stops are retained. Tighter first-hit targets are only aggregated
as an unconditional mean when every seed attains them.

A 72-run numerical audit changes Adam epsilon from 1e-8 to 1e-12, extends through
4096 states and keeps rates/model/initializations fixed. The large size advantage
survives; it is not solely the numerical stabilizer. Another 20 runs complete
the full sixteen-seed cohort at 4096. Total new toy runs: 164.

There is an exact weight-space identity in the affine toy. If e is the deviation
from the compiled posterior coefficients, the wake gradient is M(B)e, using the
current recognizer Fisher matrix. The forward/sleep term is the average of M
along the coefficient segment between the useful posterior and the recognizer,
applied to e. The combined treatment adds these two gradients. This links the
nonlocal recovery mechanism directly to weights; near the posterior the two
operators agree. A 36-case audit verifies posterior compilation and both
identities with maximum error below 1.5e-15. This does not prove a global LLM
weight identity, an Adam convergence rate, or a universal exponent.

All CPU processes and Slurm jobs completed. All outputs/checkpoints/logs are
retained. No paid API calls or rented compute in this follow-up; nothing pushed
or externally published. The canonical local report is updated.

Key new artifacts: `expanded_size_analysis.json`, `wide_analysis.json` in each
model's result directory, `wide_matched_evaluation.json`,
`wide_epsilon_analysis.json`, and `weight_mechanism_audit.json`.

The next comparable LLM size test should grow the hypothesis family while
holding a useful generator fixed, matching the toy's scaling axis. Simply
extending this already nearly solved eight-rule task cannot establish that law.

## Earlier experiment details

The following sections document the shorter-budget results preceding this
follow-up; their numbers are retained as historical comparisons.

## Toy: cost to a fixed error as the latent space grows

A two-layer sigmoid belief network generates six binary observations. The
generator is fixed and exactly matches the data distribution in the size sweep;
an affine full-configuration recognizer learns. All states and observations can
be enumerated. Moderate recognition neglect is induced by independent random
initial logit biases with standard deviation 3. There is no temperature change,
selection, or task-specific checker. Adam rates come from an expanded grid on
separate tuning seeds (ELBO 0.1, sleep 0.3).

Across 16 new seeds and K = 16, 64, 256, 1024 configurations, mean estimated
compute to posterior KL <= 0.1 has descriptive fitted exponents:

- ELBO: 1.6625; paired-seed bootstrap interval [1.5336, 1.8173].
- ELBO plus sleep: 1.0657; interval [1.0341, 1.0999].
- Difference interval: [0.4861, 0.7295].

All 128 independent-replication cases reached this primary target. The mean
cost ratio grows from 2.05x at 16 configurations to 29.79x at 1024. The ELBO
adjacent-size slopes vary (2.49, 1.21, 1.45), so the fitted exponent is not a
universal asymptotic law. Diffuse initial recognition removes the large benefit.

A 64-run fairness audit retuned rates separately at each size on independent
tuning seeds, with equal grids {0.03, 0.1, 0.3, 1}. Restricted mean time to the
same target selected ELBO 0.1 and sleep 0.3 at every size. Thus this audit did
not require rerunning any confirmation cell; the original rate choices survive.

A forecast from the earlier three-seed sweep was saved before testing 4096
configurations. Sleep reached the target at mean cost 2.367 billion FLOPs,
versus forecast 2.360 billion, within checkpoint resolution. ELBO was partially
censored: costs 922.6 billion, unattained, and 162.0 billion. Its mean lower
bound is 743.8 billion versus point forecast 472.5 billion. Do not interpret
the censored budget as a successful target time or claim both forecasts passed.

The broader depth sweep has 1296 main cases, 324 rate follow-ups, 48 initial size
cases, 96 size controls, 128 independent size replications, six extrapolations,
and 376 pilot/tuning cases. It includes population/sampled updates, fixed/coupled
generators, affine/deep recognition and SGD/Adam. Strong neglect creates large
recovery improvements, but sampled sleep can hit a noise floor; later slopes
flatten or reverse. Deeper models alone do not establish different asymptotic
error-versus-compute exponents.

The full-state recognizer has exponentially many outputs in the number of
latent bits. An unrestricted algorithm could compile its exact affine posterior
from the generator. The result concerns the specified learning rules, not a
lower bound on all nongenerative inference algorithms.

## LLM: actual weight learning without a correctness checker

Qwen2.5-1.5B-Instruct has two separate rank-four q/v LoRA adapters on a frozen
backbone. The generator predicts one of ten output digits given a named rule
and an input. The recognizer chooses among eight rules from two examples.
Uniform rule prior, temperature 1, constrained digit/letter supports in every
arm. Real examples expose only inputs and outputs; latent rules are removed
before updates. The model itself supplies every dreamed output. All dreams,
including semantically incorrect ones, are used.

The dataset has 480 training observations and 240 held-out observations with
disjoint ordered input pairs. ELBO enumerates the eight latent rules exactly.
Sleep adds inverse cross-entropy on self-generated pairs. Exact-posterior
teaching is a diagnostic control using the current model, not true latent
labels. Both generator and recognizer train in the final comparison.

Six short fixed/coupled pilots and six rate-tuning runs preceded three new
optimization seeds (9610–9612), three arms, 512 updates. Common recognition
AdamW rate 0.0003, generator rate 0.000001, batch 8, zero decay, gradient clip 1.
The pretrained generator's exact posterior initially gives 81.7% compatible-rule
accuracy; direct recognition starts at 12.5%.

For actual matched-budget evaluation, sleep resumes from saved update 448 and
continues to update 459. ELBO and posterior teaching use update 512. Estimated
compute is 3.0111e16 versus 3.0148e16 FLOPs, a 0.13% difference with sleep lower.
These are evaluated weights, not interpolation between checkpoints.

| Method | Two-example accuracy | Two-example posterior KL | Fresh three-example accuracy |
|---|---:|---:|---:|
| ELBO | 72.36% | 2.1974 | 75.26% |
| ELBO + sleep | 82.22% | 0.9558 | 84.11% |
| Exact posterior teaching | 80.28% | 0.7217 | 83.85% |

Sleep improves greedy accuracy in all three seeds on both cohorts. The fresh
cohort has 640 observations with three examples and was generated after
selecting the setup. No fresh-cohort information enters training or tuning.
The two-example cohort had been used in exploratory calibration/tuning, so the
new optimization seeds are not a wholly untouched benchmark confirmation.
Initializations are paired, but real-batch orders differ across arms: sleep
consumes additional RNG draws. The IID pool/distribution is identical.

At the halfway matched-compute budget, the interpolated probability of an
incompatible rule is 0.3716 for ELBO and 0.2703 for sleep, with all three seeds
favoring sleep. At the largest budget, probability-error differences are more
variable. Posterior teaching is competitive, often better in posterior KL.
Generator marginal likelihood does not show a comparable sleep advantage.

Finite-window probability-error exponents are ELBO/sleep 0.61/0.88 over
64–256 baseline-update-equivalent compute and 0.41/0.46 over 128–512. These
are transient effective slopes, not stable asymptotic exponents. The LLM still
has eight hypotheses; its hypothesis-count scaling was not measured.

## Interpretation and validation

Generation gives an inverse mapping labels for causes it underweights. This
differs from imitating samples of the same conditional distribution. Both KL
directions have identical local Fisher curvature at a matched posterior, even
through deep networks. Persistent recovery of increasingly difficult directions
is a more plausible route to exponent changes than depth alone.

Gradient audits cover 144 directional derivatives, maximum error 9.52e-11.
Source/runtime hashes, per-stage protocols, trajectories, and weights are
retained. Long-run resume infrastructure saves adapters, optimizer state, RNGs,
step and data-stream position every 32 updates; the exact-budget evaluation
exercised resume. Timing diagnostics from a mid-evaluation-interval resume are
not used for compute claims; token counters are saved directly. Checkpoints
and model downloads use cache storage. No paid model API calls, rentals, pushes, or
external publication. All compute jobs finished and allocations were released.

Main code: deep_wake_sleep.py, llm_wake_sleep.py,
analyze_deep_wake_sleep.py, analyze_wake_sleep_sizes.py,
analyze_llm_wake_sleep.py, evaluate_llm_wake_sleep_matched.py.
Raw artifacts: results/deep_wake_sleep and results/llm_wake_sleep.
Each archived runtime uses its adjacent archived protocol; current experiment
protocols include the current required accuracy_stop field.

The next discriminating experiment is to replace flat state enumeration by
autoregressive recognition over compositional causes, grow the same hypothesis
family in the LLM, and retain compute-matched posterior-inference controls.
