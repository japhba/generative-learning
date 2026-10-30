# Coupled wake–sleep results

Canonical report: http://127.0.0.1:8095/coupled-wake-sleep.html

## Outcome

Self-generated joint cause/observation pairs improve finite-budget generator learning in the high-overlap setting. The effect requires a useful generator and an adequately expressive recognizer, and depends on the learning rule and rate. Different asymptotic scaling exponents are not established.

The model has six fair binary causes, eight Gaussian observations, and a linear decoder. The primary recognizer is one affine layer with a 64-state softmax, which can represent the exact posterior. Both sets of weights learn. Current sleep adds recognition cross-entropy on self-generated (h,x) pairs, with no external checker; wake updates remain anchored to real x. The initial generator already contains useful teacher-aligned structure.

## Evidence

- Twelve paired confirmation seeds, disjoint from tuning, four overlaps, five methods. Leading dense-arithmetic costs include dream generation and recognition learning; the primary allowance is equivalent to 8192 ordinary wake updates. Baseline uses 1,048,576 real examples; current sleep uses 696,320 real examples plus 696,320 dreams at essentially the same arithmetic budget.
- At overlap 0.8, fresh evaluation on 16,384 observations per seed gives stable-ELBO marginal KL 0.00707488, current-sleep KL 0.00038688: ratio 18.29, paired bootstrap 95% CI 16.13–20.96; 12/12 seeds favor sleep. This is an error ratio, not an equal-sized compute speedup.
- Shared recognition learning rate 3: error ratio 8.17, CI 5.57–10.81. With shared rate 1 the improvement is only about 8%; at shared rate 10 it is much larger. Orthogonal causes show no reliable shared-rate gain. Sleep permits a wider useful range of recognition rates under overlap.
- Initially selected ELBO rates give a larger 24.13 ratio but that baseline's marginal fit deteriorates, despite improved variational inference loss. It is not the most conservative comparison. The slower baseline and shared-rate checks are explicitly post-confirmation follow-ups, not newly preregistered discoveries.
- Exact posterior teaching on real x matches or improves on sleep. This supports an inference-objective explanation, and does not establish that generating data is uniquely necessary.
- Frozen dreams underperform current dreams (main-grid error ratio 74.52), but this partly reflects a persistent mismatch to an outdated generator.
- The factorized-recognizer control has a large residual error in overlapping settings and does not reproduce the large gain.
- Measured training CPU-time allowances also favor sleep: stable-baseline/sleep error ratio 14.93. Timing is implementation-specific.

## Mechanistic interpretation and its limit

Posterior pair interactions are exactly determined by off-diagonal decoder overlaps WᵀW. A uniform recognizer removes the data–latent cross moment in the generator gradient, initially shrinking W. Sleep teaches an inverse mapping using latent labels supplied by the generator itself. The high-overlap runs show better forward posterior KL, better generator weight recovery, and more accurate recognition pair coefficients.

However, instantaneous squared gradient bias is not uniformly reduced: at the primary budget it is 0.006645 for sleep versus 0.006223 for the initially selected ELBO baseline. A blanket explanation through a smaller gradient-bias norm is unsupported. Inclusive versus exclusive posterior fitting, mode coverage, and tolerance to faster recognition updates are better-supported interpretations.

## Scaling

Against the stable baseline, finite-window exponents are:

| Compute window | Stable ELBO | Current sleep |
|---|---:|---:|
| 512–4096 | 1.086 | 1.416 |
| 1024–8192 | 0.944 | 1.221 |
| 4096–16384 | 1.030 | 0.811 |

The slope advantage reverses across windows. These are transient learning curves, not established distinct asymptotic laws. The prescribed decaying SGD rates can also influence late noise-limited slopes.

## Validation and reproduction

Finite-difference checks cover wake and sleep gradients (maximum error 7.48e-11), plus exact-likelihood and forward-posterior gradients (5.70e-11). The categorical recognizer reproduces the exact posterior to machine precision; orthogonal decoder columns produce factorized posteriors (maximum discrepancy 5.55e-16). Every run retains its protocol, metrics and final weights; stage runtime copies and hashes preserve the code actually used. Fresh evaluation uses separate randomness and deterministic replay to the primary budget, rather than relying on interpolation of the original evaluation grid.

Main command from the project root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /var/tmp/jbauer/venvs/generative-learning/bin/python experiments/coupled_wake_sleep.py --protocol experiments/coupled_wake_sleep_confirmation.json --workers 8

Analogous protocol files reproduce factorized_confirmation, rate_robustness, and fresh_evaluation. The analyzer requires all four completed stages. All exploratory tuning results remain under results/coupled_wake_sleep. No paid APIs or rented compute were used; no work was pushed or published externally.
