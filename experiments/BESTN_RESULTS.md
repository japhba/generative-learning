# b2k results

The tested LLM setting supports a benefit from refreshing the generator under a fixed correctness-selection rule. A conditional Boltzmann toy supplies exact weight-overlap dynamics, and prospective gradient overlaps predict local LLM learning. The stronger one-scalar whole-curve approximation remains quantitatively incomplete.

## Untouched-task confirmation

All approximately 1.54B weights of Qwen2.5-1.5B-Instruct were trained with FP32 plain SGD, fixed completion-loss normalization, batch 16 and two temperature-1 candidate traces per training question. Uniform maximum-correctness selection retains all-failure groups. The current generator updates with the learner; the frozen generator always uses its original weights. There is explicit correctness feedback.

The final test uses exactly update 16, 512 candidate rollouts and 256 training presentations. Its 128 questions were unseen in training, calibration and recurring evaluation. Each condition generates eight responses per question. This test was frozen after discovering the earlier effect and before generating its responses; it is independent confirmation, not an original preregistered endpoint.

Untrained accuracy is 19.92%. Mean frozen-source accuracy is 23.44% and current-source accuracy is 34.64%. The advantage is 11.20 percentage points, hierarchical 95% interval [7.42,14.94]. Only three training seeds are available; bootstrap resampling preserves the shared questions across conditions.

| Seed | Frozen | Current | Current−frozen |
|---|---:|---:|---:|
| 86001 | 22.46% | 34.77% | +12.30 pp |
| 86002 | 23.83% | 32.23% | +8.40 pp |
| 86003 | 24.02% | 36.91% | +12.89 pp |

## Mechanism and predictive scope

For arbitrary full-trajectory policies with binary success and uniform best-N ties, the expected selected score is c_N(p) times the success gradient. For N=2, c_N=1. In the Gaussian-visible/binary-hidden RBM, the population update is A*mu+B*w, giving an explicit positive normalized-overlap velocity under feedback. The frozen proposal instead approaches a fixed selected-target sufficient statistic. Population toy curves were saved before simulation, with no fitted time constant.

For the LLM, diagnostic rollout banks estimate training/evaluation gradient overlaps before each four-update block. Confirmation-seed errors for the current-source learner are:

- importance_gain: forecast RMSE 1.59 pp, versus 3.30 pp for a zero-gain forecast, across 13 blocks.
- fresh_gain: forecast RMSE 2.33 pp, versus 3.51 pp for a zero-gain forecast, across 13 blocks.

These are descriptive errors on correlated blocks, not uncertainty intervals for a universal scaling law. Frozen-source forecasts are noisier and include ordinary off-policy score fluctuations. Actual token costs are reported because current-source responses get longer; equal candidates do not exactly match compute.

Equation-mode frequency explains a substantial portion of the LLM gain. A one-gate Boltzmann reduction was discovered from seed 86001 and numerically frozen at 2026-09-07T12:48:21.149133+01:00. Seed 86003 current checkpoints through 8 updates already existed, so only later checkpoints are prospective. Its point forecasts underpredict current-source learning. No gate coefficient was retuned after confirmation; this stronger global model is not an established prediction of the full LLM learning curve.

## Separate no-feedback fixed-norm result

Projecting the eight-weight toy to its initial unit norm after each self-sampled conditional-score update improves its unlabeled marginal density from a poor initialization. On 16,384 fresh replicas at initial overlap 0.3, the cross-entropy change is −.114070 nats [−.116391,−.111749], compared with −.031034 for trace-matched isotropic noise. Covariance-matched Gaussian noise reproduces the effect. The second-order analytic drift agrees with an independent finite-difference audit.

Classification still worsens from 32.64% to 46.45% error, with 44.27% sign reversals. The model's unlabeled density is sign-invariant; label predictions are not. Starting at overlap 0.8 instead worsens density by+0.216463 nats. Thus this is a bounded representation benefit with an explicit norm constraint, not a universally improving learner. Real input samples supply the information. No no-feedback LLM benefit was established.

## Retained limitations and verification

- The original unconstrained no-feedback experiment had worse density and classification. Its outputs remain intact.
- The original BF16/canonical-likelihood differential predictions failed. FP32 restored local numerical agreement; the generation/scoring audit agreed within 0.001 nats per tested trajectory.
- A smaller original fresh-sampling probe was less favorable; all outcomes were retained, and the later 1,024-response checks confirmed actual gains without rate/target tuning.
- A stopping correction replaced a mistaken AND condition with the required OR condition. Initial source/protocol snapshots are retained. The final test's fixed update 16 avoids choosing a favorable endpoint.
- First candidate/target banks and first losses agree exactly within each current/frozen seed pair. Full resume state reproduced the next update exactly. Source hashes, raw responses, models and checkpoint provenance are retained.
- No universal exponent, architecture generality, ICL/human equivalence or advantage over every recognition learner is claimed.

The canonical report is http://127.0.0.1:8095/bestn-learning.html. Paid API/rental spend for b2k is $0; no rented instance was created. Code and reports were not pushed to a repository or public website.
