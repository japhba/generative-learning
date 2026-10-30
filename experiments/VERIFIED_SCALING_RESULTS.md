# r5s: observed sample-efficiency improvement

Completed 7 September 2026, UK time. All six pilot runs and all six fresh confirmation runs are retained. The model is Qwen/Qwen2.5-1.5B-Instruct with full-weight FP32 plain-SGD updates. The task counts a target digit in a four-digit list. Both methods generate two temperature-1 one-token candidates per input and train only when a correct candidate exists, averaging loss over accepted examples. Only the generator source differs: current student versus initial pretrained weights. Both students keep learning.

The symmetric pilot tried learning rates 1.4e-6, 2.8e-6 and 5.6e-6. Each method independently selected 5.6e-6. Three fresh paired seeds then ran 4096 updates, with 32 IID question draws and 64 generated candidate tokens per update. All six completed their planned horizons. No paid API calls or rented instances were used.

## Primary evidence

The strict final cohort contains 2444 literal inputs absent from all r5s pilot training and monitoring and all confirmation training and monitoring. Its unequal class counts are handled with an equal-weight average over answer classes. The original 4095-question pool and the monitoring cohort remain reported as sensitivity checks. The strict subset was defined by IDs after early confirmation monitoring but before any final-test outputs. All final-test evaluations began after every confirmation trajectory finished.

Expected sampled-answer error, computed exactly as 1 minus correct-token probability:

| Generated candidates | Current source | Frozen source |
| --- | --- | --- |
| 32,768 | 0.5153% | 3.8166% |
| 65,536 | 0.2099% | 1.9300% |
| 131,072 | 0.0910% | 1.0723% |
| 262,144 | 0.0436% | 0.5733% |

The current-source model at 65,536 candidates beats the frozen-source model at 262,144: lower error with four times fewer candidates. The paired hierarchical bootstrap interval for the difference is 0.2041–0.5504 percentage points. This direct comparison needs no interpolation. All five specified secondary cross-budget comparisons are retained, including an eightfold comparison whose interval crosses zero. Comparisons were selected using monitoring curves before reading final-test predictions; intervals are pointwise, with only three training seeds.

Interpolating between saved checkpoints estimates a 6.54-fold candidate saving to reach 1% error, with bootstrap interval 5.27–8.15. The interval covers seed/question uncertainty, not interpolation-model uncertainty. The corresponding descriptive savings are 4.94-fold in accepted examples and 5.34-fold with a token-work proxy that skips rejected training rows. Candidate counts do not measure wall-clock speedup.

## What scales, and what remains uncertain

Both methods have approximately power-like finite learning curves. Over updates 512–4096, fitted exponents are 1.19 current / 0.91 frozen on the strict test but 1.06 / 1.14 on the monitor. The robust finding is a sample-efficiency gain, not a universal exponent ordering. The primary probability metric includes confidence sharpening after greedy decisions become correct; greedy error is reported separately.

The strict-test exponent difference has a paired bootstrap interval of 0.074–0.448. This is positive evidence for a steeper finite-range learning curve, not merely an inconclusive comparison of coefficients. A descriptive diagnostic added on 8 September uses the same five existing checkpoints for all cohorts. The strict-test error ratio rises from 2.31 to 13.16; an exact common exponent with constant coefficients would instead give a constant ratio. The last-doubling slope gap is only 0.16, and the monitor's late ratio declines, so persistence beyond the measured range remains uncertain. The diagnostic is experiments/analyze_verified_slope_diagnostic.py; no further training or evaluations were run for it.

The independent-mode Bernoulli toy, embedded exactly as a conditional RBM, predicts the same inverse-budget exponent for both methods and a candidate-efficiency coefficient gain of mean(q0) times mean(1/q0). Fixed toy simulations confirm the predicted amplitudes; population weight overlaps also have explicit unfitted asymptotic laws. The derivation is in VERIFIED_SCALING_THEORY.md.

A further construction on 8 September replaces independent weights with one shared logistic skill theta and an exponential difficulty tail of rate a. For 0<a<1, the same verified-only update has population error exponent 1 for current generation and a for frozen generation. Both greedy and sampled error show this separation. The explicit Boltzmann model, proof and numerical checks are in SHARED_SKILL_SCALING_THEORY.md and shared_skill_scaling.py. If difficulty is bounded, the asymptotic sampled-error exponents agree again at 1. This establishes a possible mechanism, not a quantitative explanation or new experimental result for the LLM. Its assumptions have not been validated in the LLM.

In the LLM, the frozen generator supplies only about 15% of accepted examples from the initially difficult count-0 class; current-source training brings that share toward one third. All raw accepted tokens were independently verified. A global answer-bias correction cannot explain the trained accuracy. Fixed-state gradient cosines support an overlap diagnostic, but do not alone explain immediate progress: at one probed state the frozen gradient gives a larger immediate population-step gain.

Plugging initial LLM probabilities into the independent-mode coefficient formula gives 5.17, but this quantitative extrapolation is not validated: the final same-budget error ratio is 13.16, and finite-window exponents differ. No claim of a common quantitative LLM/toy law is warranted yet.

This experiment uses an external correctness checker. It does not demonstrate learning from unvalidated generations, isolate generation versus recognition, compare against uniform gold-label supervision, or test multistep reasoning. The evidence supports a narrower claim: refreshing the generator improves the sample efficiency of verified self-training on this task.

## Reproduction and retained evidence

Training protocol: experiments/VERIFIED_SCALING_PROTOCOL.md. Training implementation: experiments/train_scaling_verified.py. Complete results, raw candidates, all rate choices, source snapshots, data audits, diagnostics and resume metadata: results/scaling_r5s/. Full checkpoints are separately retained at the durable locations recorded in their archive metadata.

Report builder: experiments/build_verified_scaling_report.py. Canonical report: reports/self-generation-scaling.html, served at http://127.0.0.1:8095/self-generation-scaling.html. Archive builder: experiments/archive_verified_scaling.py. Six meaningful sampling/analysis tests passed; task identities, accepted labels, split disjointness, theoretical amplitudes, RBM normalization and weight-overlap identities were also checked.
