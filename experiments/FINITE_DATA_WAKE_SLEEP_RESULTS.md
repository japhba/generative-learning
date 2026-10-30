# Finite-data wake–sleep: an approximation-dependent learning-curve improvement

Updated 09 September 2026, 20:15 BST.

Canonical report: http://127.0.0.1:8095/wake-sleep-finite-data.html

## Result

This study actually varies the number of independent real observations N at fixed task distribution and architecture. A useful imperfect generator is fitted from the same finite data, then frozen. Real examples reveal no latent labels. Recognition either replays real contexts or adds actual unfiltered ancestral dreams.

The fully expressive recognizer shows no sleep advantage. A standard factorized recognizer shows a lower observable prediction-loss curve with sleep. A paired check on the same task, datasets and fitted generators confirms that making recognition expressive removes the benefit. Exact posterior teaching on real contexts performs similarly or better than sleep, so novel imagined contexts are not established as necessary.

## Strict compute check

All values below are mean held-out excess conditional log loss, in nats, over 12 paired seeds. Evaluation enumerates the true observation distribution but never supplies it to training or stopping. N includes seven eighths training and one eighth validation.

| Real observations N | Wake | Wake + sleep | Posterior teaching | Sleep / wake, 95% paired-seed interval |
|---:|---:|---:|---:|---:|
| 512 | 0.026729 | 0.022597 | 0.022259 | 0.845 [0.770, 0.916] |
| 2,048 | 0.012291 | 0.007901 | 0.008122 | 0.643 [0.551, 0.734] |
| 8,192 | 0.010578 | 0.004370 | 0.004111 | 0.413 [0.384, 0.445] |

At N=8,192, sleep lowers excess loss by 58.7% (95% paired-seed interval 55.5%–61.6%) and wins in all 12 seeds. Wake and forward teaching use 512 updates. Sleep uses 239 combined updates and 15,296 unfiltered generated contexts. Counting its ancestral generation and extra matrix products, sleep uses 510.59 ordinary-wake equivalents: 0.28% less dense recognition compute. Generator fitting is shared identically. This is a proxy based on executed leading dense operations, not full hardware FLOPs; elementwise work, setup and evaluation are excluded.

This is a post-confirmation prefix replay on the same datasets and frozen hyperparameters. It is not an independent replication sample. Its purpose is to remove early-stopping differences from the compute comparison.

## Larger allocation and the capacity control

At a maximum recognition allocation of 4,096 wake equivalents, factorized recognition has losses 0.015397 (wake), 0.006291 (sleep), 0.004143 (forward teaching) and 0.008747 (direct prediction) at N=8,192. Wake frequently stops earlier on the prespecified training/validation stationarity criterion. Some subsequent optimization improves its own reverse-KL objective while worsening observable prediction, which is why the 512-unit values and the larger-allocation values differ. Both budgets retain the sleep advantage.

The separate capacity check uses exactly the same generator, true task and real datasets as the factorized cohort. All executed dense budgets across the four family/method combinations are within 0.4%.

| N | Full-family wake | Full-family sleep | Sleep / wake, 95% interval |
|---:|---:|---:|---:|
| 512 | 0.021785 | 0.021766 | 0.999 [0.965, 1.026] |
| 2,048 | 0.006749 | 0.006940 | 1.028 [0.974, 1.086] |
| 8,192 | 0.001438 | 0.001892 | 1.316 [1.201, 1.462] |

Increasing recognition capacity removes the sleep advantage on these same data. Both families use previously selected, separately tuned rates. This is a standard approximation restriction, not an artificial initial neglect intervention.

## What scales, and what does not follow

The original fully expressive cohort uses a separate fixed world. Over N=512–8,192, effective powers for L(N)=A N^(-beta) are 0.922 for wake [0.837,0.993] and 0.889 for sleep [0.800,0.966]. Both track the learned generator; smaller-N bias changes the full-window fit to about 0.81 and 0.80. These are finite-window descriptions, not universal exponents.

For the factorized family, a nonzero recognition approximation error remains. Different apparent floors can explain the widening gap. The result supports lower held-out loss at the same real data and compute; it does not establish a different asymptotic data exponent. Plotting a zero-floor power fit through the plateau would conflate estimation and approximation.

The common generator fitting stage can stop at different times for different N. Main data curves hold subsequent recognition allocation fixed; total fitting-plus-recognition compute is matched across arms within N, not asserted constant across N. The strict prefix checks additionally match executed matrix-product work.

## Mechanism

The one-layer sigmoid generator has four hidden binary causes and eight observed bits. The task predicts the final two bits from the first six. The full recognizer has 112 affine-softmax parameters. Its exact posterior coefficients can be compiled from the learned generator. The factorized recognizer has 28 parameters and cannot express posterior interactions between causes.

At N=8,192, factorized wake achieves a smaller training reverse KL (0.140) than sleep (0.148), while its observable predictions are worse. Sleep improves forward posterior KL from 0.172 to 0.142 and coefficient overlap from 0.830 to 0.849. These support a different compromise between inference objectives under restricted recognition; they do not certify a global optimum.

The model does not acquire external facts from its own samples. The forward joint supplies cause labels and conditional moments to a distinct approximate inverse. Posterior teaching extracts related supervision on existing contexts. The experiment therefore supports internal knowledge transfer, with an informative control against attributing the effect solely to exploring new inputs.

## Scope, selection and validation

There are 480 original and 480 factorized confirmation runs, plus 108 compute and 72 paired capacity checks. The two tuning cohorts contain 288 recognition trajectories total. Main grids span N=16 to 8,192, ten dataset sizes and 12 paired seeds per world. All checkpoints and logs are retained.

Truth world 9820 and dataset seeds 9821–9822 select rates using an equal six-rate grid. Full-family confirmation uses world 9830 and seeds 9831–9842; factorized confirmation uses world 9850 and seeds 9851–9862. Rates are frozen before each confirmation. The later paired capacity check uses this same second world and the previously selected full-family rates.

Finite-difference gradient checks cover both recognition families and all three analytic objectives; maximum absolute discrepancy is below 4.4e-11. The compiled categorical posterior matches enumeration to 1.2e-16. An ancestral-sampling cross-moment audit passes. Full on-disk resume for both families reproduces weights and Adam moments bit exactly. Replay controllers also reuse completed runs without retraining.

All work used local CPU. Paid API/rental spend is $0. There is no new LLM run in this update. The next LLM match should test an imperfect learned forward model and an approximate inverse, with observable held-out prediction and the same posterior-teaching control.

Files: finite_data_wake_sleep.py, analyze_finite_data_wake_sleep.py, confirm_finite_data_compute.py, FINITE_DATA_WAKE_SLEEP_PROTOCOL.md, FINITE_DATA_WAKE_SLEEP_THEORY.md. Source snapshots, protocols, hashes, learned weights, raw examples and all trajectories are under results/finite_data_wake_sleep. For a fresh rerun, use a new protocol output directory rather than overwriting archived cohorts.
