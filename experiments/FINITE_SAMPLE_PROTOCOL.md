# Finite-sample self-training at equal temperatures

Date: 2026-09-07. Prospective specification, frozen before the sweep. This tests stochastic self-training in a toy; it is not new LLM training.

## Primary question

Can finite, unfiltered self-generated targets improve normalized weight overlap or population classification error, despite zero expected score? Does the covariance of the model's gradient noise explain the effect?

## Model and update

Use the existing Gaussian-visible, one-binary-unit RBM conditional:

```
y_true = +/-1 equiprobably
x = y_true * mu + Normal(0,I)
mu = (1.5, 0, ..., 0)
p_w(y|x) = exp(y*w.x) / (2*cosh(w.x))
```

The true label is discarded after input generation. Training samples an independent pseudo-label from the model's conditional at temperature 1, then takes one ordinary likelihood SGD step at temperature 1. Every sampled target is retained. No clipping, decay, gradient normalization, correctness filtering, or extra optimization steps are used. Initial weights have norm 1 and specified positive overlap with mu.

## Arms

1. **Self-sampled score:** average x*(y_sample-tanh(w.x)) over the batch.
2. **Matched Gaussian:** replace each scalar residual by a centered Gaussian with variance 1-tanh(w.x)^2, retaining the same input vector. At a given current w and input batch this matches the entire conditional covariance of the batch-average score exactly.
3. **Isotropic Gaussian:** centered isotropic noise with the same conditional total variance as that batch-average score.

All arms receive the same input batch in each paired replica and use their own current weights when computing the noise covariance. Gaussian draws and pseudo-label draws are independent across arms. A zero-population-gradient learner is represented by the unchanged baseline. Gaussian arms do not literally generate response tokens; their horizontal sample axis denotes the same number of presented inputs used to construct the matched covariance.

## Frozen grid and budget

- Dimensions: 2 and 8.
- Initial normalized overlap: 0.3 and 0.8; norm 1 throughout the grid.
- Batch sizes: 1, 8, 64.
- Learning rates: 0.025, 0.1, 0.4.
- 512 updates per run, 512 independent stochastic replicas per setting, all three arms.
- Checkpoints: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512.
- Primary setting: dimension 8, overlap 0.3, batch 8, learning rate 0.1, endpoint 512.
- Case RNG: SeedSequence([20260907, case_index]), with index from the fixed Cartesian-product order in the source. Replicas use independent array entries in this stream; the stream and replica index jointly identify a run.

There are 36 settings and 55,296 arm-replica trajectories. This is a local CPU experiment, with no paid API or rented compute. Raw trajectories go to FAST_CACHE_DIR; a verified persistent copy will be saved under CACHE_DIR.

## Measurements and interpretation

Primary outcomes are normalized overlap rho and exact population classification error Phi(-1.5*rho). Also record raw overlap w.mu, squared norm, and squared distance to mu. Evaluate all checkpoints analytically, avoiding a finite test-set sampling error.

Report mean, standard error/95% pointwise interval, median, 5th/95th percentiles, and fraction of replicas improving over the fixed initial model. All failures and negative results remain in the analysis. Error improvement on a minority of trajectories is not a reproducible average benefit. Pointwise intervals across a sweep are exploratory; report multiplicity-adjusted endpoint evidence before selecting favorable settings.

Compare arms using paired replica differences. Track the accumulated conditional variance and verify the martingale prediction that the expected norm increase equals this accumulated variance. Compute the local Fisher matrix by Gaussian quadrature and contract it with independently checked Hessians of error and normalized overlap.

For small updates the predicted per-update metric drift is eta^2/(2B) * trace(F*H_metric). Plot the same curves against update count, presented inputs, and the diffusion clock updates*eta^2/B. A clock collapse at small steps would support a diffusion mechanism, not a power-law learning exponent. Larger steps and higher moments can break this approximation.

## Separate nonlinear sanity control

To make sure a negative main result is not mistaken for a universal impossibility, also evaluate the scalar model p_theta(y)=Normal(theta^2,1). Its fresh self-sampled score is 2*theta*epsilon. Use initial predicted means 0.1 and 1, evaluate against true mean 1, with eta=0.1, batch=1, 24 updates, and 200,000 replicas (seed 2026090701).

Let u=4*eta^2/B and initial predicted mean m0. The exact expected squared mean error after t updates is

```
m0^2 * (1 + 6*u + 3*u^2)^t - 2*m0 * (1 + u)^t + 1.
```

Report this exact expectation alongside sampled mean, median, and fraction improving. This is a deliberately simple existence/sanity control for nonlinear parameterization; it is not a latent-structure model, evidence for an RBM benefit, or a selected substitute for the primary experiment.

## Scope

No temperature mismatch, target selection, new task tuning, exponent fitting, or early stopping based on favorable test outcomes. Conclusions apply to these models and this specified grid. If they do not improve, the next unresolved candidates include nonlinear representations, optimizer transformations, and sample reuse, each requiring a distinct experiment.
