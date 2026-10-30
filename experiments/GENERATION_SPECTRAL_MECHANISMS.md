# Generation and exponent changes: mechanisms and an equivalence audit

Status: initial theoretical investigation on 2026-09-08. The gradient audit below
was followed by the completed [self-contrastive spectral experiment](SELF_CONTRASTIVE_SPECTRAL_RESULTS.md).
That follow-up finds distinct shallow/deep population exponents, but equal late
exponents under frozen/refreshed sources within each architecture. Its actual
sampled SGD runs show a finite-budget source-refresh benefit. No LLM was trained.

## What the feature-learning paper contributes

[Bordelon, Atanasov, and Pehlevan](https://arxiv.org/html/2409.17858v2),
Section 4 and Appendix G, separate a fixed feature kernel from an evolving one.
For source exponent beta < 1, their limiting time exponent improves from beta
to 2 beta / (1 + beta). For beta > 1, the corresponding exponent stays beta.
Continued growth in weight norms and effective kernel strength supports the
improved hard-task scaling. Their comparison is feature learning versus lazy
learning; it does not identify generation as its cause. Their finite offline
dataset-size bottleneck has the same exponent in the two regimes (Appendix E.4).
Time, data count, and compute therefore require separate treatment.

## A diagnostic for a generation hypothesis

For independent error modes let lambda_k be the baseline learning rate and a_k
their initial contributions to the common evaluation error. Suppose

```
lambda_k ~ k^(-alpha)
sum_(j>k) a_j ~ k^(-alpha beta)
L(t) = sum_k a_k exp(-2 lambda_k t) ~ t^(-beta).
```

The integral argument only needs the corresponding regularly varying spectral
tail, rather than exact powers for every eigenvalue. Infinitely many modes, or
a growing intermediate range before a finite-size cutoff, are essential here.
At fixed finite dimension with strictly positive mode rates, the eventual
population decay is exponential. Fixed-step noisy learning can instead hit a
noise floor.

If a method changes all relevant rates by a scalar g(t), its effective time is
tau(t) = integral_0^t g(s) ds. A finite nonzero limiting g changes the coefficient.
If g(t) ~ t^delta with delta > 0, the same model instead gives
L(t) ~ t^[-beta(1+delta)]. This is a conditional calculation: prescribing g(t)
does not explain how generation produces it. Another route is mode-dependent
rates lambda_k^r with 0 < r < 1, yielding exponent beta/r under the same tail.
That route requires a gain that grows toward the increasingly slow modes, even
without a time-dependent scalar gain.

For a variable per-update cost c(t) ~ t^kappa, total work grows as t^(1+kappa).
The scalar-gain exponent in work is beta(1+delta)/(1+kappa). Increasing rollout
length can therefore consume an apparent advantage in optimizer steps. Neither
formula is itself a guarantee for noncommuting, changing eigenvectors or noisy
gradients; those need separate dynamics and variance analysis.

An operational research target is thus: does self-generation produce a growing
gain on the remaining error directions per unit of work? Weight overlap is a
useful diagnostic, but cannot answer this alone. Track the error projected into
the effective update operator, its slow spectrum, and finite-sample variance.

## New diagnostic of our existing posterior-refresh toy

The existing generator is x = A z + noise, z ~ N(0,I), with noise variance sigma^2.
Write

```
C = A A^T + sigma^2 I
H = I + A^T A / sigma^2
B* = A^T C^(-1)
E = B - B*.
```

With fixed Gaussian recognition covariance, the B-dependent part of the
negative variational evidence lower bound is

```
V(B) = 1/2 E_x [ ||Bx||^2 + ||x - A Bx||^2 / sigma^2 ]
     = 1/2 tr(H E C E^T) + constant
grad_B V = H E C.
```

Our current-source posterior rollout has conditional mean

```
E[z1 | x, B] = (I - delta H) Bx + delta A^T x / sigma^2.
```

Fitting Bx to a detached z1 with half squared error gives

```
E[Delta B | B] = -eta delta H E C = -eta delta grad_B V.
```

Thus one-step refinement and direct variational optimization have the same
population vector field after constant learning-rate rescaling. This comparison
was not the original current-versus-frozen experiment: its frozen teacher has a
different, biased optimum. It explains why that positive result alone cannot
establish a generation-specific exponent advantage.

The equivalence survives an arbitrary differentiable recognition mean f_phi(x).
At fixed A and fixed recognition covariance,

```
grad_phi V = E_x [ J_phi(x)^T (H f_phi(x) - A^T x / sigma^2) ]
E[Delta phi | phi] = -eta delta grad_phi V.
```

Consequently, replacing B by a two- or three-layer network does not create a
new mean-gradient advantage for the rollout procedure over this matched
variational control. Both can learn features. This is not a general impossibility
theorem: changing the latent distribution family, refinement operator, target
loss, covariance parameters, or decoder dynamics requires a new analysis.
Equal mean gradients also do not imply equal finite-sample behavior. Our sampled
rollout includes extra target noise; its noise covariance matters.

The numerical audit uses the original A, 24 random recognition initializations
across one, two, and three linear layers, and finite differences of separately
implemented variational and detached-target losses. The largest absolute
directional-derivative discrepancy was 6.24e-11. Antithetic posterior noises
check the conditional target mean. These are checks of the identity, not new
training curves. Code: `audit_generation_gradient_equivalence.py`; evidence:
`results/generation_gradient_audit/analysis.json`.

The broader initialize-refine-distill idea is also studied as
[amortised MCMC](https://arxiv.org/abs/1702.08343). That literature motivates
changing the inference operation; it does not supply the desired exponent theorem.

## A canonical candidate with an actual changing source

[Self-contrastive estimation](https://arxiv.org/html/1412.6515), Section 2,
compares real samples with samples from a detached copy of the current model.
With a density model p_theta and a reference source q, the discriminator is

```
D_theta,q(x) = p_theta(x) / (p_theta(x) + q(x))
V(theta;q) = E_data log D + E_q log(1-D).
```

Refresh q to p_theta before each update, and hold q fixed while differentiating.
Then grad V = (1/2) grad E_data log p_theta. This is a known expected-gradient
identity, not a new result here. Our script checks it independently for three
Gaussian mean-estimation cases. Generated samples are labeled by their origin,
so no task-specific answer checker is required. A real-data anchor is still
required; this is not improvement from unconditional self-imitation alone.

For a fixed reference q, at a correctly specified regular optimum p*, the
negative NCE Hessian is

```
H_q = E_p* [ q(x)/(p*(x)+q(x)) s*(x) s*(x)^T ],
s*(x) = grad_theta log p_theta(x) at theta*.
```

Refreshing the source gives a local update operator equal to one half the Fisher
information. A fixed source discounts directions supported where it puts little
probability mass. This supplies a concrete operator to compare, with no manually
ranked curriculum. Poor reference distributions are known to cause optimization
difficulties even for Gaussian models; see
[Liu et al.](https://arxiv.org/abs/2110.11271).

This does not by itself prove different exponents. If the overlap weight is
bounded away from zero, it gives a uniformly bounded change of quadratic metric.
A proposed many-mode separation must show a changing spectral tail or adaptive
feature dynamics, not just a small overlap constant. A single regular Gaussian
mean problem only gives finite-dimensional optimization-rate differences, not
the requested pair of asymptotic power laws. Further, source choice also changes
gradient variance, and sampling an RBM can have a substantial mixing cost.

This candidate directly studies refreshing a model's generated samples. It is
an alternative contrastive density-learning experiment, not an equivalence to
our earlier arithmetic self-training or ICL experiments. Both fixed and refreshed
arms are discriminative objectives with generated negatives; the precise claim
would be about adaptive self-generation, not all generation versus recognition.

## Criteria for the next positive experiment

1. Derive the expected update and common optimum before fitting curves. Reject a
   proposed mechanism if its supposed advantage reduces to a constant rescaling
   of the matched control, as in the Gaussian one-step audit.
2. Use the same trainable architecture in both source conditions, and add a
   fixed-feature control in both. A generation benefit confined to the changing
   feature condition would be evidence for an interaction, not proof by itself.
3. Establish the slow-mode spectrum and target-error tail in a family of models.
   Derive the slope gap from those quantities before simulations. Vary spectral
   details, basis rotations, initializations, and dimension; retain easy regimes
   where the theory predicts no gap.
4. Measure a common held-out error above the same optimum. Separate gradient-flow
   dynamics, finite-sample learning, fixed-data limits, and total compute,
   including generation, inference, filtering, and training.
5. Compare sampled updates with their exact expected-gradient counterpart where
   tractable. If both have the same advantage, the cause is the induced learning
   objective or geometry; stochastic sampling is an implementation, not evidence
   that sampling randomness itself creates the benefit.

The concrete new result is an equivalence that rules out a naive depth-only
extension of the previous Gaussian mechanism. A canonical, generation-specific
exponent separation remains an open target.
