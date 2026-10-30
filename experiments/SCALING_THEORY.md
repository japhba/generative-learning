# From a selected score to a scaling law

This calculation concerns the b2k best-of-two algorithm: sample two complete outputs from the current policy, prefer a correct output when available, break reward ties uniformly, and fit the chosen output. Both failures are retained. There is explicit correctness feedback.

For any normalized differentiable policy, write `p(theta)` for success probability and `g(y)=grad log pi_theta(y)`. The selected distribution is

```
q_theta(y) = pi_theta(y) [1 + r(y) - p(theta)].
E_selected[g] = grad p(theta).
```

Thus small-step population SGD gives

```
d theta / d t = grad p(theta),
d p / d t = ||grad p(theta)||².
```

In the full-trace LLM implementation, `t = eta * updates / 128`, because128 is the fixed completion-loss normalization. For the one-token implementation there is no division by128: `t = eta * updates`. With batch sizeB, each update costs2Bcandidate trajectories. This is an optimization-time relation; actual generated tokens also change with the policy. On a distinct training and test distribution, the success derivative is the overlap of their two gradients, not necessarily a positive squared norm.

Let `epsilon = p_infinity - p`. If, in a stable regime,

```
||grad p||² ~ c epsilon^gamma,
```

then for `gamma>1`,

```
epsilon(t) ~ [c (gamma-1) t]^[-1/(gamma-1)].
```

For gamma1 the corresponding solution is exponential. This links a measurable weight-space quantity to a learning exponent. It requires knowing or identifying the ceiling and controlling finite-step and stochastic corrections. The basic connection between gradient geometry and policy-gradient convergence is established literature; the one-gate1/t result is consistent with [Mei et al., ICML2020](https://proceedings.mlr.press/v119/mei20b.html). We make no novelty claim for that result or the elementary ODE implication.

## A single Boltzmann gate: exponent1

For `p=sigmoid(z)`, normalized by the partition function `1+exp(z)`,

```
grad_z p = p(1-p),
d p/d t = p²(1-p)².
```

As p approaches1, gamma2 and `1-p ~ 1/t`. An exact implicit solution follows by integrating

```
t = F(p)-F(p0),
F(p) = -1/p + 1/(1-p) + 2 log[p/(1-p)].
```

If this gate chooses between fixed output experts with success ratesa>b, the ceiling isa and the exponent of the excess error remains1, up to the appropriate constants.

Finite two-sample selection also has a controlled small-error limit here. Selected failure probability is `epsilon²`; the scalar selected-score variance is `O(epsilon²)`. In the transformed variable `exp(z)`, leading drift is constant and fluctuations are smaller than the growing mean. This gives the same leading inverse-time error law for fixed moderate eta and finiteB, unlike the RBM's persistent boundary noise below.

An additional fresh finite-sample check used512 replicas,262144 updates, eta0.2, and batches1 and8. The final-factor8 slopes were1.00073 and1.00084; endpoint errors divided by `1/(eta*updates)` were1.00028 and1.00033. Those narrow finite-window slope intervals exclude exactly1 because finite-time corrections remain measurable. Both slopes and amplitudes approach the independently derived limit; no amplitude was fitted.

## Gaussian-visible/binary-hidden RBM: population exponent1/2

Keep the same normalized model as b2k:

```
p_w(x,h) proportional to exp(-||x||²/2 + h w·x),
Z(w) = 2 (2pi)^(d/2) exp(||w||²/2),
pi_w(h|x) = exp(h w·x) / [2 cosh(w·x)].
```

The real inputs are the balanced Gaussian mixture `x=h_true*mu+Normal(0,I)`. Write `m=||mu||`, `R=||w||`, and `rho=<w,mu>/(R*m)`. The signed margin before multiplying byR has distribution `Normal(m*rho,1)`. Bayes classification error is `Phi(-m)`.

The normalized overlap increases under the population selected-score dynamics. Its exact derivative is

```
d rho/d t = (m/R) (1-rho²) A,
A = E[0.5 sech²(R Z)],  Z~Normal(m*rho,1).
```

For largeR and aligned weights, define `a=m*phi(m)*pi²/24`. Expanding the smooth margin density near zero gives

```
epsilon = error - Phi(-m) ~ a/R²,
dR/dt ~ 2a/R³,
R^4 ~ 8a t,
epsilon ~ sqrt[a/(8t)].
```

Consequently `||grad p||² ~ (4/a)*epsilon³`: gamma3 and exponent1/2. The angular transient shrinks faster than this radial population tail. Atm1.5 the predicted error coefficient is0.0999330804. No time constant is fitted to numerical learning curves.

The important difference from the homogeneous gate is the continuous mass of examples close to a decision boundary. A common selected-score identity does not force a common exponent. Here the shared phenomenon is success-gradient optimization through a probabilistic output model; the margin geometry determines its rate.

This criterion optimizes sampled correctness, not the RBM's joint data likelihood. The normalized overlap tends to1 while the norm diverges; the weights do not converge in Euclidean distance to the true generative parameter mu. That distinction must remain visible in interpretations of weight overlap.

In the aligned population tail, greedy classification error is already at the Bayes floor. The remaining1/2 sampled-error exponent describes confidence sharpening. In the noisy multi-weight tail, angular error also changes the decision boundary; it accounts for about97.2% (batch1) and93.3% (batch2) of the final excess sampled error in the fresh confirmation. Thus the finite-noise1/3 law also appears in excess greedy error. This connection uses the RBM conditional policy and success objective; contrastive divergence and joint-likelihood learning require a separate analysis.

## Finite sampling and the frozen source

For finite steps, the second-order expected success change also involves the Hessian and the update covariance. Angular diffusion in a multi-weight RBM can affect the accessible scaling regime. Population laws are therefore not automatically finite-batch SGD laws.

### A finite-batch RBM prediction: exponent1/3

Write the selected score as `s=(h-tanh(w·x))x`, the step size as eta, and the batch size asB. In the small-step diffusion approximation,

```
dw = grad p dt + sqrt(eta/B) Sigma(w)^(1/2) dW,
Sigma(w) = Cov_selected(s),  t=eta*updates.
```

At aligned largeR, every direction perpendicular to mu has selected-score variance `2 phi(m)/R` to leading order. This follows from the exact conditional second moment

```
E_selected[(h-tanh(RZ))² | Z]
    = sech²(RZ) [1-tanh(RZ)],
integral sech²(v)[1-tanh(v)] dv = 2.
```

Define `lambda=(d-1)*eta*phi(m)/B`. Ito's formula for the norm then adds a positive radial drift `lambda/R²`. It eventually dominates the population radial drift `2a/R³` when d>1. The transverse coordinates relax on a faster timescale, with approximate variance

```
E||w_perpendicular||² ~ (d-1)*eta*R/(B*m).
```

Their angular contribution to excess error is

```
epsilon_angular ~ m*phi(m)*E||w_perpendicular||²/(2R²)
                ~ lambda/(2R).
R³ ~ 3 lambda t,
epsilon ~ lambda^(2/3) / [2*3^(1/3)] * t^(-1/3).
```

This is an asymptotic diffusion and timescale-separation calculation, not an exact finite-eta theorem. It predicts the amplitude and its batch-size dependence before any noisy-SGD curve fit. For d1 the transverse term vanishes and the population1/2 regime survives. At a fixed candidate budget `C=2B*updates`, the noisy asymptote is proportional to `(eta/B)^(1/3)*C^(-1/3)`; the population regime is proportional to `sqrt(B/eta)*C^(-1/2)`. The constants also depend on m and, in the noisy case, d.

The second-order covariance correction is essential here. In the slowly evolving angular distribution, leading tangential success-gradient and diffusion terms balance. Applying `dp/dt=||grad p||²` directly to noisy iterates would miss this balance and need not predict the observed exponent.

The fresh s4g confirmation used512 independent replicas and262144 updates. The finite-window slopes were0.50192 in d1,0.35314 in d8/B1, and0.34016 in d8/B2. Final errors divided by the predictions with no fitted amplitude were1.00113,0.99710, and1.03500. Windows were the final factor8 in updates. Some bootstrap intervals narrowly exclude the eventual asymptotic exponent; these are finite-window slope intervals, not a proof of an exact asymptote. All pilot and confirmation curves are retained.

### Frozen-source dynamics

For a frozen proposalq, the selected-score expectation decomposes as

```
E_q[g] + Cov_q(r,g).
```

The first term generally does not vanish at the student's updated weights. In the one-gate model, the student approaches the frozen source's initial selected-success probability. Its local convergence to that interior optimum is exponential, to a different error floor. Comparing raw curves without treating these ceilings separately would confuse a change in attainable error with a change in exponent.

The LLM tests will therefore ask whether measured gradient overlaps predict future improvement and whether a stable relation between overlaps and excess error exists. Reproducing a power fit alone is insufficient.

## A conditional one-weight bridge to a full LLM

The short-output LLM pilot uses the full pretrained vocabulary and updates every model parameter. It is not restricted to a scalar architecture. The scalar model is a separately frozen forecast of those updates.

For every calibration/evaluation question, record its initial correct-token probability `p_i0`. Define fixed biases `b_i=logit(p_i0)` and a shared adjustable weightz:

```
p_i(z) = exp(b_i+z) / [1+exp(b_i+z)].
A_train(z) = mean_i p_i(z)[1-p_i(z)].
A_eval(z) = mean_j p_j(z)[1-p_j(z)].
```

This is a normalized conditional Boltzmann model. It collapses all incorrect outputs into one event while retaining heterogeneity across questions. Its metric factor is calibrated from the actual initial LLM weight-gradient overlapG:

```
k = eta*G / [A_train(0)*A_eval(0)].
current: z_(n+1) = z_n + k*A_train(z_n).
frozen:  z_(n+1) = z_n + k*[mean_i(2*p_i0-p_i0²)-mean_i p_i(z_n)].
```

The numerical curves are written before sampled training. They assume a common additive logit shift and a constant metric; a full network can violate both. Exact per-question probabilities, current gradient overlaps, and actual parameter displacement projected onto the initial held-out success gradient test those assumptions. No gold gradient is used for an actual learner update: training fits only its selected sampled token. A separate initial population-gradient Taylor check restores all weights before sampled training begins.

For a finite set of positive initial probabilities, this shared-shift model eventually has1/t excess error under current sampling, but it can have long heterogeneous transients. Frozen sampling converges to an interior optimum with a floor and locally exponential relaxation. A finite power fit to raw frozen error is therefore not directly comparable to a current-source exponent. The simple one-token task isolates the statistical update; it cannot establish that generating multistep reasoning without feedback improves a model.

## Limits revealed by the LLM pilot

The first full-trace arithmetic forecast agreed closely with one later endpoint, but both runs stopped at64updates on selected-trace NLL stationarity. This is too short to identify an exponent.

The short-output counting pilot provided a longer counterexample to a fixed early geometry. Error fell from0.3580 to0.00673 by1024updates. Its32..128 overlap/error relation suggested gamma3, but a numerical gamma3 forecast frozen after384updates predicted0.06469 error at512; the observed error was0.02844. The weight displacement cosine with the initial success gradient fell from0.974 at32updates to0.620 at512. Different answer groups improved at different times. These observations support the shared gradient identity while rejecting a fixed initial direction or a constant cubic coefficient across the whole run.

The toy's Gaussian-mixture Bayes uncertainty is not the same statistical condition as deterministic counting labels with an evolving large representation. For example, a fixed separating margin yields exponentially small error in a growing logit and typically inverse-time optimization error; continuous mass near a boundary can instead yield polynomial norm/error relations. To predict an LLM exponent from the toy, one must identify the corresponding margin/gradient regime and its floor rather than assume those features are shared. The fresh-seed test retains this distinction explicitly.

## Completed fresh LLM test and a more specific overlap diagnostic

Three fresh Qwen2.5-1.5B-Instruct runs completed2048 updates. At the common4096-candidate budget (64 updates), current-source accuracy was82.895% and frozen-source accuracy69.431%, a paired13.465percentage-point gain with a95% hierarchical interval12.110–14.865points. Current-source final sampled accuracy averaged99.744%; frozen-source students stopped at64,64,192updates. The endpoints have different budgets and must not be used as a matched-budget contrast.

The mean-curve unshifted zero-floor finite-window exponent was1.460 over128..1024updates and1.708 over256..2048updates. Local exponents varied strongly, and the early cubic-coefficient forecast failed on all fresh runs. A secondary, prospective quadratic-geometry forecast anchored at1024 predicted2048 errors within23% relative error across the three runs. This is limited evidence for local rate forecasting over a twofold window, not a global exponent or a known asymptotic floor. Observer gradients use512 inputs; their finite-input uncertainty remains despite exact output probabilities.

A retrospective reduction using three initial class-success gradients also failed to reproduce the whole pilot curve. Its coefficients came entirely from initial probabilities, the learner's rate, and initial gradient overlaps, but its three-mode structure was chosen after seeing class-specific learning phases. It predicted93.95% at64updates versus83.40% observed. The span of those three initial gradients captured42.12% of final squared parameter displacement. Its closer late endpoint does not validate its dynamics.

Direct checkpoint measurements make the missing state dependence concrete. Define `p_a` as success conditional on answer classa, `g_a=grad p_a`, and `P(a)` as the class frequency. Within the corresponding population, or exactly within the fixed empirical observer cohort,

```
dp_a/dt = sum_b P(b) <g_a, g_b>.
```

The initial gradient cosine between count0 and count1 was+0.295, then−0.195 at32updates and−0.588 at128updates. A direction that helps one class can locally hurt the other; the full update can still improve both through their diagonal terms. A fixed initial overlap matrix cannot reproduce this measured change. Gradient rotation alone does not prove new hidden-feature learning: nonlinear probabilities and changing example weights can also produce it with fixed underlying features.

These experiments establish a shared selected-score mechanism and several toy scaling regimes. They do not establish an intrinsic generation-versus-recognition exponent advantage or improvement without a checker. In the LLM, the monitoring set is held out of optimizer data but is used for rate calibration, diagnostics and stopping; it is not an untouched final test set. The new study does not establish cross-model or ICL equivalence.
