# Self-contrastive Gaussian scaling: completed experiment

Completed on 2026-09-08. The experiment separates two effects: feature learning
changes the population error exponent, while refreshing generated negatives
improves finite-budget error without a distinct late exponent in the tested
Gaussian family. No LLM was trained and no paid API or rented compute was used.

Report: http://127.0.0.1:8095/self-contrastive-scaling.html

## Model and intervention

The true conditional distribution is y | x ~ N(w* · x, 1), with x ~ N(0, Lambda).
The student has the same observation variance and mean f(x). Its two
parameterizations are f(x)=v·x and f(x)=a(v·x). The latter is a two-layer linear
network with one hidden unit, training both its feature vector v and readout a.
Initial v=0 and a=1 give identical zero predictions and identical initial tangent
kernels. Both source arms train all parameters within their architecture.

We minimize the negative of

    V = E_real log D(y|x) + E_source log(1-D(y|x))
    D(y|x) = p_student(y|x) / [p_student(y|x) + q_source(y|x)].

Frozen-source negatives use the initial student N(0,1). Refreshed negatives use
a detached copy of the current student before every update. Source densities and
generated targets are held fixed when differentiating. Labels indicate whether
a draw came from the real-data distribution or the source; there is no task
answer checker. Both methods receive real observations throughout training.

This is a test of adaptive negative generation in density learning. It is not
unconditional self-imitation, a wake-sleep latent-inference experiment, or a
reduction of our LLM arithmetic runs.

## Expected updates and the spectral prediction

For the effective coefficient vector w, self-contrastive estimation gives

    G_current(w) = 1/2 Lambda (w* - w).

This is the known maximum-likelihood gradient identity
[Goodfellow, Section 2](https://arxiv.org/html/1412.6515), applied to the Gaussian
conditional model. The frozen gradient also has a closed Gaussian-moment form

    G_frozen(w) = Lambda [c(w) w* - d(w) w].

The implementation computes c,d by Gaussian quadrature. A preliminary
one-million-sample check agreed with the analytic gradient within 2.3 Monte Carlo
standard errors in every tested coordinate. At w=w*, c=d, giving the same
optimum exactly, including in the quadrature implementation.

For the deep model, let w=a v and s=a^2. Population gradient flow preserves
a^2-||v||^2=a0^2. Therefore

    s = [a0^2 + sqrt(a0^4 + 4 ||w||^2)] / 2
    dw/dt = [s I + w w^T / s] G(w).

Both source choices have this evolving kernel. The extra factorization is
available in both arms; it is not given only to the refreshed source.

For initial error mass dA(lambda) proportional to lambda^(beta-1) d lambda,
the source/feature-learning prediction recorded before the main runs was:

- Shallow: exponent beta under either source.
- Two layers, beta<1: exponent 2 beta/(1+beta) under either source.
- Two layers, beta>1: exponent beta under either source.

The feature-learning prediction is motivated by
[Bordelon et al.](https://arxiv.org/html/2409.17858v2) and the growing kernel in
the factor dynamics above. These experiments check its application to this
particular minimal factorization; they do not prove a global rate theorem for
the nonlinear frozen-NCE objective.

## Why the frozen source preserves the slow spectral order here

Let S=w*^T Lambda w*, and m=w*·x. Define the scalar conditional NCE curvature

    h(m) = E_(y~N(m,1)) [
        q0(y)/(p_m(y)+q0(y)) (y-m)^2
    ].

For a Gaussian input, conditioning x on m gives the exact optimum Hessian

    H_frozen = Lambda^(1/2)
               [h_perp (I-u u^T) + h_parallel u u^T]
               Lambda^(1/2)
    u = Lambda^(1/2) w* / sqrt(S)
    h_perp = E_m h(m)
    h_parallel = E_m [h(m) m^2 / S].

For any fixed, finite, positive S, both scalar coefficients are strictly
positive. The matrix is bounded above and below by constant multiples of Lambda,
independent of the small eigenvalues. Refreshing instead gives H_current=Lambda/2.

In the main case S=4, the two frozen bounds are approximately 0.24095 and
0.37977. Thus the frozen source weakens learning but does not introduce a new
power of lambda in the local spectrum. This is a structural explanation for the
equal source exponents, not an assertion that a local Hessian bound alone proves
every global deep-network convergence rate.

## Population results

The main case has beta=0.5 and signal standard deviation 2. Exponents were fitted
in the predeclared population-time window 100,000 to 10,000,000:

| Architecture | Frozen source | Refreshed source | Prediction |
| --- | ---: | ---: | ---: |
| Shallow linear | 0.501785 | 0.500001 | 0.5 |
| Two-layer linear | 0.666231 | 0.666023 | 2/3 |

All nine settings support the same distinction. We varied beta across 0.5, 0.8,
and 1.5; signal standard deviation across 0.5 and 2; the initial factor scale; and
a smooth perturbation of the spectral mass. At beta=1.5, all methods have exponent
approximately 1.5. At beta=0.8, shallow fits are approximately 0.8 and deep fits
0.882–0.891; the weak-signal deep crossover remains incomplete at this budget.

The population integrator uses logarithmic spectral quadrature from lambda=1e-18
to 1. It represents the continuous error-mass measure; its 1,024 quadrature
coordinates must not be interpreted as 1,024 equally weighted real-world modes.
Time is integrated expected-gradient time, not a generated-sample count. At a
fixed finite spectral cutoff the ultimate behavior can change again.

The finite-dimensional SGD experiment below uses ordinary discrete eigenvalues
instead. It is a finite-sample check of the mechanism, not a pointwise replication
of the continuum trajectories.

## Actual sampled weight training

There were 24 runs: six paired seeds in each of four arms. Each run used 1,024
input coordinates, lambda_k=k^-2, initial modal error proportional to k^-2,
batch size 32, learning rate 0.05, and 8,192 updates. Every run received 262,144
real conditional observations and 262,144 generated negatives. Inputs and Gaussian
noise were coupled across methods. All runs completed their budgets; evaluation
ran throughout.

The final evaluation is the exact excess Gaussian negative log likelihood,

    L_excess = 1/2 (w-w*)^T Lambda (w-w*).

| Architecture | Frozen error | Refreshed error | Frozen/current ratio |
| --- | ---: | ---: | ---: |
| Shallow | 0.064921 | 0.052452 | 1.238 |
| Two layers | 0.023614 | 0.021019 | 1.123 |

The paired bootstrap 95% intervals for frozen minus refreshed error are
[0.011432, 0.013820] in the shallow case and [0.000830, 0.004431] in the deep case.
Six of six shallow seeds and five of six deep seeds favor refresh. The bootstrap
resamples six training seeds; it quantifies training randomness in this model,
not uncertainty across model families.

The noisy finite curves are not asserted to establish asymptotic exponents.
The negative-source ratio is a finite-budget error ratio, not a measured compute
saving factor. Counts of real observations, generated negatives, and optimizer
updates match within each architecture; dominant compute per update is O(BD)
in both source arms. The source mean can reuse the student's existing prediction.
No population integration time is relabeled as actual sample consumption.

## Numerical checks and evidence

- Doubling integration steps, spectral quadrature size, and Gaussian quadrature
  order changes the main curves by at most 0.156%.
- The shallow current curve agrees with its independent exact modal solution
  within 0.036%.
- Explicit RK4 on the original a,v weights agrees with the reduced deep solver
  within 0.094% at the checked time near 30.
- A finite-sample orthogonal rotation check passes to machine precision.
- The runtime source copy matches the hash recorded before the main experiment.

A small initial solver check used lambda_min=1e-12 and showed finite-cutoff effects
in the deepest late tail. The main run moved this numerical cutoff to 1e-18.
The equal-source-exponent prediction and main fit windows were unchanged.
The persisted protocol in results/self_contrastive_spectral/protocol.json records
the main settings and hypothesis; the exact lambda cutoff is also in its hashed
source copy. This note was written after completion and is not a preregistration.

Reproduce with the project environment and a new output directory:

    OPENBLAS_NUM_THREADS=1 /var/tmp/jbauer/venvs/generative-learning/bin/python \
        experiments/self_contrastive_spectral.py --output results/self_contrastive_reproduction

The analysis script reads the canonical results directory:

    OPENBLAS_NUM_THREADS=1 /var/tmp/jbauer/venvs/generative-learning/bin/python \
        experiments/analyze_self_contrastive_spectral.py

Canonical evidence: results/self_contrastive_spectral/. Final finite-run factor
weights, all metric checkpoints, protocol, source copy, continuum trajectories,
and numerical refinement results are retained. Static figures and the local HTML
report are under reports/.

The remaining target is a mechanism by which adaptive generation changes the
slow spectral order or its ongoing evolution. Merely adding depth to this
Gaussian source-refresh comparison does not supply that mechanism.

