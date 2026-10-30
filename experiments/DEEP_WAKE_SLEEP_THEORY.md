# Wake–sleep, depth, and what would actually change a scaling law

This investigation extends the previous six-cause Gaussian decoder to directed
sigmoid belief networks with one or three stochastic latent layers, and affine
or three-layer tanh recognition networks. It also implements the same two
conditional distributions using LLM adapters. Temperature is always one; no
correctness checker, acceptance rule, or true latent label enters sleep or wake
updates. All negative outcomes and tuning runs are retained.

## A local constraint that survives depth

Fix an observation x and let a parameterize recognition logits. At q=p,
both KL(q||p) and KL(p||q) have the same logit Hessian:

    F = diag(p) - p p^T.

For any differentiable recognition architecture with logit Jacobian J, their
parameter Hessians are both J^T F J. Terms involving second derivatives of
the network vanish because the logit gradient is zero at equality. Thus adding
depth does not itself distinguish these two local quadratic objectives.

This extends to coupled learning at a correctly specified regular solution
p_theta(x)=p_data(x), q_phi(z|x)=p_theta(z|x). Write the ELBO gradient's
linearization in (theta,phi) as H. Adding sleep with a fixed weight gamma leaves
the generator row unchanged and multiplies the recognition row by 1+gamma.
The resulting linearized flow is D H, where D=diag(I, (1+gamma)I). For a
positive semidefinite H, its nonzero eigenvalues equal those of
H^(1/2) D H^(1/2), whose eigenvalues are bounded between the corresponding
eigenvalues of H and (1+gamma)H. This is a bounded preconditioner, not a new
slow spectral-tail exponent. A bounded extra cost per update does not alter
this conclusion. The claim is local and concerns exact gradient flow under
these assumptions, not Adam, misspecified recognition families, noise-limited
SGD, nonregular solutions, or globally nonlinear trajectories.

The numerical audit checks directional derivatives for one-, two-, and
three-layer belief and recognition networks. It separately checks that the
relative difference between forward- and reverse-KL logit gradients vanishes
linearly with perturbation size near q=p.

## The nonlocal candidate: neglected explanations

For a binary posterior with p(z=1|x)=1/2 and q=sigmoid(a),

    d KL(p||q)/da = q - 1/2,
    d KL(q||p)/da = q(1-q) a.

At a=-M the first gradient is order one, while the second is order M exp(-M).
Under Euclidean logit gradient flow, time to reach q=1/4 scales as M for
forward fitting, and exp(M)/M for reverse fitting. This is an optimization
recovery law in initial neglect, not an error-versus-compute power law. It is
also parameterization/optimizer dependent, which motivates checking Adam and
separately tuning rates instead of deliberately choosing a stalled baseline.

Sleep can supply labels for causes that the current recognizer seldom proposes.
This requires an already useful generator. In the toy, random output biases
with standard deviations 0, 3, and 6 deliberately vary recognition neglect,
independently of generator parameters. The intervention is explicit: these
runs do not establish that ordinary pretraining naturally produces such biases.

## Why a broad family is required

In a mode approximation, let initial errors satisfy a_j ~ j^(-1-s), and
learning rates r_j ~ j^(-b). Then

    E(C) ~ sum_j a_j exp(-r_j C) ~ C^(-s/b).

Changing rates by uniformly bounded factors changes coefficients, not the
exponent. An exponent change needs an unbounded relative gain on increasingly
slow directions, a different error tail, or persistent feature evolution. At
fixed finite support, eventual exponential convergence, optimizer noise, or
misspecification floors can supersede apparent powers. The previous toy's
window-dependent slopes are therefore unsurprising.

The initial expansion checked different depths, 16–1024 latent configurations,
population and sampled learning, and fixed and coupled generators. No modes are
ranked, oversampled, or rejected. A different exponent must survive shifts of
the fitting window, finite-size changes, rate controls, and generation cost.
Fitting a selected transient is insufficient. The original four-size experiment
was exploratory. The million-state follow-up now checks nine sizes with the
fixed two-layer generator and affine recognizer, including moving fit windows,
numerical stabilizers and more than one evaluation criterion.

## LLM match and limits

The LLM has two separately trainable rank-four q/v LoRA adapters on the same
Qwen2.5-1.5B-Instruct backbone. The generator predicts output digits given a
named rule and input; the recognizer predicts the rule from two examples.
There are eight rules and a uniform latent prior. The generator is a normalized
conditional distribution over ten digit tokens; recognition is normalized over
eight letter tokens. These constrained supports are identical across arms and
are part of the finite probabilistic model, not correctness-based selection.

Real observations are generated externally, with their latent labels removed
before the training loop. In sleep, the LLM itself samples every output digit
conditioned on a sampled rule; all sampled pairs are used, including examples
that violate the rule's intended arithmetic semantics. Evaluation alone uses an
executor. This is a specialized latent-variable learning experiment in an LLM,
not unrestricted chain-of-thought self-imitation or RLVR.

ELBO uses an exact eight-state expectation, avoiding a weak REINFORCE baseline.
The treatment adds inverse cross-entropy on imagined examples. The diagnostic
forward arm directly teaches the exact current-model posterior on real examples;
it gets no true latent labels. Fixed-generator and coupled-generator variants
are both tested. Generation/scoring and backward tokens contribute to the
compute axis, and checkpoint state includes adapters, optimizers, RNGs, step,
and data-stream position. Pilot metrics are exploratory, not confirmed exponents.

## Relation to existing work

[Le et al., Revisiting Reweighted Wake-Sleep](https://proceedings.mlr.press/v115/le20a.html)
describes separate generative and inference updates, including the distinction
between sleep targets averaged over model observations and forward-posterior
targets on real observations. It supplies motivation and practical tradeoffs,
not a general claim of different error-versus-compute exponents.

[Bordelon, Atanasov, and Pehlevan, How Feature Learning Can Improve Neural Scaling Laws](https://arxiv.org/html/2409.17858v2)
shows why task difficulty and persistent feature evolution can matter for
exponents. Its result is not a wake–sleep theorem; the local constraint and
the mechanism tests above are our deductions for this project.

## An exact connection to the recognition weights

For the affine full-state recognizer in the size experiment, the generator's
exact posterior is representable by another affine softmax. Write the design
vector, including a constant, as f(x), the recognition coefficient matrix as B,
and the compiled posterior coefficients as B*. Centering logits across latent
states fixes the irrelevant common-logit gauge. Let e = vec(B-B*) and let
J_x map coefficient errors to logit errors. Define

    M(B) = E_x [J_x^T F(q_B(.|x)) J_x],
    F(q) = diag(q) - q q^T.

The wake recognition gradient is exactly

    grad_B KL(q_B || p) = M(B) e.

The expected sleep/forward gradient, when model and data marginals match, is
exactly

    grad_B KL(p || q_B) = [integral_0^1 M(B* + s(B-B*)) ds] e.

The ELBO-plus-sleep treatment adds this forward gradient to the wake gradient;
the posterior-teaching control uses the forward gradient on its own.

The first identity follows by writing the centered log-ratio as J_x e. The
second follows by integrating the softmax Jacobian along the straight segment
between the two natural-parameter vectors. These are global identities for
this affine, realizable recognition family, not just local approximations.
They explain the mechanism in actual weights: wake uses the covariance under
the current recognizer, which can be very small in neglected directions;
forward fitting integrates covariances along the path to the useful posterior.
Both already use information from the data. Generation is not required for
every valid way to compute the forward gradient, as the posterior-teaching
control demonstrates.

Under Euclidean gradient flow, both objectives decrease squared distance to B*:
the radial contractions are nonnegative. For wake, the contraction is the
data-average variance under q of the log-ratio log(q/p); for forward fitting,
it is the data-average Jeffreys divergence KL(q||p)+KL(p||q). This does not
guarantee that forward fitting is faster in every state, nor does it establish
an ordering under Adam or finite step sizes. Near B*, the two operators agree,
recovering the local constraint above. A nonlocal recovery advantage therefore
need not remain a different late-time exponent.

`audit_wake_sleep_weights.py` verifies posterior compilation, both weight-gradient
identities, and the radial contraction formula across 36 combinations of
generative depth, initialization seed and neglect. The nonlinear LLM recognizer
does not have this exact global weight identity; that connection remains a
mechanistic analogy and a local linearization, rather than an exponent theorem.

## Separating optimizer units from a size exponent

The million-state extension preserves the earlier Adam rates and compares
epsilon = 1e-8 and 1e-12 at every new size. This control becomes more consequential
when a probability distribution is spread across many output coordinates.

For a positive rescaling of the objective by a, Adam's moments scale as
m -> a m and v -> a^2 v along a common weight trajectory. The resulting update is

    -eta (a m) / (sqrt(a^2 v) + epsilon)
      = -eta m / (sqrt(v) + epsilon/a).

Thus epsilon sets an absolute gradient scale; it is not invariant to the units
of the loss. Where a typical per-state gradient scales approximately as 1/K,
the effective numerical stabilizer grows approximately as K epsilon. This
per-state approximation is not an exact distribution-wide law for our belief
networks: posterior probabilities and coordinate gradients are heterogeneous.

In an epsilon-dominated direction, updates behave more like gradient descent
with step size eta/epsilon. If that direction also has curvature proportional
to 1/K, its relaxation can require order K updates rather than a constant
number. Multiplying by an order-K dense update cost can then produce an
apparent quadratic cost law. A neglected wake direction can enter this regime
earlier than its sleep counterpart. This is a candidate finite-range numerical
mechanism, not a claim that either method has a universal quadratic exponent.
Eventually both local objectives retain the same curvature constraint above.

We therefore report update counts separately from matrix-product cost, repeat
the stabilizer control over the whole enlarged range, and fit both the original
and the new size windows. A nearly linear cost law with approximately constant
update count is useful, but its factor of K also comes from a recognizer that
has K output classes. It does not by itself establish an error-versus-samples
law for a fixed architecture or an exponent for an LLM.

The GPU implementation caches the fixed generator. Its dense recognition cost
is 4 n_x d K per wake update and 6 n_x d K per combined wake-plus-sleep update:
one forward matrix product, one wake gradient product, and one additional
sleep gradient product. Elementwise kernels, setup and evaluations are excluded
from that leading-dense count. The previous uncached reference FLOP convention
is also retained for comparison, explicitly labeled as reference accounting.
Every size is rerun on the same backend and evaluation schedule because a
first-hit accuracy target can be sensitive to brief error dips between the
earlier, more widely spaced checkpoints.

The size experiment increases K while holding the average posterior-KL target
at 0.1. This is a different limit from sending error to zero at a fixed K.
A small average KL need not bound every relative probability error or centered
logit error uniformly as the number of rare configurations grows. The shared
local Fisher matrix at q=p therefore does not rule out a size-dependent cost
to this fixed target. Conversely, a measured size exponent does not establish
a different late-time error exponent at a fixed architecture.
