# A one-vector probabilistic toy for self-training

Date: 2026-09-06. Status: a mathematical reduction and a small numerical sanity check, not a reproduction of the LLM result or a claim of theoretical novelty. No new LLM training or paid compute was used.

## Claim and assumptions

A linear probabilistic classifier can improve its direction, and hence its greedy classification accuracy, by training on its own unfiltered sampled labels. The construction uses lower temperature for sampling than for training, continuously refreshed targets, ordinary Euclidean gradient flow, structured unlabeled inputs, and an initial positive overlap with the true signal.

The exact population dynamics close in two scalar weight overlaps. The small-weight limit is a linear dynamical system. This establishes a possible mechanism for self-improvement; identifying the mechanism in an LLM and predicting its learning curve require additional work.

## Generative model and RBM interpretation

Take a balanced binary Gaussian mixture:

\[
y\in\{-1,+1\},\qquad x=y\mu_\star+\epsilon,
\qquad \epsilon\sim\mathcal N(0,I_d).
\]

Write \(a=\|\mu_\star\|>0\). Training observes only \(x\). The true labels are used to define evaluation error, never to construct the update.

The learner has one vector \(w\in\mathbb R^d\), with the joint model

\[
p_w(x,y)=\frac12(2\pi)^{-d/2}
\exp\left[-\frac12\|x-yw\|^2\right]
\propto\exp\left[-\frac12\|x\|^2+yw^\top x\right].
\]

This is a Gaussian-visible RBM with one binary hidden spin and zero biases. Its conditional is

\[
p_w(y\mid x)=\frac{e^{yw^\top x}}{2\cosh(w^\top x)},
\qquad \mathbb E_w[y\mid x]=\tanh(w^\top x).
\]

The Bayes conditional is represented exactly at \(w=\mu_\star\). There is one learned linear projection; the tanh/sigmoid is the probabilistic output link. The latent variable is the class label, not a multistep reasoning trajectory.

Probabilistic grounding does not make the update below maximum marginal likelihood for the RBM. Marginal-likelihood ascent would instead use \(\mathbb E_x[x\tanh(w^\top x)]-w\).

## Sampling and learning

Given an input \(x\), sample an unfiltered pseudo-label at temperature \(0<\tau<1\):

\[
\widetilde y\sim p_w(\cdot\mid x;\tau),\qquad
p_w(y\mid x;\tau)=\frac{e^{yw^\top x/\tau}}{2\cosh(w^\top x/\tau)}.
\]

Then take an ordinary conditional log-likelihood ascent step at training temperature one:

\[
\Delta w=\eta x\,[\widetilde y-\tanh(w^\top x)].
\]

The sampled label is treated as a target: there is no differentiation through sampling. In the population, infinitesimal-step limit, refreshing the sampling weights continuously gives

\[
\dot w=\mathbb E_x[x f_\tau(w^\top x)],
\qquad f_\tau(s)=\tanh(s/\tau)-\tanh(s).
\]

At matched temperatures, \(\tau=1\), this vector field is exactly zero. Individual sampled gradients still have variance. Zero expected raw gradient does not imply that finite-step SGD or Adam leaves a model unchanged.

For \(\tau<1\), \(f_\tau\) is odd and strictly positive on the positive half-line.

## Exact overlap dynamics

Define

\[
m=w^\top\mu_\star,\qquad q=w^\top w,\qquad
\rho=\frac{m}{a\sqrt q}.
\]

For \(Z\sim\mathcal N(0,1)\), set

\[
A(m,q)=\mathbb E_Z[f_\tau(m+\sqrt q Z)],\qquad
B(m,q)=\mathbb E_Z[f'_\tau(m+\sqrt q Z)].
\]

Gaussian integration by parts and the mixture symmetry give

\[
\dot w=A\mu_\star+Bw.
\]

Consequently,

\[
\dot m=a^2A+mB,\qquad \dot q=2mA+2qB,
\]

and the normalized overlap obeys

\[
\boxed{\dot\rho=\frac{a}{\sqrt q}(1-\rho^2)A(m,q).}
\]

To determine its sign, let \(\varphi_q\) be the zero-mean Gaussian density with variance \(q\). Oddness of \(f_\tau\) implies

\[
A(m,q)=\int_0^\infty f_\tau(s)
\left[\varphi_q(s-m)-\varphi_q(s+m)\right]ds.
\]

For \(m>0\), both factors in the integrand are strictly positive. Thus \(A>0\), and \(\dot\rho>0\) whenever \(0<\rho<1\). This is an exact population statement for every finite nonzero weight norm, not just a small-weight approximation.

The greedy classification error is also exact:

\[
\boxed{\mathcal E(w)=\Pr(\operatorname{sign}(w^\top x)\ne y)
=\Phi(-a\rho),}
\]

where \(\Phi\) is the standard normal CDF. Therefore increasing overlap strictly decreases error. The Bayes error is \(\Phi(-a)\).

At zero overlap, the population signal stays zero. Negative overlap is driven toward the opposite label orientation. The unlabeled distribution cannot resolve this sign ambiguity; the initialization must supply it.

## A linear approximation that exposes the mechanism

For fixed \(\tau\) and sufficiently small weights,

\[
f_\tau(s)=(\tau^{-1}-1)s+O(s^3).
\]

Writing \(\kappa=\tau^{-1}-1>0\), the leading dynamics are

\[
\dot w\simeq\kappa\mathbb E[xx^\top]w
=\kappa(I+\mu_\star\mu_\star^\top)w.
\]

Decompose \(w\) into components parallel and perpendicular to \(\mu_\star\). In this linearized system,

\[
\dot w_\parallel=\kappa(1+a^2)w_\parallel,\qquad
\dot w_\perp=\kappa w_\perp.
\]

Hence, for an initial angle \(0<\theta_0<\pi/2\),

\[
\tan\theta(t)=\tan\theta_0\,e^{-\kappa a^2t}.
\]

The input covariance amplifies the initially useful component more quickly. This is closely related to principal-component amplification. The approximation applies only while the margins remain small compared with the sampling temperature; it is not an asymptotic learning law for the full nonlinear model.

## Frozen targets and complete distillation

Freeze the sampling vector at \(w_0\), but allow fresh pseudo-label samples on every presentation. The expected teacher distribution is exactly \(p_{w_0/\tau}(y\mid x;1)\). The unique minimizer of its population conditional cross-entropy is therefore

\[
w_{\mathrm{frozen}}^*=w_0/\tau.
\]

The input distribution has full support, so this parameter is identifiable. Complete fitting leaves the normalized overlap equal to its initial value. Intermediate frozen-teacher gradient steps can still improve the angle: the initial vector field is the same as for the refreshed teacher. No claim of dominance at every finite budget follows from the endpoint calculation.

If every refreshed-teacher round is itself fitted completely, each round similarly just rescales the weights. Directional improvement in the derived flow relies on refreshing the teacher after partial optimization. This is a concrete distinction between target sharpening alone and its interaction with shared weights, input geometry, and the optimization schedule.

## What information is available?

The unlabeled inputs contain information about the signal axis through their mixture geometry. The initial weights contain information about the label orientation. Training uses both. The generated labels do not add independent ground-truth information.

One may give both learners the same fixed unlabeled corpus at the outset and ask how effectively they extract its structure. The exact equations above use population expectations; finite-corpus adaptive reuse needs a separate analysis.

If both \(x\) and \(y\) are sampled entirely from the current model, without informative inputs or another grounded operation, this argument supplies no improvement toward an unknown external truth. With isotropic Gaussian inputs unrelated to \(\mu_\star\), integration by parts instead gives \(\dot w\parallel w\), so there is no directional improvement.

## Limits of the reduction

- **Overlap and norm differ.** For \(\tau<1\), \(\dot q=2\mathbb E[s f_\tau(s)]>0\). Directional error improves under the stated assumptions, but Euclidean distance to \(\mu_\star\) need not decrease, and calibrated probabilities are not guaranteed. Track norm separately.
- **Sampling is an implementation of the update.** Replacing the sampled label by its conditional mean produces the same expected gradient. The vector field also equals the gradient of \(\mathbb E_x[\tau\log\cosh(w^\top x/\tau)-\log\cosh(w^\top x)]\). Thus this construction does not establish a unique advantage of stochastic generation over every deterministic learning procedure.
- **The covariance assumption matters.** Here the informative direction is also the leading covariance direction. Nuisance structure could break that relationship. The toy models exploitation of useful input structure and initial alignment; it does not yet model exploration of reasoning trajectories.
- **A strictly linear Gaussian conditional is insufficient for this particular temperature mechanism.** Changing only its sampling variance preserves the pseudo-target mean. Squared-error mean learning then has zero expected update at the sampling parameters. The binary output link is the minimal nonlinearity used here, not a proof that every fully linear self-training construction is impossible.
- **Learning-rate predictions remain bounded in scope.** Population time has to be related to step size and the number of updates. Finite-batch noise, dimension, repeated inputs, optimizer choice, rollout cost, and teacher-refresh frequency must be included before comparing sample or compute efficiency. No LLM scaling exponent is derived or fitted here.

## Numerical sanity check

A local CPU calculation used 160-node Gauss-Hermite quadrature and fourth-order Runge-Kutta integration with step 0.02, for 1,000 steps. Parameters were

\[
\mu_\star=(1.5,0),\qquad
w_0=(0.3,\sqrt{1-0.3^2}),\qquad \tau=0.5.
\]

At population time 20, the weight was approximately \((1.68507,1.24546)\). Overlap increased from 0.3 to 0.80418; the analytic classification error decreased from 0.32636 to 0.11386; norm increased from 1 to 2.09539. These are deterministic toy calculations, not sampled training or held-out LLM measurements.

Additional checks verified positive angular drift at 27 combinations of temperature, norm, and positive overlap; agreement between the angular formula and the vector derivative; exact cancellation at matched temperature; and agreement of the Stein decomposition with independent two-dimensional Gaussian quadrature.

## Relationship to prior work

[A Model Can Help Itself](https://arxiv.org/html/2510.18814v4) motivates the sampled-response, conditional-likelihood, temperature-mismatch, and teacher-refresh ingredients. This note replaces an autoregressive LLM by a binary linear conditional and analyzes a continuous population limit; it is not a reproduction of that paper's experiments.

[Oymak and Gulcu (2021), A Theoretical Characterization of Semi-supervised Learning with Self-training for Gaussian Mixture Models](https://proceedings.mlr.press/v130/oymak21a.html) provides prior theory for linear self-training in Gaussian mixtures. The specific low-temperature sampled-label flow derived here should be compared carefully with that literature before any novelty claim.

[Takahashi, The Role of Pseudo-labels in Self-training Linear Classifiers on High-dimensional Gaussian Mixture Data](https://arxiv.org/abs/2205.07739) is further relevant work on high-dimensional self-training dynamics and directional improvement. Its assumptions and algorithm are not identified with this flow.
