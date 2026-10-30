# Finite-sample self-training: axis recovery and prediction error

Finite sampling produces a reproducible change in representation geometry in this toy. Sign-invariant alignment with the true latent axis improves and is reproduced by covariance-matched Gaussian noise. Signed overlap, mean classification accuracy, and marginal generative likelihood worsen. These outcomes must be kept separate.

## What was run

The learner is the one-vector Gaussian-visible/binary-hidden RBM conditional p(y|x)=exp(y w.x)/(2 cosh(w.x)). Inputs follow x=y_true*(1.5,0,...,0)+Normal(0,I), with balanced y_true. The true labels are discarded before training. Each batch supplies fresh, unfiltered self-sampled labels at temperature 1, followed by one plain SGD likelihood update at temperature 1. No clipping, normalization, reward, teacher validation, or selection is used.

A frozen sweep varied dimension (2,8), initial signed overlap (0.3,0.8), batch (1,8,64), and learning rate (0.025,0.1,0.4), with 512 updates and 512 replicas per setting. All 36 settings and all three arms were retained. The unchanged initial model is the zero-expected-gradient baseline.

The Gaussian control replaces each sampled label residual by a centered Gaussian of the same conditional variance while retaining the input vector. Its batch-average gradient covariance matches exactly at any given weights and input batch. The isotropic control preserves only the covariance trace. Both controls compute these quantities from their own current weights.

## Original endpoints and independent precision check

The original primary endpoints were signed overlap and exact population classification error. Of 36 self-sampling settings, 35 had worse mean error, 1 had better mean error, 33 showed worsening under simultaneous 95% intervals, and 0 showed improvement. These intervals use Bonferroni correction across the 36 endpoints.

An independent 16,384-replica confirmation of d=8, initial overlap 0.3, norm 1, batch 8, eta 0.1, 512 updates found self-sampled error 41.496% from a 32.636% baseline. The self-minus-matched-Gaussian difference was -0.106 pp [-0.618, +0.406]. Its entire 95% interval was inside the predeclared +/-1 pp practical-equivalence margin.

A second 16,384-replica check targeted the largest squared-norm martingale diagnostic in the original grid. The original isotropic residual z=-3.866 became z=-2.058. It remains a modest pointwise fluctuation; it is inside an interval adjusted for the six arm/group norm checks. The conditional covariance identity is exact and was independently checked numerically. The original outlier is retained.

In that d=2, initial-overlap=0.8 check, 56.65% of runs improved and median error fell from 11.507% to 10.121%, while mean error rose to 21.101%. A typical-run improvement can coexist with a worse expectation because damaging trajectories have larger effects.

## Axis diagnostic: discovery followed by fresh confirmation

Inspecting those completed runs suggested a distinction between the signal axis and its sign. The diagnostic rho² was introduced post hoc. Its confirmation protocol was then fixed before generating another 16,384 independent replicas of the original primary setting, using disjoint random seeds. It is not being relabeled as an original endpoint.

The metric rho²=(w.mu)²/(||w||²||mu||²) measures alignment with the one-dimensional signal subspace and ignores its sign. Both mu and -mu define the same unlabeled Gaussian mixture. Random isotropic directions have expected rho²=1/d=0.125 here, so an increase from the initial 0.09 alone would not establish useful structure recovery. The isotropic comparison is essential.

| Fresh axis-confirmation arm | Mean rho² | Signed overlap | Error | Reversed sign |
|---|---:|---:|---:|---:|
| Self-generated labels | 0.23855 | 0.16660 | 41.472% | 35.62% |
| Covariance-matched Gaussian | 0.23620 | 0.16799 | 41.404% | 35.61% |
| Isotropic Gaussian | 0.12489 | 0.16234 | 41.162% | 30.08% |

The self-minus-isotropic difference in mean rho² was +0.11367, 95% paired interval [+0.10961, +0.11772]. The self-minus-matched-Gaussian difference was +0.00235 [-0.00242, +0.00711], entirely within the predeclared +/-0.01 axis-overlap margin.

This supports a covariance-driven effect on the latent signal axis. It does not establish that stochastic target tokens are uniquely necessary: the matched noise control reproduces the endpoint. Inputs still contain the useful mixture structure; the experiment does not generate both inputs and targets from an isolated incorrect model.

## Local mechanism

For a smooth metric f and a fresh self-sampled batch, the leading conditional drift is eta²/(2B) * trace(F(w) Hessian(f)(w)), despite E[delta w|w]=0. For rho², its Hessian is 2 grad(rho) grad(rho)^T + 2 rho Hessian(rho). The initial coefficients of eta²/B in the primary setting are:

- Signed overlap: -0.914164.
- Classification error: 0.782901.
- Squared axis overlap, self-sampled/matched: 1.032342.
- Squared axis overlap, isotropic: 0.189237.

The opposite signs for signed and squared overlap explain why a zero-score argument alone is insufficient to characterize the representation. These are local small-step predictions, not a closed-form global learning curve. Norm also grows; neither true parameter recovery nor calibrated density learning follows from axis alignment.

## Generative likelihood check

As an additional post-hoc diagnostic, evaluate the marginal p_w(x), integrating out y. Up to a w-independent constant, its population cross-entropy is ||w||²/2 - E[log cosh(w.x)]. The following endpoint changes are in nats per input; positive is worse:

- Self-generated labels: +0.36243 [+0.35695, +0.36791].
- Covariance-matched Gaussian: +0.36737 [+0.36184, +0.37290].
- Isotropic Gaussian: +0.59662 [+0.58923, +0.60401].

Thus the axis effect also does not imply improved marginal likelihood. Gaussian quadrature used 160 nodes; maximum discrepancy from 100 nodes was 4.43e-05 nats over all evaluated replicas.

## Scalar sanity control and scope

The separately predeclared model p_theta(y)=Normal(theta²,1) supplies an exact nonlinear example where expected error can initially improve under zero-mean self-sampled scores. Its median error worsens and its expected error later grows. It is an existence check, not evidence for an RBM or LLM benefit.

The central empirical finding is a finite-sample improvement in sign-invariant axis recovery, accompanied by worse predictions and density fit. It is a candidate component of a self-learning mechanism, not a successful end-to-end learner. No power-law exponent was fitted, and no conclusion about human or LLM learning rates is claimed.

## Reproducibility

The original simulation and five numerical tests were frozen before the sweep; the score, covariance controls, metric Hessians, quadrature, and scalar moment identities passed. The squared-axis Hessian was also checked against independent finite differences. Confirmations use disjoint case seed ranges (1000–1063 and 2000–2031), while the original sweep uses 0–35. Raw weights, accumulated gradient variance, code snapshots, protocols, and hashes are retained; archive.json records persistent copies under CACHE_DIR.

The three simulation phases took 18.67 seconds of measured wall time using 8 local CPU workers, excluding analysis and reporting. They produced 202,752 classifier trajectories in total. No paid API, rented compute, GPU, or new LLM training was used. Plot element sizes and fonts use matplotlib defaults.
