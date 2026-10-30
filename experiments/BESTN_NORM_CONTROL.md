# Fixed-norm no-feedback extension

This is a separate, exploratory toy extension to the completed unconstrained finite-sample experiment. It does not change the b2k LLM learner or retroactively turn the earlier negative results positive.

The learner uses the same normalized Gaussian-visible/binary-hidden RBM: `p_w(x,h) ∝ exp(−||x||²/2 + h w·x)`, with `h` in `{-1,+1}`. Fresh inputs come from the real mixture with signal norm 1.5. Labels are sampled from the model's own conditional at temperature 1. There is no feedback, validation-based selection or temperature change in the update. The added inductive bias is explicit: after each ordinary score update, project the weights back to their initial unit norm.

The primary setting preserves the old dimension 8, initial overlap 0.3, batch 8, rate 0.1 and 512 updates. An exploratory 4,096-replica run was followed by 16,384 fresh replicas with the same parameters. A separately frozen 4,096-replica limitation case starts at overlap 0.8. Source hashes and protocols precede their respective simulations.

## Mechanism

For unit vector w, let `P = I − w wᵀ` and let `F = E_x[x xᵀ sech²(w·x)]` be the conditional score covariance per input. Before projection the score mean is zero. After projection, to second order,

```
E[Δw] = (eta²/B) b
b = −P F w − 0.5 tr(P F) w
Cov(Δw) = (eta²/B) P F P
```

For marginal cross-entropy, ignoring a w-independent constant,

```
L(w) = ||w||²/2 − E_x[log cosh(w·x)]
E[ΔL] = (eta²/B) [∇L·b + 0.5 tr(P F P ∇²L)] + higher-order terms.
```

The initial analytic coefficient is −0.647452 for self-sampled score noise and −0.110038 for trace-matched isotropic noise. An independent antithetic finite-difference audit gave −0.636154 ± 0.007220 standard error and −0.107257 ± 0.003302, respectively. Projection prevents norm growth from overwhelming the unsigned-axis benefit. It also means the projected weight update itself is no longer an exact martingale; there is no contradiction with the zero-mean score identity.

## Fresh confirmation

At initial overlap 0.3, self-sampling reduces true marginal cross-entropy by 0.114070 nats, 95% interval [−0.116391, −0.111749]. Covariance-matched Gaussian noise gives −0.111883 and isotropic noise −0.031034. The self-minus-isotropic difference is −0.083037 [−0.085844, −0.080229]. The self-minus-matched difference is −0.002188 [−0.005447, +0.001072].

Squared axis overlap rises from 0.09 to 0.2701. Classification error nevertheless rises from 32.64% to 46.45%, with 44.27% of replicas reversing sign. Sign is a nuisance symmetry for the unlabeled mixture density, but is essential for a labeled classifier.

At initial overlap 0.8, self-sampling instead worsens marginal cross-entropy by 0.216463 nats [0.211666, 0.221259]. Thus this is an improvement from a poor representation under a particular inductive bias, not a universally improving or consistent learner.

Fresh real inputs contain the useful information. Self-generated labels are one way to induce the data-dependent covariance; matched Gaussian noise reproduces the benefit. This is not evidence that an isolated model creates new information, nor that this no-feedback mechanism improves LLM accuracy. The b2k LLM confirmation remains a separate experiment with explicit correctness selection.
