# A linear toy with an actual sampler-refresh contrast

The original wake–sleep toy holds its generator fixed and teaches recognition from
exact internal `(x,z)` pairs. This follow-up instead draws a posterior particle
initialized by the recognition weights. Updating that initialization changes the
distribution of the next internally produced target. No external labels are used.

Keep exactly the original 8×8 generative matrix A and observation noise sigma. Let

```
z ~ N(0,I),   x | z ~ N(Az, sigma² I)
C = A Aᵀ + sigma² I
H = I + Aᵀ A / sigma²
B* = Aᵀ C⁻¹,   H B* = Aᵀ / sigma²
p(z | x) = N(B*x, H⁻¹).
```

The learner's current recognition mean is Bx. Its latent rollout is

```
z0 = Bx + epsilon0,                    epsilon0 ~ N(0,I)
z1 = (I-delta H) z0 + delta Aᵀx/sigma² + epsilon1
epsilon1 ~ N(0, 2 delta I - delta² H).
```

For `0 < delta < 2 / lambda_max(H)`, the innovation covariance is positive definite.
Writing `R = I-delta H`, its covariance is exactly
`H⁻¹ - R H⁻¹ Rᵀ`. Together with `H B* = Aᵀ/sigma²`, this verifies that the linear
Gaussian transition preserves the exact posterior distribution. A single step from
an imperfect initialization generally does not produce an exact posterior sample.
The transition uses A and its implied posterior precision, not an oracle latent or
the separately computed B*. It is equivalent to a noisy linear update using the
reconstruction residual and Gaussian prior.

Training fits Bx to the actual sampled z1 using one squared-error SGD update. Both
main arms receive the same internally generated x values and coupled Gaussian
random numbers, and take one posterior step per training pair. The latent originally
used to generate x is discarded before target construction.

- Current initialization uses the current B for z0.
- Frozen initialization always uses the initial B0=0 for z0.
- Raw self-imitation omits posterior refinement and fits its own z0.

The conditional mean of the refined target is

```
E[z1 | x,B] = R Bx + (I-R) B*x.
```

Consequently the expected population update in the current arm is

```
B_(t+1) - B* = (B_t - B*) - eta delta H (B_t - B*) C.
```

For the stable step sizes used here this approaches B*. The frozen arm approaches
the mean of its permanently shallow teacher,

```
B_frozen = delta Aᵀ / sigma² = (I-R) B*.
```

Raw same-policy self-imitation has zero expected raw gradient. Finite stochastic
updates can nevertheless diffuse its weights. Neither the population recursion nor
this contrast implies different asymptotic power-law exponents: these are geometric
mode dynamics and, for the frozen arm, a biased limiting predictor.

`simulate_posterior_refresh.py` froze its source, parameters, and predictions in
`results/posterior_refresh/protocol.json` before sampling. It uses the original A,
`delta = 0.8 / lambda_max(H)`, SGD step 0.08, batches of 256, 1,024 updates, and 32
independent repetitions (seed 703000). The exact population recurrences accompany
the stochastic curves. Checks verify posterior invariance, the posterior normal
equations, stability, and identical first-step targets for the two main arms.

Mean final prediction error is 2.1024 with current initialization versus 5.3060 with
frozen initialization. The optimal error is 1.8422; the frozen population limit is
5.2922. Normalized weight overlap is 0.9300 versus 0.2969. All learners started at
error 8 and zero recognition weights. These comparisons use 262,144 internal
training pairs per repetition and no newly observed facts.

The computation asymmetry is explicit: the teacher takes a posterior-refinement
step that the direct recognition mean does not take. Thus this is a concrete model
of repeated elicitation and amortization, not a counterexample to the zero-score
identity. It models refreshing the initialization of posterior sampling, not an
autoregressive RL policy. It still does not establish a mechanistic equivalence to
the LLM's arithmetic traces or predict their learning-rate exponents.

```bash
OPENBLAS_NUM_THREADS=1 "$VENV_LOCAL/generative-learning/bin/python" experiments/simulate_posterior_refresh.py --output results/posterior_refresh_reproduction
```
