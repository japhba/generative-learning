# Best-of-N selection, weight dynamics, and prospective LLM predictions

Research batch: b2k. Started 2026-09-07 (Europe/London). The active objective is to find a setting that combines an interpretable mechanism, a probabilistic learner, predictive value for real LLM learning, and a benefit from generation. Correctness-based selection is now explicitly allowed.

## Requirements and current evidence

- Probabilistic grounding: conditional Boltzmann/softmax policies, including the one-vector Gaussian-visible/binary-hidden RBM. The selected distribution and its likelihood objective can be written exactly.
- Mechanistic interpretation: derive actual weight updates and normalized overlaps. Generic parameter-gradient overlaps describe transfer between prompts in an LLM.
- Predictiveness: state the expected effect of N, initial success probability, and source refresh before running the corresponding LLM updates. Validate learning and held-out performance, not merely the binomial probability of selecting a correct answer.
- Generative benefit: compare current-policy sampling with a frozen proposal policy at matched prompts, candidate counts, selection rules, and update budgets. Report costs of sampling and feedback separately. An external correct-answer SFT reference is a stronger-information baseline, not something generation is presumed to beat.

The preceding finite-sample experiments found improved unsigned signal-axis recovery but worse classification and density fit. They do not satisfy the present objective. Existing Qwen full-trace consensus training supports an on-policy source benefit, but its learning curve was not predicted prospectively from a mechanistic model. Neither finding completes this goal.

## Exact selection identity

For one fixed prompt, let p be the current probability of a successful full trajectory, r(y) in {0,1} be correctness, and g(y)=gradient_theta log pi_theta(y). Draw N independent trajectories, select uniformly among those with maximal r, and fit the actual selected full trajectory. If all fail, select a failure; do not silently omit the group.

The selected success probability is q_N(p)=1-(1-p)^N. Uniform tie-breaking preserves the current conditional distribution inside each reward class. With ordinary likelihood ascent at the same sampling/training temperature,

```
E[g(selected)] = c_N(p) * gradient_theta p
c_N(p) = [q_N(p)-p] / [p*(1-p)]
       = sum_{k=0}^{N-2} (1-p)^k,  for N>=2.
```

N=1 has zero expected score. N=2 follows the gradient of success probability exactly. This identity applies to arbitrary autoregressive policies and their full trajectory scores; it does not require a two-action output space.

With one small plain-SGD step, expected improvement on that prompt is eta*c_N(p)*||gradient p||². A fixed normalization of the summed trajectory log likelihood rescales eta. Dividing by random trajectory length, clipping, or adaptive optimizer transforms changes the identity and must be modeled separately.

At equal candidate count, the first-order gain is proportional to c_N(p)/N. Thus N=4 is locally more efficient than N=2 when p<(3-sqrt(5))/2, approximately 0.382; the ordering reverses above that threshold. This is an idealized per-prompt prediction. Multiple prompts have interacting gradient directions, variable lengths have different costs, and finite SGD introduces noise. We must not apply the threshold blindly to an average success rate.

For prompts i and j, define K_ij=<gradient p_i, gradient p_j>. The first-order change of average held-out success depends on the overlaps between evaluation and training gradients, weighted by c_N(p_j). The required predictive bridge is to check this geometry or validate a justified reduction of it, instead of fitting learning-rate constants after seeing the curves.

## RBM reduction with an explicit overlap equation

Use x=y_true*mu+Normal(0,I), with y_true=+/-1 balanced, and pi_w(y|x)=exp(y*w.x)/(2*cosh(w.x)). Reward is agreement with the simulator's label. This introduces correctness feedback explicitly. In this binary reduction one verifier outcome reveals the class label; the toy does not model the information complexity of checking a long reasoning trajectory.

Let z=y_true*x, so z~Normal(mu,I), m=w.mu, q=||w||², and rho=m/(||mu||*sqrt(q)). Write u(s)=sigmoid(-2*s). The selected-label likelihood gradient has population flow

```
h_N(s) = 2 * [u(s) - u(s)^N]
dw/dt = E_z[z*h_N(w.z)] = A_N*mu + B_N*w
A_N = E_Z[h_N(m+sqrt(q)*Z)]
B_N = E_Z[h_N'(m+sqrt(q)*Z)]
d rho/dt = ||mu||/sqrt(q) * (1-rho^2) * A_N.
```

For N>1, A_N>0, so every nondegenerate interior overlap improves in population. Greedy classification error is Phi(-||mu||*rho). Norm/calibration are distinct from this reward-learning claim.

The same selected-target rule can be simulated with a frozen initial proposal. Its target sufficient statistic stays fixed, whereas current-policy selection refreshes it. Both variants receive identical feedback and input budgets.

## Execution sequence

1. Validate the exact selection-score identity and the RBM overlap dynamics numerically. Freeze predictions and run a finite-budget current-versus-frozen toy comparison, including N/candidate-budget dependence.
2. Calibrate arithmetic task difficulty on Qwen2.5-1.5B-Instruct, using new calibration-only problems. Seek useful low- and high-success regimes with reliable completion; keep every failed task cell.
3. Freeze a separate LLM prediction/training protocol using fresh task cohorts. Use full sampled trajectories, equal temperatures, uniform reward ties, and fixed likelihood normalization. Measure gradient overlap and prospective changes before updates. Include actual held-out learning and source-refresh controls.
4. Confirm successful claims on independent seeds/cohorts and, if needed, a second model. Publish one stable local report with no-cache metadata and retain all failures.

The goal remains active until the LLM predictive requirement and the generative benefit are verified. A toy proof, a selected-target quality curve, or a successful post-hoc fit alone is insufficient.

## Relevant prior work

[RAFT: Reward rAnked FineTuning](https://arxiv.org/abs/2304.06767) studies sampling, ranking, and fine-tuning on selected outputs. [STaR](https://arxiv.org/abs/2203.14465) is a related reasoning-bootstrapping method, with additional answer-conditioned regeneration. These are related algorithms; this project must independently verify the proposed mechanism and predictions.
