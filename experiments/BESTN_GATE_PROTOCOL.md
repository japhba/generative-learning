# A single Boltzmann gate between two existing output modes

This model form was discovered by inspecting seed 86001. Its numerical parameters use that seed's untrained baseline, initial training-forecast bank, and gradient forecast saved before its first update. No time constant was fitted to its learning curve.

The first quantitative forecast was frozen in `results/bestn_b2k/gate_prediction/prospective.json` at12:48:21BST on7September2026. Seed86003 current-source checkpoints through8updates already existed; only subsequent checkpoints are prospective. The frozen-source condition had no completed checkpoint. The file records these cutoffs. Earlier or discovery-seed agreement must not be labeled independent confirmation.

## Observable mode

The primary mode indicator H is the presence of at least one `=` character in a full generated response. It indicates an explicit equation, not guaranteed valid reasoning. It never selects training data. Correctness feedback remains the fixed final-answer verifier. Multiline output and completion length above32tokens are fixed sensitivity diagnostics; they gave similar discovery results.

In seed86001's untrained evaluation bank, equation responses occurred58.59%of the time and were32.89%correct; responses without an equation were0%correct. At24current-source updates, those values were91.67%,32.67%,and0%. This motivates a gate-probability explanation rather than assuming that arithmetic competence improved inside the worked-output mode.

## Normalized probabilistic learner

Use a Bernoulli gate with logit z and fixed normalized experts Q_0,Q_1 on disjoint output supports:

```
P_z(H=1)=lambda=sigmoid(z)
P_z(Y,H)=P_z(H)*Q_H(Y)
P(success)=b+(a-b)*lambda
```

The corresponding energy is minus H*z minus log Q_H(Y), with partition function1+exp(z). This is a one-weight conditional Boltzmann model. The Bernoulli/logistic factor is the same factor appearing in a binary-hidden RBM; the long-trajectory experts are a reduction of the LLM, not a claim that the entire LLM is an RBM.

With binary correctness feedback and uniform maximum-reward ties,

```
E[H_selected] = lambda + c_N(p)*(a-b)*lambda*(1-lambda).
```

For N2,c_N=1. If a>b, current-source likelihood distillation keeps increasing the gate until lambda approaches1. Frozen-source distillation instead approaches its initial selected-mode frequency. Thus this toy predicts different attainable performance even though the experts are unchanged.

## Calibration, weight geometry, and predictions

Training and evaluation can have different initial gate logits and component success estimates; a shared logit displacement shifts both. Let k=eta*||v||²/128, where v is the hypothesized gradient of the gate logit in LLM weight space. The scalar update is

```
z_next = z + k*(mean_selected_H - lambda_train(z)).
```

Calibrate k from the already-saved initial four-update success-gradient forecast Gamma:

```
k = (Gamma/4) / [(a_eval-b_eval)*lambda_eval0*(1-lambda_eval0)
                *(a_train-b_train)*lambda_train0*(1-lambda_train0)].
```

This gives k=1.38923454 and a gate-logit gradient norm of1210.67054 for the discovery initialization. Both were fixed before the future confirmation checkpoints. Population trajectories use the exact expected scalar SGD update;4096finite-gate replicas use batch16and sampled selected-mode counts. The stored intervals initially represent SGD variation only, not all parameter or forecast uncertainty.

The model also predicts a weight displacement z/||v|| along its gate direction. Comparing that number with the LLM's projection onto its initial success-gradient direction tests an additional approximation: fixed experts imply these directions align. Approximate constancy of conditional accuracy alone does not prove this gradient identity in a real LLM. Report failures of this approximation as well as successes.

## Primary prospective assessment

Compare the frozen numeric predictions with seed86003's future checkpoints beyond the recorded cutoffs, at actual candidate budgets. Evaluate total sampled accuracy, equation frequency, conditional accuracy in each mode, and the weight projection. Show current and frozen sources together. Include the uncertainty in fresh generation and avoid treating a finite set of noisy endpoints as an exponent estimate.

This is a prospective test of a reduced mechanism, introduced after a discovery run. It does not establish an intrinsic generation advantage over all recognition methods, an improvement without feedback, or a universal scaling exponent. The final interpretation must incorporate the independent full-training seeds, the earlier exact best-N identity, and all failed precision/likelihood checks.

## Analytic asymptotics and the normative objective

For one homogeneous gate with N=2, the population continuous-time equation is `dλ/dt = k(a−b) λ²(1−λ)²`. If `a>b`, `1−λ` eventually decays as `1/[k(a−b)t]`, so excess error relative to the gate's ceiling has a 1/t asymptotic. With a frozen source, `dλ/dt = k λ(1−λ)(λ_selected_initial−λ)`, giving exponential local convergence to an interior, lower ceiling. These equations describe the toy approximation, not an estimated LLM scaling exponent. Different train/evaluation offsets change the constants while retaining these local asymptotics.

More generally the exact best-N score identity is the gradient of the increasing utility `J_N(p) = sum_{j=1}^{N−1} [1−(1−p)^j]/j`. Its derivative is `c_N(p)`. N=2 therefore optimizes success probability in expectation; larger N shapes that utility and puts greater relative gradient weight on lower-success prompts. This gives a defined population objective for the update, without claiming that it maximizes the model's joint data likelihood.
