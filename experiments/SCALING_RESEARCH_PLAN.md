# s4g: scaling of feedback-guided self-generation

Started 7 September 2026, after the completed b2k experiment. This is a new investigation, not a change to its frozen protocols or conclusions.

The question is whether adaptive self-generation has predictable learning curves, and whether a normalized probabilistic toy predicts the mechanism and scaling of a real LLM. We will not infer a common exponent merely because both settings use the same update identity.

## Shared identity and prospective theory

With two independent trajectories, binary correctness, uniform reward ties, and likelihood fitting of the selected trajectory, the expected current-policy score is exactly the success gradient. Under small-step population SGD, `d p / d t = ||grad p||^2`, where `t` includes learning rate and the fixed loss normalization. On a separate test distribution, replace the squared norm by the training/test gradient overlap.

If the excess error is `epsilon = p_infinity - p` and squared gradient norm is proportional to `epsilon^gamma`, then `epsilon` has power exponent `1/(gamma-1)` for `gamma>1`. This implication requires a stable regime, an identified ceiling, sufficiently small steps, and negligible or modeled stochastic corrections. A common update identity alone does not establish a common gamma.

Two analytically distinct predictions will be checked before new LLM learning curves:

- A homogeneous one-logit Boltzmann success gate has gamma=2 and excess-error exponent1.
- The existing Gaussian-visible/binary-hidden RBM, at large aligned weight norm, has gamma=3 and population excess-error exponent1/2. The reference error floor is the known Bayes error, not a fitted endpoint. Finite-batch angular diffusion may change the accessible regime and will be measured rather than ignored.

The existing frozen-source objective approaches a fixed selected-target distribution. We will distinguish its floor and transient convergence from an on-policy power law.

## Existing-data audit

Keep all six b2k trajectories. Compare shifted powers and exponentials with free nonnegative error floors; inspect exponent/floor ambiguity and prediction of withheld later checkpoints. Any fits to these already-inspected runs are explicitly retrospective. Do not describe them as newly confirmed scaling laws.

## First LLM follow-up

Use the already-calibrated easier arithmetic regime: four operations with addition/subtraction arguments at most29 and at most three multiplications by2..5. Keep Qwen2.5-1.5B-Instruct, full FP32 weights, temperature1, plain SGD, complete sampled trajectories, and best-of-two correctness selection including all-failure groups. This preserves the b2k algorithm and changes the task difficulty to allow more progress before the early plateau.

Pilot seed95001 is exploratory. Subsequent confirmation parameters, budgets, fitting windows, and held-out tasks must be frozen after the pilot and before confirmation outputs. Both current and frozen sources are required. All attempted settings remain reported. Measure training/evaluation gradient overlaps and actual weight displacements where feasible, and record candidate, token, and diagnostic costs separately.

Use geometric evaluation budgets and stop on stationarity of either loss or evaluation. Do not resume a stationary run just to extend a fitted line. Runs longer than30minutes require full resume state, local-first checkpoints and persistent asynchronous copies. Use only authorized institutional partitions or on-demand Vast within the existing$10 allowance; release rented compute immediately after use.

## Criteria for a scaling claim

A visually straight segment is insufficient. Report fitting-window sensitivity, exponent/floor uncertainty, and comparison with exponential and saturating alternatives. A prospective prediction must use only measurements made before its target checkpoints. A claim of different exponents needs evidence beyond different final accuracy or different fitted floors. Cross-model and ICL equivalence remain separate tests unless actually performed.

Primary literature checked at the start: [IsoCompute Playbook](https://arxiv.org/abs/2603.12151), [Faster WIND](https://arxiv.org/abs/2410.20727), and [Self-Distillation Amplifies Regularization in Hilbert Space](https://arxiv.org/abs/2002.05715). These provide related results, not verification of the new calculations or experiments here.

## Outcome of this bounded study

The population gate and RBM laws were derived and checked numerically; fresh finite-SGD replicas supported the1,1/2, and1/3 regimes and their unfitted amplitudes. The LLM's refreshed-source advantage replicated at matched budget, but its local exponents changed substantially and the fixed early RBM extrapolation failed. Both the scalar gate and a retrospective three-initial-mode reduction missed intermediate learning. Direct gradient-overlap measurements revealed changing class interactions. See SCALING_THEORY.md and the canonical served self-generation-scaling.html report for full results and limitations.

All three fresh current-source runs completed2048updates; all frozen controls halted under the predeclared stationarity rule. A full scratch disk interrupted the first confirmation attempts. Verified full checkpoints enabled identical replay, all failed attempts and partial checkpoint pieces were preserved, and checkpoint storage was changed to move scratch copies into separately retained durable copies. The learning protocol remained unchanged. No paid API calls or rented instances were used.
