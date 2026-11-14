# Model 4 feasibility result

14 November 2026, UK time. Canonical report: http://127.0.0.1:8095/model4_feasibility.html

## Finding

A qualified construction works: a separate stochastic encoder and hierarchical decoder, local parity constraints learned from finite demonstrations, and offline importance-weighted variational updates. The final sampler uses 80% ancestral trajectories and 20% uniform random observations followed by recognition; its full proposal probability corrects the weights. Both trainable pathways learn during Watching and practice. The learned evidence memory is retained.

The objective KL(pi_X(x) q_phi(h|x) || p_theta(x,h)) upper-bounds KL(pi_X || p_theta). The resulting total-variation bound is TV(p_star,p_theta) <= TV(p_star,pi_X)+sqrt(F/2). This does not promise monotonic true population KL or global optimization. All network gradients use local score eligibilities and scalar third factors. Supplied parity detectors remain a strong inductive bias.

Prospective final run: seeds 300–331, sixteen demonstrations, one hundred initial Watching updates and two thousand continuation updates, batch 128, learning rate 0.5. Mean population KL starts at 2.192 nats. Final values (mean ± standard error):

| Method | Population KL |
|---|---:|
| Offline practice with exploration | 0.156 ± 0.036 |
| Pure ancestral practice | 0.714 ± 0.304 |
| Continued Watching, stored examples | 0.375 ± 0.084 |
| Fresh iid teacher examples | 0.164 ± 0.037 |
| End-to-end stochastic layerwise reconstruction | 1.191 ± 0.104 |

Paired bootstrap 95% intervals for comparator KL minus practice KL: stored Watching [0.077, 0.403], fresh data [-0.066, 0.080], reconstruction [0.822, 1.259]. The practice learner wins 29/32 seeds against stored Watching and 32/32 against reconstruction; no advantage over fresh data is established. Normalized root information reaches 0.991; branch representations generally become informative earlier.

## Limits and preserved failures

- Pure reverse joint KL, symmetric joint KL with finite-batch sampling, and pure ancestral forward sampling each exhibited coverage failures. All seeds and learning curves are retained.
- The final model changes the original setup: overlapping local constraints, supplied parity features, a retained learned evidence memory, and exploratory proposals. It is not a homogeneous equal-expressivity deterministic MLP pair.
- The stochastic layerwise reconstruction baseline is fully differentiated but is not the literal deterministic root-only autoencoder.
- In exact expectation the update is variational fitting of the learned constraint distribution. There is no proof of an intrinsic own-sample advantage; any correctly weighted proposal has the same expected direction.
- Coupled initialization is not necessary in the secondary controls. The two-level example does not establish generic depth scaling or transformer ICL.
- The comparison is sample-presentation matched, not FLOP matched. Observational baselines use an exact per-observation control variate; sampled practice uses a fixed baseline.
- Fixed finite-budget runs are not claimed to be converged. The median terminal log-objective/log-step slope is approximately -1.17.

## Artifacts and validation

Durable results, complete parameters, source snapshots and SHA-256 manifest:

[/nfs/gatsbystor/jbauer/generative-learning/model4_feasibility/manifest.json](/nfs/gatsbystor/jbauer/generative-learning/model4_feasibility/manifest.json)

Simulation source:

[/nfs/nhome/live/jbauer/generative-learning/experiments/model4_feasibility.py](/nfs/nhome/live/jbauer/generative-learning/experiments/model4_feasibility.py)

The report has interactive learning curves, coverage and weight inspection, per-level latent information, sample-size/coupling controls, and a sample-budget allocation heatmap. Analytical gradients agree with finite differences below 1e-8; normalization, KL decomposition and population-label isolation checks pass. Browser validation checks KaTeX, chart controls and offline rendering. No GPU, LLM API call, or README edit was used.
