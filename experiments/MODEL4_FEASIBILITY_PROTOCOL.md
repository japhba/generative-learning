# Model 4 feasibility: theory before simulation

14 November 2026, 12:00 UK time. This protocol was written before running this study. README.md and model_4.md are unchanged.

## Claim to investigate

A feasible relaxation of Model 4 retains teacher evidence in learned local constraint detectors. A separate stochastic encoder infers causes, and a stochastic hierarchical decoder generates observations. During fully offline practice the decoder receives a scalar score from the constraint detectors and encoder, while the encoder learns from generated latent trajectories. The constraint weights are learned from the same finite demonstrations; no true grammar or evaluation labels enter an update. This deliberately changes a pure RHM to overlapping constraints and adds a fixed evidence memory inside recognition. It is not yet the original homogeneous two-MLP proposal.

## Derivation

Let E_X(x) be a learned log-potential, pi_X(x)=exp(E_X(x))/Z_X, p_theta(x,h) the decoder joint, and q_phi(h|x) the encoder. The offline objective is

J = KL[p_theta(x,h) || pi_X(x) q_phi(h|x)]
  = KL[p_theta(x) || pi_X(x)] + E_p(x) KL[p_theta(h|x) || q_phi(h|x)].

For an own trajectory, A=E_X(x)+log q_phi(h|x)-log p_theta(x,h). Gradient descent on J gives decoder update E_p[(A-b) grad log p_theta] and encoder update E_p[grad log q_phi]. The normalizer Z_X is not needed. Each categorical conditional gives the local eligibility pre*(one_hot(post)-conditional_probability). A scalar third factor multiplies decoder eligibilities. The entropy term is essential; dropping it defines a different, potentially collapsing objective. The learned critic is an explicit source of target evidence, not a claim that dreams create information.

This decomposition is a joint-KL guarantee, not monotonic forward population KL. Parameter nonconvexity, posterior approximation, imperfect evidence and stochastic optimization can all prevent success. The decoder and encoder are both learned during Watching: use the finite-data variational likelihood bound, with local score-function gradients for the encoder and a wake gradient for the decoder. No dream is required during Watching.

## Small exact world

Draw independent uniform signs z,u,v. The six observations before an independently chosen fixed sign mask are (u,zu,z,v,zv,z). There are eight equally likely valid strings. The root z and two four-state branch causes (z,u), (z,v) form a two-level hierarchy. The decoder has a two-state root, two four-state categorical children, and three Bernoulli leaves per child. Recognition reverses these connections. Parameters are separate and randomly initialized with tunable correlation, independent of the teacher. Internal labels are evaluated permutation-invariantly.

The evidence memory tests every parity feature of order one through three. Features constant over the demonstrations are retained, including false constraints at small sample sizes. A deterministic GF(2) elimination retains an independent basis, favoring low order. Each retained constraint gets strength beta=0.5 log(2N+1), corresponding to a half-count smoothing convention for its observed sign. This is a learned finite-sample algebraic prior; parity features and the sparse-constraint inductive bias are supplied. It is a significant relaxation, to be stated prominently.

## Experiments and controls

First validate analytical gradients by finite differences and normalization by exact enumeration. Pilot seeds and any redesign are disclosed, then run fresh confirmation seeds. Measure exact forward population KL, validity, entropy/coverage, joint objective, posterior mismatch, and per-level latent information, throughout Watching and Practice. Evaluate all observation and latent states; true labels are diagnostics only.

Compare continued Watching on the fixed demonstrations, fresh iid teacher observations, offline variational practice, no-evidence practice, no-entropy practice, and a frozen encoder. Include an end-to-end reconstruction autoencoder and an exact-posterior likelihood reference if practicable. Check coupling and sample-size sensitivity. Exact expectation dynamics isolate whether the rule can work; sampled trajectories test the local implementation. Match and state update and sample budgets; do not imply exact expectations cost the same as a sampled update.

Success requires lower population KL and improved meaningful latent inference, not only increased validity or recognition of own codes. Negative results remain in the report. This pilot cannot establish a quality-matched own-vs-foreign advantage, generic depth scaling, or a faithful transformer ICL implementation.

## Artifacts

Source: experiments/model4_feasibility.py and experiments/model4_report.py. Durable simulation arrays, provenance and figures: /nfs/gatsbystor/jbauer/generative-learning/model4_feasibility/. Canonical report: reports/model4_feasibility.html, served at http://127.0.0.1:8095/model4_feasibility.html. No GPU or external LLM calls.

## Amendment after first pilot, 14 November 2026, UK time

The first three seeds (0–2, 500 Watching steps and 2,500 continuation steps, learning rate 0.5) showed that reverse joint KL can discard valid modes. Seed 0 worsened population forward KL while improving the fitted objective. This is a mechanism failure, not infrastructure noise; the pilot remains archived.

Before further simulations, change the main proposal to forward joint KL F=KL[pi_X(x)q_phi(h|x) || p_theta(x,h)]. Under own proposal p_theta(x,h), the unnormalized weight is w=exp(E_X(x)+log q_phi(h|x)-log p_theta(x,h)). Its expectation is Z_X, independent of all network weights. Weighted decoder eligibilities and weighted encoder eligibilities multiplied by (log p_theta-log q_phi) are the negative gradients of F. Exact expectations test the objective; self-normalized finite-batch importance estimates test feasibility and have finite-sample bias. This avoids computing p_theta(x), exact posterior inference, or the partition function in a sampled update. Full-support probabilities and importance-weight variance remain practical constraints.

Keep reverse-KL practice as a comparison, add unweighted self-wake, a frozen encoder for forward practice, and forward practice with evidence removed. Confirmation seeds begin at 100, disjoint from all pilot seeds. No claim that self-normalized sampling is unbiased or that the fitted forward bound guarantees monotonic population KL.

A further theoretically motivated candidate, specified before its simulations, is the equally weighted sum (F+J)/2 of the two joint KLs. The forward term penalizes missing modes; the reverse term supplies a direct sleep gradient for recognition. Its decoder and encoder gradients are the averages of the two already-derived updates. This retains a single scalar objective and bounds KL(pi_X || p_theta) by twice the objective. Sampled experiments must charge two independent batches per symmetric step (or reuse the same own batch); they must not silently compare double the samples to a baseline.

## Confirmation plan, 14 November 2026, UK time

The symmetric pilot used seeds 0–7, N=16, 100 Watching updates and 2,000 continuation updates, step size 0.5. Mean population KL was 0.171 for symmetric practice, 0.645 for continued Watching, 1.277 for the stochastic reconstruction baseline, and 0.119 for population Watching. This motivates testing the symmetric rule without further parameter tuning. It does not establish a win over fresh teacher data.

Confirmation: seeds 100–131, N=16, the same step sizes and horizons. Sensitivity: N in {4,8,16,32,64}, coupling in {0,0.8,1}; use seeds 100–115. Sampled confirmation: batch 128, one shared own batch for both terms of the symmetric gradient, identical number of trajectory samples for each baseline. Sampled decoder generation is ancestral, not a draw from an enumerated target. Diagnostics enumerate states but do not feed true labels into updates. The stochastic end-to-end autoencoder minimizes expected reconstruction of both the leaves and the branch codes; its full gradient passes through the stochastic encoder, unlike a frozen-recognition baseline. It is not a deterministic root-only bottleneck autoencoder.

## Final prospective check after sampled failure, 14 November 2026, UK time

Symmetric practice did not survive finite-batch sampling reliably: at batch 128 its mean population KL was 3.48, versus 0.244 for forward-only practice. Forward-only practice is therefore the retained candidate. Its update is fully offline, but in exact expectation equals variational learning on the learned constraint distribution; it is not proof of an intrinsic own-sample advantage. Evaluate final seeds 200–231 separately, with no additional hyperparameter tuning: N=16, Watching 100, continuation 2,000, learning rate 0.5, coupling 0.8. Run exact and batch-128 versions. Keep all previous pilots/confirmation, including failures, in the report.

Secondary checks on seeds 200–215: sample size and coupling for forward practice; a strictly local evidence variant retaining all constant short-parity detectors, without GF(2) basis selection. This removes a nonlocal preprocessing operation at the price of redundant factors and a different softened target. It is a declared architectural variant, not pooled into the primary estimates. A sampled allocation surface uses seeds 200–207, N=16 fixed, total steps {100,250,500,1000,2000}, own fractions {0,.25,.5,.75,.9,1}, batch 128 throughout. Each grid point starts from the same coupled random network; the finite evidence memory is built first. The horizontal sample budget counts network-training trajectories and excludes the common N-example evidence pass.

## Coverage redesign before the next simulations, 14 November 2026, UK time

Final sampled forward-only practice had a large failure in an independent seed (maximum population KL 6.52); its mean gain over continued Watching was not resolved relative to its paired standard error. The all-detector local variant also had rare severe failures despite a low median. These are proposal-support failures, not grounds to discard seeds. Preserve these results.

Test a changed sampling mechanism: proposal r(x,h)=0.8 p_theta(x,h)+0.2 u(x)q_phi(h|x), where u is uniform over all six-bit observations. The second component uses internal noise then recognition, not data or an external checker. The importance ratio becomes exp(E_X)q_phi/r; gradients of the target forward joint KL are unchanged in exact expectation. Crucially pi_X q_phi/r <= 64 max_x pi_X(x)/0.2, so every target-relevant trajectory has a bounded importance ratio. This is an explicit departure from exclusively ancestral own generation and is to be disclosed. It is not a claim of dimension-independent scalability.

Prospective validation uses new seeds 300–331, batch 128 and all other settings unchanged. Run both the basis memory and the all-detector local memory. Baselines receive identical numbers of trajectory presentations. No further optimization based on these seeds is planned. This experiment tests the theoretically motivated coverage repair, rather than repeating the failed sampler as-is.

The prospective all-detector exploration run gave population KL 0.156 (mean), versus 0.375 for stored-data Watching and 0.164 for fresh-data Watching. Retain this model without tuning. Secondary characterization uses the same seed family: frozen-encoder and missing-evidence ablations, and sample-size/coupling sensitivity (eight seeds per cell). These are mechanism checks, not a new independent confirmation. An allocation surface for this final rule is run on seeds 300–307; the two earlier allocation surfaces must not be conflated.

## Artifact chronology (UK timestamps)

These are completion timestamps recorded by the filesystem, rather than claimed exact amendment-writing times: reverse-KL pilot 12:02:51; forward pilot 12:05:34; symmetric pilot 12:06:58; sampled symmetric study 12:09:19; first reserved-seed sampled study 12:12:39. Later JSON files also embed their creation timestamps and source hashes. Each redesign above was specified before running its associated simulation. This is an in-session prospective protocol, not an externally registered study.
