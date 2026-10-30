# Coupled wake–sleep with controlled latent overlap

This is a new coupled-generator experiment, not the previous fixed Gaussian decoder or self-contrastive Gaussian study. All computation is local CPU; there are no API calls, rented GPUs, or correctness checkers.

## Model and initialization

Six independent fair binary causes h in {-1,+1} generate eight-dimensional observations x = W h + epsilon, epsilon ~ N(0, 0.7² I). Each target column has unit norm. The target Gram matrix is (1-rho) I + rho 11ᵀ, with a random rotation per seed. Increasing rho creates posterior interactions without changing the individual column norms or total signal energy. It also changes observable conditioning and identifiability: this is an overlap intervention, not a guarantee of a monotonic increase in every measure of difficulty.

The initial generator is 0.8 W_true plus Gaussian weight noise of standard deviation 0.15/sqrt(8). This deliberately models an already useful generator paired with an untrained recognizer. It is not learning from no prior information. Real examples expose only x; the training routines never receive their generating h. The initializer uses the teacher to specify a controlled experimental starting condition; no teacher parameters enter updates. Recognition weights and biases start at zero in all methods.

The primary recognizer is a single affine map followed by a 64-way softmax over all binary states. It can represent the exact posterior. No hidden neural-network layers are used. A secondary factorized linear recognizer tests the effect of insufficient posterior capacity.

## Why overlaps matter mechanistically

For fixed W:

    log p_W(h | x) = xᵀ W h / sigma² - hᵀ Wᵀ W h / (2 sigma²) + constant(x).

Off-diagonal weight overlaps produce pairwise posterior interactions. For orthogonal columns the posterior factorizes. For the categorical recognizer, exact recognition weights are R = W Hᵀ / sigma² and state biases b_s = -||W h_s||²/(2 sigma²), up to softmax gauge freedom. This makes the toy interpretable and supplies an exact computational oracle; an unrestricted algorithm could directly compile these recognition parameters. The experiment tests learning under specified local update rules, not an information-theoretic advantage over every possible algorithm.

The generator's negative-ELBO gradient is:

    grad_W = [ W E_data E_q(hhᵀ | x) - E_data x E_q(h | x)ᵀ ] / sigma².

At uniform recognition, E_q h = 0, so the data-dependent cross moment vanishes and the update shrinks W. Recognition learning restores the cross moment. Exact enumeration measures how far this update is from the exact marginal-likelihood gradient.

## Methods

- ELBO: analytic expectation over q for both generator and recognition updates. No Monte Carlo latent-gradient estimator is used.
- ELBO + current sleep: the same wake update plus a recognition cross-entropy update on freshly sampled (h,x) pairs from the current generator. The generator is detached during the sleep update. Gradients are summed at the same pre-update parameters.
- ELBO + frozen sleep: identical, but dreams come from the initial generator. This tests whether updating the dream source matters.
- Forward recognition on real data: the same approximate generator update, with recognition trained toward the exact current-model posterior on real x. This is an inclusive-KL control; it receives no true latent labels or teacher posterior. Enumeration is feasible only because the toy is small.
- Exact likelihood: use the exact posterior for generator learning, bypassing recognition. This diagnoses the inference bottleneck. Its unused recognition metrics are not interpreted.

## Budgets, optimization, and pilot selection

Primary x-axis: cumulative leading dense multiply/add FLOPs, with the costs of generating dreams and updating the recognizer included. This is an explicitly approximate compute model, not an exact hardware FLOP measurement. Elementwise operations and evaluation are excluded. Actual training CPU time, real samples, and dream samples are also recorded. Different methods can receive different numbers of real examples at equal compute. Secondary comparisons use matched real-example counts and measured CPU time.

Plain SGD uses separate generator and recognition learning rates. Exploratory constant-rate pilots exposed late instability in some recognition settings. The confirmation schedule therefore multiplies both rates by 1/sqrt(1 + cumulative_compute / (1024 ordinary wake updates)). Separate, equal-grid tuning for each method uses seeds 9202 and 9203, overlaps 0 and 0.8, and recognition rates {1,3,10,30}, generator rates {0.003,0.01,0.03}. Selection minimizes mean log marginal KL on a common log-spaced compute grid from 512 to 8192 equivalent wake updates. Stopped models retain their final value for this budget-allocation score. Tuning directories and runtime copies preserve all exploratory results.

The main protocol JSON is written before confirmation runs and specifies disjoint seeds, selected rates, overlaps, budgets, metrics, and comparison windows. The same selected rates are used across overlaps. Frozen sleep inherits the current-sleep rates for a controlled source intervention.

## Evaluation and interpretation

The likelihood sums all 64 latent states exactly. Expectations over real observations use an independent fixed test set per seed, paired across methods. The primary marginal KL estimator is the nonnegative integrand E_ptrue[exp(d)-1-d], d=log p_W(x)-log p_true(x); its expectation equals KL(p_true || p_W) because both densities normalize. We also record the ordinary excess NLL, density-ratio mean, and Monte Carlo standard error. Finite test integration is not described as an exact population expectation.

Other metrics: both directions of current-model posterior KL; KL from the true posterior to recognition; posterior total correlation; generator weight error/alignment after optimal signed permutation; exact-gradient bias/alignment; recognition linear-weight and pair-interaction errors. A lower recognition KL alone is insufficient evidence that generator learning improves.

Runs evaluate regularly. Once both marginal KL and reverse recognition KL have log-log slopes above -0.02 over the preceding factor of four in steps (after at least 2048 updates), training stops. NaN or clearly exploded trajectories abort. Individual runs take seconds and do not require long-run resume infrastructure. All completed records and final weights are retained.

Error ratios, paired-seed bootstrap intervals, and local slopes are descriptive finite-budget results. Fit windows are fixed before confirmation. Different fitted slopes will not be called different asymptotic power laws unless stable across windows and free of evident floors/crossovers. A benefit removed by exact forward recognition would identify an inference-objective mechanism, rather than an intrinsic information gain from synthetic samples.

## Post-confirmation checks

The selected recognition rates are 3 for ELBO, 10 for current/frozen sleep, and 30 for exact posterior teaching; the generator rate is 0.01 throughout. The primary budget is 8192 equivalent wake updates, with runs allowed up to 16384. Main seeds are 9300–9311 and overlaps are 0, 0.4, 0.8, 0.95. The factorized capacity control uses seeds 9300–9307 and its own equal-grid tuning.

After observing deterioration in the initially selected ELBO baseline, add a slower recognition rate of 1 and comparisons with shared rates 1, 3, and 10 at overlaps 0 and 0.8. These use the existing twelve paired seeds and are labeled follow-up robustness checks. The stable baseline is used for conservative learning-curve comparisons; it was not selected before the original confirmation.

Fresh evaluation replays the deterministic updates to the primary budget and uses 16,384 new observations per seed, with evaluation RNG offset 700000 instead of 300000. Evaluation RNGs are independent of both real-training and dream RNGs. Both the slower baseline and shared-rate-3 comparison are checked this way. The original finite-window slope intervals are retained; they are not changed to seek a preferred exponent result.
