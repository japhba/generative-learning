# r5s: sample efficiency with a learning frozen-source control

Started 7 September 2026 after s4g. The target is a reproducible reduction in candidate samples or learner compute needed to reach a given error. A common toy/LLM exponent, or even a difference in fitted exponents, is not required.

## Contrast

Keep the s4g four-entry digit-counting task, Qwen2.5-1.5B-Instruct, full FP32 weights, temperature 1, unrestricted first-token sampling, plain SGD, and two candidates per question. Both arms now discard groups without a verified correct candidate. Choose uniformly among correct candidates otherwise. The current arm generates with its student's current weights; the frozen arm generates with the initial pretrained weights while its student continues updating.

Both arms therefore train only on correct sampled tokens and can approach perfect accuracy. This removes the old frozen teacher's wrong-label floor. Normalize each update by its number of accepted examples, rather than by its original batch size, so a higher acceptance rate does not directly multiply the mean gradient. Compute all 32 training rows with a zero loss weight for rejected rows, keeping the main forward/backward workload matched. Record accepted examples as an additional budget axis.

This is feedback-guided self-generation, related to rejection-sampling fine-tuning. It is not learning without external validation and does not isolate generation versus recognition. With a unique one-token answer, the mechanism can only involve which inputs supply successful training examples, their frequency, and sampling noise; there are no alternative successful reasoning traces.

## Pilot and confirmation

An exploratory pilot uses seed 99501, both sources, and the same fixed learning-rate grid: 1.4e-6, 2.8e-6, 5.6e-6. These are one half, one, and twice the approximate median s4g rate. The pilot horizon is 1024 updates. For each source, select the rate with the lowest area under log monitoring-error versus log candidate budget over updates 128 through 1024. A stopped or failed run remains reported and is ineligible for complete-window selection. Report every rate, including matched-rate comparisons.

After the pilot, freeze the selected rates and the fresh confirmation protocol before training. Use three independent training/sampling seeds, both methods, and a horizon of 4096 updates unless stationarity or a broken trajectory stops a run. No restart of stationary runs. If the pilot fails to show an advantage, report that result and use its measured mechanism to choose any subsequent redesign explicitly; do not suppress the unsuccessful comparison.

Each stage has a common, stratified monitoring cohort of 1023 unique questions (341 per answer) and an independent final cohort of 4095 unique questions (1365 per answer), disjoint by literal input ID. Training draws choose the answer count uniformly, then sample an input uniformly conditional on avoiding both cohorts. Monitoring and final cohorts use separate seeds. Training retains incidental repeats. The final cohort is evaluated only after rates and training trajectories are fixed. It is not used for rate selection, stopping, or gradient calibration. Splitting exact inputs does not establish semantic or compositional out-of-distribution generalization.

## Primary measurements

Report full curves, all seeds, sampled error from exact full-vocabulary probabilities, and greedy error. Primary comparisons are error ratios at updates 512, 1024, 2048 and 4096, plus candidate budgets required to reach errors 0.10, 0.05, 0.02, 0.01, 0.005 and 0.002 when both curves cross. Interpolate in log budget and log error within observed adjacent checkpoints; no extrapolation. Report non-crossings as censored. Bootstrap training seeds and whole final-test questions, preserving source/checkpoint pairing. Report exponent window sensitivity as descriptive evidence rather than the success criterion.

Record generated candidates, prompts presented, accepted examples, unique inputs, prompt/completion tokens, and learner seconds. The fixed one-token architecture makes candidate and theoretical forward/backward budgets proportional. Report curve comparisons against accepted examples too: an advantage only per candidate would support improved generation yield, while an advantage at matched accepted counts also involves the accepted input distribution or noise.

## Mechanistic reduction

Let p_theta(x) be correct-answer probability and q_s(x)=1-(1-p_s(x))^2 the chance that a group supplies a correct example. Ignoring the negligible probability of an entirely rejected batch, the expected accepted-mean update is

    d theta / d t = E_x[q_s(x) grad log p_theta(x)] / E_x[q_s(x)].

For current sampling q_s=q_theta; for frozen sampling q_s=q_0. Thus both have correct-label fixed points, but current sampling changes the weights assigned to inputs as the model improves. A homogeneous scalar gate has identical population dynamics in the two arms after accepted-mean normalization. Heterogeneity or finite-batch noise is required for an advantage. A multi-gate normalized Bernoulli model tests this mechanism without forcing an exponent match.

## Execution and provenance

Use institutional gpu_lowp H100, inherited Slurm GPU allocation, authenticated cached Hugging Face weights. No paid API calls or rental. Full resumable state at initialization, 128, 512, 1024, 2048, 4096 and any early stop; write locally and asynchronously move to durable storage while preserving original paths as symlinks. Preserve all raw candidates, checkpoints, logs and failed attempts. Do not push code or commit Slurm scripts.

At each scheduled checkpoint from update 64 onward, fit the latest four positive checkpoints in log error versus log updates and accepted-target NLL versus log updates. Stop when either absolute slope is below 0.02. Abort on nonfinite values, a greater than 0.15 accuracy drop, or valid-digit mass below 0.95. Retry transient infrastructure failures without changing the learning protocol. All absolute times are UK local time.

Related primary work: [RAFT](https://arxiv.org/abs/2304.06767) and [STaR](https://arxiv.org/abs/2203.14465). These motivate filtered self-training, not the outcome of this new comparison.
