# Cross-model replication of Zipf ICL curves

Mistral Small 3.2 24B reproduces Gemini Flash Lite's active-generation exponent
closely on the frozen task. Qwen3 30B A3B does not give a similarly clean result:
its example-selection policy is inconsistent, producing curved learning curves and
much wider uncertainty. All selected outcomes are included.

## Protocol and spending

Before requests, we selected Qwen3-30B-A3B-Instruct-2507 through Nebius and
Mistral-Small-3.2-24B-Instruct through DeepInfra, using family diversity, price,
and strict structured-output support. Qwen remained selected despite its irregular
four-world pilot. Live endpoint prices were $0.10/$0.30 and $0.075/$0.20 per million
input/output tokens, respectively. The shared $0.80 spending ceiling and request
price caps were unchanged.

Both models use the original 32 dictionaries (seeds 4000–4031), task prompts,
three conditions, eight budgets, and stratified test sets. The primary fit remains
`error = A * n^(-beta)` over 8, 16, 32, 64, and 128 observations. Intervals resample
whole dictionaries with 1,500 paired bootstrap draws. Cross-model comparisons use
the same bootstrap world indices across models. These are finite-range fits,
not asymptotic laws or weight-training results. Intervals are pointwise and are
not adjusted for multiple model, condition, or fitting-window comparisons. Reusing
the same dictionaries supports paired model comparisons, but is not an independent
replication over newly sampled task worlds.

The run completed all 64 model/dictionary combinations and 1,536 condition/budget
observations. It made 2,024 successful paid requests, with no new provider failures
or unresolved charges. Replication cost was $0.191703: $0.1130505 for Qwen and
$0.0786525 for Mistral. Including previous pilots and confirmation, settled charges
are $0.3732745; the conservative total is $0.37499892 after retaining the original
$0.00172442 reservation for an earlier interrupted request. No paid requests remain
running. The audit verifies complete pairing, saved code hashes, and actual providers.

## Mistral result

- IID study: beta 0.373, 95% interval [0.334, 0.409].
- Active generation: beta 0.428, interval [0.402, 0.456].
- Yoked study of the same chosen examples: beta 0.445, interval [0.411, 0.482].
- Active minus IID: 0.055, paired interval [0.011, 0.101].
- Yoked minus IID: 0.072, paired interval [0.019, 0.128].
- Active minus yoked: -0.017, paired interval [-0.061, 0.023].

The active exponent differs from Gemini's by -0.00064, paired interval
[-0.0300, 0.0295]. This is a close finite-range replication; absence of a detected
difference is not a formal equivalence test. The IID exponent is slightly higher
than Gemini's: difference 0.0210, paired interval [0.0013, 0.0420]. Thus the data
do not establish one exactly shared exponent for every condition and architecture.

At 128 examples, mean error is 13.42% IID, 6.25% active, and 5.46% yoked.
Active minus IID error is -7.17 percentage points, interval [-8.11, -6.13].
Active is 0.78 points worse than yoked study, interval [0.16, 1.45]. The full
selection/guess transcript provides no improvement over the same correct labels
at this checkpoint. This comparison changes transcript content, length, and message
structure together, so it does not identify guessed-label interference alone.
Mistral chooses the optimal ranked curriculum at every budget in all 32 worlds.
The same-example control supports coverage as the explanation of the sample advantage.

Sensitivity remains similar to Gemini: active-minus-IID beta is 0.111 over 4–64
but 0.033 over 16–128, with the latter interval [-0.031, 0.099] including zero.
At 128 examples, active deployment uses 8,569 tokens versus 1,115 for IID study
and 1,214 for yoked study. Yoked token counts exclude the active learner's cost
of obtaining its curriculum. The sample benefit does not demonstrate compute efficiency.

## Qwen result

- IID study: beta 0.323, 95% interval [0.279, 0.365].
- Active generation: beta 0.531, interval [0.355, 0.759].
- Yoked study: beta 0.457, interval [0.270, 0.718].
- Active minus IID: 0.208, paired interval [0.033, 0.439].

Although the fitted active exponent is larger, the active curve fits a power law
poorly: log-error RMSE is 0.142, compared with 0.034 for Mistral and 0.024 for
Gemini. Qwen starts its curriculum at key 42 or 123 in 19 of 32 worlds, despite
being told that low-numbered keys are more frequent. Only 13 worlds have the
optimal ranked set at the final budget. Different acquisition histories generate
a mixture of curves, so quoting its fitted slope alone would be misleading. The
most frequent key is first observed at budgets 1, 16, 32, or 64 in 13, 3, 4, or 9
worlds, respectively; three worlds never observe it within the tested budget.

At 128 examples, errors are 13.83% IID, 12.30% active, and 13.21% yoked.
Active minus IID error is -1.53 points, interval [-7.41, 5.79], so the endpoint
benefit is inconclusive. A steep fitted slope can coexist with similar endpoint
error because the active condition starts much worse and catches up late. This
does not reproduce the clean Gemini/Mistral coverage mechanism reliably.

## Artifacts

The frozen selection record, endpoint snapshots, raw paired results, analysis,
cross-model contrasts, and audit are saved under `results/zipf_replication`.
The three model learning curves and cross-model exponent figure were visually
inspected. Original Gemini artifacts remain unchanged.

- `results/zipf_replication/cross_model_exponents.png`
- `results/zipf_replication/mistralai_mistral-small-3.2-24b-instruct_learning_curve.png`
- `results/zipf_replication/qwen_qwen3-30b-a3b-instruct-2507_learning_curve.png`
