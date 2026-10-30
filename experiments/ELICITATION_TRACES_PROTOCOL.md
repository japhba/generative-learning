# Preserving full self-generated reasoning

This is an exploratory follow-up to the failed answer-only consolidation recipe.
The source and data hashes in `results/elicitation_traces/protocol.json` were frozen
before the run. Correct answers are computed only in the offline analysis, after all
three arms finish.

The learner is Qwen/Qwen2.5-1.5B-Instruct at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. There are 96 new arithmetic practice
questions (seed 69100), a disjoint set of 144 evaluation questions (seed 69200), and
one optimization seed (69000). Each cohort has equal numbers of programs containing
4, 6, and 8 operations. Four shuffled passes give 384 practice presentations; this is
not 384 independent questions or 384 external labels.

The two main arms obtain four reasoning samples per presentation, either from the
initial frozen model or from the current weights. Among completed, parsed answers,
plurality selects an answer, breaking ties by first occurrence. The target is the
actual full token sequence of the first trace yielding that answer. Incorrect traces
are retained. An entirely invalid group contributes no supervised tokens and still
counts toward practice and generation cost. The frozen arm obtains independent
samples for every presentation, including repeated questions. Its precomputed
samples enter learning only according to the shared causal presentation schedule.

The third arm generates one unfiltered current-model reasoning trace and trains on
its actual tokens, including incomplete attempts. It is a mechanism control with
fewer rollouts, not an equal-compute comparison. Sampling is at temperature 1 without
top-k or nucleus truncation; each reasoning attempt permits 512 new tokens.

All arms teach the original reasoning prompt and update all weights using FP32
weights and AdamW state, BF16 autocast, learning rate 1e-6, no weight decay, gradient
clipping at 1, and one update per batch of 16. Completion-token NLL is summed and
divided by the fixed value `128 * batch_size`. Random completion length therefore
does not appear in the loss denominator. The score-identity argument concerns the
expected raw gradient at the sampling parameters; clipping, AdamW, finite precision,
and finite samples can still produce parameter drift in self-imitation.

Relative to answer-only consolidation, the target representation, training prompt,
learning rate, update count, and normalization change together. This tests a revised
recipe. It does not identify trace preservation as the sole cause of a difference.

The primary outcome is final held-out reasoning accuracy: current versus frozen
source, and each versus their common baseline. Greedy reasoning permits 512 tokens;
short-answer evaluation permits 96. Both are measured at 0, 16, 64, 192, and 384
presentations. Completion rate and output length are recorded. Evaluation questions
and responses never enter practice or updates. Practice accuracy and target quality
are offline diagnostics. The evaluator requires a completed answer.

`analyze_elicitation_traces.py` verifies source/data hashes, the exact sampled tokens
used in every target, causal source paths, presentation/rollout/update counts,
supervised-token counts, matching initial teacher draws, and bitwise-identical
baseline token sequences. Paired bootstrap intervals resample evaluation questions
within depth strata. With only one optimization seed, these intervals exclude
training-seed uncertainty. No scaling exponent is fitted.

The run uses one H100 through the institutional `gpu_lowp` partition. All final
models are written to the configured fast cache and copied to the shared cache with
SHA-256 verification before the job exits. No paid API calls or Vast rentals are used.

Reproduction, using the environment created by
`UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync --extra training --locked`:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/train_elicitation_traces.py prepare --output results/elicitation_traces_reproduction
"$VENV_LOCAL/generative-learning/bin/python" experiments/train_elicitation_traces.py run --output results/elicitation_traces_reproduction
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_elicitation_traces.py --output results/elicitation_traces_reproduction
```

Run inside an allowed GPU allocation with an unused configured fast-cache destination;
the runner deliberately refuses to overwrite existing model directories.
