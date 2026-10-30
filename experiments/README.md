# Learning from supplied examples and generated solutions

This project studies whether self-generated computation can make existing knowledge
more useful, through a linear probabilistic learner, frozen-model in-context learning,
and full-weight LLM fine-tuning. Earlier experiments separately studied example
coverage, retention, and corrective feedback. None of these acquisition comparisons
by itself isolates producing an answer from recognizing one.

The [strict source-only search](MATCHED_SOURCE_RESULTS.md) holds rehearsal questions,
their order, initial examples, learner, and final inference settings fixed. It varies
only who supplies the answers or full worked solutions, without judging generated
samples for inclusion. The search spans 32 synthetic settings, 16 full-solution
tasks, and 23 BBH classification configurations, with fresh confirmation cohorts
for selected exploratory gains. The [protocol](MATCHED_SOURCE_PROTOCOL.md) records
selection rules, source invariants, dataset checks, and the shared $3 spending cap.
This is the comparison that addresses the user's source-only constraint.

The earlier [unjudged inverse-compilation experiment](COMPILED_INVERSE_RESULTS.md)
uses a known arithmetic encoder and frequent-first forward rehearsal. Qwen3.5 Flash
and Gemini 3.1 Flash Lite show steeper finite-range error declines than IID teacher
examples, with matched-curriculum teacher controls reproducing the gain. Longer
direct reasoning is competitive; Qwen's late fit window loses the slope advantage.
The [theory](COMPILED_INVERSE_THEORY.md) derives exact coverage curves and separates
finite-size effects from asymptotic exponents. The [protocol](COMPILED_INVERSE_PROTOCOL.md)
preserves calibration changes before the 32-encoder main and eight-encoder second-model
confirmation. All generated samples remain unjudged during inference.
Its curriculum advantage does not satisfy the later source-only constraint and
should not be interpreted as evidence for that narrower hypothesis.

The [Self-ICL experiment](SELF_ICL_RESULTS.md) reproduces the paper's
GPT-3.5 Instruct direct-prompting setting on 368 held-out BBH questions and extends
it to CoT and demonstration-count curves. No generated examples are judged or
filtered. The direct gain is small and uncertain; a larger CoT exact-match gain
shrinks under an option-text audit. Fresh Boolean/navigation curves show no stable
power law. The [protocol](SELF_ICL_PROTOCOL.md) records all operational changes;
`self_icl.py`, `analyze_self_icl.py`, and `audit_self_icl.py` reproduce the results.

The latest [finite-sample toy](FINITE_SAMPLE_PROTOCOL.md) uses equal sampling and
training temperatures, unfiltered self-generated labels, and plain SGD. Its
[results](FINITE_SAMPLE_RESULTS.md) distinguish worse classification from improved
sign-invariant recovery of a latent signal axis. The latter diagnostic was discovered
after the original sweep and then confirmed with independent samples; matched
Gaussian noise reproduces it. The [report](../reports/finite-sample-self-training.html)
includes the full grid, fresh confirmations, negative outcomes, and data downloads.
Rebuild the existing analyses with `analyze_finite_sample.py`, then run
`build_finite_sample_report.py` to combine all three completed phases. Raw weights and
protocols have verified persistent copies recorded in each result's `archive.json`.

The LLM elicitation experiment fixes the question distribution and supplies no
oracle corrections. Qwen2.5-1.5B-Instruct first solves new arithmetic questions with
short answers, reasoning, or internally selected consensus. Training then compares
practice from the frozen starting model with practice from the current weights.
The [elicitation protocol](ELICITATION_PROTOCOL.md),
[answer-only consolidation protocol](ELICITATION_CONSOLIDATION_PROTOCOL.md), and
[full-trace follow-up protocol](ELICITATION_TRACES_PROTOCOL.md) record the designs.
The [fresh-data confirmation](ELICITATION_TRACE_CONFIRMATION_PROTOCOL.md) repeats
the unchanged full-trace main contrast with three new optimization seeds.
The [linear Gaussian toy](LINEAR_ELICITATION_TOY.md) gives an explicit weight-distance
mechanism, while keeping its generator fixed. It is a possibility proof for closing
an inference gap, not a matched model of on-policy refresh or the arithmetic task.
The [posterior-refresh extension](POSTERIOR_REFRESH_TOY.md) adds an explicit current
versus frozen initialization contrast with analytically predicted weight dynamics.
It establishes that mechanism within a linear Gaussian model, without claiming a
matched account of the LLM arithmetic results.
The [results](ELICITATION_RESULTS.md) and
[standalone report](../reports/elicitation-pilot.html) retain the failed recipes and
distinguish elicitation, consolidation, and a benefit from refreshing the sampler.

The direct bridge across all three settings is the matched Zipf dictionary experiment.
Qwen3-0.6B-Base and SmolLM2-135M learn the same original 32 dictionaries and example
streams used in ICL, with separate training-only calibration worlds. Fixed IID,
ranked and historical query streams are compared with actual current-model query
generation followed by oracle feedback. The ordinary four-label supervised loss is
shared across all acquisition conditions; evaluation contains no labeled examples.
The [prospective protocol](ZIPF_FINETUNING_PROTOCOL.md),
[results](ZIPF_FINETUNING_RESULTS.md), and
[standalone report](../reports/zipf-finetuning.html) document the experiment, retained
calibration failures, same-data parameter checks and compute accounting. Rebuild its
analysis with `analyze_zipf_finetuning.py` and its report with
`build_zipf_finetuning_report.py`. The analyzer's `--family qwen` or `--family smollm`
option audits and plots a single completed family.

The earlier modular-planning fine-tuning experiment compares teacher solutions with
the model's own successful rollouts.

The fine-tuning setup, three-arm control, compute accounting, and reproduction commands
are in [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md). Its results are in
[TRAINING_RESULTS.md](TRAINING_RESULTS.md); the separate cross-model ICL results are in
[REPLICATION_RESULTS.md](REPLICATION_RESULTS.md).

The follow-up in [AUGMENTATION_PROTOCOL.md](AUGMENTATION_PROTOCOL.md) preserves every
teacher example and adds verified model-generated supplements, with a control that
repeats teacher targets at exactly the same supplemental positions. Its runner is
`train_augmentation.py` and its independent analysis is `analyze_augmentation.py`.
Completed results and the additional fresh-identity validation are reported in
[AUGMENTATION_RESULTS.md](AUGMENTATION_RESULTS.md).

The [linear weight-coverage toy](WEIGHT_COVERAGE_TOY.md) supplies explicit teacher
weights, Bayesian inference, row-wise SGD and a categorical Boltzmann connection.
Its exact sample curves reproduce the coverage exponents. At a smaller learning
rate, ranked acquisition improves classification while IID repetition fits weight
magnitudes better, showing why prediction error and weight convergence need not
share a learning curve. `simulate_weight_coverage.py` verifies this with dense
weight matrices across 128 paired dictionaries.

The further investigation tests corrective feedback on learner-visited states.
[FEEDBACK_PROTOCOL.md](FEEDBACK_PROTOCOL.md) records a modular-planning pilot whose
models remained constant-action policies. [TRACKING_PROTOCOL.md](TRACKING_PROTOCOL.md)
defines a simpler control task, its retained calibration pilots, and the frozen
Qwen3/SmolLM2 replication. [LINEAR_FEEDBACK.md](LINEAR_FEEDBACK.md) gives the exact
two-weight Gaussian mechanism: demonstrations can omit a corrective input direction,
while exploration supplies informative labeled states. A process-noise sensitivity
run shows how weak excitation replaces non-identifiability with slower learning.
The completed two-model, three-seed results, behavioral probes and exact endpoint
checks are in [FEEDBACK_INVESTIGATION_RESULTS.md](FEEDBACK_INVESTIGATION_RESULTS.md).
The [standalone HTML report](../reports/generation-and-learning.html) embeds the
figures, an interactive checkpoint explorer, and downloadable methods and data.
Rebuild it from the completed artifacts with
`"$VENV_LOCAL/generative-learning/bin/python" experiments/build_html_report.py`.

## Zipf association experiment

The main experiment uses a new random dictionary in each world: 65,536 integer keys,
each assigned an independent uniform label from A/B/C/D. Test queries have probability
proportional to rank^-1.5. The learner is told these frequencies, but never the secret
world seed or the label-generation algorithm. Each practice example reveals a correct
label. The budgets are 1, 2, 4, 8, 16, 32, 64, and 128 label observations.

- `iid_study` receives IID examples from the query distribution, including repeats.
- `active_generate` chooses key IDs, guesses their labels, and receives verified
  feedback. Its guesses and feedback remain in context.
- `active_yoked` receives exactly the active learner's chosen correct observations,
  with no query-selection or guessing transcript.

This tests active exploration and example coverage. It does **not** isolate generation
from query selection, additional computation, or longer context. The yoked condition
checks whether chosen examples explain an advantage. Study is observation of labeled
examples, not an explicit recognition/judgment task. Generation with oracle feedback
also differs from unverified self-generated examples. No weights are updated.

With perfect recall, conditional Bayes error is 0.75 times the unobserved probability
mass. Exact expected IID error is `0.75 * sum_r p_r * (1-p_r)^n`; querying ranks 1..n
gives `0.75 * sum_{r>n} p_r`. For an infinite Zipf distribution with exponent s>1,
these scale as `n^(-(s-1)/s)` and `n^(-(s-1))`, respectively. At s=1.5 the asymptotic
exponents are 1/3 and 1/2. Over the actual finite-vocabulary budget range 8–128, the
exact reference curves have effective exponents approximately 0.327 and 0.502.
These are task predictions, separate from measured model exponents. The general
connection between heavy-tailed frequencies and learning curves is developed in
[Hutter's Learning Curve Theory](https://arxiv.org/abs/2102.04074).

The completed cross-model replication uses the same task, seeds 4000–4031,
conditions, and primary 8–128 fitting window for Qwen3-30B-A3B-Instruct-2507
through Nebius and Mistral Small 3.2 24B through DeepInfra. Models were selected
using family diversity, structured-output support, and price; Qwen remained included
despite its irregular pilot. The selection record and live endpoint pricing snapshots
were saved before requests. Both models completed all 32 dictionaries, with no new
provider failures. Replication cost $0.191703; the conservative total including earlier
experiments is $0.37499892, below the unchanged cumulative $0.80 ceiling.

Mistral's effective exponents are 0.373 for IID study, 0.428 for active generation,
and 0.445 for yoked study. Its active exponent closely matches Gemini's 0.428:
paired difference -0.0006, 95% interval [-0.0300, 0.0295]. Qwen's active fit is
less reliable: beta 0.531, interval [0.355, 0.759], with substantial curvature
and inconsistent selection of common keys. The full report, including unchanged
fit-window sensitivity checks and same-example controls, is in
[REPLICATION_RESULTS.md](REPLICATION_RESULTS.md).

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/openrouter_zipf.py run --output results/zipf_replication --worlds 32 --seed 4000 --models qwen/qwen3-30b-a3b-instruct-2507 mistralai/mistral-small-3.2-24b-instruct --concurrency 8
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_zipf.py results/zipf_replication
"$VENV_LOCAL/generative-learning/bin/python" experiments/compare_zipf_models.py results/zipf_main results/zipf_replication --output results/zipf_replication
```

The cross-model comparison bootstraps complete dictionaries with the same sampled
world indices across models. It reports paired exponent differences from Gemini;
intervals that cover zero do not by themselves establish exponent equivalence.
Intervals are pointwise and unadjusted for multiple contrasts. Reusing the same
dictionaries supports paired model comparisons; these are additional architectures,
not another independent draw of task worlds.

Evaluation branches never enter practice histories. Each world uses the same 40
stratified test items at all budgets and across conditions: ranks1–8 are enumerated,
and eight remaining rank strata each contribute four draws with importance weights.
Batch predictions can depend on the other items in the batch; this estimates the
specified stratified batch protocol, not necessarily isolated-query performance.
The analyzer verifies paired test sets and exact weighted-error accounting, and
resamples entire paired worlds for uncertainty intervals.

Four pilot worlds per model (seeds3000–3003) compared Qwen3-30B-A3B-Instruct-2507
with Gemini2.5FlashLite. Gemini consistently selected the optimal ranked examples;
Qwen had an irregular selection failure. The unchanged Gemini protocol was then
frozen for 32 new worlds (seeds4000–4031). A selection record is saved before requests
in the main output directory. The primary analysis estimates effective sample
exponents over budgets8–128, with paired active-minus-IID exponent differences.
Windows4–64 and16–128 are sensitivity checks, not independent confirmations.

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/openrouter_zipf.py run --output results/zipf_pilot --worlds 4 --seed 3000
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_zipf.py results/zipf_pilot
"$VENV_LOCAL/generative-learning/bin/python" experiments/openrouter_zipf.py run --output results/zipf_main --worlds 32 --seed 4000 --models google/gemini-2.5-flash-lite --concurrency 8
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_zipf.py results/zipf_main
```

The finite fit range spans only a factor16. Report it as a finite-range scaling
pattern, with recall/coverage diagnostics, rather than an established asymptotic law.
Prompt/completion tokens are API-reported counts per logical world, excluding shared-
cache amortization; they are not measured hardware compute. The
yoked learner's own token cost excludes the cost of obtaining its curriculum from the
active learner, so it is an information control rather than a complete competing system.

## Initial composition pilot

Each episode independently samples two uniform random permutations of four states.
The learner sees 0, 2, 4, 6, or 8 primitive transitions and answers 24 new programs
containing two, three, or four operations. An exact simulator grades all answers.
Enumerating all 576 possible machines gives a Bayesian MAP reference. Six balanced
examples identify the machine completely; eight display its complete transition table.

The two conditions receive identical correct examples in the same order:

- `study`: worked examples only.
- `generate_feedback`: attempt each two-question practice batch, then receive the
  correct answers. Attempts are retained, including errors.

Evaluation calls are separate branches: their questions and outputs never enter the
practice history. The zero-example baseline is shared. Comparisons are paired across
worlds, and confidence intervals resample whole worlds, not individual test questions.
The generation condition also has a longer transcript and additional computation;
this pilot cannot separate those effects or establish a human generation effect.
Operators and task worlds differ between episodes, so pretrained factual knowledge
cannot supply their particular transition tables. The composition algorithm is given.

## Environment

Keep `OPENROUTER_API_KEY` in `~/.env` (or a gitignored project `.env`). Never commit it.
The code loads these files with `python-dotenv`. From an interactive shell:

```bash
UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync
```

## Pilot and main run

The pilot uses three worlds and two models (78 maximum inference requests):

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/openrouter_learning.py prepare
"$VENV_LOCAL/generative-learning/bin/python" experiments/openrouter_learning.py run
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_openrouter.py results/openrouter_pilot
```

After inspecting pilot difficulty, a separate 24-world run can use fresh seeds:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/openrouter_learning.py run --output results/openrouter_main --worlds 24 --seed 2000
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_openrouter.py results/openrouter_main
```

Both runs share a persistent $0.80 experiment spending ceiling, leaving $0.20 below
the user's $1 maximum. The ledger reserves a conservative upper cost before each
request, counts in-flight reservations, and replaces them with reported costs after
responses. Requests have fixed output limits, provider price ceilings, no provider
fallbacks, and no automatic application-level paid retries. Models and providers are
pinned in the manifest. Cached successful requests are reused on restart; unresolved
reservations and failed requests stop reruns until inspected. Do not delete the ledger
to bypass the cumulative spending ceiling.

Costs are OpenRouter-reported inference charges, not hardware FLOPs or any separate
credit-purchase fee. Input-token counts measure observed context size; total practice
input/output tokens and practice costs are also recorded. API credentials are never
saved in experiment artifacts.

## Offline verification

```bash
"$VENV_LOCAL/generative-learning/bin/python" -m unittest discover -s experiments -p 'test_*.py' -v
```

Checks cover exact labels, Bayesian identifiability, budget reservations, restart/cache
behaviour, and isolation of held-out evaluation from practice. They make no paid calls.

The analysis creates per-model accuracy, paired-difference, and context-token PNGs,
plus machine-readable summary statistics. Plots use Matplotlib's default element and
font sizes. Run data and the persistent request ledger are gitignored under `results/`.

## API references

- [OpenRouter pricing ceilings](https://openrouter.ai/docs/guides/routing/provider-selection#max-price)
- [Key status and usage](https://openrouter.ai/docs/api/reference/limits)
- [Reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
