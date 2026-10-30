# Unjudged generation as compilation of a known inverse task

Completed 10 September 2026, Europe/London.

Generated forward examples can give a steeper finite-range error curve than IID
teacher examples in this controlled task. The mechanism is a prescribed curriculum
that covers frequent latent states, together with forward computation that makes
their inverse answers easy to retrieve. A teacher supplying the same curriculum
matches the result. This experiment therefore supports a tractable computational
mechanism, while leaving an intrinsic benefit of generating rather than receiving
identical content unestablished.

The main Qwen3.5 Flash run used 32 fresh random encoders and 1,024 stratified query
instances. Gemini 3.1 Flash Lite used eight of the same encoders and their 256 query
instances. Both planned cohorts completed; no encoder or condition was dropped.
The [prospective protocol and amendments](COMPILED_INVERSE_PROTOCOL.md) distinguish
calibration from confirmation. The [theory note](COMPILED_INVERSE_THEORY.md) derives
the exact coverage references and their limiting exponents.

## Task and comparison

A world provides a known one-to-one encoder

$$x=(az+b)\bmod 257,\qquad z\in\{1,\ldots,256\},\qquad p(z)\propto z^{-1.5}.$$

Each world independently draws `a` from 17–239 and `b` from 0–256. The learner sees
the formula, its coefficients, and the prior. Its final task is to return `z` from
`x`. This is a deliberately constrained inference task: ordinary evaluation asks
for answer lines without reasoning, while forward generation permits a calculation
trace. An unrestricted arithmetic solver is perfect at zero examples.

For ranked rehearsal, a program schedules labels `1,...,N`, and the same LLM computes
each forward code in an independent call. The latent label is known by construction.
Parsed outputs become `Code x -> label z` pairs; unparsed outputs are retained in
full. Nothing is checked, corrected, selected for correctness, or rerolled before
evaluation. Test queries never enter the generation requests. The schedule is
designed by the experimenter; the LLM does not discover it.

Four example conditions use nested prefixes of length N:

| Condition | Latent labels | Code source |
|---|---|---|
| Generated, ranked | 1 through N | LLM forward computation |
| Teacher, IID | IID draws from the query prior, with replacement | Exact encoder |
| Teacher, same ranks | 1 through N | Exact encoder |
| Generated, IID | Exactly the teacher's IID latent draws | LLM forward computation |

All conditions present examples in the same canonical format. Identical successful
API requests share a cache. Consequently, equality between correctly generated
examples and their matched teacher control is a content check by construction,
not an independent repeated evaluation of identical prompts.

Queries are sampled independently of rehearsal but may revisit rehearsed latent
states. That overlap is the intended cache mechanism. This does not measure
generalization exclusively to unseen states or learning an unknown encoder from
a few initial examples. There are no weight updates.

## Error curves

Values are percent error estimated under the specified query prior and stratified
batch protocol. They are not unweighted fractions of the 32 test items per world.

| N | Qwen generated, ranked | Qwen teacher, IID | Gemini generated, ranked | Gemini teacher, IID |
|---:|---:|---:|---:|---:|
| 0 | 99.82 | 99.82 | 100.03 | 100.03 |
| 1 | 60.51 | 85.38 | 58.92 | 87.29 |
| 2 | 44.88 | 72.67 | 45.67 | 70.95 |
| 4 | 32.84 | 58.69 | 32.91 | 55.82 |
| 8 | 23.02 | 42.21 | 22.98 | 45.29 |
| 16 | 15.25 | 34.30 | 17.78 | 31.80 |
| 32 | 11.21 | 27.26 | 10.95 | 28.14 |
| 64 | 7.38 | 19.29 | 4.87 | 18.98 |
| 128 | 6.63 | 14.39 | 3.30 | 15.33 |

At the primary N=64 comparison, ranked generation minus IID teacher error is
**−11.91 percentage points**, paired 95% bootstrap interval **[−14.17, −8.76]**,
for Qwen. Gemini's corresponding difference is **−14.11 points [−16.17, −11.98]**.

Teacher-ranked is exactly equal to generated-ranked through N=64 on Qwen and at
all budgets on Gemini. Qwen's N=128 teacher-ranked error is 6.40%, versus 6.63%
for generated-ranked. Generated-IID equals teacher-IID at every budget in both
models. These controls attribute the principal difference to curriculum coverage,
not to the origin of correct code-label pairs.

![Qwen error curves](../reports/compiled-inverse-qwen_flash.png)

![Gemini error curves](../reports/compiled-inverse-gemini31.png)

Overlapping curves reflect identical contexts. The direct-reasoning points on the
Qwen plot use the generation allowance associated with the indicated N; those
calls receive no examples. Bands are pointwise 95% encoder-bootstrap intervals.

## Finite-range slopes and exact reference

Fit `log(error) = intercept - beta * log(N)` to the mean error curve. The fixed
primary window is N=8,16,32,64; the other two windows are declared sensitivity
checks. Intervals resample entire paired encoders 6,000 times, preserving all
conditions, budgets, IID curricula, and query samples within each encoder.

| Model | Window | Generated beta | IID teacher beta | Paired beta difference, 95% interval |
|---|---|---:|---:|---|
| Qwen, 32 encoders | 4–32 | 0.525 | 0.362 | +0.163 [0.069, 0.248] |
| Qwen, 32 encoders | **8–64** | **0.537** | **0.372** | **+0.165 [0.016, 0.296]** |
| Qwen, 32 encoders | 16–128 | 0.421 | 0.426 | −0.005 [−0.209, 0.209] |
| Gemini, 8 encoders | 4–32 | 0.513 | 0.347 | +0.166 [−0.029, 0.356] |
| Gemini, 8 encoders | **8–64** | **0.741** | **0.394** | **+0.347 [0.239, 0.453]** |
| Gemini, 8 encoders | 16–128 | 0.846 | 0.373 | +0.473 [0.352, 0.626] |

The Qwen primary beta intervals are [0.387, 0.662] for generated-ranked and
[0.337, 0.410] for IID teacher. Gemini's are [0.682, 0.806] and [0.312, 0.508].
These are finite-window statistics, not established asymptotic scaling laws.
Qwen's advantage in slope disappears in the late window. Gemini also has visible
curvature: its consecutive local generated slopes over 8–64 are 0.370, 0.700,
and 1.168. Confidence intervals are pointwise and unadjusted; the secondary
model uses matched worlds, not a new independent draw of encoders.

For a perfect lookup recognizer that fails whenever an input is absent from its
memory, exact expected errors are

$$R_{\rm ranked}(N)=\sum_{z>N}p_z,\qquad
\mathbb E R_{\rm IID}(N)=\sum_z p_z(1-p_z)^N.$$

For an infinite Zipf distribution with exponent s, the asymptotic exponents are
`s-1` and `(s-1)/s`: here 1/2 and 1/3. With the experiment's finite support, the
exact reference slopes over 8–64 are instead **0.723 and 0.378**. Approaching
complete finite coverage steepens the ranked reference. Qwen falls below that
reference as lookup errors accumulate; it should not be described as accurately
recovering an infinite-support exponent simply because its fitted beta is near 1/2.

The cache reference is deliberately restricted. It does not use the prior to
guess unobserved labels. Always guessing label 1 already gives exact error 59.80%,
well below the answer-only zero-example LLM error. This trivial baseline makes the
zero-example weakness explicit; the N>=2 ranked results improve beyond it. An
unrestricted decoder has zero error without any rehearsal, so no information is
created by generation.

![Qwen sample scaling and inference work](../reports/compiled-inverse-qwen_flash-scaling.png)

![Gemini sample scaling and inference work](../reports/compiled-inverse-gemini31-scaling.png)

## Computation and response diagnostics

Qwen's direct-reasoning control receives no examples and permits a visible
calculation trace. Its extra output allowance equals the actual prompt-plus-output
tokens spent on ranked generation, divided over four evaluation batches. It is
an allowance comparison; actual token work can be much smaller, and tokens are
not measured hardware FLOPs. Both generation and inverse calls have hidden
reasoning disabled.

| Qwen condition | N / allowance index | Error | Mean total API tokens per encoder |
|---|---:|---:|---:|
| Generated, ranked | 32 | 11.21% | 13,434 |
| Teacher, IID | 32 | 27.26% | 2,390 |
| Direct reasoning | 32 | 9.43% | 8,820 |
| Generated, ranked | 128 | 6.63% | 65,751 |
| Teacher, IID | 128 | 14.39% | 6,193 |
| Direct reasoning | 128 | 5.68% | 8,696 |

The generated-minus-direct differences are +1.77 points [−5.53, 8.23] at the
N=32 allowance and +0.95 points [−4.26, 5.90] at N=128. There is no demonstrated
advantage over direct reasoning. These controls also use 32 inverse queries per
encoder; a much longer query stream could amortize generation differently.
Teacher-token costs exclude obtaining the teacher examples. Logical generation
work counts repeated IID requests even when the cache avoids a new paid call.

All **2,048/2,048** ranked forward codes through N=64 were correct for Qwen. At
N=128, **4,090/4,096** were correct; six calculations reached their 1,024-token
limit without a final CODE marker. Their complete unfinished calculations remain
in context. Gemini produced **1,024/1,024** correct ranked codes. The correctness
checks are offline diagnostics after evaluation, never a gate on rehearsal.

Qwen's conditional error on queries whose correct codes were present rose from
2.57% at N=64 to 4.69% at N=128. Thus unreliable retrieval and output handling,
in addition to six unfinished generations, contribute to its flattening curve.
The primary 8–64 example conditions had **no missing parsed inverse answers**.
At N=128, Qwen omitted/misassigned four query IDs in each ranked condition and
two answers in each IID condition. Full-response inspection found an ID-confusion
response, not a harmless numeric-format variant. Gemini had no missing parsed
inverse answers at any budget.

At the N=32 direct-reasoning allowance, nine evaluation calls reached the output
limit and 68 query answers lacked a parseable final line. A full-response spot
check contains correct intermediate answers before truncation; the reported
direct risk conservatively counts missing final lines as errors. No post-hoc
rescoring changes the primary curves. At the N=128 allowance, no direct call
reached the limit and all final answers parsed. Full selected responses and their
ground truths are retained in `spot_checks.json` without truncation.

## Sampling, calibration, and verification

Each encoder evaluates all labels 1–4, four uniformly sampled labels from each
of 5–8, 9–16, 17–32, 33–64, and 65–128, and eight from 129–256. Selection is
without replacement within each stratum. An item's weight is its true probability
divided by its inclusion probability. All 32 queries are shuffled into four fixed
batches of eight and reused across methods and budgets.

The weights are not self-normalized, so estimates can occasionally exceed 100%
slightly. With batch-sensitive prediction, this estimates the specified sampling
and batch procedure, not necessarily performance on isolated or IID-batched
queries. Resampling whole encoders respects the experiment's pairing; it does
not treat repeated evaluations of the same query as independent samples.

Earlier pilots were retained separately:

- `pilot`: compositional subset sums; four complete cases before interruption
  after output repetition and truncation. Partial requests remain cached.
- `pilot_fast`: 18 complete subset-sum cases across Qwen3.5 Flash, Gemini 3.1
  Flash Lite, and Qwen3.6 35B A3B. Fast inversion remained near floor even with
  correct partial tables.
- `pilot_individual`: three complete cases, then a provider-reported output-token
  overage triggered an old pre-save assertion. That one HTTP response body was
  not preserved; its conservative reservation remains. The revised client saves
  successful responses before accounting checks and records token overages.
- `compiled_pilot`: four modular encoders; a 256-token generation cap cut many
  CODE markers, and the original numeric parser rejected valid answer variants.
- `compiled_interface`: two encoders; a compact 512-token calculation interface
  parsed reliably but produced incorrect arithmetic.
- `compiled_calculation`: two encoders; restored calculation traces with a
  1,024-token cap gave 128/128 correct forward codes and motivated the frozen
  confirmation. These data are not pooled with the final cohorts.

The final prompt, parser, source hashes, seeds, budgets, primary fit window, and
secondary cohort were frozen after calibration. This is a prospective confirmation
of the final design, not a claim that the entire exploratory program was fixed
before its first pilot. Source snapshots and logs preserve the redesigns.

Offline verification passed for all 40 final cases and **8,202 unique referenced
API records**. It checks frozen source and request hashes, regenerated worlds,
all planned conditions, target-blind generation, unchanged erroneous content,
paired query identities and weights, exact scoring, reconstructed token allowances,
and spending reconciliation. Three modular-task tests cover inversion, numeric
parsing, and evaluation/generation isolation; three subset-task tests also passed.

## Models, cost, and reproduction

The primary endpoint is `qwen/qwen3.5-flash-02-23` through `alibaba`, with pinned
prices of $0.065 input / $0.26 output per million tokens. The secondary is
`google/gemini-3.1-flash-lite` through `google-ai-studio`, at $0.25 / $1.50.
Temperature is zero, top-p one, provider fallbacks are disabled, and price
ceilings are enforced. Provider catalog snapshots are retained. These are the
prices used for this run; see the official [Qwen routing page](https://openrouter.ai/qwen/qwen3.5-flash-02-23/providers)
and [pricing page](https://openrouter.ai/qwen/qwen3.5-flash-02-23/pricing).

All new pilots and final cohorts together recorded **$1.341926025** in settled
API charges plus **$0.083107030** of unresolved conservative reservations:
**$1.425033055** committed against the unchanged $3 cap. There were 9,329 settled
calls and 9,354 reserved attempts. Unique cached calls referenced by Qwen's final
cohort cost $0.681388500; Gemini's cost $0.418021750. These totals exclude the
earlier, separately budgeted Self-ICL experiment.

From the project root, using the existing environment outside NFS:

```bash
UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync
"$VENV_LOCAL/generative-learning/bin/python" experiments/compiled_inverse_icl.py --phase compiled_confirmation --worlds 32 --seed 202000 --budgets 0 1 2 4 8 16 32 64 128 --iid-generated --long-budgets 32 128 --world-concurrency 8 --concurrency 32
"$VENV_LOCAL/generative-learning/bin/python" experiments/compiled_inverse_icl.py --phase compiled_gemini --models gemini31 --worlds 8 --seed 202000 --budgets 0 1 2 4 8 16 32 64 128 --iid-generated --world-concurrency 4 --concurrency 24
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_compiled_inverse.py results/inverse_icl/compiled_confirmation --plot
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_compiled_inverse.py results/inverse_icl/compiled_gemini --plot
"$VENV_LOCAL/generative-learning/bin/python" experiments/verify_compiled_inverse.py results/inverse_icl compiled_confirmation compiled_gemini
"$VENV_LOCAL/generative-learning/bin/python" -m unittest discover -s experiments -p 'test_compiled_inverse.py' -v
```

The final runner requires `OPENROUTER_API_KEY` from `~/.env`. Completed cases and
successful requests are reused on resume; one writer holds the persistent ledger
lock. Analyses and verification make no paid calls. Raw data are linked at
`results/inverse_icl`; `archive.json` there records the persistent CEPH archive
and its verified SHA-256. The archive includes pilot outputs, full cached request
and response records, all final curves, code, environment lockfile, and protocols.
