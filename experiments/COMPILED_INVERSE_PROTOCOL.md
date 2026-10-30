# Unjudged compilation of a known encoder

Started 10 September 2026, Europe/London. Shares the inverse-inference study's
$3 cap and ledger, including all subset-sum pilots and infrastructure failures.

The subset-sum pilots did not provide a clean measurement: answer-only inverse
accuracy was at floor, forward arithmetic was noisy, and visible reasoning often
exhausted the allowance. A provider-reported completion-token count also exceeded
the requested allowance in one pilot call. The client now saves successful raw
responses before accounting checks, records reported overages, and continues to
bound actual charges by its conservative reservations. All prior artifacts remain.

## Sharper task and predicted mechanism

A world specifies an affine encoder x=(a*z+b) mod 257 for labels z=1,...,256.
The encoder is one-to-one. Query labels follow the known prior p(z) proportional
to z^-1.5. The LLM is given a and b. Forward multiplication/remainder calculation
is easier than inverse modular arithmetic without a reasoning trace.

Generated rehearsal supplies labels 1,...,N, asks the LLM to calculate each code
independently, and stores the resulting code-label pairs in context. The latent
label is known because it was chosen before generating the observation. No hard
inverse pseudo-label is guessed. Wrong codes are retained without correction,
filtering, resampling, or judging. Evaluation queries are hidden during generation.

Controls provide N correct IID examples from the query distribution, or N correct
examples with exactly the same ranked latent curriculum. All final examples have
the same canonical format. A longer-direct control omits examples and permits
reasoning with the total generation output allowance apportioned among evaluation
batches. Ordinary decoding is answer-only, with hidden reasoning disabled.

For an ideal recognizer that caches examples and otherwise fails, the exact
miss probabilities are sum_{z>N} p(z) for perfect ranked generation and
sum_z p(z)*(1-p(z))^N for IID teacher data. On an infinite Zipf distribution
with exponent s, these have exponents s-1 and (s-1)/s: here 1/2 and 1/3.
Finite support, a nonzero model baseline, incorrect generated codes, and inference
errors can change measured curves. Exact finite-support reference curves will be
reported alongside finite-window fits. The unrestricted ideal decoder has zero
error before rehearsal; this is computational compilation, not new information.

Unlike a held-out novel-state task, independently drawn evaluation queries can
match rehearsed states. That overlap is the intended amortization mechanism.
Teacher IID examples have the same opportunity for overlap. No test labels or
test-specific curriculum selection enter generation. The effect, if present,
should be reproducible by a teacher using the same useful curriculum; it is not
an intrinsic advantage of having authored identical text.

## Sampling and confirmation

For each random encoder, evaluate the first four labels and stratified uniform
draws from ranks 5–8, 9–16, 17–32, 33–64, 65–128, and 129–256: 32 queries total.
Each sampled item's weight is its true probability divided by its inclusion
probability. The resulting sum estimates risk without renormalizing sample
weights. Identical queries and ordering are reused across conditions. Confidence
intervals bootstrap whole encoders, not individual correlated query tokens.

The operational pilot uses four encoders (seeds 201000–201003), Qwen3.5 Flash,
and N=0,8,32,64. Check forward-code accuracy, parsing, inverse headroom, and cost.
Any model or task change is recorded before fresh confirmation; pilot outcomes
are not pooled into confirmation. Freeze the main dimensions, budgets, model,
seed range, and controls before dispatch. Preserve all curves, including null
or reversed effects. Any fitted exponent is explicitly a finite-range statistic.

## Interface calibration amendment

The first modular pilot revealed two interface issues: verbose forward calculations
often exhausted 256 tokens before their final CODE marker, and inverse outputs
used variants such as `id: label z` or `id: code x -> label z`. Spot checks found
correct calculations and lookup answers within these responses. The next pilot
requests exactly two short forward lines, allows 512 tokens, and accepts these
unambiguous numeric answer forms. Earlier primary scores and raw responses are
retained; the corrected parser is frozen before fresh confirmation.

A generated-IID control uses exactly the teacher's sampled latent labels, but the
LLM computes the codes itself. This separates forward generation from ranked
coverage. Repeated identical computations may reuse the persistent request cache;
logical generation work still counts each requested occurrence. The direct
reasoning control's extra output allowance is now based on the actual total
prompt-plus-completion tokens used for ranked generation, apportioned among
evaluation batches. Token allowances and observed tokens are distinct from FLOPs.

## Forward-computation calibration

The concise-output interface pilot parsed all answers but its generator made many
arithmetic errors for larger labels. Correct teacher-ranked examples followed
the predicted coverage curve. The earlier verbose generator's completed codes
were correct; most missing codes came from its 256-token cutoff, with correct
intermediate arithmetic visible in full-response checks. Therefore the next
`compiled_calculation` pilot restores the original calculation prompt with a
1,024-token allowance. No generated sample is judged, selected, or rerolled.
This changes available forward computation, not the curriculum or task. The
same numeric-answer parser is retained. All earlier pilots remain separate.

## Frozen confirmation

The calculation pilot completed with 128/128 correct ranked forward codes and
no generation truncation or answer parsing failures. It is only calibration.
Before new outcomes, freeze Qwen3.5 Flash on 32 fresh encoders, seeds
202000–202031, with N=0,1,2,4,8,16,32,64,128. Retain ranked generation,
IID teacher data, identical-ranked teacher data, and generated-IID data.
Run direct-reasoning allowance controls at N=32 and 128. Each encoder has the
same 32 stratified queries in four batches of eight for all conditions.

Primary inference is the generated-minus-IID-teacher finite-window exponent
difference over N=8,16,32,64 and the paired risk difference at N=64. Windows
4–32 and 16–128 are declared sensitivity checks. The exact finite-support
cache references have slopes about 0.723 (ranked) and 0.378 (IID) over 8–64;
the infinite-support exponents 1/2 and 1/3 are not expected to hold exactly
over this finite range. Report world-bootstrap intervals for all contrasts.

Also freeze a secondary Gemini 3.1 Flash Lite run on the first eight confirmation
encoders (202000–202007), all the same budgets and four example conditions, with
no additional direct-reasoning control. This is a second model on matched worlds,
not an independent sample of encoders. Both cohorts and all earlier pilots share
the unchanged $3 cap. No model is removed based on a disappointing outcome.

When generated codes are all correct, the teacher-ranked prompt is literally
identical to the generated-ranked prompt; successful identical API requests are
reused from the cache. Generated-IID and teacher-IID can likewise coincide. This
is a content control by construction, not an independent replication of an
identical input. Logical costs charge generation to the generating method only.

## Operational record and estimand clarification

After 12 main encoders had completed, request concurrency increased from 16 to
32 and concurrent encoders from four to eight. Completed requests were reused
on restart; unresolved request reservations remain in the cumulative ledger.
Seeds, prompts, budgets, sampling, source hashes, and scoring were unchanged.

The importance-weighted estimate describes this stratified, shuffled, eight-query
batch protocol. Predictions can depend on other queries in a batch, so it need
not equal risk when queries arrive singly or in IID batches. The reference cache
has no such batch dependence. Sample weights are not self-normalized and can sum
to more than one; occasional risk estimates slightly above one are expected from
the sampling estimator, not impossible per-query errors.
