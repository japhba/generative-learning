# Strict source-only in-context rehearsal search

Started 10 September 2026, Europe/London, following the user's correction that
coverage, curriculum, and every other deliberate intervention must remain fixed.
This study has a new persistent $3 OpenRouter cap including all calibration and
confirmation. Earlier studies and their ledgers remain unchanged.

## Invariant comparison

Each paired world fixes the task, initial four correct examples, N rehearsal
questions, their ordering, held-out queries, learner checkpoint, system message,
final prompt template, decoding parameters, and final output allowance. Only the
source of rehearsal answers changes. The primary teacher supplies exact answers
from the task simulator. The learner solves the same rehearsal questions from
the initial four examples in independent calls; the final answer from each call
becomes the corresponding demonstration answer. Its reasoning is discarded in
this answer-only screen. Wrong answers and malformed output are never corrected,
filtered, selected by confidence, or replaced with another attempt. Unparsed
responses occupy their sample slot as UNPARSED; full raw responses stay in logs.

Questions are sampled by the same program once per world and shared literally
between arms. No ranked, balanced, targeted, or adaptive curriculum is supplied
only to generation. Initial examples, rehearsal inputs, and test inputs are
disjoint by their exact question strings. Final calls have no access to their
own test labels. Generation never sees any test question or label, future
rehearsal question, or other generated answer. Teacher labels never enter a
generation request. Both final arms use exactly N question-answer pairs.

A frozen model has no extra memory of having authored text in a separate API
call. This experiment tests whether its generated answers form more useful
context than external answers on matched inputs. It does not claim that identical
final token sequences can yield different prediction distributions because of
their provenance. Identical requests share a persistent cache; equality when
all answers agree is an identity control, not independent replication.

Final inference allowance is identical across arms. Generating a sample requires
additional work, which is recorded separately and in logical total-token curves.
Oracle-teacher acquisition cost is excluded. Equal N and equal final allowance
do not imply equal total computation; no extra test-time reasoning control is
given preferentially to either arm.

## Broad screen and selection

Sixteen exact task families, each with two settings, cover modular inversion,
modular arithmetic, hidden affine and linear rules, Boolean expressions,
permutation programs, finite-state machines, stack programs, string editing,
shortest paths, Bayesian urn inference, multiplication in unfamiliar bases,
bracket completion, sorting with an unfamiliar alphabet, substitution grammars,
and parity codes. Task descriptions expose an exact rule except the two explicit
rule-induction families. The oracle's hidden parameters are never put in prompts
for those families.

An interface check uses separate seeds before freezing the main screen. The
initial screen uses Qwen3.5 Flash on four worlds per cell, 24 test questions per
world in four batches of six, and N=0,4,16. Each source's N prefixes are nested.
The endpoint and provider are qwen/qwen3.5-flash-02-23 through alibaba, temperature
zero, top-p one, hidden reasoning disabled, generation allowance 1,024 tokens,
final allowance 512. Both source arms share that final allowance. The secondary
model is Gemini 3.1 Flash Lite through google-ai-studio.

The screen is explicitly exploratory across 32 cells. Positive screen contrasts
are not confirmations. Rank candidates by paired generated-minus-oracle mean
accuracy at N=16, breaking ties by family name and setting. Before further calls,
freeze up to four leading cells with positive screen differences and adequate
non-ceiling test error for fresh confirmation, using a separate seed range.
Report all screened cells, including null and negative results. If no positive
cell emerges, report that result and broaden the representation/source comparison
without changing the matched-question invariant. Do not manufacture a curriculum
difference to obtain a positive effect.

Fresh confirmation uses N=0,1,2,4,8,16,32 and paired whole-world bootstrap intervals.
Primary endpoints are generated-minus-oracle accuracy at N=16 and the exponent
difference over N=4,8,16,32, when strictly positive errors permit a fit. Final
cohort sizes and selected cells are frozen before outcomes, based on remaining
budget. Apply Holm correction across selected cells to paired world-level tests;
keep exploratory results separate. A slope from four points is a finite-range
statistic, and a flat, curved, or worsening curve does not establish a power law.

An additional teacher-LLM control may use another model's solutions to exactly
the same questions, with the same generation instructions and allowance. When
present it is an additional source arm, never a replacement for the exact teacher
chosen after observing its quality. An advantage over a less accurate teacher
alone is not evidence for the proposed intrinsic generation benefit.

## Records and checks

All prompts, full raw responses, intermediate predictions, answer-source lists,
accuracy by world, generation correctness diagnosed only after evaluation,
token work, charges, errors, source snapshots, and frozen manifests are saved.
No data are deleted. Learning curves are saved as PNGs with Matplotlib defaults.
Offline tests verify task semantics, held-out isolation, unchanged wrong samples,
and that changing source changes only the answer fields of final prompts.
Analyses refuse incomplete planned cohorts. Transient network failures receive
bounded retries with unresolved reservations retained. Persistent archives include
code, environment lockfile, negative pilots, and plots.

## Interface amendment before the screen

The six-task interface check found valid final-answer parsing but some lengthy
generation calls exhausted their allowance. The original fallback would insert
their entire reasoning into an answer-only demonstration. That can give only the
generated arm an unintended rationale, violating the representation invariant.
Before the main screen, unparsed outputs instead occupy their original sample
slot with the literal answer UNPARSED. No row is removed and no correctness
information enters this handling. The same rule applies to any model teacher.
All original interface data and source snapshots remain separate.

The common generation prompt now requests at most eight short working lines and
no repeated verification. Its token allowance is unchanged at 1,024. This is an
interface correction, not a correctness filter or a search for favorable examples.

## Frozen answer-only confirmation

The complete screen has three positive cells away from floor: permutation
programs setting 2, Boolean expressions setting 1, and shortest paths setting 1.
Their screen differences are +3.125, +2.083, and +1.042 percentage points; all
paired intervals include zero. Before confirmation, operationalize usable
headroom as exact-teacher accuracy strictly between 5% and 95% and generated
accuracy below 95%. Positive floor cells remain reported but cannot support a
useful scaling study. The exact selection and screen hash are in selection.json.

Freeze Qwen on 32 new worlds per selected cell and Gemini on the first 16 of
those worlds. Seed base 600000; N=0,1,2,4,8,16,32; 24 held-out queries per world;
all prompts, sample rules, parsing, and allowances unchanged from the screen.
The primary contrast remains generated versus exact teacher. Confirmation
data are separate from selection; no selected cell is dropped after outcomes.

## Additional worked-solution screen

To test richer samples under the same source-only constraint, also retain complete
worked solutions from both the learner and an external LLM teacher. Both models
receive identical rehearsal questions, initial examples, generation instructions,
and 1,024-token allowance. Both final arms receive N complete solutions under the
same template and 512-token final allowance. No answer repair, correctness filter,
or source name enters the final prompt. Unfinished solutions remain in full in
both source arms; unlike the answer-only screen, reasoning is part of every sample
by design. Differences in actual wording, length, and errors are consequences
of the source and are recorded; they are not separately assigned treatments.

Freeze this exploratory screen on all 16 families at setting 1, two new worlds
per family, seed base 900000, 12 held-out queries, and N=0,2,8. The same generated
solutions are also evaluated as answer-only samples with the exact teacher retained
as a reference in a separate cohort. The primary worked-solution contrast is Qwen
using its own solutions versus Gemini's solutions; the reciprocal Gemini contrast
is secondary. A gain against a less accurate external teacher will be identified
as such. This does not replace or reinterpret the exact-teacher primary experiment.

Any promising worked-solution task must be frozen for fresh confirmation before
claiming a benefit. Contexts and query sets stay matched throughout; only the
source of the complete demonstration changes. A small N=2,8 exploratory contrast
does not establish a scaling exponent.

## Frozen worked-solution confirmation

The two leading primary Qwen cells are hidden affine inference and permutation
programs, both at setting 1 and both with +8.33-point exploratory N=8 differences.
All 16 rehearsal answers from each source were correct in each selected screen
cell; reasoning steps were not judged. These are two-world exploratory results.
The reciprocal Gemini contrasts were zero and −4.17 points, respectively.

Before new outcomes, freeze these two cells, both source models, 24 new worlds
per cell, seed base 1200000, 12 held-out queries per world, and N=0,1,2,4,8,16.
Reuse exactly the same question streams and raw source solutions across every
source arm. The primary endpoint is Qwen own-minus-Gemini accuracy at N=8, with
Holm adjustment across the two selected cells. The reciprocal model contrast is
secondary. The primary finite-range slope window is N=2,4,8,16; N=1,2,4,8 is
a sensitivity window. No fit is reported unless both window endpoints exist.
The paired answer-only and exact-teacher references are preserved separately.

Selection is stored with the exploratory analysis hash in trace_selection.json.
No chosen cell is removed after confirmation, regardless of effect sign, teacher
quality, or arithmetic error. Report quality and token-work differences as
possible explanations of any observed source effect, not as sample filters.

## Additional BBH classification screen

Freeze all 23 classification task configurations used by Self-ICL, Qwen3.5 Flash,
two question splits per task, four initial labeled examples, 12 test questions,
and N=0,2,8. Use original BBH inputs and published labels at the pinned commit
recorded in bbh_inputs/original_provenance.json. Before any calls, an input audit
found comma-split option texts in the cache. Checking all original BBH files
confirmed the same three invalid questions in the original release: their
published target maps to none of the displayed choices. Preserve every original
record and list these three exclusions in bbh_inputs/validation.json. Exclude
these malformed corpus questions identically from both source arms before any
sampling; this is independent of generated outputs. The same preflight check
found two identical-question pairs with conflicting causal-judgment labels and
two consistent duplicate pairs in sports understanding. Exclude both conflicting
pairs and retain one instance per consistent pair, identically for both sources.
The 5,511 raw records yield 5,502 eligible unique questions; every removed record
and reason is preserved in validation.json. This prevents duplicate question text
from crossing rehearsal, test, or phase boundaries.
Descriptions remain the shared Self-ICL descriptions. Exact option-text or letter
normalization uses only displayed options and is identical for both sources.

Uniformly sample a single shared question sequence for both source arms. Reuse
the answer-only prompts and allowances from the synthetic study: 1,024 generation
tokens, 512 final tokens, batches of six questions, temperature zero, and no
hidden reasoning. No generated answer is checked, repaired, or discarded before
evaluation. The control is a published teacher label, not a proof-checked oracle.

Before split sampling, partition each task's dataset into two disjoint halves
using seed 43210 plus its sorted task index. Screen worlds use only the first
half with seed base 1500000; confirmation worlds use only the second half.
Rehearsal, support, and tests are disjoint within each world. Different worlds
may reuse items within their phase half, so world-bootstrap uncertainty concerns
random splits of this fixed corpus, not independent new natural-language tasks.

The primary endpoint is self minus published teacher accuracy at N=8. Report
all task results. Select at most two leading positive screen cells with teacher
accuracy strictly between 5% and 95% and self accuracy below 95%. If any qualify,
freeze all selected cells for 16 fresh splits per task, seed base 1800000,
N=0,1,2,4,8,16, and the same 12-test protocol. Keep N=8 primary with Holm correction
over selected tasks; N=2,4,8,16 is the descriptive slope window. Save the selection
with the screen analysis hash before confirmation calls. All phases share the
existing new-study $3 ledger ceiling; stop before starting a cohort that cannot
fit the remaining conservative budget.

Before any BBH confirmation calls, the selected tasks are formal fallacies and
salient translation error detection, each +8.33 points in the two-split screen.
The maximum-token charge bound for 16 splits/task is $1.215501, above the remaining
$0.650339. Freeze eight confirmation splits per task instead, keeping both tasks,
the disjoint confirmation half, all budgets, and the endpoint unchanged. This
budget amendment reduces precision and is recorded in bbh_selection.json before
outcomes; it is not a response to confirmation results.

## BBH parser sensitivity after confirmation

Full-response review found seven formal-fallacy generations with an unprefixed
final INVALID label (six bare responses, one reasoning response ending INVALID).
The frozen parser required ANSWER:, so these occupied UNPARSED slots. Preserve
the original screen, selection, confirmation, and all raw generations. In a
separate explicitly post hoc sensitivity cohort, accept an exact last-line label
when the ANSWER: marker is absent. This rule consults only the generated text and
displayed options, never the published correct answer; wrong labels are retained.
Replay all original confirmation worlds and budgets with the same teacher/zero
rows, questions, prompts, limits, and shared ledger. Do not regenerate samples.
Both original and corrected results will be reported; this is not fresh data.
