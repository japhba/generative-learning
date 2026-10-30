# OpenRouter exploration and learning curves, 2026-09-05

The confirmed result is a difference in finite-range **sample efficiency caused by
example selection**. This experiment does not demonstrate an intrinsic advantage of
generating answers, recognition training, weight updates, or asymptotic compute laws.

## Search and confirmation

The initial four-state permutation-composition pilot used three worlds and two cheap
models. It had a sharp identifiability threshold and unreliable model performance,
so it was unsuitable for estimating power-law exponents. Its results remain saved.

The replacement task is a random dictionary of 65,536 independent four-way labels,
with Zipf-distributed test queries: probability of key r is proportional to r^-1.5.
The learner knows the frequency distribution. Each practice observation supplies an
exact oracle label. Active generation selects keys, guesses labels, then receives
feedback. IID study observes randomly sampled examples. Yoked study observes exactly
the active learner's selected labels without its selection/guess transcript.

Four pilot worlds per model favored Gemini2.5FlashLite: it consistently chose the
optimal ranked curriculum, while Qwen3-30B-A3B-Instruct-2507 had an irregular coverage
failure. The Gemini protocol was frozen before 32 fresh worlds (seeds4000–4031),
using Google through OpenRouter, temperature0, reasoning disabled. There were eight
budgets1–128 and40 weighted stratified test items per world. Evaluation never entered
the learning history. All32 worlds and768 condition/budget observations are included.
No outcomes were excluded to improve a fit. All main active curricula covered keys1..n.

## Primary finite-range result

Fit `error = A * n^(-beta)` to log mean error at n=8,16,32,64,128.
Percentile95% intervals resample complete paired worlds (1,500 replicates).

- IID study: beta=0.352, interval[0.305,0.396].
- Active generation with feedback: beta=0.428, interval[0.402,0.456].
- Yoked study of the same selected labels: beta=0.462, interval[0.416,0.504].
- Active minus IID exponent:0.077, paired interval[0.029,0.130].
- Yoked minus IID exponent:0.110, paired interval[0.044,0.178].
- Active minus yoked exponent:-0.033, paired interval[-0.077,0.011].

At128 observations, mean errors are13.29% IID,6.14% active, and5.62% yoked.
Active-minus-IID error is-7.15 percentage points, interval[-8.19,-6.15].
Active-minus-yoked is+0.52 points, interval[-0.32,+1.26]: there is no demonstrated
benefit of retaining generation when the correct examples are held constant.

Exact perfect-recall coverage theory predicts effective exponents0.327 IID and0.502
ranked over the same finite range (infinite-vocabulary limits1/3 and1/2). The LLM
curves approximately track these coverage predictions. The difference is readily
explained by avoiding repeated low-information examples. The ranked optimum is
simple and the frequency ordering is given to the learner; this is an intentionally
constructed mechanism test, not autonomous discovery of an unknown demand function.

## Limits and sensitivity

The primary fit spans a factor16 in samples. Over4–64, active-minus-IID beta is0.116,
interval[0.063,0.169]. Over16–128, it is0.052, interval[-0.010,0.120]. Thus the primary
difference is supported, but sensitivity to the upper fitting window prevents a
strong claim of stable asymptotic exponent separation.

The active learner makes occasional errors even on seen labels: conditional seen
error at128 is1.28%, versus0.14% IID and0.81% yoked. In the pilot it sometimes repeated
its own earlier incorrect guesses despite feedback. Those guesses cannot contain
information about independent random labels, so they plausibly introduce interference.
This observation is suggestive, not a causal identification of that interference.

Supplementary fits compare power/exponential curves with and without nonnegative
floors. Simple power fits beat simple exponentials in descriptive held-out-world
prediction, but floor variants remain plausible. For active generation, a power-plus-
floor fit gives beta approximately0.555 and floor approximately0.023; its floor
interval is approximately[0.001,0.040]. These short-range shape/floor tradeoffs do not
identify an asymptotic floor or exponent. These diagnostics were added after the
pilot; they do not change the primary log-error fit.

Raw weighted test error is primary. A separate diagnostic adds exact unseen coverage
error to weighted mistakes on seen keys, marginalizing unknown random labels.
Because predictions are made in40-item batches, conclusions concern that batch
protocol rather than isolated queries. Fresh worlds control pretrained label knowledge;
this remains associative ICL, with no test of compositional transfer or weight learning.

## Tokens and charges

At128 examples, cumulative practice plus the endpoint evaluation uses about1,177
tokens IID,8,450 active, and1,292 yoked. Generation buys sample efficiency at a higher
token cost in this implementation. The yoked count excludes the cost of generating
its curriculum and is an information control, not an end-to-end acquisition system.

These are API-reported token counts assigned to each logical world; shared-cache
amortization is excluded. Token slopes are descriptive: the fitted token ranges span only1.84x IID,3.71x active,
and2.00x yoked because prompt overhead is substantial. They are not hardware FLOPs or
credible asymptotic compute exponents. A second token plot includes all earlier test
checkpoints and measures total experimental expenditure rather than endpoint use.

Across both tasks, pilots, confirmation, and reviewed retries, reported settled
charges are$0.1815715. One interrupted request has an unknown original charge, so its
full$0.00172442 reservation remains counted. The conservative total is$0.18329592,
below$0.19 and the user's$1 limit. A final read-only key-usage check reported
$0.1816223 total usage, also within that conservative bound. No further paid runs are active. The persistent
ledger keeps a shared$0.80 ceiling; request-level price bounds and reservations remain
in force. Use the ledger for actual spending, not sums that may double-count shared
cached condition rows.

Five zero-charge provider errors were inspected and repeated with unchanged payloads.
One sibling request interrupted during error propagation was also repeated under a
separate billing ID, while retaining the original charge reservation. All failure
and recovery records remain saved. Execution fixes serialize identical concurrent
requests and await sibling requests before reporting an error. Original manifests
and notes preserve these execution-only changes; task prompts and seeds were unchanged.

## Artifacts and checks

`results/zipf_main` contains the frozen manifest, pre-run selection record, worlds,
all final results, analysis summary, figures, execution notes, and spending snapshot.
`results/openrouter_budget` contains cached raw requests/responses and the ledger.
Both pilot directories are retained. Credentials are absent from saved requests.

All15 offline tests pass. The final audit verifies complete pairing, fresh seeds,
the recorded runner hashes, actual Google provider use, and optimal active coverage.
The main learning and endpoint-token PNGs were visually inspected. Nothing was pushed.
