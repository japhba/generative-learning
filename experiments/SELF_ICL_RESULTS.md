# Self-ICL reproduction and demonstration-count curves

Completed 10 September 2026 (Europe/London). The sampled direct-prompting
reproduction finds a small, uncertain gain. A CoT extension finds a larger
exact-match gain, part of which comes from answer formatting. Fresh
Boolean-expression and navigation curves are nonmonotone and do not establish
a power-law advantage from self-generated demonstrations.

## What was reproduced

[Self-ICL](https://aclanthology.org/2023.emnlp-main.968/) generates k pseudo-inputs
conditional on each test question, labels each independently using the same
frozen model, and supplies those input-answer pairs in the final prompt. We used
the released prompt text, temperature 0, top_p 1, and native OpenRouter text
completions with `openai/gpt-3.5-turbo-instruct`, pinned to the OpenAI provider.
CoT demonstrations include the full generated reasoning. No judge, correctness
filter, confidence ranking, or corrective feedback enters generation or inference.
The test question and task description are provided; no initial gold examples
are supplied in either the zero-shot or Self-ICL conditions.

GPT-3.5 Instruct appears in **Table 3** of the paper, where direct accuracy rises
from 48.52% to 49.72% over the full benchmark. The main Table 2 and original
shot-count analysis use **text-davinci-003**. Thus our direct comparison samples
an original paper setting; the GPT-3.5 CoT comparison is an extension. Current
endpoint naming cannot guarantee identical model weights or serving behaviour
to the 2023 experiment. See the paper's [PDF](https://aclanthology.org/2023.emnlp-main.968.pdf).

We completed 16 selected questions from each of all 23 classification tasks,
368 questions total, instead of the full 5,511. Each task's seeded permutation
reserves its first 32 questions for real demonstrations. The operational pilot
uses the next two; this cohort uses positions 34–49. Test texts are disjoint
from both that pool and the official demonstrations. All conditions use the
same held-out questions, and published BBH labels are retained unchanged.

The k=3 input generation and all CoT calls allow 1,024 output tokens. Direct
prediction follows the released code's short-label convention: two output
tokens, with a parenthesis continuation prefix for letter choices. This differs
from the paper's generic 1,024-token statement. Direct calls often end at this
intentional label limit; that is not itself evidence of a truncated reasoning
trace. All input-generation and CoT calls in the completed cohorts finished
normally without reaching their output limit.

## Main paired results

Percentages are macro-averaged across tasks. Equal per-task sample counts also
make them equal to pooled accuracy. Intervals resample paired test items within
each task 4,000 times, keeping the 23 task identities fixed. They are pointwise
95% intervals, without correction for multiple comparisons; they do not estimate
generalization to new task families or randomness from repeated API runs.

| Prompting | Zero-shot | Self-ICL, k=3 | Paired change, percentage points | Two-sided McNemar p |
|---|---:|---:|---:|---:|
| Direct | 50.8% (187/368) | 51.6% (190/368) | +0.8 [−3.8, +5.4] | 0.828 |
| CoT | 43.2% (159/368) | 53.0% (195/368) | +9.8 [+4.3, +15.2] | 0.00186 |
| CoT, option-text sensitivity | 51.4% (189/368) | 57.1% (210/368) | +5.7 [0.0, +11.4] | 0.0617 |

The direct result is compatible with the paper's modest +1.20-point GPT-3.5
effect, but this sample does not resolve it. The official three real examples
give 56.3% direct accuracy (207/368); there is no teacher-rationale CoT control.

The third row is **post-hoc evaluation only**. Full-response checks found
answers such as a correct date or movie title where the requested output was
an option letter. The sensitivity parser maps exact option text to its displayed
letter, using no gold-label information to choose the mapping. It also accepts
a final answer letter with a relaxed marker. It does not perform semantic judging.
This scoring change reduces the apparent CoT advantage by 4.1 points, leaving
an uncertain positive remainder. It cannot establish that reasoning itself
improved. Both the primary and sensitivity scores are retained.

![Self-ICL on 368 held-out BBH questions](../reports/self-icl-replication.png)

## Demonstration-count extension

Short Boolean expressions and navigation permit more demonstrations within
the available context. The direct cohort uses 32 fresh questions per task
(64 total, permutation positions 64–95). It compares independent k-instance
generations at k=1,2,3,4,8,16, plus zero-shot. The real-example reference uses
prefixes of a fixed disjoint 32-item labeled pool. These inputs are not matched
to the self-generated inputs or conditioned on each particular test question.

After seeing the main CoT contrast, a follow-up used the first 24 scaling
questions per task (48 total) and k=0,1,2,3,4,8. Its pseudo-input lists are
identical to the corresponding direct lists; CoT labels and final answers
come from separate calls. This is a motivated follow-up, not independent
confirmation of a prospectively predicted exponent. Direct and CoT aggregate
curves have different sample counts and should not be treated as a paired
cross-mode comparison.

Input generation allows **192 output tokens per requested example**, including
3,072 at k=16. This declared scaling extension differs from the historical k=3
cohort's fixed 1,024-token allowance. Each k is a separately requested list,
so the generated sets need not be nested. No correctness-based rerolls occur.

| Demonstrations k | Self-ICL direct, n=64 | Real-example direct, n=64 | Self-ICL CoT, n=48 |
|---:|---:|---:|---:|
| 0 | 65.6% | 65.6% | 70.8% |
| 1 | 71.9% | 64.1% | 70.8% |
| 2 | 78.1% | 68.8% | 79.2% |
| 3 | 76.6% | 64.1% | 70.8% |
| 4 | 76.6% | 62.5% | 66.7% |
| 8 | 78.1% | 64.1% | 64.6% |
| 16 | 73.4% | 76.6% | — |

Every paired self-minus-zero 95% interval includes zero in both scaling
cohorts. The direct curve improves early and plateaus; the CoT curve peaks
at k=2 and then declines. The real-example reference is also irregular.
These samples give **no evidence for a stable, steeper power law**. They do
not exclude a benefit under other tasks, models, or larger samples.

The descriptive zero-floor fit `error = A * k^(-beta)` over direct k=1–16
gives beta≈0.015. Its local slopes change sign. An unknown error floor,
near-ceiling Boolean performance, independently generated curricula, and
small samples make an asymptotic exponent claim inappropriate. No exponent
comparison is presented as an inferential result.

![Demonstration-count curves; shaded bands are pointwise 95% intervals for Self-ICL](../reports/self-icl-scaling.png)

Task-level curves explain part of the aggregate pattern: direct Boolean
accuracy is already 90.6% at zero-shot and falls to 81.3% at k=16; navigation
rises from 40.6% to 65.6%. Both tasks remain included. The observed pattern
therefore mixes a task with little headroom and a harder task with an early
demonstration benefit.

![Per-task curves; direct uses 32 and CoT uses 24 questions per task](../reports/self-icl-scaling-gpt35.png)

## Work, failures, and generated-label diagnostics

All logical token costs include input generation, independent labeling, and
final inference, even when the API cache avoids paying for an identical call
twice. Tokens are not measured hardware FLOPs. Real-example costs omit the
cost of obtaining teacher examples.

| Main condition | Mean prompt plus completion tokens per question |
|---|---:|
| Zero direct | 132 |
| Self-ICL direct, k=3 | 1,243 |
| Official real direct, k=3 | 431 |
| Zero CoT | 281 |
| Self-ICL CoT, k=3 | 2,105 |

The main CoT gain uses about 7.5 times as many total API tokens as zero CoT.
This comparison does not isolate a benefit of generation from extra computation,
query-conditioned curriculum selection, or merely receiving the resulting
demonstrations. At final inference the frozen model receives a prompt; the
experiment does not endow it with an additional memory of having generated
that text. There are no weight updates or online corrective signals.

![Accuracy versus mean total API token work](../reports/self-icl-work.png)

All 368 main cases completed with the requested three generated inputs.
Malformed instance counts at k=1 affect 3/64 direct scaling cases and 2/48
CoT cases. They count as incorrect pipeline outcomes, with no replacement
samples. No other scaling budget has this failure. Exact test-query copies
were absent. Summing uniqueness within individual k-shot sets gives 1,103
of 1,104 main generated inputs, and 931 of 1,024 at direct k=16; this is not
a count of global semantic uniqueness.

A Boolean grammar evaluator was used **only after inference** to diagnose
pseudo-label accuracy. In the main cohort, 34/48 direct labels and 43/48 CoT
labels were correct. In direct k=16 scaling, 409/512 were correct. These
labels were not repaired or filtered. The diagnostic checks the final Boolean
answer, not every step of the generated rationale.

## Operational record and reproducibility

The initial modern-model attempts used Qwen3-30B-A3B-Instruct-2507, Mistral
Small 3.2, and later Gemini 2.5 Flash Lite. Chat formatting failures and repeated
upstream 429s led to an operational narrowing to GPT-3.5 Instruct. Earlier
pilots and the interrupted multi-model cohort remain in the raw results;
they are not combined with the completed cohorts. The chronological
[protocol](SELF_ICL_PROTOCOL.md) records these amendments. No final model
was chosen for a positive treatment effect.

The entire exercise, including diagnostics and failed attempts, records
**$3.79424** in settled OpenRouter charges. A further **$0.40627** remains
conservatively reserved for requests with unresolved usage, for a committed
total of **$4.20052**, below the $5 cap. There were 11,370 settled calls and
11,793 reserved attempts. Unresolved reservations are not reported as
confirmed charges. Credentials are never stored with requests.

Raw outputs and immutable cohort manifests live in `results/self_icl`, a
symlink to `/var/tmp/jbauer/self_icl_20260910`. They include all requests,
responses, token usage, per-case predictions, generated demonstrations,
provider snapshots, and source snapshots. A persistent archive location and
checksum are recorded in `results/self_icl/archive.json` after verification.

Source revisions:

- Self-ICL: `3368f359508d001c1e697c54bbe04795aa113ef6`.
- BIG-Bench-Hard: `9ee07bd481feebf959a6b59d61ea57bdcf30964d`.
- Prepared task data SHA256: `db16f1a203ce436be935b03c19b9f907f790586362c1c3947e0fda0f677b563c`.

Rebuild all final analyses and PNGs without making API calls:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/audit_self_icl.py results/self_icl/historical
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_self_icl.py results/self_icl/historical --plot replication --audit results/self_icl/historical/audit.json
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_self_icl.py results/self_icl/scaling_cot
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_self_icl.py results/self_icl/scaling --plot scaling --cot-phase results/self_icl/scaling_cot
"$VENV_LOCAL/generative-learning/bin/python" experiments/audit_self_icl.py results/self_icl/scaling
"$VENV_LOCAL/generative-learning/bin/python" experiments/audit_self_icl.py results/self_icl/scaling_cot
"$VENV_LOCAL/generative-learning/bin/python" -m unittest discover -s experiments -p test_self_icl.py -v
"$VENV_LOCAL/generative-learning/bin/python" experiments/verify_self_icl.py results/self_icl
```

The final inference commands, which reuse completed cases and cached calls
when the original result directory is present, were:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/self_icl.py --phase historical --models gpt35 --tasks all --count 16 --start 34 --budgets 0 3 --modes direct cot --real
"$VENV_LOCAL/generative-learning/bin/python" experiments/self_icl.py --phase scaling --models gpt35 --tasks boolean_expressions navigate --count 32 --start 64 --budgets 0 1 2 3 4 8 16 --modes direct --real --input-tokens-per-shot 192
"$VENV_LOCAL/generative-learning/bin/python" experiments/self_icl.py --phase scaling_cot --models gpt35 --tasks boolean_expressions navigate --count 24 --start 64 --budgets 0 1 2 3 4 8 --modes cot --input-tokens-per-shot 192
```

The runner refuses to alter a cohort's frozen manifest. The request ledger
retains uncertain charges on restart. Offline checks verify prompt identity
against the released implementation, held-out target isolation, disjoint
teacher/test inputs, count-failure accounting, answer-audit ambiguity handling,
Boolean operator semantics, and budget-reservation persistence. The analyzer
refuses incomplete cohorts. Code and figures accompany the persistent archive.

The supplied strategy-emergence paper's expansion/contraction grammar remains
a candidate for the next controlled modelling step. This experiment establishes
the empirical Self-ICL baseline and its limitations; it performs no SFT, RL, or
mechanistic-model fitting.
