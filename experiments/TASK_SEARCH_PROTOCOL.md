# Broad local task search

This study keeps Qwen3.5-9B fixed and searches task families. It follows the user's requirement that the final learner's only experimental change is the supplied examples. The earlier matched-source study is preserved separately.

## Model and data

- Checkpoint: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; BF16 on the local RTX 4500 Ada 24 GB, served by vLLM 0.28.0. Thinking disabled for every call. No parameter updates.
- Reasoning Gym 0.1.25: 103 standalone procedural families, 385 distinct default/curriculum configurations (including the easiest level after a baseline-only pilot showed many floor settings). Preflight records every setting, including unavailable corpora, invalid generator configurations, oracle/scorer disagreements, generation timeouts, and context exclusions.
- Four initial problem/answer examples per episode. Distinct procedural questions partitioned into support, rehearsal, and held-out tests. Every phase has separately hashed seeds.
- Calibration uses only the four initial examples and four test questions per setting. It selects difficulty using baseline accuracy alone, before examining any source contrast. Selection and fresh screening seeds are recorded.

## Conditions

The final prompt uses the same initial examples, demonstration template, evaluation questions, decoding settings, and output allowance in all arms. Examples never carry an origin label.

1. `zero`: four initial examples only.
2. `teacher`: N procedural rehearsal problems with generator-provided valid answers.
3. `self_fixed` (primary): those **identical N rehearsal problems**, with answers independently produced by Qwen from the four initial examples. Generation uses the same native answer prompt as final inference, with temperature 0.7; only its answer enters the final context. There is no scoring, correction, rejection, or feedback on these answers before inference.
4. `self_free` (secondary): N complete problem/answer examples independently generated from the same four initial examples and a generic request for the same family and approximate difficulty. No test question, curriculum hint, or external validator is supplied. This arm measures the combined effect of changing question and answer source and cannot establish an effect at fixed question distribution.

The optional broader complete-example interpretation was presented to the user. Pending further steering, the identical-question comparison remains primary and complete examples are separately reported.

Generations are drawn once and reused as prefixes across N. Incorrect and duplicate samples remain included. Truncated generations produce an explicitly recorded empty-answer placeholder; they are not replaced by retries or selected for quality. Inference transport errors are retried with backoff and logged separately.

Final inference uses temperature 0, no thinking, and one question per response in the task's native answer format, with 1,024 output tokens. Generation uses temperature 0.7, top-p 0.9, and 1,024 output tokens per sample. Both final conditions share their evaluation seed. Sample counts are the main budget axis; actual prompt/completion token counts are retained because equal sample counts need not imply equal token counts or total compute.

## Constructive tasks and scoring

Multiple-valid-answer tasks are especially informative: two correct solutions can differ even when their question is identical. Valid alternative graph colorings, arithmetic expressions, logical conclusions, and plans must be accepted by constraints rather than exact string matching.

Preflight verifies that the teacher answer receives full credit before and after serializing its metadata. For graph coloring, propositional logic, and Rubik's cube, the documented metadata solution replaces the upstream null answer. A strict rational-arithmetic AST checker enforces the exact supplied number multiset and target for puzzle24/countdown; the upstream puzzle24 scorer did not enforce the multiset and converted the result to an integer. Symbolic-expression scoring is preceded by a restricted AST check.

Primary accuracy counts only full-credit answers. Partial reward is retained separately. Model output and grader failures are inspectable in raw records. Data/preflight exclusions happen without access to model comparison outcomes. A prompt exceeding the server's context budget is marked unavailable, rather than silently shortened; comparisons use complete paired episodes and report missingness.

## Search and confirmation

The broad screen is exploratory. All task-level results, including negatives, are retained. Promising tasks must be tested on newly seeded episodes under a frozen configuration before claiming an effect. Confirmed improvement must exceed both the teacher condition and the no-extra-example baseline. A source gap caused solely by teacher deterioration is not evidence of learning from generated experience.

Uncertainty is clustered by independently generated episode, preserving paired tasks/queries. Primary endpoints use the largest prespecified N and adjust for the number of selected confirmation tasks. Learning curves are always saved. A finite-window slope or favorable point estimate is not sufficient to claim a steeper asymptotic power law.

All requests, full raw responses, token usage, engine fingerprint, source snapshots, data seeds, and selection decisions are stored under `results/task_search`. No generated-sample LLM judge is used.

## Baseline-only format pilot

The initial four-question JSON-array calibration was stopped after 26 of 230 preflight-ready settings. A sorting response confused the outer list of problem answers with the inner list required by its answer. An eight-task one-question comparison favored native plain answers over a JSON string wrapper (the sorting and graph-coloring examples became correct). All pilot data remain under `calibration` and `wrapper_spot_checks.jsonl`. The main phase, `calibration_single`, uses native one-question answers and includes level-zero curricula. No generation-source comparisons preceded these decisions. The preflight token check conservatively retains the larger four-query teacher prompt; the API additionally checks the actual rendered single-query prompt.

The malformed-input scorer audit exercised 99 available families with six invalid formats each. Prime factorization raised an uncaught ValueError on prose; Game of Life raised KeyError on a JSON object. Targeted format checks now award zero for these invalid answers. The interrupted `calibration_single` phase is preserved, and `calibration_native` reuses its exact data and cached model responses with corrected scoring; the data-origin record documents this replay. Both scorer changes precede all source comparisons.

## Focused strategy-family extension

While the broad screen was running (before observing any positive two-episode source contrast), a 36-setting graph-coloring grid was added. It varies vertices (6, 10, 14, 18), palette size (half, equal to, or twice the vertex count), and edge probability (0.15, 0.4, 0.7). Every arm receives the same problem constraints and initial examples. When the palette is at least the vertex count, assigning every vertex a distinct color is always valid; the canonical teacher continues to use the library's ordinary greedy color reuse. No prompt suggests either strategy. Baseline calibration uses four fresh test graphs per setting before any source comparisons. Thirty-five settings meet a 14,000-token teacher-prompt bound. This grid tests whether an easier valid strategy can emerge from unjudged self-generated answers; it is one task family with multiple settings, not 36 unrelated families. The extension registers native Reasoning Gym datasets and uses the unchanged inference implementation.

The focused `palette_screen` retains all nine six-vertex palette/density settings, with four fresh episodes each, four initial examples, 32 assigned rehearsal graphs, eight held-out graphs, and N=0,2,8,32. The two arms are canonical teacher answers and unjudged answers to the identical graphs from Qwen. The six-vertex grid was selected for context capacity and demonstrated baseline headroom; no palette source outcomes were inspected before selection. Low-palette cells remain as task controls. Any selected source advantage requires a separate fresh confirmation.

## Early number-format confirmation

The first completed broad-screen fixed-question candidate exceeding both controls by 12.5 percentage points was `number_format__default` (own 15/16 correct; exact teacher and initial-only each 13/16). A fresh 32-episode confirmation was frozen before its inference, with 16 held-out questions per episode and N=0,1,2,4,8,16,32. Its primary endpoint remains N=8; other N values provide the learning curve. A prespecified teacher control copies the winning candidate exactly in its displayed notation instead of returning the library's canonical decimal representation. The learner receives the same questions, initial examples, and final instructions. This attempt will remain in the final primary multiplicity family even if later screen candidates displace it from the final shortlist. Raw generated answers remain unjudged and unfiltered. The selection file stores hashes of both selecting episodes.

For the early number-format confirmation, the prespecified descriptive slope window is N=2,4,8,16,32: fit log(error)=intercept−beta*log(N), with episode-paired bootstrap uncertainty where finite. There is no pseudocount or fitted error floor, and no asymptotic claim. Its notation control uses N=2,8,32, retaining N=8 as the primary mechanism check. These analysis/control choices were recorded before inspecting confirmation accuracies.

## Completed-screen confirmation plan

The completed broad screen produced exactly six positive fixed-question candidates and six positive complete-example candidates at N=8. All are retained for confirmation. The already completed number-format confirmation is negative and remains in the primary multiplicity family. The remaining five primary tasks receive 32 fresh episodes with 16 held-out questions each. All six secondary tasks receive 32 fresh episodes with eight held-out questions each. Both phases use N=0,2,8 and eight independently generated, unjudged rehearsal samples per episode. The smaller secondary cohort reflects its less restrictive hypothesis and is frozen before any secondary confirmation inference.

Primary candidates: graph coloring, maze, number format (already completed), counting primes, palindrome partitioning, and path star. Secondary candidates: self reference, bit counting, coin flip, cube-color rotation, graph coloring, and path star. The primary endpoint is N=8. Apply a paired episode sign-flip test against both teacher and initial-only, take the maximum of the two p-values for the conjunction, and apply Holm separately across all six primary and all six secondary hypotheses. Also report correction across all twelve as a sensitivity check. The plan, selection hashes, cohorts, and UK freeze time are recorded in `confirmation_plan.json`. No partial confirmation accuracies determine stopping or cohort size.

The nine palette settings had no candidate exceeding both controls at their prespecified N=32 endpoint. None advances to a palette confirmation. Their full curves and post-inference strategy audit remain part of the reported search.

Before the new confirmation inference, a source inspection found an upstream coin-flip scoring defect: the partial-credit prefix score can equal 1 for an incorrect short answer (for example, `0` against `0.5`). The original selecting screen and native-score records are preserved. Coin-flip confirmation claims will instead require absolute probability error at most 1e-4, computed from the exact combinatorial fraction in metadata, with no prefix credit. This audit changes no prompt, sample, model output, or selection decision. Native and corrected metrics will both be reported, and coin flip remains in the six-task secondary multiplicity family even if its screening gain disappears under the corrected score. The correction was specified before any new primary or secondary confirmation request.

## Palette strategy diagnostic

After the negative palette screen and its completed strategy audit, an additional teacher control was specified while the fresh broad confirmations were still running. It includes all six six-vertex settings with six or twelve allowed colors, using the same 24 original episodes, initial examples, rehearsal graphs, and tests. Only the additional teacher answer strings change: each vertex receives a different allowed color. This solution is valid by construction, without reading or selecting self-generated samples. N=2,8,32 and all final inference settings remain matched. The diagnostic asks whether the model can learn an easy valid strategy when it is consistently demonstrated; it does not establish a benefit of unjudged generation. The original teacher/self contrasts and all their null results remain unchanged.

## Shared task wordings

A small exploratory extension expresses the identical allocation constraints using meetings/time slots, items/storage bins, records/codes, transmitters/radio channels, or vertices/colors. The complete wording is identical across teacher and self conditions within each comparison. No strategy instruction is added to generation. All wordings state only the pairwise inequality constraints, with six entities, six or twelve allowed resources, and edge probability .4 or .7. Standard greedy teacher assignments remain valid. Underlying graph seeds are paired across wordings at a fixed geometry.

The 20-setting baseline calibration uses four initial examples and four held-out queries per setting. One geometry per wording is chosen using baseline accuracy closest to .5, with shorter prompts breaking ties. All five wordings are retained. The source screen then uses four new episodes per selected setting, eight held-out queries, and N=0,2,8 under teacher/self_fixed. Any candidate exceeding both controls requires a fresh, frozen confirmation and inclusion in the primary multiplicity family. This is one mathematical task under several wordings, not five independent task families. The hypothesis is that shared task wording changes the prior probability of producing a simple valid assignment; no source-specific cue or correctness feedback is allowed.

## Sequential rehearsal

To test the original online-learning motivation directly, an exploratory sequential-generation arm reuses all nine completed palette settings and their 36 episodes. The same ordered rehearsal questions, four initial examples, held-out questions, N=0,2,8,32, final instructions, and decoding settings are retained. At generation step i, the model sees its own previous i unjudged answers as additional examples. It sees neither future questions nor held-out questions or teacher rehearsal answers. The deterministic teacher oracle is unchanged. There is no scoring, filtering, repair, source label, or strategy hint in the generation loop. Temperature and per-question RNG seeds match independent generation, so the first generation request is identical. Generation pools and raw calls are incrementally resumable.

This is a distinct generation protocol, reported separately from independent draws. It supplies a growing rehearsal history and therefore uses more generation input tokens; equal N is not equal compute. All final source comparisons still differ only in the provided answers to identical questions. The online screen's primary endpoint is N=32. Up to two uncensored candidates exceeding both teacher and initial-only may advance to fresh confirmation; every additional attempted primary confirmation must join the earlier primary multiplicity family. No source gain is inferred from the reused exploratory screen alone.


## Continuation: fresh wording and online confirmations

The completed five-wording screen selected exactly one positive fixed-question candidate, vertex_colors_v6_c6_p40. Its fresh confirmation uses 32 episodes, 16 tests each, and N=0,1,2,4,8,16,32, with the selecting endpoint N=8 fixed as primary. The completed six original primary attempts stay in the correction family.

The ongoing nine-setting sequential palette screen has primary endpoint N=32. At most two uncensored candidates positive against both controls, ranked by their smaller gain, will receive fresh confirmation with 32 episodes, 16 tests, and the same seven N values. The full selected cohort is completed before accuracy inspection. All additional attempts join the original six plus the shared-wording attempt for final primary Holm correction; the complete-example family retains all six attempts. The final cross-family correction includes every attempt. Teacher and final-evaluation requests remain matched; sequential generation alone can see prior unjudged own rehearsal answers. No filtering, repair, oracle feedback, or strategy hint is introduced.

An administrative status read inadvertently emitted a few early secondary and online screen episode summaries. They are not used to alter sample sizes, task membership, prompts, stopping, endpoints, or confirmation selection. Subsequent monitoring reads counts and infrastructure status only until each full phase completes.

For every new seven-point confirmation, the common descriptive slope window is fixed at N=2,4,8,16,32 before phase completion: log(error)=intercept-beta log(N), no pseudocount or fitted floor, paired episode bootstrap. Both source betas and their difference are reported regardless of direction. Censored windows or zero empirical errors are marked unidentifiable rather than repaired. These fits are descriptive, not additional corrected discovery tests or evidence for an asymptotic law.


## Arithmetic decomposition extension

A compact additional task search tests sum and product decompositions, with two or three integer parts and digit scales3,6,12. Each task admits multiple valid answers, including identity-based answers, without naming a strategy in the model prompt. A sum teacher samples uniformly from ordered weak compositions of the target, including possible zero parts. A product teacher samples independent uniform positive factors, including possible ones, and supplies their product as the question. No special restriction excludes easy teacher answers. Initial examples and assigned questions are literally matched across final source conditions; only the supplied answer strings change. Every own sample is an independent unjudged draw from the four initial examples under the unchanged native prompt and decoder.

Twelve configurations receive four baseline queries each. One magnitude per operation/arity is selected by baseline headroom only, keeping all four settings even at floor/ceiling. The source screen uses four fresh episodes per setting, eight tests, andN0,2,8,32. At most two uncensored candidates exceeding both controls atN32, ranked by their smaller gain, advance to32fresh episodes of16tests andN0,1,2,4,8,16,32, with primary endpointN32. Every additional attempt joins all seven previous primary attempts in the final Holm family; all six secondary attempts remain included separately and in the overall correction. Full cohorts complete before accuracy interpretation. The common descriptive slope window remainsN2,4,8,16,32. Ten exact scoring checks and all12world generators passed before model requests. These are two operations with arity/magnitude settings, not12 independent task families.
