# Self-ICL reproduction and demonstration-count curves

Started 10 September 2026. This study implements the three-stage procedure from
Chen et al., EMNLP 2023: generate k pseudo-inputs conditional on each test question,
answer every pseudo-input independently with the same frozen model, and prepend
the resulting examples before answering the test question. No answer correctness,
judge, confidence ranking, or rejection sampling enters these stages.

## Sources and fidelity

Original implementation: https://github.com/ntunlplab/Self-ICL, commit
3368f359508d001c1e697c54bbe04795aa113ef6. BBH data are downloaded from the original
suzgunmirac/BIG-Bench-Hard repository, with its commit and input hashes saved.
The 23 classification tasks match the paper. Task descriptions and prompt text
match the released StreamPrompt implementation, including its spelling.

Model scope: GPT-3.5 Turbo Instruct is the paper's additional-model experiment
(Table 3), which reports direct prompting. The paper's main Table 2 and original
shot-count analysis use text-davinci-003. Our direct three-shot comparison samples
the Table 3 setting; our GPT-3.5 CoT comparison and extended shot counts are method
extensions. The same API model name does not establish identity of checkpoint or
serving implementation across 2023 and 2026.

All stages use temperature 0 and top_p 1. The k=3 input-generation and CoT calls
have the paper's 1,024-token limit. GPT-3.5 Turbo Instruct uses native text
completion and the released direct-prediction convention: a parenthesis prefix
for letter choices and a two-token label allowance. The authors' implementation
reduces the direct output limit despite the paper's generic 1,024-token statement.

Qwen3-30B-A3B-Instruct-2507 and Mistral Small 3.2 are modern-model extensions.
They use chat endpoints with a shared completion instruction and 16-token direct
answer limit, since the unmodified prompt elicited explanations in smoke tests.
The semantic task prompts remain identical. Modern results are a method
replication, not exact historical model replication. Providers and price ceilings
are pinned. Smoke tests and all subsequent calls share a $5 cumulative cap.

## Pilot and confirmation

The initial pilot uses two randomly selected cases per task and all three models,
with zero-shot direct, zero-shot CoT, three-shot Self-ICL direct and CoT, and the
official three-example real-data direct reference. Models/tasks will not be
selected for a positive treatment effect. The pilot checks operational validity,
token lengths, cost, baseline headroom, and response parsing. Its data remain
separate from confirmation. Final confirmation sizes are frozen before dispatch.

The first 32 entries in each seeded dataset permutation are reserved exclusively
for real-data demonstration pools. Pilot and confirmation test indices follow
this pool and are disjoint. Test target values are never included in Self-ICL
generation or labelling calls. Official examples are checked for exact overlap.

## Scaling extension

Use independently generated k-shot sets for each k, preserving the actual meaning
of generating k examples. Do not generate a large pool and retrospectively charge
small prefixes as though only k examples had been generated. Generation allowance
above k=3 increases in proportion to k; label and evaluation allowances stay fixed.
Report both demonstration count and total prompt-plus-completion tokens, including
input generation, all independent label calls, and final inference. These are
token-work counters, not hardware FLOPs. Real-data controls omit teacher-production
cost and are identified as data/ICL references.

Four compositional BBH tasks are nominated before outcome inspection:
boolean_expressions, navigate, logical_deduction_seven_objects, and
tracking_shuffled_objects_seven_objects. Candidate budgets are 0,1,2,3,4,8,16.
This is a finite-range curve study; it cannot establish an asymptotic power law.

## Failure accounting and analysis

All responses and requests are preserved without truncating spot checks. Input
parsing normalizes presentation syntax only. Malformed generation counts produce
an explicit pipeline failure and count as incorrect in primary intention-to-treat
accuracy; no correctness-based rerolls or exclusions are allowed. Truncated
responses and output-format failures are reported. Transient HTTP/transport errors
get bounded retries, with unresolved charges conservatively reserved.

Evaluation extracts the final answer using the released answer conventions with
markdown punctuation normalization. The CoT answer must have the requested final
answer marker; direct predictions use the first answer token. Parser diagnostics
and full-response spot checks distinguish task errors from formatting effects.

Report macro-average accuracy across the 23 tasks and paired treatment differences.
Resample test items within tasks jointly across conditions; optionally resample
tasks as a separate generalization analysis. Scaling analysis retains all nominated
tasks and budgets, shows uncertainty and local slope stability, and permits flat
or worsening curves. No universal exponent is inferred from a zero-floor fit.

## Suggested subsequent task

The supplied local paper, How New Strategies Emerge in RL Post-Training: A
Controlled Study, uses symbol expansions and contractions in a known grammar.
This offers a later compositional experiment with auditable traces. The present
study first reproduces Self-ICL on its original task family and performs no RL,
SFT, or new mechanistic modelling.

## Pilot amendments, frozen before confirmation

The historical-model pilot completed all 46 cases. The initial modern chat
instruction sometimes elicited reasoning in direct mode or only one requested
input. Stage-specific system instructions now separately request all input
instances, label-only direct answers, and the explicit CoT final-answer sentence.
An intermediate instruction induced literal ellipses in the final answer; it was
replaced with an explicit LABEL placeholder instruction. No task or model was
chosen for a positive treatment effect. All earlier outputs and code snapshots
are retained as operational pilots, separate from confirmation.

Mistral was excluded from confirmation after repeated upstream 429 errors at both
DeepInfra and Parasail. Gemini 2.5 Flash Lite through Google AI Studio replaces it;
its hidden reasoning is disabled. Qwen uses StreamLake and GPT-3.5 Instruct uses
OpenAI native text completion. The eight-case final interface check completed.
Malformed instance counts remain counted as failures, not silently repaired.

Confirmation: 16 items per task, all 23 tasks, all three retained models, permutation
positions 34 through 49. Conditions are zero direct, zero CoT, Self-ICL three-shot
direct and CoT, and official real three-shot direct. The same test cases are used
across all models and conditions. Total: 368 held-out questions per model.

Scaling: 24 fresh items per nominated task, Qwen and Gemini, permutation positions
64 through 87, budgets 0,1,2,3,4,8,16. Real answer-only controls use the first k
examples of each task's reserved pool; gold tests never enter that pool. Every k
uses a separately requested k-instance generation. Input generation gets 1,024
tokens per requested instance at every k in this extension, preventing a fixed
list-length cap from masquerading as a demonstration-count effect. Thus its k=3
generation allowance differs from the 1,024-token historical replication. All
pseudo-label and final CoT calls retain 1,024 tokens. Both stages share the $5 cap,
including conservative reserves for unresolved failed requests.

## Infrastructure-driven narrowing

The attempted three-model confirmation was interrupted after StreamLake also
returned repeated upstream 429s. Modern-model operational outputs remain archived;
they also expose a substantial short-list generation failure rate. To complete
the user's requested paper reproduction with a stable native completion interface,
the final `historical` cohort retains GPT-3.5 Instruct on the unchanged 368 selected
test cases and all five conditions. Its already completed identical requests are
reused from the cache. This narrowing was based on endpoint and interface failures,
not on selecting a positive treatment effect.

The final `scaling` cohort uses the same historical model, direct prompting,
Boolean expressions and navigation, 32 fresh cases per task at permutation
positions 64..95, and k=0,1,2,3,4,8,16. These short tasks permit 16 demonstrations
inside the model's 4,095-token context; CoT traces and the nominated seven-object
tasks generally do not. Input generation gets 192 tokens per requested instance
(3,072 at k=16); final answers retain the original two-token allowance. This is
a declared context-budget extension, not an exact replication of the paper's
fixed 1,024-token input-generation setting. The main k=3 historical replication
retains the original settings. Real answer-only controls use the same held-out
queries and the reserved example pool. Both tasks remain included regardless of
the treatment effect.

Retries and restarts keep every unresolved earlier reservation in the cumulative
budget. Reissuing an interrupted request receives a new attempt ID rather than
erasing the previous attempt or its potential charge. No correctness-based sample
rerolls occur. Operational pilots, the interrupted cohort, and final cohorts have
separate immutable manifests and source snapshots.

## CoT scaling follow-up

The completed historical cohort shows a positive three-shot CoT exact-match
contrast and little direct-prompting change. Before collecting CoT scaling
outcomes, a follow-up is fixed on the first 24 of the 32 scaling items per task,
using k=0,1,2,3,4,8. It reuses the exact cached pseudo-input lists from the direct
curves, obtains independent CoT pseudo-labels, and evaluates new CoT predictions.
The k=8 limit respects the historical model's context size. No teacher rationale
baseline is added. Logical method costs include input generation even where the
experiment reuses that already-paid cache. This is a motivated follow-up, not an
independent replication of a prospectively predicted scaling exponent.
