# Cross-task reader–source specificity search

Frozen 5 October 2026 before any inference of this study. It extends the
12 September specificity experiment (SPECIFICITY_PROTOCOL.md), which found no
setting where *both* readers prefer their own unjudged rehearsal answers. Here
we search the whole procedural reasoning_gym catalog for such settings.

## Fixed inference (unchanged from the 12 September experiment)

- Readers/generators: pinned local Qwen3.5-9B BF16 (identical vLLM command and
  engine fingerprint 14f2e0f1…) and Gemma 4 26B A4B IT via OpenRouter, pinned to
  DekaLLM BF16, no provider fallback. Thinking disabled for both. The local
  launcher no longer prepends the private NVML after the kernel-driver update to
  580.173.02; the vLLM command is identical.
- Prompt: `task_search_prompts.answer_prompt` (one problem per call, "Return only
  the answer … without working"). K initial exact teacher examples, then N
  additional rehearsal examples, then the test question. No source labels.
- Generation: each model answers each rehearsal question independently,
  conditioned only on the K initial examples, temperature 0.7, top-p 0.9, same
  seed for both generators. Evaluation: temperature 0, top-p 1. 256 output
  tokens in both roles. Saved stripped content, including truncated prefixes and
  empty strings, is replayed verbatim without validation, filtering or repair.
- Scoring (post-inference only): reasoning_gym native full credit
  (`task_search_scoring.score_answer` ≥ 1−1e−9) and the strict
  `audit_task_search_constraints.answer_valid` checks (coin-flip tolerance,
  graph-colour and palindrome-partition containers). Truncated final answers are
  incorrect.
- OpenRouter billed-cost ceiling for the whole study: USD 9, checked before each
  uncached call (`xspec_api.BUDGET`).

## Phase 1 — calibration (baseline only)

Every catalog specification (295 specs, 103 task families; same exclusions as
the original search) × K ∈ {0, 4}; one fresh world per spec with 8 test
questions; both readers at N=0. Unconstructible worlds are recorded exclusions.

Screen eligibility of a setting (spec, K), using only baselines and lengths:
1. worst-case Qwen prompt at N=16 fits the 16,384-token context assuming every
   rehearsal answer uses all 256 tokens;
2. longest teacher answer ≤ 200 Qwen tokens (otherwise a correct answer cannot be
   produced within the cap);
3. not both readers ≤ 1/8 and not both readers ≥ 7/8 at N=0.

Within each task family and K, keep the eligible variant with the largest mean
over readers of p(1−p) (ties: catalog order). If more than 80 settings remain,
keep the 80 with the largest mean p(1−p). No source-comparison data exist when
this is decided.

## Phase 2 — screen

4 fresh episodes per selected setting, 16 rehearsal questions, 8 tests. Both
readers evaluated at N=0 (initial only) and at N=16 with Qwen, Gemma and teacher
rehearsal. a[r,s] is pooled accuracy over the 32 tests. Own-source preferences
P_Q = a[Q,Q] − a[Q,G] and P_G = a[G,G] − a[G,Q]; H = (P_Q + P_G)/2.

Selection: settings with **both P_Q ≥ 2/32 and P_G ≥ 2/32**, at most one K per task
family (higher H, then higher min(P_Q,P_G)), ranked by H then min(P_Q,P_G);
advance the top 6. If none qualifies, report the screen as null and do not
confirm. Screen numbers are selection statistics only, never evidence.

## Phase 3 — confirmation

16 fresh episodes per selected setting, 32 rehearsal questions, 12 tests,
N ∈ {0, 2, 8, 16, 32}; N=32 is dropped for a setting if any actual N=32 Qwen
prompt would exceed the context (length-only check before evaluation).

- Primary endpoint per setting: H at **N=16** (the screened count). One-sided exact
  sign-flip test over the 16 episode-level H values (65,536 assignments), Holm
  correction across the confirmed settings.
- A setting counts as a *mutual own-source advantage* only if Holm p < 0.05 **and**
  both P_Q and P_G have 95% paired episode-bootstrap intervals (20,000
  resamples) above zero.
- Descriptive: the full 2×3 matrix, initial-only baselines, own − teacher,
  own − initial, all N, source correctness/length/truncation, and example
  rehearsals/answers. Learning curves for every setting.

## Interpretation limits

As before, a positive interaction can reflect differences in correctness,
verbosity, formatting or strategy portability rather than anything intrinsic
to authorship; it does not establish shared neural representations. Source
correctness and length are not matched.

## Amendments (recorded before any screen outcome was inspected)

1. 5 Oct, calibration: transport-only retry of HTTP-200 responses with an
   unparseable body (design.json `amendments`).
2. 5 Oct, screen at 9.2k/19.5k evaluations: scheduling only. Individually
   shuffled jobs gave a 0.8% vLLM prefix-cache hit rate; jobs are now grouped by
   (model, world, source, N), groups shuffled reproducibly, each group's first
   request sent alone, separate per-model worker pools. Prompts, seeds and
   decoding unchanged; completed records reused.
3. Confirmation only (the screen stays on DekaLLM): the pinned DekaLLM endpoint
   is upstream rate-limited at ~4 requests/s. Confirmation episodes are
   therefore **blocked by Gemma provider**: episode e uses BF16 endpoint
   [DekaLLM, NextBit, Novita][e mod 3] for every Gemma request of that episode
   (its generations and its evaluations under every source), each request pinned
   with fallbacks disabled and the serving provider asserted. Own-versus-other
   contrasts are within episode, so the provider is a blocking factor; the
   episode-level sign-flip test and bootstrap are unchanged. Provider-wise
   results are reported descriptively. All three endpoints were checked for
   seed/top-p support, disabled reasoning and zero reasoning tokens.
