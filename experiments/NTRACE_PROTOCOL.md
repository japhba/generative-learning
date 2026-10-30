# Native traces: own versus foreign worked solutions as in-context rehearsal

Frozen 5 October 2026 before any inference of this study. Motivation: the
cross-task search (XSPEC) used answer-only rehearsal; 35% of Qwen and Gemma
rehearsal strings were identical and most were ≤ 20 characters, so foreign
examples were hardly foreign. Hypothesis (user): a reader benefits more from
its own practice because foreign examples are less recognizable — there is
less for it to sensibly attend to. Rehearsal must therefore carry each
model's idiom.

## Design

- Seven readers = generators, spanning family, training domain and size:
  Qwen3.5-9B (pinned local), Qwen3-Coder-30B-A3B (code; Novita FP8),
  Llama-3.1-8B (Novita FP8; DeepInfra was persistently rate-limited in a pre-run smoke test), Llama-3.2-3B (Parasail BF16), Gemma 4 26B A4B
  (BF16; DekaLLM/NextBit/Novita by episode mod 3), Phi-4 (DeepInfra BF16),
  Mistral-Nemo 12B (DeepInfra FP8). Every request is pinned to its provider,
  fallbacks disabled, serving provider asserted. Thinking disabled where the
  model has it.
- Prompt (`ntrace_search.prompt`): problem family, optional worked examples
  (“Problem: … Solution: <trace>”), then the instruction to reason step by step
  in one's own words and end with “Answer: <answer>”. **K = 0 teacher examples.**
- Generation: each model writes one worked solution per rehearsal question,
  temperature 0.8, top-p 0.9, identical seed across generators, 1024 tokens (raised from 768 after a smoke test showed Gemma/Coder overrunning 200 tokens on a trivial problem).
  Every saved text (wrong, truncated or empty) is replayed verbatim; no
  filtering, labels, or feedback.
- Evaluation: temperature 0, 1024 tokens. Readers see N ∈ {0, 8} worked examples
  from one source (all seven sources, same questions in the same order).
- Scoring (post-inference only): text after the last “Answer:” line, with
  markdown emphasis, code fences, `\boxed{}`/`$…$` and a trailing period
  removed; native reasoning_gym full credit and the strict
  `audit_task_search_constraints.answer_valid` checks.

## Calibration and task choice (baseline only)

20 candidate reasoning_gym tasks (default difficulty), one world × 12 tests,
all seven models at N = 0. Keep tasks whose mean accuracy over models is in
[0.2, 0.8]; choose the **3** with the largest mean over models of p(1−p).
No source-comparison data exist when this is decided.

## Matrix phase (single confirmatory phase)

3 tasks × 6 fresh episodes × 12 tests, 8 rehearsal questions, N ∈ {0, 8},
7 readers × (initial + 7 sources).

- **Primary:** S = mean diagonal − mean off-diagonal accuracy at N = 8 (reader
  and source main effects cancel), pooled over the 18 task-episodes;
  one-sided exact sign-flip test over task-episodes.
- Secondary/descriptive: S per task; every pairwise H; double-centred own
  effect per reader; family gradient (off-diagonal accuracy from a same-family
  source vs cross-family sources, for the Qwen and Llama pairs); gain over
  N = 0; source correctness and length.
- Recognition (post hoc, local models): mean per-token log-likelihood of each
  source's traces under Qwen3.5-9B (and the Llamas if served locally); test
  whether within-reader ICL gain tracks familiarity.
- OpenRouter ceiling for this study USD 4 (`ntrace_api.BUDGET`); combined with
  XSPEC (USD 3.91) this stays under the USD 10 allowance.

## Limits

Source quality differs strongly across these models; S removes source and
reader main effects but not reader-specific sensitivity to quality. Providers
differ in quantization (FP8/BF16), recorded per model.

## Amendments before the matrix phase (no source-comparison data existed)

1. Calibration: a provider response with `content: null` (finish_reason stop) is
   recorded as status `content_missing` with empty text instead of crashing.
2. Qwen3-Coder-30B-A3B (Novita) returned no text in 103/240 calibration calls,
   and its other endpoints lack seeds or cost ~5×. It is replaced by
   **Qwen3-30B-A3B-Instruct-2507** (Nebius FP8, seed supported), which keeps a
   Qwen-family pair for the family gradient; the code-domain contrast is
   dropped. Task choice used the calibration as frozen (the replacement was not
   calibrated): calendar_arithmetic, number_sequence, time_intervals.
3. Matrix, after all rehearsal generations and ~3% of Qwen evaluations: the
   local Qwen server was relaunched with `TASK_SERVER_FAST=1` (max-num-seqs 48,
   max-num-batched-tokens 8192, CUDA graphs instead of eager mode; same
   checkpoint, vLLM and sampler) and client concurrency 48, because free-form
   evaluation was decode-bound (~25 calls/min). Qwen's rehearsal traces come
   from the original configuration; its remaining evaluations from the new one
   (engine fingerprints recorded per record via the cache key).
