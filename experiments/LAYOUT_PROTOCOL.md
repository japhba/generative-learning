# Laid-out reasoning: own versus foreign reasoning as in-context practice

Frozen 5 October 2026 before inference. User hypothesis: learning may require
laying out one's reasoning in front of oneself to become clear about it, so a
model should profit most from reasoning it wrote out itself. The native-traces
study could not test this: readers reasoned at test time too, so practice
reasoning was redundant. Here **tests are answer-only** and **every episode
has a hidden rule** to become clear about.

## Design

- Families (`layout_tasks.py`): mystery operator a◇b = p·a + q·b + r; Caesar
  cipher decoding with an episode-specific shift; alien unit conversions with
  two hidden ratios. 6 episodes per family; K = 4 input/answer examples (units
  examples cover each conversion type), 8 practice problems, 12 tests.
- Six models = readers = generators: Qwen3.5-9B (local), Qwen3-30B-A3B-2507
  (Nebius FP8), Llama-3.1-8B (Novita FP8), Gemma 4 26B A4B (BF16, provider by
  episode mod 3), Phi-4 (DeepInfra BF16), Mistral-Nemo (DeepInfra FP8).
- Practice: each model reasons out loud on each practice problem (work out the
  rule, reason in its own words, end with "Answer:"), T = 0.8, 1024 tokens,
  same seed across models. Saved texts replayed verbatim; no filtering.
- Test: answer-only ("Do not show any working"), T = 0, 32 tokens. Sources per
  reader: none; teacher answers; and for each model its 8 practice solutions
  either **full** or reduced to **answers only** (its extracted final answers).
- Scoring: exact match of the extracted answer (last "Answer:" line, else first
  line), case/space/comma-insensitive.

## Analysis

Per reader r, with F_s / A_s the accuracy after source s's full / answers-only
practice:

- **Primary L** = mean_r [ (F_r − A_r) − mean_{s≠r} (F_s − A_s) ]: value of
  laying out *one's own* reasoning beyond its answers, relative to others'
  laid-out reasoning. Own errors appear in both own arms, so they cancel.
- One-sided exact sign-flip test over the 18 episodes (per-episode L).
- Secondary: own value of laying out V = mean_r (F_r − A_r); specificity of
  full traces S = mean diag F − mean off-diag F; the same per family; teacher
  and no-practice references; practice correctness per model.
- OpenRouter ceiling USD 1.8, raised to 2.5 mid-run when long full traces made prompts costlier than estimated (`layout_api.BUDGET`); the session total stays
  under the USD 10 allowance.
