# Same content, own versus foreign expression

Frozen 6 October 2026 before inference. User request: hold solution quality
constant so that only the "language" / form of expression differs, without one
model rewriting another model's text. Every source re-expresses the SAME correct
solution in its own words ("rewrite in your own words … keep every step and every
number"), T = 0.8, 4 samples.

## GSM8K arm (learning in the weights; in context as secondary)
- 2,000 GSM8K training problems (same seeded shuffle as SMALL); content = the
  human reference solution. Renderers: Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B,
  Phi-4-mini, Gemma 4 26B, Qwen3-30B-A3B.
- Valid rendering: finished, correct final answer, and every intermediate result
  of the reference calculation (GSM8K <<…=r>> annotations) present. Problems where
  all six have a valid rendering are kept; each source contributes the valid
  rendering closest to the problem's median length. The raw human solution is a
  seventh source.
- Fine-tuning: each small student × each of the 7 sources, LoRA as in SMALL
  (`msft_train.py`), **training seeds 0 and 1**; GSM8K test (500, step-by-step).
- **Primary:** mean over students of acc(own rendering) − mean acc(three peer
  students' renderings), averaged over both seeds; per-student and per-seed values
  reported. Secondary: own vs large renderings, vs raw human; in-context arm
  (300 test items, 4 rendered examples, same problems for every source).

## Operator arm (in context, answer-only)
- 16 fresh episodes (rule p·a + q·b, q ≥ 2), pool 12, 12 tests. Content = a
  programmatic canonical solution (check the rule on the four examples, then
  compute). Renderers/readers: Qwen3.5-9B, Qwen3-30B-A3B-Instruct, Gemma 4.
- Valid rendering: finished, correct answer, both partial products present;
  first 6 pool questions with valid renderings from all three; length-matched.
- Test: answer-only with "Answer:" prefill. Conditions: none, answers only,
  canonical text, each model's rendering. **Primary:** S = mean over readers of
  own rendering − mean(other renderings), one-sided sign-flip over episodes.
