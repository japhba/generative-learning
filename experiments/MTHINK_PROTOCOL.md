# Arm 1: own versus foreign thinking traces as practice

Frozen 5 October 2026 before inference. User intuition: reasoning a model lays
out for itself is mostly legible only to itself. Thinking traces (private
reasoning, thinking mode on) are the most self-directed text a model writes.

- Models: Qwen3.5-9B, Gemma 4 26B-A4B, Nemotron-3-Nano-30B-A3B (three families,
  all with a thinking toggle), pinned revisions, vLLM 0.28, 32k context,
  reasoning parsers qwen3 / gemma4 / nemotron_v3, gpu_lowp H200.
- Operator family (a◇b = p·a + q·b, q ≥ 2), 16 fresh episodes, 4 examples,
  pool of 20, 12 tests. Each model generates 8 samples per pool question with
  thinking ON (T = 0.8, 16,384 tokens; raised from 12,288 after a smoke test truncated a Gemma trace).
- Qualifying trace: finished, correct final answer, both partial products p·x
  and q·y present in reasoning or answer, reasoning ≤ 30,000 characters (smoke-test traces ran 10–27k). For each
  pool question (in order) where every model has qualifying traces, each model
  contributes the one whose reasoning length is closest to the median of all
  qualifying traces for that question; 3 questions (context budget); episodes with < 3 excluded.
- Practice shown as "Reasoning: <trace> / Answer: <answer>". Test: thinking OFF,
  reply prefilled "Answer:", 12 tokens, exact match. Conditions: none; answers
  only; each model's thinking traces.
- Primary S = mean over readers of own − mean(other two), one-sided sign-flip;
  secondary: per reader, logit own effect, trace lengths.

## First-cohort result and replication (frozen before replication inference)

First cohort (16/16 episodes): S = +3.2 pp [+1.0, +5.3], one-sided p = 0.0075;
logit +0.16 [0.00, +0.32]. Per reader: Qwen3.5-9B +1.0, Gemma 4 +8.1 [+3.9,
+12.5], Nemotron +0.5 pp; Qwen3.5 and Nemotron answer directly at 10–17% (floor).
Scoring note: the "Answer:" extractor was corrected on 6 October (it crossed line
breaks, e.g. "Final Answer:\n\nAnswer: 7"); selection in this cohort ran with the
old extractor (conservative: fewer qualifying traces); answer-only scoring is
unaffected.

**Replication (`mthink2_search.py`)**: 16 fresh episodes; same design with a fourth
thinking model that answers directly well, Qwen3-30B-A3B (hybrid thinking,
revision ad44e777, reasoning parser qwen3). Primary: S over the four readers
(one-sided sign-flip). Secondary (pre-registered): Gemma 4 own − mean(others);
Qwen family pair (Qwen3.5-9B, Qwen3-30B-A3B); logit own effect.
