# Correctness-matched laid-out reasoning

Frozen 5 October 2026 before inference. Follow-up to LAYOUT (laid-out reasoning),
whose positive own-source contrast was explained by (1) condition-dependent
answer-only compliance, (2) ceiling × answer-quality effects, (3) a single
strong source meeting a strong reader, and differing correctness of own vs
foreign practice. User request: own versus foreign practice must contain
solutions of the same correctness; no rewriting of one model's text by another.

## Design

- Readers = generators: Qwen3.5-9B, Qwen3-30B-A3B-Instruct-2507, Gemma 4 26B
  A4B IT, Phi-4. Each is served from pinned Hugging Face revisions with vLLM
  0.28.0, one H100 per job on gpu_lowp (gpu-xd670-30), generation config
  `vllm`, thinking disabled through the chat template. Server command and
  revision are recorded per request (cache fingerprint).
- Families (`mlayout_tasks.py`): mystery operator a◇b = p·a + q·b (no offset,
  operands 11–49) and alien units with two hidden ratios 4–12 and quantities
  3–19. 8 episodes per family; 4 input/answer examples (units examples cover
  each conversion type); a pool of 16 practice questions; 12 tests.
- Practice generation: each model samples 16 worked solutions per pool question
  (T = 0.8, top-p 0.9, 2048 tokens, same seed across models). Set after a
  pilot on separate seeds: at 1024 tokens / 4 samples most Qwen solutions were
  truncated and Phi-4 found the operator rule in 6% of samples.
- **Correctness matching (frozen rule):** walk the pool in order; keep a question
  if every model has a finished sample whose final "Answer:" is correct; use
  each model's first such sample; stop at 6 questions. Episodes with fewer
  than 3 matched questions are excluded (recorded). Within an episode all four
  sources therefore show correct solutions to the same questions in the same
  order; only the solution text differs. Test questions never enter selection.
- **Test:** answer-only with the assistant reply prefilled as "Answer:"
  (`continue_final_message`), T = 0, 12 tokens; the first line is the answer.
  No reader can reason or ignore the format. Exact-match scoring.
- Conditions per reader: none (examples only); answers only (the matched
  questions with their correct answers — identical for every source); full
  correct solutions from each of the four models.

## Analysis

- **Primary S** = mean over readers of [F(r,r) − mean_{s≠r} F(r,s)], F = test
  accuracy after full solutions; one-sided exact sign-flip test over included
  episodes. Source and reader main effects cancel; correctness is matched.
- Secondary: S per family; double-centred own effect per reader; pairwise H;
  value of laid-out reasoning F(r,s) − answers-only; Qwen sibling comparison
  (own vs same-family vs cross-family); solution length per source; matched
  questions per episode and per-model sample accuracy.
- Interpretation limit: correctness is matched, but solutions can still differ
  in clarity, explicitness of the rule and length; the double-centred design
  removes these only insofar as they act as source main effects.
- Compute is local cluster GPUs only; no OpenRouter spend.

## Results of the first phase and refined phase (frozen before refined inference)

First phase (14/16 episodes included): S = +3.0 pp [+0.8, +5.2], one-sided
p = 0.015, entirely from the operator family (+7.3 pp); units saturated at
98–100% for every reader and source. A post-hoc logit model with reader and
source effects gives an own effect of +0.16 log-odds [+0.05, +0.28]. Phi-4's
"correct" operator solutions often reach the answer by nonsense arithmetic, so
final-answer matching did not equalise reasoning quality.

**Refined phase (`mlayout2_search.py`)**: operator only, 16 fresh episodes, rules
p·a + q·b with q ≥ 2, pool 24, 16 samples, models Qwen3.5-9B, Qwen3-30B-A3B,
Gemma 4 (all regularly find the rule). A practice solution qualifies only if it
finishes with the correct answer **and** contains both partial products p·x and
q·y for its question (process-correct). Same answer-only prefilled test and
conditions. Primary: S as above (one-sided sign-flip over included episodes);
pre-specified secondary: the logit model (reader + source + own indicator,
episode bootstrap), per-reader own effects, Qwen sibling comparison.

## Refined-phase results and confirmation (frozen before confirmation inference)

Refined phase (13/16 episodes): S = +2.5 pp [−0.9, +5.7], p = 0.098; logit own
effect +0.12 [−0.07, +0.31]. Per reader: Qwen3.5-9B +1.0, Qwen3-30B −1.0, Gemma 4
+7.4 [−0.3, +14.7] pp; Gemma's solutions do not help the other readers more.
Post-hoc pooling of both cohorts (19 operator episodes, three models): Gemma 4
own − other two = +9.4 pp [+3.3, +15.4]; Qwens +0.7 and −3.3. Gemma's own
solutions are shorter (~2,100 vs ~3,000 characters).

**Confirmation (`mlayout3_search.py`)**: 16 fresh operator episodes; same models,
pool, sampling, process-correct criterion and prefilled answer-only test.
Selection is additionally **length-matched**: per matched question, each model's
qualifying solution closest to the median length of all qualifying solutions
for that question. **Primary: Gemma 4's own − mean(other two sources)**, one-sided
exact sign-flip over included episodes. Secondary: the same for each Qwen
reader, S, logit own effect, solution lengths after matching.
