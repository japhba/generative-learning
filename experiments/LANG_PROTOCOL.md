# Language-dominant models: own vs same-language vs other-language worked solutions (LANG)

Frozen 7 October 2026, 01:22 UK, after two pilots and before any confirmatory inference.

**Question (user).** An own-source advantage should appear when foreign examples are correct but less comprehensible
to the reader. Foreign examples should still help, only less efficiently, with the fraction correct held constant.
Ideally the reader recognises the foreign examples (probe) yet applies them worse.

**Panel** (vLLM 0.28, pinned revisions, 3 servers per H100 on gpu_saxe):

| Language | Models |
|---|---|
| English | Llama-3.1-8B-Instruct, OLMo-3-7B-Instruct |
| Japanese | Llama-3.1-Swallow-8B-Instruct-v0.5, Gemma-2-Llama-Swallow-9B-it |
| Chinese | GLM-4-9B-0414, InternLM3-8B-Instruct |

Every prompt is in the model's own language, and every model writes its reasoning in that language (100% compliance
in the pilots).

**Task: `units4`.** Four alien units; the examples show only composite conversions; questions in both directions. The
problems are language-neutral, and only the language of the reasoning differs between sources.

## Pilots (exploratory; data in `lang_pilot`, `lang_pilot2`)

- **Pilot 1:** the operator task is beyond 8B models (attempt accuracy 0–6%). Only 1 of 6 `units4` episodes was
  usable with 4 attempts per problem. The probe asked for adjacent ratios, but models answer with cumulative factors.
- **Pilot 2:** 8 attempts per problem; 8 of 12 episodes usable (≥ 4 problems where every source has a correct
  attempt); probe asks for cumulative factors. Results on the double-centred reader × source accuracy matrix:

  | Contrast | N = 1 | N = 4 |
  |---|---|---|
  | Same-language peer | +6.6 pp [+4.0, +9.1] | +2.1 pp |
  | Other language | −1.2 pp | −0.2 pp |
  | Own model | −1.7 pp | −1.3 pp |
  | Recognition, same-language peer | +11.8 pp | — |
  | Recognition, other language | −5.4 pp | — |

  Every source gives a large uplift over no practice: +30 to +44 pp.

## Confirmatory design

- 40 fresh episodes (seed prefix `lang_main_20261007`); 12 pool problems × 8 attempts per model (T = 0.8, 1,024
  tokens); 8 test problems.
- An episode is usable if at least 4 pool problems have a finished correct attempt from every source.
- Practice is all correct, on identical problems for every source; per source, the correct attempt closest to its
  own median length.
- Conditions per reader: none; answers only (4 problems); N ∈ {1, 2, 4} worked solutions from each of the 6 sources
  (always the first N matched problems).
- Test: step by step in the reader's language, greedy.
- Recognition probe: a separate call with the same practice, asking for the cumulative factors from the largest unit
  ("Ratios: x, y, z"), 256 tokens.
- **Scoring:**
  - Tests: the last number on the final answer line (`Answer`, `答え`, `答案`).
  - Probe: the numbers after `=` on the `Ratios:` line, or all numbers on it.

**Statistic.** For each episode and N, D is the double-centred 6 × 6 reader × source accuracy matrix. It removes how
good each reader is and how helpful each source is for everyone.

- **Primary:** the language contrast at N = 1, C₁ = mean D(same-language peer cells) − mean D(other-language cells),
  averaged over usable episodes. Two-sided sign-flip test over episodes (100,000 draws), 95% episode bootstrap CI.
- **Secondary:**
  1. C₂ and C₄, and the efficiency trend C₁ − C₄. The pattern the user predicts is C₁ > C₄ ≥ 0.
  2. Self beyond language: mean D(own) − mean D(same-language peer), at each N.
  3. The same contrasts on recognition-probe accuracy.
  4. Application given recognition: test accuracy in reader × source × episode cells whose probe was correct,
     same-language vs other-language sources (episode bootstrap).
  5. Uplift over no practice by relation and N. Other-language sources must still give a positive uplift.
  6. Per-reader contrasts.
