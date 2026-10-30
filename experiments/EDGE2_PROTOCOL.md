# Edge of capability with flash-tier models (EDGE2)

Frozen 6 October 2026 (UK) before any EDGE2 inference (only single probe calls to choose endpoints).

Same question and design as EDGE (EDGE_PROTOCOL.md, including the numeric scorer of its amendment 4), with larger
models via OpenRouter and a hard budget of USD 5.

- **Panel (flash tier, thinking off so that every attempt is fully visible):** DeepSeek-V4.1-Flash (DeepInfra, fp8,
  reasoning disabled), Qwen3.8-Flash (Alibaba, reasoning disabled), Mistral-Small-2603 (Mistral), Gemma-4-31B-it (Crusoe,
  bf16). Providers pinned, no fallbacks, provider asserted on every response. Gemini 3.8 Flash, GLM-5.3-Flash and
  Cohere Command A+ were excluded because their endpoints make reasoning mandatory: their own attempts would be partly
  hidden from the reader. GPT-5.6-luna was excluded because it accepts no temperature.
- **Scoring:** numeric answers are correct if the last number on the final "Answer:" line equals the true answer;
  word answers (cipher) if the last word on that line matches, ignoring case.
- **Calibration:** 3 episodes each of `operator`, `offset`, `units`, `cipher` (EDGE generators, fresh seeds); 8 pool
  problems × 4 attempts per model (T = 0.8, top-p 0.9, 1,024 tokens). **Rule:** a family is eligible if every model's
  attempt accuracy lies in [10%, 90%] and the mixed-set yield is at least 30%. The eligible family with the highest
  yield is chosen; if none is eligible, the family with the highest yield, provided it is at least 30%.
- **Main:** 40 fresh episodes of the chosen family; 10 pool problems × 4 attempts per model; 8 test problems. Matching
  as in EDGE: the mixed set has 2 right + 2 wrong (identical problems, right/wrong pattern and order for every source;
  per slot the qualifying attempt closest to the median length); the all-correct set has 4 right. Conditions per
  reader: none, mixed from each source, all-correct from each source. Test: step by step, greedy, 1,024 tokens. Tests
  run episode by episode. If the budget ceiling stops the run, the analysis uses the episodes whose tests are complete.
- **Primary:** S(mixed) = mean diagonal − mean off-diagonal of the 4 × 4 reader × source matrix, averaged over
  episodes; two-sided sign-flip test (100,000 draws) and episode bootstrap CI (EDGE's one-sided own-advantage test is
  reported too). **Secondary:** S(all-correct); S(mixed) − S(all-correct); own effect per reader; all six pairwise H;
  practice effect relative to no practice; strict-format vs numeric scoring per cell (format-copying check).

## Amendment 1 (6 October 2026, 21:04 UK; after calibration and probes, before any main-phase inference)

Calibration (USD 0.60) found no family with a usable mixed set. At this tier the models almost never finish with a wrong
answer: on `operator`/`offset` they fail by running into the 1,024-token limit while searching for the rule (40–98%
truncated), on `units` they are at ceiling (99–100%), and on `cipher` two models are at ceiling. A probe asking for
brief working (≤ 120 words, 400 tokens) did not change this. A harder four-unit family (`units4`, ratios inferred by
division, both directions) put DeepSeek-V4.1-Flash, Gemma-4-31B and Llama-4-Maverick at 100% and Qwen3.8-Flash and
Mistral-Small at 62–72%, again failing only by truncation. Probes cost USD 0.16 in total.

Changes for the main phase:

- **Unsuccessful attempt = any attempt that is not correct, including an unfinished (truncated) one**, provided its text is
  not empty. It is still the model's own visible reasoning; a mixed set is 2 successful + 2 unsuccessful attempts, with
  identical problems and success pattern for every source.
- **Family `operator`**, where the probes put three models from three families at the edge: DeepSeek-V4.1-Flash 42–58%,
  Gemma-4-31B 42–66%, Llama-4-Maverick 38% (Novita, fp8). **Panel = these three** (3 × 3 matrix). Qwen3.8-Flash
  (4–22%) and Mistral-Small-2603 (0–2%) are too weak on this family and are dropped.
- Budget: 36 episodes, 8 pool problems × 3 attempts, 8 tests; conditions none and mixed only (no all-correct arm).
  Study ceiling USD 4.70 on recorded costs (probes outside the recorder took USD 0.16), i.e. at most USD 5 overall.
- Analysis as frozen, with the 3 × 3 matrix: S(mixed) two-sided, the three pairwise H, own effect per reader,
  practice effect, strict vs numeric scoring per cell.

## Amendment 2 (6 October 2026, 22:00 UK; before any test result was scored)

Main-phase attempts (36 episodes) gave only 5 episodes with a mixed set: for rules with small coefficients Gemma-4-31B
and Llama-4-Maverick are at ceiling (88–100%), and the three models fail together only for large coefficients
(p ≥ 4, q ≥ 2). One tested episode costs about USD 0.08. Therefore a second set of **36 fresh episodes restricted
to hard rules (p ∈ {4, 5}, q ∈ {2, 3, 4}, p ≠ q)** is added (phase `main_hard`, otherwise identical). The primary analysis
pools every tested mixed-set episode from `main` and `main_hard`; each phase is also reported separately. The recorded
budget ceiling for `main_hard` is USD 4.35, because the `main` tests run in parallel.
