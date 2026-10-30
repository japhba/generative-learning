# Edge of capability: results (6 October 2026)

Protocol: EDGE_PROTOCOL.md (amendments 1–4). Code: edge_search.py, edge_label.py, edge_analyze.py, edge_plot.py.
Figure: reports/edge-of-capability.png. Valid data: results/task_search/edge/main2 and label2. The first main/label run is void
(answer-format scoring artefact, amendment 4).

- Calibration: only alien units is at the edge for Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B and Phi-4-mini. Operator, offset
  and cipher are near 0%.
- main2: 120 units episodes. 65 have a matched mixed set (2 right + 2 wrong; same problems and right/wrong pattern for
  every source) and 115 a matched all-correct set. Test: step by step, numeric scoring.
- **Primary S (mixed) = −6.2 pp [−8.4, −4.0]**, one-sided p (own advantage) = 1.0. The own effect is negative.
- Per reader (mixed, vs the 3 others): Qwen3-1.7B +5.0, SmolLM3 +6.6, Phi-4-mini +1.7, Llama-3.2-3B −38.0 (its own
  cell: 25% accuracy, 22% truncated).
- Llama-3.2-3B is far weaker (no-practice accuracy 13% vs 66–84%). Among the three similar models:
  - S(all-correct) = −0.1 pp [−1.2, +0.8], a tight null.
  - S(mixed) = −5.3 pp [−7.9, −2.7]; all three pairwise H are negative (−4.6 to −5.8).
- Labelled arm (wrong attempts marked "Check: wrong."): S = −6.1 pp. Change vs unlabelled +0.0 [−1.8, +1.9]. Practice
  gain unchanged.
- Practice helps strongly: +12.6 pp (mixed) and +26.5 pp (all correct) over no practice.
- Error adoption (wrong factor copied): own 18.5% vs peers 17.5%, no difference. The earlier difference was the format
  artefact.

Reading: correctness-matched, own attempts are worth the same as peers' attempts when all are right. When half are wrong,
own attempts help less, and marking the wrong ones does not help. Nothing here supports an own-comprehension advantage
at the edge of capability.
