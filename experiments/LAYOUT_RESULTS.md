# Laid-out reasoning: completed 5 October 2026

Protocol LAYOUT_PROTOCOL.md; report reports/own-reasoning-report.html; data results/task_search/layout.

Hidden-rule episodes (operator, cipher, units); six models reason out loud on 8
practice problems; tests are answer-only after one source's practice, full or
reduced to answers. 18 episodes × 12 tests.

- Primary L (own full − own answers) − (foreign full − foreign answers) = +6.1 pp
  [+2.3, +10.2], p = 0.0045. V (own full − own answers) = +16.7 pp; S (own full −
  foreign full) = +4.9 pp, p = 0.004.
- Not clean evidence for legibility: (1) Gemma and Phi-4 often ignore "no
  working" (32-token cap); Gemma complies 65% after its own full traces vs 35%
  after foreign ones. (2) Units: Llama/Nemo's own answers are mostly wrong, so the
  answers-only arm is low while any full trace reveals the ratios (ceiling).
  (3) Operator: Gemma alone derives the rule in its (often truncated) traces and is
  the best direct-answer reader, so good source × capable reader coincide on the
  diagonal. Four fully compliant readers: L = +3.0 pp [+0.7, +5.7], all from units;
  own full vs foreign full −0.1 pp.
- Decisive next test: identical content rewritten in another model's idiom
  (paraphrase both ways) plus an explicit-rule control.
