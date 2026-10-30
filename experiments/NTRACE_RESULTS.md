# Native traces: completed 5 October 2026

Protocol NTRACE_PROTOCOL.md; report reports/own-reasoning-report.html; data results/task_search/ntrace.

Seven models wrote free-form worked solutions (K = 0); readers reasoned step by
step at test after 8 traces from one source. 3 tasks × 6 episodes × 12 tests.

- **Own traces help least.** S = mean own − mean other accuracy = −3.9 pp
  [−6.0, −1.7]; one-sided p for S > 0 = 0.998 (lenient scoring −3.7 pp).
  calendar_arithmetic −6.9, number_sequence +0.3, time_intervals −5.0.
- Double-centred own effects: Qwen3.5-9B +1.0, Qwen3-30B −4.0, Llama-3.1-8B −6.0,
  Llama-3.2-3B −0.1, Gemma 4 −0.8, Phi-4 −7.1, Nemo −6.3 pp.
- No family gradient: own − same-family source −5.6 to +1.4 pp, same-family −
  cross-family −8.4 to +2.6 pp (intervals mostly cover 0).
- Weaker readers' own-trace deficit is answered-but-wrong questions, not format;
  Qwen3-30B's is partly truncation. Reading: own unjudged traces add no new
  information and reinforce the reader's own errors.
- Panel change before the matrix: Qwen3-Coder replaced by Qwen3-30B-A3B-2507
  (provider dropped 103/240 texts). Server/scheduling changes listed in the protocol.
