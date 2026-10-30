# Further search for own-source advantages: completed 6 October 2026

Report: reports/own-reasoning-report.html (section 5). Protocols: MTHINK_PROTOCOL.md,
MSFT_PROTOCOL.md (arm 2 reuses MLAYOUT confirmation practice; `mcot_search.py`).
All compute on gpu_lowp (gpu-sr675-35, H200), pinned HF revisions, vLLM 0.28;
no OpenRouter spend.

| Arm | Design | Primary result |
|---|---|---|
| 1 Thinking traces | Private thinking (Qwen3.5-9B, Gemma 4, Nemotron-3-Nano) as practice; process-correct, length-matched; answer-only test | S = +3.2 pp [+1.0, +5.3], p = 0.0075 (Gemma +8.1, others at floor) |
| 1b Replication | 16 fresh episodes + Qwen3-30B-A3B (hybrid) | **S = +0.7 pp [−1.1, +2.5], p = 0.25; Gemma −0.5 pp** |
| 2 Reasoning at test | Matched practice from MLAYOUT confirmation, N = 1 or 6, step-by-step test | Ceiling: 100% for every reader and source |
| 3 Fine-tuning | LoRA on 557 identical-question, process-correct, length-matched solutions per source; fresh operator step-by-step test | **Own − foreign −0.4 pp [−4.6, +3.7]**; Gemma's data best for all; GSM retention 98–99% |

Own fine-tuning data starts at 0.10–0.15 loss per token and barely moves; foreign
data starts at 0.32–1.09. Familiarity is real but did not improve learning.

Corrections: the "Answer:" extractor crossed line breaks ("Final Answer:\n\nAnswer:
7" -> "Answer: 7"); fixed in all modules on 6 October. It changed 140/4,860 arm-2
scores (removing a spurious +18 pp Qwen3-30B effect), 7/12,096 native-trace scores
and 125/53,248 practice checks. Transport fixes (timeouts, concurrency, a
preemption-driven move from H100 to H200) are listed in each design.json.
