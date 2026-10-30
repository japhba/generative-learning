# Correctness-matched laid-out reasoning: completed 5 October 2026

Protocol MLAYOUT_PROTOCOL.md; report reports/own-reasoning-report.html (section 3);
data results/task_search/mlayout, mlayout2, mlayout3. Models served from pinned
HF revisions with vLLM 0.28 on gpu_lowp (H100, then H200 after a preemption);
no OpenRouter spend.

Own and foreign practice: correct worked solutions to the same questions;
answer-only tests with an "Answer:" prefill; no rewriting.

| Phase | Matching | Episodes | Own − foreign S | Notes |
|---|---|---|---|---|
| 1 (4 models, operator+units) | correct final answer | 14/16 | +3.0 pp [+0.8, +5.2] | units at ceiling; Phi-4's "correct" solutions often nonsense |
| 2 (3 models, operator) | process-correct | 13/16 | +2.5 pp [−0.9, +5.7], p = 0.098 | Gemma +7.4, Qwens ±1 |
| 3 (confirmation) | process-correct + length-matched | 15/16 | −2.1 pp [−4.9, +0.6] | **primary Gemma own − other two +1.1 pp [−4.4, +7.5], p = 0.40** |

Post-hoc pooling of phases 1–2 had suggested a Gemma-specific preference
(+9.4 pp [+3.3, +15.4]); it did not replicate once solution length was matched
(Gemma's natural solutions are shortest). Qwen siblings never preferred each
other's solutions over cross-family ones beyond noise.

Conclusion: with matched correctness, process and length, in-context practice
shows no own-source advantage for these three models on this task.
Infrastructure: node-local /var/tmp on gpu-xd670-30 was full of the user's own
data (805 GB), so env/weights/caches live in /ceph/scratch/jbauer/mlayout;
serve.sbatch there (not version-controlled).
