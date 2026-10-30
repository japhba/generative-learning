# Small students: own versus foreign GSM8K solutions

Frozen 6 October 2026 before inference. Rationale: all earlier studies used
capable 9–30B models; the literature's learnability gap suggests small models
learn better from reasoning closer to their own (own or similar-size) than from
strong models. Students: Qwen3-1.7B, Llama-3.2-3B-Instruct, SmolLM3-3B,
Phi-4-mini-instruct (four families); large sources Gemma 4 26B-A4B and
Qwen3-30B-A3B-Instruct-2507. Pinned revisions, vLLM 0.28, thinking off.

- Data: 2,000 GSM8K training problems (seeded shuffle); each model writes 4
  zero-shot solutions (T = 0.8, 1,024 tokens, final "Answer: <number>"). A problem
  is kept if all six models have a finished, numerically correct solution; each
  model contributes the one closest to the problem's median length.
- Fine-tuning arm: each student LoRA-fine-tuned (r = 16, α = 32, dropout 0, all
  linear layers, 2 epochs, lr 1e-4, batch 16, `msft_train.py`) on each source's
  matched set (identical problems); evaluated on 500 GSM8K test problems
  (step-by-step, greedy). Base models too.
- In-context arm: 300 of those test problems; reader = student; 4 worked
  examples (the same 4 matched training problems per test item for every
  source) or none; step-by-step, greedy.
- **Primary (each arm):** mean over students of acc(own) − mean acc(three peer
  students as sources). Secondary: own − mean(large sources); per student;
  initial training loss vs gain. Paired item bootstrap (20,000).

## First result and seed replication (frozen before replication training)

Seed 0: in-context own − peers −2.8 pp [−4.8, −0.8] (own − large −4.8 pp);
fine-tuning own − peers **+1.5 pp [+0.2, +2.9]** (Qwen3-1.7B +5.3, Llama +2.2,
SmolLM3 +3.3, Phi-4-mini −4.7), own − large −1.7 pp (Qwen3-30B data best for all
students). Item-level intervals ignore training-run variance, so all 24
fine-tuning runs are repeated with training seed 1 (LoRA initialisation and data
order; identical data and evaluation). Primary for the replication: own − peers
averaged over students using seed-1 runs alone; then pooled over both seeds with
run-level variability reported (per-student difference between seeds).
