# Arm 3: own versus foreign solutions as fine-tuning data

Frozen 6 October 2026 before training. Question: does a model learn more (in its
weights) from its own correct worked solutions than from other models' equally
correct solutions to the same questions? Reported for SFT in the literature
("I learn better if you speak my language"; self-distillation reduces
forgetting); untested here so far.

- Students = sources: Qwen3.5-9B, Qwen3-30B-A3B-Instruct-2507, Gemma 4 26B-A4B.
- Data (`msft_data.py`): every pool question of the two process-matched cohorts
  (mlayout2, mlayout3; 768 questions) where all three models have a
  process-correct solution (557 questions); each model's solution length-matched
  (closest to the question's median). Identical questions for every source.
- Training (`msft_train.py`): LoRA r = 16, α = 32, dropout 0 (PEFT's wrapper for
  fused MoE expert weights rejects dropout; set to 0 for every student before any
  run completed), all linear layers for Qwen3.5-9B and Gemma 4; **attention
  projections only for Qwen3-30B-A3B** (all-linear wraps its fused experts, ~1B
  trainable parameters, out of memory; the same choice for all three of its sources)
  layers, 2 epochs, lr 1e-4 cosine with 10 warm-up steps, effective batch 16,
  ≤ 3072 tokens, one sequence per micro-step (first attempt ran out of memory
  computing full-vocabulary logits); loss only on solution tokens (inference prompt + solution +
  end-of-turn markup). Loss curve logged per step; resumable per epoch. Adapter
  merged for evaluation. One run per student × source (no seed replication).
- Evaluation (`msft_eval.py`, vLLM, T = 0, thinking off) on fresh seeds
  (`msft_sets.py`): operator problems (24 episodes × 8) answer-only with
  "Answer:" prefill and step-by-step (2048 tokens); alien-units answer-only
  (transfer, 128); GSM-symbolic step-by-step (retention, 120). Base models too.
- **Primary:** mean over students of acc(trained on own) − mean acc(trained on
  the two others), operator step-by-step. Secondary: the same for operator
  answer-only, units and GSM (forgetting); per-student contrasts; loss curves.
  Item-level bootstrap only (training-seed variance unmeasured).
