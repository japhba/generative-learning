# Consolidating self-generated computation

The corrected elicitation pilot passed an independent 144-question arithmetic
confirmation: short greedy accuracy 0.7%, greedy reasoning 80.6%, and four-sample
plurality 88.2%. This establishes an inference-compute gap under the specified
prompts. It does not establish learning or an advantage from policy refresh.

The consolidation pilot starts Qwen2.5-1.5B-Instruct from the same pinned weights
in three conditions and three optimization seeds. It uses 96 new fixed arithmetic
programs, balanced across 4/6/8 steps, for four shuffled passes. A separate 144-question
test set is fixed before training. All conditions have identical question orders
within each seed. No task simulator or gold answer is consulted to create targets.

1. Frozen reasoning source: draw four new reasoning traces from the initial model
   for every scheduled practice presentation. Distill their plurality final answer
   into the short-answer prompt. Repeated questions get independently sampled
   candidate groups, not a reused target bank.
2. On-policy reasoning source: the identical procedure, using the current weights
   after each 16-question practice batch. Sampling seeds and batch sizes match the
   frozen condition; the source weights are the intended intervention.
3. Direct self-imitation: sample one current-model short response and train on its
   exact sampled tokens. This measures finite-batch self-imitation; it is not
   assumed to leave realized weights unchanged.

Reasoning targets use temperature 1, four 512-token attempts, explicit final-integer
parsing and first-occurrence tie breaks. No correctness or confidence filter is
used. Entirely invalid groups supply zero target tokens; all their sampling work
and practice presentations remain in the budget. Frozen-source draws can be
precomputed, but only the matching causal prefix is exposed during updates or
attributed to intermediate budgets.

Training uses full FP32 weights and AdamW state with BF16 autocast, learning
rate 1e-5, no weight decay, gradient clipping at 1, batch size 16 and four updates per batch.
Completion NLL is summed and divided by the fixed 8 * batch_size; random response
length does not change that normalization. The existing LM head is evaluated at
supervised positions only, with a pre-update check against the standard HF loss.
There is no added classifier or acquisition head. No hyperparameter is chosen
from the new held-out outcomes.

Checkpoint budgets 0/16/64/192/384 count practice presentations, with only 96 unique
questions. Both short-answer (96 tokens) and reasoning (512 tokens) accuracy are
evaluated at every checkpoint. Training-question predictions diagnose whether
the optimizer learned the supplied self-targets. The primary comparison is final
on-policy-minus-frozen-source short-answer accuracy. All candidate targets are
graded for diagnostic analysis only after the learning runs.

An exploratory ICL comparison uses the initial model and the first 4 or 16 frozen
practice examples from seed 68000. Context contains either their selected answers
or the corresponding actual generated reasoning traces. Both evaluation modes
are retained, including truncations. This is one demonstration sequence, not a
three-seed ICL replication or an assertion that prompt and weight learning have
identical dynamics.

Report all generation/training tokens, optimizer updates and timings separately.
The direct self-imitation condition has fewer rollouts and is a mechanism control,
not a matched-total-compute competitor. The main frozen/on-policy pair has matched
question, rollout-count and optimizer schedules, but variable response lengths
still require token accounting. No scaling exponents are fit in this pilot.

Source/data hashes are frozen before training. Final seed 68000 models are archived
to the configured shared cache. SLURM scripts remain under ignored results; use
only gpu_lowp and release GPUs when execution and archiving finish. No paid API
calls or Vast rentals are needed.
