# Learning from states reached by a model's own actions

The earlier successful-rollout distillation experiment discarded most mistakes.
This follow-up tests corrective supervision on learner-generated states, motivated
by [DAgger](https://arxiv.org/abs/1011.0686) and
[on-policy language-model distillation](https://arxiv.org/abs/2306.13649).
The hypothesis is that a learner's state distribution reveals errors and missing
coverage that expert demonstrations do not adequately train. Generation without
an informative teacher is not assumed to create information.

## Task and learner

The underlying modular planning dynamics are retained: moduli 29/31, increments
1 through 7, and actions add, subtract, or double. The interface is changed to a
closed-loop next-action policy. The model sees the full current state, goal and
transition rules, selects one action, and the simulator computes the next state.
Test success means reaching the goal within eight actions. State arithmetic,
trajectory formatting, and termination at the goal are handled by the simulator.
Scores therefore cannot be compared numerically with the earlier free-text task.

Qwen3-0.6B-Base is fine-tuned from its pinned pretrained revision with all weights
updated. Its three legal action-token logits define the conditional policy.
The target is uniform over every action that decreases exact BFS distance by one;
this removes arbitrary lexical tie-breaking as a training objective. Conditional
cross-entropy is also forward KL from this oracle distribution up to its entropy.
This is a constrained LM action policy, not unrestricted text generation or
policy-gradient reinforcement learning.

Whole goal contexts `(modulus, add, subtract, goal)` are hashed into 80% training,
10% validation and 10% test. Every current state within a goal context shares its
split, preventing rollouts from reaching held-out inputs during training. Initial
training states have shortest distance 3–4. Evaluation strata are distances 1–2,
3–4 and 5–6. The main test split is reserved while piloting on validation contexts.

## Acquiring and training on states

All arms share a seed-specific warm start: 2,048 expert-visited states, two shuffled
passes. Subsequent collection uses 128 parallel trajectories, reset on success or
after eight actions. Each round acquires 512 oracle-labeled nonterminal states.
Every state is retained, whether the executed action was optimal or erroneous.

- `expert` follows randomly tie-broken optimal actions.
- `on_policy` samples the current learner's actions at temperature 0.8.
- `random` executes uniformly random actions, controlling for generic exploration.
- `frozen_policy` samples the warm-start learner once to construct the whole state
  stream, controlling for adaptation during generation.

Each round trains its 512 new examples plus 512 examples drawn uniformly with
replacement from accumulated data, including warm examples and the new round.
These 1,024 examples are shuffled and trained in 16 batches of 64. All weights
and AdamW states use float32, with bfloat16 forward autocast, learning rate 3e-5,
zero weight decay and gradient clipping at norm 1. There is one target action
distribution per example, so label presentations and optimizer updates match.
This is incremental imitation learning with replay, inspired by DAgger; it does
not retrain to convergence on the full aggregate after every collection round.

Different policies finish episodes at different rates, so equal state-label
budgets do not equalize initial episodes, distinct contexts, prompt lengths or
compute. All acquisition rows, actions, oracle labels, state distances, replay
indices, minibatch order, counters and raw test trajectories are saved. Frozen
acquisition cost is paid in full before its first update. No cost-efficiency claim
can be based only on the matched label count. The goal split and evaluator are
identical across arms. Test evaluation is greedy; its actions never enter training.

## Pilot and decision rule

Before observing this task's outcomes, the pilot is fixed to seed 7310, 4,096
post-warm-start labels, 64 validation problems per stratum, all four arms, and
checkpoints 0, 512, 1,024, 2,048 and 4,096. It checks learnability, runtime and
distributional differences. Any changes prompted by the pilot must be recorded
and use a new output directory. Pilot results will be retained regardless of sign.

The intended continuation is three fresh training seeds and the separate test
split, extending to 16,384 labels if the task learns and runtime permits. Its exact
selection record is saved before any test predictions. The primary contrast is
on-policy minus expert ID success at the final label budget; random and frozen
controls are secondary. Learning curves and uncertainty are retained at every
prescribed checkpoint. No power-law exponent is assumed, and no fit window is
chosen to obtain a desired difference.

## Resources

Use the permitted institutional H100 allocation first. The user's authorized Vast
ceiling is $10, with on-demand rental and immediate destruction after use. No Vast
instance is rented unless institutional compute proves unsuitable. No new paid
OpenRouter calls are required. Model checkpoints use FAST_CACHE_DIR, with selected
models and all probes copied and verified into CACHE_DIR before job release.
Sources and dataset hashes are frozen per run. Previous experiments are unchanged.
