# Teacher fine-tuning versus verified model rollouts

The full-weight Qwen3-0.6B-Base comparison is complete. At 4,096 new prompts,
ordinary teacher SFT outperforms verified rollout distillation in all three seeds
and every evaluation stratum. Rollout training has a small, inconsistent advantage
over its matched teacher control. Sampled solution diversity declines substantially,
and the final greedy rollout policies use repetitive addition/subtraction strategies.
The curves do not establish a training scaling law.

The frozen main comparison uses three seeds, three training arms, and 4,096 new problems per arm.
The protocol is in [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md), with the separate
ICL replication in [REPLICATION_RESULTS.md](REPLICATION_RESULTS.md).

## What was trained

All 596,049,920 pretrained parameters were fine-tuned. Each problem supplies rules
for modular addition, subtraction, and doubling, then asks for a valid path from
start to target within eight moves. The exact verifier checks every state, the
target, the full output format, the horizon, and completion with EOS. Held-out
identities never enter training. Evaluation contains 128 problems in each of
three shortest-distance strata per seed, with greedy decoding and no worked
examples in the prompt.

Each seed starts with 2,048 teacher examples trained for two epochs. Its identical
warm checkpoint initializes all three arms:

- Teacher SFT learns the canonical shortest teacher solution for every new problem.
- Verified rollout distillation samples four fresh current-model solutions and
  learns the first valid completed one, if any. There is no teacher fallback.
- Yoked teacher SFT learns the canonical solution for exactly the problems accepted
  by the rollout arm, matching its example order and optimizer minibatches.

Teacher SFT is also an autoregressive generative objective. The experiment isolates
aspects of the **source of training targets**, not recognition versus generation
as architectural categories. The generated arm is rejection distillation using a
verifiable reward; it is not an implementation of PPO or GRPO.

## Main outcomes

Mean exact held-out accuracy at the final endpoint:

| Training condition | Easy, 1–2 moves | In distribution, 3–4 | Long horizon, 5–6 |
| --- | ---: | ---: | ---: |
| Shared warm-start, before new prompts | 25.26% | 15.10% | 7.81% |
| Teacher SFT | 47.40% | 35.68% | 20.57% |
| Verified rollout distillation | 34.38% | 23.70% | 14.06% |
| Yoked teacher SFT | 33.07% | 21.88% | 13.54% |

The final in-distribution accuracies, without pooling seeds, are:

| Seed | Teacher SFT | Rollout distillation | Yoked teacher SFT |
| --- | ---: | ---: | ---: |
| 6200 | 35.94% | 28.12% | 18.75% |
| 6201 | 27.34% | 17.97% | 22.66% |
| 6202 | 43.75% | 25.00% | 24.22% |

The rollout-minus-SFT difference averages **−11.98 percentage points** in
distribution and **−6.51 points** on long problems. Both differences are negative
in every seed. Relative to the yoked control, rollout differences average only
**+1.82 points** and **+0.52 points**, respectively, with inconsistent signs across
seeds. In-distribution rollout-minus-yoked differences are +9.38, −4.69, and +0.78
points. This does not establish a consistent benefit from the generated targets.

The early mean in-distribution rollout curve is higher than ordinary SFT at several
prompt budgets: at 128 prompts, 20.57% versus 16.67%; at 1,024, 21.61% versus
20.05%. Those are prompt-count comparisons, not equal-compute comparisons. By
4,096, SFT is substantially higher. All nine mean arm-by-stratum curves contain
declines between checkpoints. We retain these irregular curves and do not fit
training power-law exponents.

### Samples and compute

Costs below are means per seed for the post-warm-start training/rollout phases:

| Arm | New prompts | Actually trained examples | Optimizer updates | Counted tokens | Phase seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher SFT | 4,096 | 4,096 | 256 | 469,009 | 23.71 |
| Rollout distillation | 4,096 | 1,598.67 | 100.33 | 2,671,439 | 350.45 |
| Yoked teacher SFT | 4,096 | 1,598.67 | 100.33 | 182,147 | 10.25 |

Rollout acceptance counts are 1,242, 1,798, and 1,756: 39.03% of prompts overall.
Each seed samples 16,384 candidate trajectories. Yoked SFT matches these accepted
examples and update boundaries exactly. Rollout training uses **14.78×** the
recorded phase time and **5.70×** the counted tokens of ordinary SFT at the endpoint.

These are measured phase costs, not end-to-end job durations or FLOPs. They exclude
evaluation, model loading, checkpoint writing, and common warm-start. The warm-start
itself uses 4,096 example presentations, about 469,284 counted tokens, and 24.45
training seconds per seed. Teacher BFS construction is outside the training timer;
generation timing includes decoding and verification. Yoked phase cost excludes
the rollouts required to obtain its curriculum. The protocol gives the complete
accounting conventions.

### Observed mechanism: narrow strategies, accurate local arithmetic

Sampled diversity decreases in every seed. Each entry below compares the first and
last 128 new prompts, with four sampled completions per prompt; surrounding
whitespace is ignored when counting distinct completions.

| Seed | Initial distinct completions | Final distinct completions | Final greedy action pattern |
| --- | ---: | ---: | --- |
| 6200 | 3.83 | 1.02 | Only B (subtract) |
| 6201 | 3.74 | 2.41 | One or two As, then Bs |
| 6202 | 3.95 | 1.95 | Only B (subtract) |

No final rollout output uses C (double). Ordinary teacher SFT uses all three actions
in every seed. Yoked SFT uses all three in seeds 6200 and 6201; its seed 6202 endpoint
uses only A. Action concentration therefore also occurs with teacher targets under
the selected curriculum. The action-pattern diagnostic does not imply that malformed
or unfinished outputs are valid trajectories. Strict task grading remains unchanged.

Seed 6200 gives the clearest mechanistic result. Its final success/failure vector
matches an exact **repeat-B-until-target-or-eight-moves** oracle on **384/384** test
problems. Its 4,181 parsed moves all use B; arithmetic is correct within the
eight-move evaluation horizon. Of 283 failures, 282 execute eight correct moves without reaching the target and one
stops early. There is no case where it reaches the goal within eight correct moves
and then loses credit by continuing or failing to terminate.

Seed 6202 matches the B-only oracle on **379/384** cases; the five extra misses are
two arithmetic errors and three early stops. Of 279 failures, 258 exhaust a correct
eight-move horizon without the goal, eight fail arithmetic, and 13 stop early.
Seed 6201 has 313 failures: 284 exhaust a correct horizon, 28 fail arithmetic, and
one reaches the goal but fails to emit EOS. Thus most endpoint failures concern
planning within the allowed horizon, not an inability to execute local arithmetic
or merely a too-short output-token cap.

For context, the exact B-only oracle over the complete hashed held-out pools has
accuracy 35.39%, 23.69%, and 19.81% in the three strata. These differ from a naive
uniform-target approximation because each stratum conditions on shortest-path
distance. Individual evaluation samples have their own exact baseline values.

### Weight displacement

The saved probes measure `delta_W = W_final - W_warm` in attention query
projections at layers 0, 14, and 27. Mean relative displacement norms are
0.47%/0.58%/0.47% for teacher SFT, 0.27%/0.28%/0.36% for rollout distillation, and
0.29%/0.31%/0.31% for yoked SFT. All arms change these weights.

Mean cosine similarities between flattened displacements are:

| Pair | Layer 0 | Layer 14 | Layer 27 |
| --- | ---: | ---: | ---: |
| Teacher SFT vs rollout | 0.090 | 0.073 | 0.244 |
| Rollout vs yoked teacher | 0.181 | 0.116 | 0.298 |
| Teacher SFT vs yoked teacher | 0.303 | 0.224 | 0.369 |

These show distinct update directions in the three sampled matrices. They are
neither full-network overlap nor distance to known true task weights. The
rollout/yoked pair matches example counts and optimizer steps; the full SFT pair
does not. No causal relation between these angles and generalization is established.

## Retained pilot outcomes and protocol selection

The original pilot, seed 6100, used 256 warm examples for four epochs, 512 new
problems, and a post-warm-start learning rate of 3e-5. Each evaluation stratum
contained 64 held-out problems. Its rollout arm accepted only 59/512
problems, trained in 11 optimizer updates, and ended at zero in-distribution and
long-horizon accuracy. The yoked arm also ended at zero on those strata; ordinary
SFT ended at 10.9% and 9.4%. These are retained negative results, not discarded
attempts at finding a favorable exponent.

The original implementation trained each acquisition batch immediately, including
singleton batches when only one rollout succeeded. Comparing the first and last
128 training prompts, each with 512 sampled candidates, mean distinct completions
among four samples declined from 3.68 to 1.73; surrounding whitespace is ignored.
First-transition accuracy fell from 407/512 (79.5%) to 206/512 (40.2%), counting an
unparseable first move as incorrect. All final-window outputs began with the
symbol B, although five had an unparseable first move. In the final greedy
evaluations, 171/192 rollout outputs had a wrong transition, while 181/192 yoked
outputs lacked EOS before the token limit. The same final accuracy did not imply
the same failure mechanism.

A separate warm-start-only assay, seed 6101, used 2,048 examples for two epochs.
It reached 23.4%, 12.5%, and 9.4% accuracy in the easy, in-distribution, and long
strata. No three-arm experiment was run for that assay.

Before the main arms, the protocol was frozen with that larger common warm-start,
a lower main learning rate of 1e-5, and accumulation of accepted examples into
full minibatches of 16. Only the final minibatch can be partial. All three main
seeds (6200–6202) are retained, with no intermediate hyperparameter changes.
The selection record and original source snapshots remain with the artifacts.

## Interpretation and limits

The separate ICL Zipf experiment concerns which problems the model chooses to
observe. Here, all arms receive the same externally sampled problem stream;
generation explores solution trajectories. A scaling exponent in the former does
not predict one in the latter.

The yoked control removes the difference in accepted problem identities and update
counts between teacher and generated targets. It does not equalize solution
lengths, completion tokens, or curriculum acquisition costs. The ordinary SFT arm
has access to a teacher solution even on problems where the rollout arm fails.
These comparisons answer different questions and should be reported together.

The probabilistic success-gradient identity and its implementation limits are in
the protocol. [STRATEGY_DYNAMICS.md](STRATEGY_DYNAMICS.md) gives an interpretable
softmax-strategy toy: success filtering reweights existing strategy mass by its
success probability. With fixed strategy competence and exact distribution fitting,
that narrowing improves the toy's own success probability. It does not by itself
explain harmful generalization or establish the causal mechanism in this LLM.

Three seeds provide limited uncertainty resolution. Bootstrap bands resample
paired seeds and are pointwise; they should not be read as precise population
confidence statements. The task is synthetic, small, and restricted to one model
and one main training protocol. Neither these curves nor the sampled weight probes
establish universal learning-rate exponents or superiority of generation.

## Validation and artifacts

All 29 offline tests pass. The completed-run audit verifies all 72 main metric
checkpoints, immutable source hashes, train/test separation, identical warm-start
metrics across arms, exact decoded grading plus EOS, retained rollout provenance,
token totals, and matched rollout/yoked minibatches. An independent endpoint audit
recomputes the final accuracies from all 3,456 saved endpoint verdicts. All 189
weight-cosine comparisons are available, with no missing probes.

Small configurations, datasets, source snapshots, predictions, candidates, metrics,
and analysis live under `results/training_main_seed6200`, `...6201`, `...6202`, and
`results/training_main_analysis`. The protocol selection and independent audits
are in `results/training_main`.

Seed 6200 was selected for model archival before the main outcomes. Its warm and
three final model/tokenizer checkpoints, plus every seed's weight probes, are
preserved under `/ceph/scratch/jbauer/generative-learning/training`. The archive
contains 118 files and 10,855,462,147 bytes, with source/copy SHA-256 verification
recorded in `results/training_main/archive_manifest.json`. Model weights and
tokenizers are saved; optimizer state is not. Other seed checkpoints remain in
the node's fast cache. GPU allocation 3529224 was released after training,
analysis, and archival.

The training leg used the permitted institutional `gpu_lowp` H100 allocation and
made no paid inference/fine-tuning API calls. The separate OpenRouter replication
added $0.191703; conservative cumulative API accounting remains **$0.37499892**,
including the older unresolved reservation. This is API accounting, not a monetary
valuation of institutional GPU time.

Useful figures:

- `results/training_main_analysis/in_distribution_problems_presented.png`
- `results/training_main_analysis/in_distribution_phase_seconds.png`
- `results/training_main_analysis/long_horizon_problems_presented.png`
- `results/training_main_analysis/rollout_diversity.png`
