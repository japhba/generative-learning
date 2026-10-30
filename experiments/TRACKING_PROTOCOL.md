# State coverage and corrective control

The modular feedback pilot is retained as a negative learnability result: all four
final policies repeat a constant action. Its differences do not demonstrate learned
planning. The new tracking experiment deliberately makes state excitation and the
missing corrective direction explicit. This task change is exploratory; confirmation
uses fresh training seeds and fresh evaluation command sequences after the pilot.

## Dynamics and hypothesis

The state is tracking error `e`, the requested motion `u` is uniform in -2..2, and
action `a` is an integer in -3..3. Dynamics are `e_next = e + a - u`. The expert
chooses `clip(u-e,-3,3)`, the unique action minimizing the next absolute error.
Every 16-step training episode starts at zero error. Expert trajectories therefore
stay exactly at zero error, exposing only five `(u,e)` inputs. They cannot identify
the corrective dependence on error from data alone. A learner executing exploratory
actions can visit nonzero errors and request the expert's corrective labels there.

The LLM prompt exposes the exact dynamics, objective and seven action-letter
meanings. This tests acquired behavior given pretrained knowledge and supervision,
not inference of a secret controller formula. Qwen3-0.6B-Base uses conditional
cross-entropy over its seven legal next-action-token logits; all weights are updated.
Simulator arithmetic and the 16-step interface are fixed across all conditions.

## Controls and training

Four state-collection policies are compared: expert; current learner; uniform random
actions; and frozen warm learner. Learner collection samples from
`0.9 * softmax(model_logits) + 0.1 * Uniform(7)`. The explicit exploration component
keeps acquisition informative after the nominal task is mastered; its contribution
must be acknowledged in any generation claim. Frozen collection samples its whole
dataset before updates, with the same mixture. Greedy decoding is used at evaluation.

Each seed has a shared expert-only warm start, 2,048 labels trained for two shuffled
passes. All phases use full FP32 model weights and AdamW states, BF16 autocast,
learning rate 1e-4, zero weight decay, and gradient clipping at norm1. After warm-up,
each round acquires 512 new state labels and trains them plus 512 uniform draws from
the accumulated data, including warm data, in a shuffled stream of 16 batches64.
This matches acquired labels, trained examples, target tokens and optimizer updates;
state support, input token counts, and acquisition compute differ and are recorded.

The label is always the oracle action at the actually visited state. Errors are
retained and corrected, rather than relabeled as successful generated solutions.
All acquired rows, actions, replay indices, stream order, counters and evaluation
trajectories are saved. Source/data hashes are frozen before each run.

## Evaluation and decision rule

Fresh command sequences have length16. Three initial-state conditions are fixed:
nominal `e0=0`, recovery uniform over +/-1..3, and larger shifts uniform over +/-4..8.
The primary metric is mean absolute tracking error over the16steps on recovery
episodes, averaged across episodes. Secondary metrics include nominal error,
larger-shift error and whether all final8steps have zero error. This is an explicit
robustness intervention: recovery initial states are outside expert demonstrations.
The fresh sequences are held out; individual Markov inputs may repeat across
training and test, as in policy evaluation and the earlier dictionary-coverage task.
No claim of generalization to entirely unseen state-action inputs is assumed.

Pilot: seed7510, 2,048 post-warm labels, evaluation seed94300,64episodes/condition,
all four arms, checkpoints0/512/1024/2048. All pilot outcomes are retained. If the
task is learnable, freeze a separate three-seed run with8,192labels and fresh command
sequences before observing confirmation predictions. The intended primary contrast
is current-learner collection minus expert collection in recovery error at the final
budget; random and frozen controls determine whether adaptive generation has any
advantage over generic exploration. No power-law fit is assumed or selected.

Compute uses permitted institutional H100 allocation3529401. No Vast instance or
paid API call has been used. The $10 Vast authorization remains a ceiling if needed.

## Recorded pilot adjustments and frozen confirmation

The original seed7510 pilot at LR1e-4 did not reliably retain nominal behavior.
A calibration run, seed7511 at LR3e-5, reached perfect accuracy on the five nominal
training inputs, but further expert-only updates could destroy that accuracy.
Both complete four-arm runs and their source versions are retained. Neither is
used as confirmation of an exploration effect.

The stability pilot, seed7512, changes LR to1e-5, uses label smoothing0.1, and carries
the warm optimizer moments into every arm. Its warm-up stops after two consecutive
epochs with perfect greedy predictions on the five nominal training inputs, with
a maximum of20epochs; this criterion never consults test outcomes. The smoothing
target is90%one-hot oracle plus10%uniform over seven actions. The stability pilot
uses4,096labels and fresh evaluation seed94302. Expert nominal performance stays
perfect at every checkpoint; on-policy recovery error reaches the oracle range.

Before confirmation predictions, the main plan is fixed to training seeds7520,
7521,7522, all four arms,8,192acquired labels, evaluation seed94400 and256fresh
episodes per condition. Checkpoints are0,512,1024,2048,4096,8192. The stable optimizer
and calibration procedure are unchanged. All arms receive the identical command
stream within a seed; only their visited errors and resulting oracle labels differ.
Every confirmation seed/checkpoint is retained. Its primary contrast remains
on-policy minus expert recovery mean absolute error at8,192labels. Random and
frozen-policy comparisons remain secondary controls, irrespective of their signs.

Source hashes and this decision are recorded in
`results/feedback_experiment/tracking_confirmation_selection.json` before any
confirmation predictions. Nominal mastery is checked using training inputs only.
The same fresh episode set across training seeds supports paired comparisons but
does not create additional independent training replications. Intervals resample
training seeds and are conditional on the common test set.

## Replication in another model family

Before any second-family predictions, SmolLM2-135M was selected for its different
family and smaller size. The exact revision is
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; see its
[model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
`train_tracking_smollm.py` changes only model loading and provenance constants.
Its acquisition, training, warm calibration and evaluator functions match the
Qwen runner, verified through Python AST comparison. The same three seed IDs,
command streams,8,192label budget, four arms and evaluation episodes are used.
No architecture-specific tuning is planned. The selection record is
`results/feedback_experiment/smollm_replication_selection.json`.

A fixed29input behavioral probe is separately specified before main runs finish:
`u=-2..2`, `e=-3..3`, retaining unsaturated inputs `abs(u-e)<=3`. Least-squares
regression of decoded action on `(1,u,e)` measures effective control coefficients.
The ideal coefficients are `(0,1,-1)`, while the expected action under smoothed
teacher probabilities has coefficients `(0,0.9,-0.9)`. Every warm and final checkpoint
is probed. These are behavioral response coefficients, not raw transformer weights
or evidence of a unique internal circuit. Its selection record is
`results/feedback_experiment/controller_probe_selection.json`.

The SmolLM2 replication uses a second permitted institutional H100 allocation,
3529436, to run alongside the Qwen experiment. No rented instance is used. Timings
are measurements on a shared cluster node rather than isolated hardware benchmarks.

## Completion and retained artifacts

Both three-seed replications completed every arm and checkpoint. Independent
analyses verified acquired labels, matched command streams and update counts, all
test trajectories, and disjoint training/test command sequences. Supplemental
behavioral and closed-domain policy evaluations are documented, including the
retained numerical-padding failure, in
[FEEDBACK_INVESTIGATION_RESULTS.md](FEEDBACK_INVESTIGATION_RESULTS.md).

All main final models, common warm models and optimizer states, and all 144 probe
files are archived under `/ceph/scratch/jbauer/generative-learning/tracking`, with
420 file checksums in the two family archive manifests. Both GPU allocations have
been released. No Vast instance was rented and no new paid API calls were made.
