# Teacher supervision with supplemental generated solutions

This follow-up tests whether verified model rollouts help as extra training while
ordinary teacher supervision continues. The prior replacement experiment is
retained unchanged in [TRAINING_RESULTS.md](TRAINING_RESULTS.md). Its rollout arm
trained on fewer problems and concentrated on repetitive strategies; this motivates
an intervention that gives every arm access to every canonical teacher solution.

The task, seeds 6200–6202, 4,096 new problems per seed, held-out identities, common
warm-start checkpoints, model revision, learning rate, optimizer, and decoding
settings are inherited from the prior experiment. This is an exploratory follow-up:
its design was informed by previously inspected test outcomes. Reusing those test
identities permits paired comparisons but does not supply independent confirmation.
All seeds and checkpoints are retained regardless of outcome.

## Two new arms and the historical baseline

For each acquisition batch of 32 problems, `teacher_rollout` first generates four
solutions per problem from the current model, before training on any teacher answer
from that batch. It retains the first valid completed solution, if any. Training
examples are then assembled in problem order: the canonical shortest teacher
solution, followed immediately by that problem's accepted generated supplement.
Every problem contributes its teacher solution, including those with no successful
rollout. Generated supplements preserve the model's accepted text.

`teacher_replay` uses exactly the same accepted problem identities and supplemental
positions, but replaces each generated supplement with the canonical teacher
solution. It therefore repeats the teacher solution on accepted problems. It
matches the rollout arm's complete problem-ID stream, example count, pending-buffer
boundaries, and optimizer updates. It inherits the rollout arm's adaptive selection;
it does not acquire its own curriculum.

Both arms accumulate the mixed stream into full minibatches of 16, with a partial
minibatch only at the final endpoint. The original full-weight training helper is
used unchanged: float32 parameters/AdamW state, bfloat16 forward autocast, completion
token-mean cross-entropy including EOS, learning rate 1e-5, no weight decay, and
gradient clipping to norm 1. Source snapshots and SHA-256 hashes are saved before
training, including the warm model file and exact parent dataset.

The previous ordinary SFT run is shown as a historical baseline at the same prompt
budgets. Its data, warm-start and training implementation match, but it has fewer
examples/updates than the new augmented arms. Historical phase times are contextual
measurements, not a simultaneous controlled hardware benchmark.

## What the contrast can establish

The primary comparison is paired in-distribution endpoint accuracy of
`teacher_rollout` minus `teacher_replay` at 4,096 new prompts. Long-horizon accuracy
is secondary. Curves at 0, 64, 128, 256, 512, 1,024, 2,048, and 4,096 prompts are
reported with seed variation. Three seeds provide limited uncertainty resolution.
No power-law exponent is assumed or selected after seeing the new curves.

The comparison tests supplemental generated versus teacher targets under continuing
teacher access and a shared accepted curriculum. It includes target length and
token weighting: an eight-move generated target contributes more supervised tokens
within a mixed minibatch than a two-move teacher replacement. It does not isolate
semantic content, exploration, generation compute, or architectural recognition.

Counters distinguish teacher and supplemental examples presented and actually
trained, pending examples, optimizer updates, input/completion tokens, rollout
tokens, and training/rollout phase time. The replay control's own phase time omits
the generation required to acquire its supplements. The original accounting limits
in [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md) still apply. The follow-up makes no
paid API calls and uses the permitted institutional GPU allocation.

Mechanism diagnostics include sampled completion diversity, canonical versus
alternative accepted paths, solution lengths, greedy action distributions, exact
constant-action oracle agreement, and the three original weight-displacement
probes. Weight probes compare update directions, not distance to true task weights.

## Supplemental evaluation on fresh identities

After the experiment had started and partial outcomes were available, an additional
test set was frozen before any predictions on that set. Its selection record is
`results/training_augmentation_fresh_eval/selection_record.json`, timestamped
2026-09-05 21:29:18 UTC. Using sampling seed 93201, it selects 256 problems without
replacement per stratum from the fixed held-out identity partition, excluding all
1,413 identities in earlier evaluation datasets. The same 768 problems evaluate
all nine final checkpoints: both new arms and historical ordinary SFT for each
training seed. It does not change the original primary endpoint.

The evaluator `evaluate_augmentation_fresh.py` uses greedy decoding with the
original 64-token cap, strict trajectory verification and EOS requirement. Model
parameters have gradients disabled and there are no updates. Dataset, selection,
source and model hashes, all predictions, and token counters are saved. The
selection-time source archive is retained; a separate evaluation source archive
is frozen before the first inference call and checked across all nine checkpoints.

Paired intervals resample the three training seeds, conditional on this common
test set. They do not include uncertainty from sampling the questions or supply
additional independent training replications. No subsequent model tuning or
checkpoint selection uses this supplemental evaluation.

## Reproduction and storage

The runner is `train_augmentation.py`; it imports the original training and task
helpers without modifying them. Set up the node environment outside the NFS project:

```bash
UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync --extra training --python 3.11
```

Use a fresh output directory for reproduction, repeat for each seed, and run the
generated arm before its replay control:

```bash
set -e
for mode in prepare teacher_rollout teacher_replay; do
  "$VENV_LOCAL/generative-learning/bin/python" experiments/train_augmentation.py "$mode" \
    --parent results/training_main_seed6200 --output results/augmentation_reproduction_seed6200 \
    --warm-checkpoint /ceph/scratch/jbauer/generative-learning/training/training_main_seed6200/warm
done
```

Without `--warm-checkpoint`, the runner uses the parent's recorded node-local warm
checkpoint. It requires identical warm-start evaluation metrics before any update.
Small run artifacts are stored in the project results directory. Model weights and
NPZ probes use `FAST_CACHE_DIR` from `~/.env`; the environment is loaded with
`python-dotenv`. Before observing new outcomes, seed 6200 is selected for shared
model-checkpoint archival, together with all three seeds' weight probes. The
completed archive also preserves every seed's parent warm-start checkpoint, so all
three initializations remain reproducible from shared storage. Seed 6200's existing
warm archive was reused and verified against its recorded model hash; missing warm
archives for seeds 6201 and 6202 were copied, regardless of experiment outcomes.

Archival completed at 2026-09-05 21:57:10 UTC after all training and fresh evaluation.
`results/training_augmentation/archive_results.py` verified every file list, size and
SHA-256 under `CACHE_DIR/generative-learning/training`: 98 files, 12,824,094,386 bytes,
including all 48 new probe files. Of these bytes, 10,423,975,647 were copied and
2,400,118,739 reused. Exact paths and hashes are in
`results/training_augmentation/archive_manifest.json`. New final model weights are
archived for preselected seed 6200 only; seeds 6201 and 6202 retain their final models
in node-local fast cache, whose persistence is not guaranteed. All three seeds'
warm starts and probes are in shared storage. Checkpoints do not contain optimizer
state. The GPU allocation was released after this verification.
