# Fine-tuning and verified rollout distillation

This protocol compares learning from teacher solutions with learning from a model's
own verified trajectories. It updates all weights of a pretrained causal LLM; it is
separate from the frozen-model Zipf in-context learning experiment. Teacher SFT is
itself autoregressive generative modeling. The contrast here concerns how training
targets are obtained, not a generative architecture versus a recognition classifier.

The implementation is `train_planning.py`, with the exact task and verifier in
`training_task.py`. This document specifies the main experiment; it does not assert
an outcome or a scaling law. Empirical findings are reported in `TRAINING_RESULTS.md`.

## Task and held-out evaluation

Each problem describes a finite planning system. States are integers modulo 29 or
31. Move A adds a specified integer, B subtracts a specified integer, and C doubles
the current state; the addition and subtraction parameters range from 1 through 7.
The prompt supplies these rules, a start, a target, and a maximum of eight moves.
The model must output a trajectory such as `A 12; C 24; B 19`, including the state
after every move. It receives no worked examples in the evaluation prompt.

A breadth-first search produces the canonical shortest teacher solution, breaking
ties in A/B/C order. The verifier accepts any valid trajectory within the horizon,
including longer or alternative solutions. It checks the entire decoded answer's
format, every claimed state transition, and the final target. The runner additionally
requires EOS: a valid-looking prefix cut off at the 64-token limit is rejected.

An identity hash assigns each parameter/start/target tuple permanently to an 80/20
train/test split. Changing a dataset seed never moves an identity across that split.
Within each seed, warm-start and subsequent training examples are disjoint samples
without replacement. Both use shortest distances 2–4. Evaluation uses 128 held-out
identities per stratum:

| Stratum | Shortest solution length | Interpretation |
| --- | --- | --- |
| Easy | 1–2 moves | Short problems, including distance 1 absent from training |
| In distribution | 3–4 moves | New identities at training depths |
| Long horizon | 5–6 moves | Generalization to greater shortest-path depth |

Evaluation is greedy and uses the same identities at every checkpoint and across
arms within a seed. Its random state is isolated from training rollouts. Evaluation
questions and outputs are never used as training data. Different seeds can share
some sampled identities, while their training data and sampling randomness differ.

## Model, warm-start, and three arms

The model is `Qwen/Qwen3-0.6B-Base`, pinned to revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd` (596,049,920 parameters).
Every parameter is trainable. Parameters and AdamW state use float32; forward and
loss computation run under bfloat16 autocast with native SDPA. Autocast retains
float32 for some operations; backward runs outside the autocast context using the
forward operation dtypes, and parameter gradients accumulate in float32. AdamW has
zero weight decay, and gradients are clipped to norm 1. There are no adapters or frozen layers.
The loss is next-token cross-entropy on completion tokens and EOS; prompt and
padding tokens have no loss. Its mean is over supervised tokens in the minibatch.

For each of seeds **6200, 6201, and 6202**, the common warm-start uses **2,048 unique
teacher examples for two epochs**, at learning rate **3e-5** and batch size 16.
Each arm then loads that seed's identical warm checkpoint and starts a fresh AdamW
optimizer at learning rate **1e-5**. Warm-start cost is reported separately. The
main stream contains **4,096 new prompts**, presented in batches of 32 in the same
order for all arms. The rollout arm explores alternative solution sequences for
these supplied problems; it does not choose the problem stream. This differs from
the Zipf ICL experiment's active selection of which keys to query.

| Arm | Acquired training targets | Main comparison |
| --- | --- | --- |
| `sft` | Canonical teacher solution for every presented prompt | Fixed-supervision baseline |
| `rollout_distill` | First verified success among four model samples, if any | Learning from generated trajectories |
| `yoked_sft` | Canonical teacher solution for exactly the prompts retained by `rollout_distill` | Control for selected prompts and update counts |

For rollout distillation, the current model generates **K=4** samples per prompt
at temperature **0.8**, `top_p=0.95`, and `top_k=0`, with at most 64 new tokens.
Generation minibatches contain 64 sequences. All four candidates are generated;
the first valid completed trajectory is retained in candidate order. There is no
ranking by path length, no teacher fallback, and no training on failed trajectories.
The target preserves the accepted generated trajectory rather than replacing it
with the teacher. The runner decodes with special tokens removed, strips boundary
whitespace, then tokenizes a single leading space plus the completion and appends
EOS. Thus the training token sequence is reconstructed from generated text, not
copied directly from the original sampled token IDs.

Accepted examples accumulate into full optimizer minibatches of **16**. At most 15
examples remain pending between query batches; a partial minibatch is flushed only
at the end of the run. Thus intermediate evaluation can observe accepted examples
that have not yet caused an update. Rollouts are acquired from the model current at
each query batch; buffered examples are not regenerated before each optimizer step.

The yoked arm replays the rollout arm's retained prompt identities and ordering,
replacing only their completions with canonical teacher solutions. It matches the
accepted-example count, minibatch boundaries, and optimizer-update count. It does
not match solution lengths, supervised-token counts, or the teacher and generated
trajectory distributions. Its own cost excludes the rollout computation needed to
construct this curriculum, so it is an information control rather than a separately
costed acquisition system.

## Probabilistic interpretation and its limits

For a fixed prompt x, let `pi_theta(y | x)` be the model's distribution over complete
sequences, `R(x,y)` a parameter-independent binary verifier, and `S_theta(x)` the
probability of success. For `S_theta(x) > 0`, under the usual conditions for
differentiating the sum,

```text
S_theta(x) = sum_y pi_theta(y | x) R(x,y)
grad_theta log S_theta(x)
    = E[grad_theta log pi_theta(y | x) | R(x,y)=1].
```

Thus, maximizing the unnormalized sequence log probability of an ideal untempered
on-policy success sample, treated as a fixed target, supplies an unbiased per-prompt
ascent gradient of log success probability. This statement uses sequence weighting
without token-length normalization.
For K independent samples, the first success conditional on any success has the
same success-conditioned distribution. Retaining at most one example per prompt
also weights prompt inclusion by `1 - (1 - S_theta(x))^K`; it does not make all
prompts contribute equally to an unconditional training objective.

The implemented method differs from that ideal identity: temperature and nucleus
sampling alter the proposal distribution; token-averaged cross-entropy weights
sequence lengths through the minibatch loss; decoding and boundary normalization
can change the training token sequence; and acquisition batches plus the small
pending buffer separate sampling from some subsequent updates. AdamW and
gradient clipping further alter parameter displacements. We therefore call this
**verified rejection distillation**, with fresh current-model rollouts at each
acquisition batch. It uses a verifiable success signal but is not GRPO, PPO, or a
claim of an exact RLVR policy-gradient implementation. The identity motivates the
comparison without proving that these training dynamics optimize the ideal objective.

## Learning curves, accounting, and weight measurements

Main checkpoints occur after 0, 64, 128, 256, 512, 1,024, 2,048, and 4,096 new prompts.
The primary outcome is exact held-out trajectory accuracy, separately by stratum.
Record curves against several budgets rather than treating them as interchangeable:

- New prompts presented, including prompts for which no rollout succeeds.
- Accepted examples, examples actually trained on, and pending examples.
- Optimizer updates and supervised completion/EOS tokens.
- Unpadded training input tokens plus rollout prompt and completion tokens.
- Training plus rollout phase wall time on the allocated GPU node.

Training-token counts include the prompt's forward computation even though its
positions have no loss. Token counts are not FLOPs: they omit padding overhead and
do not equate forward generation with forward/backward training. Phase time includes
CPU preparation and generation orchestration; the generation timer also includes
decoding and verifier work. Teacher-solution construction occurs before the training
timer and is excluded. Phase time is not isolated GPU kernel time or total elapsed
experiment time: model loading, checkpoint writing, and analysis are outside these timers.
The comparison axes exclude evaluation and common warm-start, whose usage is saved
separately. Token accounting does not assign teacher BFS or verification an LLM-token
cost. Full system accounting would also charge the yoked arm for acquiring its curriculum.

`analyze_training.py` aggregates paired arms by training seed and reports the
availability of each checkpoint. Its bootstrap resamples seeds, preserving pairing.
Three seeds give limited uncertainty resolution; checkpoint intervals are pointwise.
The optional exponent fit is a descriptive log-error versus log-new-prompts slope.
It does not establish an asymptotic law, an error floor, or a better fit than other
curve families. Retain irregular and negative results rather than selecting only
apparently straight curves.

The runner saves actual weight displacement arrays for the attention query
projection matrices in layers **0, 14, and 27**:
`model.layers.{layer}.self_attn.q_proj.weight`. Main-arm displacements are
`delta_W = W_checkpoint - W_warm`; warm-start displacements use the pretrained base
as their reference. Norms, relative norms, and optional pairwise cosines of flattened
main-arm displacements provide a small mechanistic diagnostic. These are three
matrices from a fully trained network, not a complete description of its changes.
They measure agreement between update directions, **not overlap with known true
task weights**, which this planning task does not specify. They cannot alone show
that one arm moves closer to a ground-truth representation or causes an accuracy gain.

## Reproduction and artifact storage

Run from the project directory on an allocated permitted GPU node. The tested
environment uses Python 3.11, PyTorch 2.10.0+cu128, Transformers 4.57.6, and an H100
80 GB. Create the environment outside the NFS project directory:

```bash
UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync --extra training --python 3.11
```

The code loads `~/.env` through `python-dotenv` and respects `FAST_CACHE_DIR` from
the environment. `HF_HUB_CACHE` is set to
`$FAST_CACHE_DIR/generative-learning/hf`. Model checkpoints and compressed weight
probes go under `$FAST_CACHE_DIR/generative-learning/training/<output-directory-name>`.
These original paths are node-local. After training, seed 6200's warm checkpoint and
all three final model/tokenizer checkpoints were archived at
`/ceph/scratch/jbauer/generative-learning/training/training_main_seed6200`.
Seed 6200 was selected for model archival before the main outcomes. Weight probes
for all three seeds were also archived under their respective sibling run directories.
The archive contains 118 files totaling 10,855,462,147 bytes, with source/destination
checksums verified and recorded in `results/training_main/archive_manifest.json`.
The saved checkpoints contain model weights, tokenizer files, and model/generation
configuration; **optimizer state is not saved**. Archival was performed separately
after the run, not automatically by the training runner.

Each output directory contains the exact sampled dataset, configuration, source
hashes, metrics, full evaluation predictions, and rollout candidates/selection logs.
The suite also copies `train_planning.py` and `training_task.py` into its `source/`
subdirectory. The model revision, locked dependencies, and source snapshots support
reproduction; bitwise identity across hardware and library stacks is not guaranteed.

The current three-seed suite is available at `results/weight_training/run_main.py`:

```bash
"$VENV_LOCAL/generative-learning/bin/python" results/weight_training/run_main.py
```

That launcher uses the existing main output names. It skips modes with completion
markers but is not a mid-arm checkpoint-resume implementation. An interrupted arm
with existing metrics fails loudly; inspect partial results before arranging a rerun.
For an independent reproduction, choose fresh output directories and run the modes
in this order, passing the same arguments each time:

```bash
set -e
training_args=(--output results/training_reproduction_seed6200 --seed 6200
  --train-size 4096 --warm-size 2048 --warm-epochs 2
  --warm-learning-rate 3e-5 --learning-rate 1e-5
  --eval-per-stratum 128 --generation-batch-size 64
  --batch-size 16 --query-batch-size 32 --rollouts 4
  --temperature 0.8 --max-new-tokens 64)
for mode in prepare warmup sft rollout_distill yoked_sft; do
  "$VENV_LOCAL/generative-learning/bin/python" experiments/train_planning.py "$mode" "${training_args[@]}"
done
```

Repeat with seeds 6201 and 6202 and distinct output directory names. The standalone
CLI defaults describe the smaller pilot, so the explicit main arguments matter.
The runner rejects a configuration or code-hash change within an existing run.
After the main experiment, analyze the three directories together:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_training.py \
  results/training_main_seed6200 results/training_main_seed6201 results/training_main_seed6202 \
  --output results/training_main_analysis --weight-cosines \
  --weight-probe-root /ceph/scratch/jbauer/generative-learning/training \
  --full-pool-baselines
```

`--weight-probe-root` resolves probes from the archive while preserving the original
paths in the run records. All 72 main-arm NPZ files are available; analysis computed
189 pairwise displacement cosines with no missing probes. This count excludes zero
budget and covers three arm pairs, seven nonzero budgets, three matrices, and three
seeds. `--full-pool-baselines` enumerates exact constant-action oracle success over
the full hashed held-out task pools; those references are not additional LLM results.
Outputs include PNG curves and JSON accounting; plotting uses Matplotlib defaults.
An optional plot summarizes diversity among each prompt's four sampled completions:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/plot_training_diagnostics.py \
  results/training_main_analysis/summary.json
```

It writes `results/training_main_analysis/rollout_diversity.png` from the saved analysis,
without further model inference.
`results/` is gitignored, including the local suite launcher and logs. Scheduler
commands and SLURM scripts are not committed. The protocol and reusable task,
training, and analysis code live in `experiments/`.

The loader explicitly uses `fix_mistral_regex=False` for this pinned Qwen tokenizer.
The CPU audit in `results/weight_training/tokenizer_audit.json` found no differences
between original and saved tokenization on all 960 pilot problems; this flag only
suppresses the known false-positive warning in the pinned Transformers version.
