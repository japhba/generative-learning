# Matched Zipf learning in context and in weights

This is a prospective extension of the completed Zipf ICL experiment. The task,
32 original dictionaries, oracle labels, acquired key histories and weighted test
items are held fixed. Full-weight fine-tuning replaces storage in the prompt.
The original data are not an independent new-world confirmation; all 32 worlds
are retained for paired comparison. Hyperparameters are selected only on separate
pilot dictionaries 51000–51003, without test evaluation.

## Fixed task and learners

The dictionary has 65,536 independent four-way labels with test mass proportional
to key^-1.5. `zipf_task.py` is reused unchanged. Both Qwen3-0.6B-Base and SmolLM2-135M
start from their original pretrained revisions, separately for every dictionary
and acquisition arm. No association labels enter the evaluation context. The
prompt is `Dictionary key: {key}\nLabel:` and prediction is the highest of the
four existing LM token logits corresponding to space+A/B/C/D. The four-way
conditional loss backpropagates through the full transformer and its ordinary
output embeddings; it introduces no new classification head.

The earlier ICL experiment predicts 40 answers in one response. Here those same
keys and weights are evaluated independently in a padded tensor batch, with no
cross-example attention. The statistical evaluation task matches; the answer
format and memory implementation differ. Exact identity of ICL inference format
is not claimed.

## Acquisition conditions

- IID: reuse each world's original 128-key IID stream, including duplicates.
- Ranked: keys 1–128. This is exactly both the Gemini and Mistral active streams;
  preparation verifies every world. Their same-example study controls supply the
  same pairs, so redundant fine-tuning conditions are not counted as replications.
- Historical Qwen: reuse its actual earlier chosen keys, retaining repeats and
  imperfect selection. No optimal policy is silently substituted.
- Current-model acquisition: at the original budgets 1, 2, 4, ..., 128, the current
  fine-tuned LM chooses the next block of keys and guesses all their labels before
  oracle feedback. The numerical key generator is grammar-constrained to valid
  IDs, analogous to the old structured-output constraints. It sees task instructions
  and previous key IDs, but no correct-label history. Each key is emitted greedily;
  within a block, later choices see earlier chosen IDs, with unchanged weights and
  no intervening feedback. Guesses use the same dictionary prediction interface.
  Every valid choice costs one observation, including repetitions. Guesses are
  logged, never treated as correct targets. There is no success filtering.

The live condition is a prospective transfer of the query-selection behavior to
the fine-tuned model; its prompt and sequential constrained key decoding differ
from the old JSON batch selector. Both successes and selection failures will be
reported. Passive replay of identical correct pairs and updates should produce
identical weights; generation adds acquisition work, not an extra gradient signal.

## Update rule and calibration

Append each oracle observation to a multiset replay buffer. Perform R updates;
each batch contains that current observation and 31 uniform draws with replacement
from the entire acquired buffer, including the newest observation. Replay index
sequences are paired across conditions. No future observation can enter a batch.
After N labels there are RN updates and32RN training presentations. All updates
use AdamW, no weight decay or smoothing, gradient clipping at norm 1, FP32 parameters
and optimizer state, BF16 autocast, and batch size 32. Fixed padding covers every
valid key, making numerical batch shape constant within training and evaluation.

The calibration grid is R∈{1,4,16}, learning rate∈{1e-5,3e-5,1e-4}. Require at least
99% unweighted accuracy on all observed unique keys at 128 labels in every one of
four pilot worlds under both ranked and IID acquisition. Select smallest R, then
smallest learning rate. A configuration is eliminated on its first failing case;
all attempted cases are completed and retained. No test labels or exponent fits
are evaluated during calibration. If no configuration passes, confirmation does
not start automatically. The immutable JSON protocol and source snapshots were
written before the first model prediction. Each family's selected configuration
is recorded before its main runs.

## Measurements and analysis

Save every checkpoint at 0, 1, 2, 4, 8, 16, 32, 64, 128 labels. Primary weighted test error is
independently regraded from the original 40 test queries. In addition, evaluate all
unique observed keys. Coverage error is exactly 0.75 times unobserved probability
mass. Adding test-weighted observed-key mistakes gives a conditional expected
error, marginalizing the independent unknown labels. That diagnostic is reported
alongside the raw test measurement rather than replacing it.

Output probability MSE and centered probability overlap are functional probes,
comparable to the toy's effective lookup matrix; they are not overlaps with a known
true transformer parameter vector. Raw parameter displacement norms are retained.

Primary fits use8,16,32,64,128 labels, with4–64 and16–128 sensitivity windows and
log-error fit residuals. Bootstrap whole paired dictionaries 1,500 times. Report
ranked-minus-IID and live-minus-IID exponent differences; inspect curves rather
than treating every fitted slope as a scaling law. The exact toy is evaluated
both under the population IID distribution and conditional on actual acquired
sets. Tokens, updates, label requests, replay presentations, generation phases,
and evaluation overhead remain distinct. Calibration cost is separate.

## Resources and retention

Use the permitted institutional `gpu_lowp` allocation 3529684, two H100s on
gpu-xd670-30. No paid API request or Vast rental is needed. The external existing
venv and configured fast model cache are used. Archive world 4000's final model
for each family and arm, selected before results; retain all other worlds' full
predictions, acquisition histories, replay indices, counters, source hashes and
functional probes. Release the allocation after execution, analysis and archival.

The main protocol is `results/zipf_finetuning/protocol.json`. The entry point is
`train_zipf_finetuning.py`; independent checks and plots use
`analyze_zipf_finetuning.py`. Original ICL and earlier tracking artifacts remain
unchanged.

### Execution scheduling addendum, 2026-09-06

After Qwen passed calibration, allocation 3529686 supplied two additional idle
H100s on the same node and permitted partition. Independent dictionaries are
distributed with `zipf_finetuning_worker.py`, using an advisory file lock. Each
case still executes the frozen training engine with its original reset and replay
seed. The original three engine/task files and their hashes remain unchanged.
The interrupted partial Qwen case was preserved under `execution_interruptions`
and rerun from its original reset; it contributes no extra experimental world.
The handoff record is retained in the result directory. Both allocations are
released after execution and archival.

Once calibration passed, SmolLM was handed to the same queue mechanism; its
interrupted first main case was likewise retained and restarted from the original
weights. Spare memory and utilization allowed two independent workers per
allocated GPU. Consequently wall times include variable contention between runs;
the causal resource comparison uses label requests, optimizer updates, fixed-width
training tokens and replay presentations. Timing curves are descriptive operational
measurements and should not be interpreted as controlled compute scaling laws.

`audit_zipf_same_data.py` replays preselected world 4000's live observations without
generation and compares every checkpoint prediction and the actual final saved
parameter tensors. `archive_zipf_finetuning.py` copies the eight preselected models
to the configured shared cache and verifies each file's SHA-256 digest. Analysis
and report builders do not alter training or select configurations from test error.

### Query termination follow-up

A query-only diagnostic found a material limitation in the original EOS boundary:
without feedback or weight updates, Qwen repeatedly emitted 10000 and SmolLM 12345,
whereas allowing a newline at a valid integer boundary made both repeat key 1.
`diagnose_zipf_query_termination.py` retains these diagnostic sequences. The original
256-case cohort remains intact; its generated-query curves describe the EOS-based
implementation and cannot cleanly separate decoder bias from acquisition behavior.

Before further training, `newline_pilot_protocol.json` specified four fresh worlds
52000–52003 per model. `pilot_zipf_newline_acquisition.py` permits newline or EOS,
with the original prompt, greedy legal prefixes, repeated-query costs, oracle
feedback and selected optimization settings. These pilots never evaluate held-out
test labels or fit exponents. Acquisition calibration requires every pilot to cover
more probability mass at 128 requests than the exact expected IID coverage 0.816121.
The independently regraded pilot summary records failure in both families: Qwen
coverage is 0.8140, 0.3839, 0.8578, 0.3839; SmolLM coverage is approximately 0.383942 in
every pilot. The corrected selector therefore does not support a reliable
beneficial-scaling claim. The fixed IID/ranked exponent comparison is unaffected.
