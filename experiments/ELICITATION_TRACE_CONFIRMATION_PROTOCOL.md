# Fresh-data confirmation of full-trace self-training

The exploratory full-trace pilot reached 100% held-out reasoning accuracy with a
current source versus 87.5% with a frozen source, from an 84.0% common baseline.
Its 144-question test and one optimization seed do not suffice to establish a robust
advantage. This confirmation was frozen before obtaining any new predictions.

Use exactly the main pair from [the full-trace protocol](ELICITATION_TRACES_PROTOCOL.md):
Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, four
reasoning samples per practice presentation, internally selected plurality, actual
full sampled traces as targets under the reasoning prompt, and either a frozen
initial source or a current source refreshed after every batch.

All training details remain fixed: 96 practice questions in four shuffled passes,
batch size 16, one AdamW update per batch, learning rate 1e-6, no weight decay,
gradient clipping at 1, FP32 weights/state, BF16 autocast, and summed completion NLL
divided by the fixed value `128 * batch_size`. Every repeated presentation obtains
new candidate samples. Neither confidence nor correctness filters targets. Raw
one-rollout self-imitation is not repeated in this confirmation.

The new practice cohort uses seed 71100; the new evaluation cohort uses seed 71200.
There are 144 evaluation programs, balanced across 4, 6, and 8 operations. Both
cohorts are disjoint from all earlier calibration, confirmation, and training data.
The three optimization seeds are 71000, 71001, and 71002. They vary question order
and sampling while sharing the new practice and evaluation cohorts. The two arms
have paired schedules and sampling seeds within each optimization seed.

The primary contrast is final current-minus-frozen reasoning accuracy, evaluated
greedily with a 512-token limit after 384 practice presentations. The original
checkpoints 0, 16, 64, 192, and 384 remain fixed. Short-answer accuracy, completion,
response length, practice accuracy, and measured token work are secondary outcomes.
Target correctness is analyzed only after all six runs finish. No intermediate
test scores select settings or stopping points.

The analyzer first audits every case's actual target token IDs, source and data
hashes, causal candidate source, counters, paired initial teacher samples, and
baseline predictions. It then uses 5,000 paired bootstrap resamples of optimization
seeds and evaluation questions, with question resampling stratified by depth. This
has only three optimization seeds and one fresh shared cohort. A perfect score can
produce a degenerate nonparametric bootstrap interval; it is not proof of perfect
population accuracy. No power-law exponent is fitted.

The main pair has equal counts of questions, rollouts, plurality selections,
and optimizer updates. Trace lengths and actual compute are allowed to differ and
are reported, so this is not an exact FLOP-matched experiment.

`replicate_elicitation_traces.py` and its imported sources are frozen in
`results/elicitation_trace_confirmation/source/`, with hashes in `protocol.json`.
The worker reuses `TraceStudent`, target selection, and evaluation from the pilot.
Three H100 workers run only on `gpu_lowp`. Final seed 71000 models are copied from
the configured fast cache to shared storage and SHA-256 verified before the job
exits. The other final models need not be retained; their complete predictions,
sampled targets, and update records are retained. No paid APIs or Vast rentals run.

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/replicate_elicitation_traces.py prepare --output results/elicitation_trace_confirmation_reproduction
"$VENV_LOCAL/generative-learning/bin/python" experiments/replicate_elicitation_traces.py run --output results/elicitation_trace_confirmation_reproduction --seed 71000
"$VENV_LOCAL/generative-learning/bin/python" experiments/replicate_elicitation_traces.py run --output results/elicitation_trace_confirmation_reproduction --seed 71001
"$VENV_LOCAL/generative-learning/bin/python" experiments/replicate_elicitation_traces.py run --output results/elicitation_trace_confirmation_reproduction --seed 71002
"$VENV_LOCAL/generative-learning/bin/python" experiments/analyze_elicitation_trace_confirmation.py --output results/elicitation_trace_confirmation_reproduction
```

Run inside allowed GPU allocations with unused fast-cache model destinations. The
runner deliberately refuses to overwrite its earlier models. This tests another
cohort and training seeds for the same model, not an independent architecture.
