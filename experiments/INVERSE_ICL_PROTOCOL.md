# Compositional inverse inference through generated rehearsal

Started 10 September 2026, Europe/London. This is a new experiment following the
Self-ICL reproduction, with a separate persistent OpenRouter cap of $3 including
operational checks. No old result directory or budget ledger is changed.

## Mechanism and prediction

Each task world supplies d random positive symbol weights. A latent subset z
generates an observation x by summing its selected weights. The forward map is
known; inferring a subset from its total is harder than evaluating a chosen
subset. Test totals have unique solutions. This studies computational access
to known information, not acquisition of new information from self-generated
data. The ideal unrestricted inverse solver is already perfect at zero samples.

Splitting the symbols in half permits compositional rehearsal: generate partial
subsets and calculate their totals. A collection of a left and b right partial
examples can represent a*b full combinations. A lookup-based recognizer using
whole examples has only N stored combinations. This predicts a possible
compositional reuse advantage, not a universal inverse-error power law. The
natural exact algorithm is meet in the middle. Any advantage must also be
compared with a teacher supplying the same useful partial representation.

The learner is given the split and combination identity in every condition.
Rehearsal is target-blind. A seeded, balanced, nonrepeating schedule supplies
latent partial subsets; the LLM itself computes their observation values. This
is forward generation with known latent labels, not guessing labels for hard
inverse inputs. Its arithmetic is never checked or repaired before inference.
All generated text, including incorrect sums or malformed rows, is retained.

## Conditions and budgets

- Zero: weights and task description only.
- Teacher whole: N IID correct whole-subset examples, excluding held-out inputs.
- Generated parts: the LLM computes N scheduled partial-subset examples.
- Teacher parts: the same N partial subsets with correct totals, isolating
  representation and curriculum from forward-generation errors.
- Teacher random parts: N IID partial examples, to separate representation from
  systematic coverage.
- Longer direct inference: no examples, with an output-token allowance increased
  by the generated-rehearsal allowance. This is an allowance control; executed
  tokens and dollars are also reported and hardware FLOPs are not measured.

Budgets are 0,2,4,8,16,32,64, limited by the half-state count. Generation proceeds
in incremental chunks without future rows leaking into smaller budgets. Final
evaluation uses identical held-out totals across conditions at each budget and
never feeds results into rehearsal. Teacher prefixes are nested. Output prompts
use the same explicit answer markers for every condition. No LLM judge is used;
evaluation uses the exact arithmetic rule after responses have been saved.

## Calibration and confirmation

Pilot dimensions are d=8,10,12 with two fresh worlds per dimension, eight test
totals per world, and budgets 0,8,32. Cheap candidates are Qwen3.5 Flash through
Alibaba and Gemini 3.1 Flash Lite through Google AI Studio. All prices and
endpoint metadata are snapshotted. Hidden reasoning is disabled for controlled
visible-token accounting. Qwen3.6-35B-A3B is an operational fallback.

Choose a working model/dimension based on output validity, forward arithmetic
accuracy, and zero-shot headroom, not by selecting a positive treatment effect.
Freeze the confirmation on fresh random worlds before collecting its outcomes.
Report paired world-bootstrap intervals, every tested budget, arithmetic-error
diagnostics, complete responses, actual token work and spending. If calibration
shows ceiling, floor, or an interface problem, record the redesign and keep all
previous results. Do not force a power-law fit onto a nonmonotone curve.

## Operational amendment: explicit fast recognition

The initial free-reasoning pilot was interrupted after Gemini repeatedly entered
literal repetition loops and exhausted 4,096 output tokens without answers.
Qwen's first d=8 baseline correctly decoded seven of eight queries by building
its own complete partial-sum tables in its reasoning. This both verifies the
intended mechanism and makes an unconstrained-reasoning baseline a different
algorithm from fast recognition. All initial responses are retained in `pilot`.

The `pilot_fast` cohort explicitly requests answer lines only, with 512 output
tokens for an eight-query batch and hidden reasoning disabled. It uses two
worlds per dimension and both original cheap models; Qwen3.6-35B-A3B is also
checked as a stronger candidate. Longer-direct controls explicitly permit
visible calculations and use the extra generation allowance; their instruction
therefore differs intentionally. This tests whether generated tables help fast
inverse recognition and whether ordinary test-time reasoning can reproduce it.
It does not claim to hold executed hardware compute exactly fixed.

## Calibration amendment: bounded reasoning on individual questions

The answer-only pilot completed but was at floor even with correct complete
partial tables. Fast recognition cannot reliably perform the remaining matching
arithmetic. Gemini additionally omitted the literal ANSWER marker; a numeric
question prefix is now accepted prospectively by the exact scorer.

The `pilot_individual` calibration uses Qwen3.5 Flash, selected for the best
forward arithmetic among the checked cheap endpoints, and d=8,10. Each query
now receives a separate call with concise calculations requested and a 1,536
output-token limit. This avoids a hard early query using a whole batch's
allowance or preventing subsequent answer lines. The first two operational
pilots retain their original saved predictions and sources. Their scores are
not pooled with the final experiment.

Generation remains target-blind and unfiltered, using the original forward
prompt and partial-subset schedule. For the longer-direct control, the total
generation output allowance is divided across the held-out queries, so the
sum of allowed output tokens approximately matches rehearsal plus evaluation.
Actual tokens and repeated input costs remain separate recorded measurements.
