# A toy mechanism for narrowing the strategy distribution

At the final 4,096-prompt checkpoint, seed 6200's rollout-distilled model uses only
B moves in all 384 greedy evaluation outputs. For seed 6201, all 384 completions
start with A, and none contains a recognized C move; the dominant pattern is one
or two initial As followed by Bs. Teacher SFT and yoked SFT still produce C moves.
These observations describe generated strings rather than establish an exact policy
identity or valid solutions: some contain malformed text, incorrect transitions,
or truncation, and extracting move fragments can skip malformed segments.
The source observations are the two seeds' `rollout_distill_n4096_predictions.json`
files under `results/training_main_seed6200` and `results/training_main_seed6201`.

The following exact toy shows one way success filtering can concentrate an existing
strategy distribution. It does not establish the cause of those observed outputs.

For one fixed task x, partition possible solutions into disjoint strategies j.
Assign each a logit z_j and probability `q_j = softmax(z)_j`. Conditional success
probabilities `s_j(x)` are fixed numbers in [0,1]; within-strategy competence does
not change. Suppress x in the notation and assume `S = sum_j q_j s_j > 0`.

```text
P(strategy j | success) = a_j = q_j s_j / S
L(z'; a) = -sum_j a_j log softmax(z')_j
At z' = z:  dL/dz_j = q_j - a_j = q_j(1 - s_j/S)
Likelihood-ascent direction: a_j - q_j = d log S / dz_j
```

Here the accepted target a is frozen during differentiation. Strategies with
above-average success gain logit weight; those below average lose it. Initial
probability matters because the instantaneous update contains q_j. This is a
sequence-level categorical objective: each accepted strategy sample counts once.

With K independent proposals, the chance of retaining any example is
`w_K(x) = 1 - (1 - S(x))^K`. Conditional on retention, the first success still has
distribution a. Treating a failed prompt as a zero update, the expected ascent
direction per presented prompt is `w_K(x) * (a - q)`. Averaging only retained
examples changes the normalization across tasks; it does not remove the increased
inclusion of tasks with higher current success. A task with S=0 contributes no
accepted examples. Thus finite K changes data availability and task weighting,
even before introducing a shared neural network.

To solve the repeated dynamics exactly, now replace a finite logit-gradient step
with a complete fit to each frozen accepted distribution. Allow probabilities on
the simplex boundary, corresponding to limiting softmax logits when a_j=0. Then

```text
q_j(t+1) = q_j(t) s_j / sum_k q_k(t) s_k
q_j(t)   = q_j(0) s_j^t / sum_k q_k(0) s_k^t
q_j(t)/q_k(t) = [q_j(0)/q_k(0)] (s_j/s_k)^t   (s_k > 0).
```

If one strategy has strictly greatest success among the initial support, its
probability tends to one. Equal best strategies preserve their initial relative
weights; if all success rates are equal, there is no change. An absent strategy
is never discovered by this update. Finite-logit softmax gives positive support,
but a very improbable strategy can remain practically absent from a finite sample.
For two equally likely strategies with success rates 0.8 and 0.4, successive exact
fits give probabilities `(1/2, 1/2)`, `(2/3, 1/3)`, `(4/5, 1/5)`, and so on.

Narrowing is not automatically harmful in this toy. In fact,

```text
S(t+1) = sum_j q_j(t) s_j^2 / S(t)
       = S(t) + Var_{q(t)}(s) / S(t) >= S(t).
```

Consequently, fixed-success, independently parameterized task distributions cannot
by themselves explain declining success on those same tasks. Explaining harmful
narrowing requires another ingredient, such as shared parameters across tasks,
changing competence within a strategy, finite-data noise, or a train/test mismatch.
These are hypotheses to test, not conclusions of the algebra. The exact-fit odds
converge exponentially in fit iteration when success rates differ; this is not a
prediction of a power-law sample exponent for an LLM.

The actual experiment samples complete trajectories at temperature 0.8 with
`top_p=0.95`, uses K=4, decodes and re-tokenizes accepted text, averages CE over
completion tokens, buffers accepted examples into batches of 16, and updates shared
transformer weights with AdamW and gradient clipping. Its proposal distribution,
target weighting, within-strategy competence, and update rule therefore differ
from the exact toy. Greedy evaluation also need not represent the full sampled
strategy distribution. Full fitting in the recurrence is distinct from the
single-gradient calculation above and from the runner's one training pass.

The toy provides an interpretable selection mechanism and measurable quantities
(strategy mass, success by strategy, and retention probability). Neither its logits
nor the experiment's three query-projection displacement probes identify known
true task weights. A ground-truth weight-overlap interpretation and predictive
real-architecture learning exponents remain unestablished.
