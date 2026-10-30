# A linear weight-overlap toy for the Zipf dictionary

The dictionary task admits an exact Bayesian learner whose weight overlap predicts
its test error. A single linear layer suffices. Different query policies produce
different sample exponents because they expose different rows of the weight matrix.
This explains a coverage advantage; it does not establish an advantage from the
learner generating the label text itself.

## Model and exact weight identities

Let a query be a rank `r` in `1,...,V`, with known probability `p_r`. Independently
draw each fixed true label `ell_r` uniformly from `1,...,K`. An oracle reveals that
label without noise when queried. These are modeling assumptions for the random
associations in `zipf_task.py`; its implementation uses seeded pseudorandom labels.

Write `u = (1/K,...,1/K)`, `c = 1 - 1/K`, and define the true matrix by

```text
W*_r = e_(ell_r) - u,          ||W*_r||² = c.
```

After observing the distinct ranks in `S`, independence and noiseless feedback give
the posterior mean exactly:

```text
M_r = E[W*_r | observations] = W*_r  if r is in S,
                             = 0    otherwise.
```

For one-hot input `x = e_r`, the affine prediction `u + xᵀM` is the posterior
predictive label distribution: one-hot on observed ranks and uniform elsewhere.
There is no hidden layer or activation function. These weights are probabilities
minus a baseline, not softmax logits; applying an additional softmax changes them.

Define test-weighted inner product and coverage by

```text
<A,B>_p = sum_r p_r <A_r,B_r>,       C(S) = sum_(r in S) p_r.
```

Then, for every realized dictionary and observed set,

```text
||W*||²_p = c,
<M,W*>_p = ||M||²_p = c C(S),
||M-W*||²_p = c [1-C(S)].
```

Thus normalized overlap `<M,W*>_p / ||W*||²_p` equals coverage. When `C(S)>0`,
the weighted cosine is `sqrt(C(S))`; it is undefined for the zero initial matrix.
The posterior Bayes classification error is also `c[1-C(S)]`, because unobserved
labels remain uniform. This is posterior expected error, not the realized error
of a particular deterministic tie-breaking rule on unknown labels. The same
quantity is the posterior predictive squared error for one-hot labels, without
an extra division by `K`.

An observed pair changes the corresponding weights in either study or active
querying. For the loss `0.5 ||M_r - (e_y-u)||²`, a unit-step SGD update gives

```text
M <- M + e_r [(e_y-u)-M_r]ᵀ.
```

Starting at zero, this reproduces the Bayesian mean exactly, including repeated
observations. The data therefore do reach the weights under ordinary supervised
learning. A repeated noiseless observation supplies no additional information.

## A numerical matrix

Take `K=3`, probabilities `(0.5,0.3,0.2)`, and labels `(1,2,1)`:

```text
      [ 2/3  -1/3  -1/3 ]
W* =  [-1/3   2/3  -1/3 ].
      [ 2/3  -1/3  -1/3 ]
```

After rank 1 is observed twice, only the first row of `M` equals its true row;
the other rows remain zero. Coverage is `0.5`, overlap is `1/3`, and weighted
MSE/Bayes error is `1/3`. Observing ranks 1 and 2 instead makes the first two rows
correct: coverage is `0.8`, overlap is `8/15`, and error is `2/15`.

| Number of queries | Expected IID error | Ranked active error |
| --- | ---: | ---: |
| 0 | 0.666667 | 0.666667 |
| 1 | 0.413333 | 0.333333 |
| 2 | 0.266667 | 0.133333 |

The IID column averages over query sequences; the repeated-rank example above is
one possible sequence, not that average.

## Exact curves and asymptotic exponents

With IID queries, rank `r` remains unseen after `n` presentations with probability
`(1-p_r)^n`. With known probabilities and one-query learning, querying the `n`
highest-probability distinct ranks maximizes coverage. Consequently,

```text
E_IID(n)    = c sum_r p_r (1-p_r)^n,
E_ranked(n) = c sum_(r>n) p_r,                 0 <= n <= V.
```

These exact finite sums also give expected weighted MSE; expected normalized
overlap is `1-E(n)/c`. A learner shown the active learner's identical observed
labels has the identical posterior and weights. Guesses generated before oracle
feedback contain no extra information about independent, unobserved labels.

For an infinite Zipf distribution `p_r = C r^(-s)`, `s>1`, `C=1/zeta(s)`,
the large-`n` limits derived from the tail integral are

```text
E_IID(n)    ~ c [C^(1/s)/s] Gamma(1-1/s) n^(-(s-1)/s),
E_ranked(n) ~ c [C/(s-1)] n^(-(s-1)).
```

For IID, substitute `t=n C r^(-s)` into the integral of
`C r^(-s) exp(-n C r^(-s))`; the resulting integral converges to
`Gamma(1-1/s)`. Ranked acquisition follows directly from the Zipf tail integral.
At the experiment's `s=1.5`, the limiting exponents are `1/3` and `1/2`.
Its `V=65,536`, `K=4` implementation uses the finite sums, whose fitted slopes
need not equal these limits. For fixed finite `V`, ranked error reaches zero at
`V` queries and the eventual IID decay is exponential, not a power law.

Here `n` counts queried labels, including IID duplicates. This is not an LLM-token,
FLOP, or wall-time law. Ranking assumes known query probabilities and does not
charge a model for choosing a rank.

## Finite learning rates change the weight curve

For the same squared loss with constant `0<eta<=1`, let `N_r` count presentations
of rank `r`. Starting at zero, the actual SGD weights satisfy

```text
M_r = [1-(1-eta)^N_r] W*_r,
||M-W*||²_p = c sum_r p_r (1-eta)^(2 N_r).
```

Argmax classification still learns a row after its first observation for any
positive `eta`, but its weights may remain far from the true row. The exact
MSE/classification-error identity above requires the Bayesian mean or `eta=1`.
For IID samples the expected squared weight error is

```text
c sum_r p_r [1-p_r(2 eta-eta²)]^n.
```

For the ranked policy that visits each of its first `n` ranks only once, it is
`c[1-(2 eta-eta²) C({1,...,n})]`. At `eta<1`, covering all ranks still leaves
the weight-error floor `c(1-eta)²`; further repetitions are needed. A policy
optimized for weight MSE need not be the once-per-rank classification-optimal
policy. This is a concrete limit on transferring coverage exponents to weight
convergence in other optimizers or architectures.

## Numerical validation with actual dense weights

`simulate_weight_coverage.py` was run on 128 dictionaries (seeds 8800–8927), using
`V=65,536`, `K=4`, `s=1.5`, both curricula, and learning rates 1 and 0.25. It
updates explicit dense matrices by row SGD; counts are used only for independent
validation. Dictionaries and IID presentations are paired across learning rates.
All 4,096 checkpoint matrices and their metrics satisfy the derived identities.
The run took 28.6 seconds on CPU and wrote no full weight matrices to disk.

At 128 label presentations, means across the 128 dictionaries are:

| Curriculum | Learning rate | Weighted weight MSE | Normalized overlap | Actual argmax error |
| --- | ---: | ---: | ---: | ---: |
| IID | 1 | 0.13767 | 0.81644 | 13.79% |
| Ranked | 1 | 0.04856 | 0.93526 | 4.85% |
| IID | 0.25 | 0.18156 | 0.70745 | 13.79% |
| Ranked | 0.25 | 0.44312 | 0.23381 | 4.85% |

The reversal at learning rate 0.25 is informative: ranked acquisition gives better
classification while its weights are farther from the teacher. Its newly observed
rows receive only one partial update; IID repetition brings important rows closer
to their true magnitudes. For example, one sampled true row is
`(-0.25, 0.75, -0.25, -0.25)`. After its first update at rate 0.25, the actual row is
`(-0.0625, 0.1875, -0.0625, -0.0625)`: the argmax is already correct.

The exact expected MSE endpoints are 0.13791/0.04856 at rate 1 and
0.18238/0.44312 at rate 0.25 (IID/ranked). Simulated MSE and classification means
are within 1.77 empirical standard errors of their exact expectations across all
nonzero-variance checkpoints; this is a descriptive numerical comparison.
Fitting the exact rate-1 curves over budgets 8–128 gives slopes 0.326899 and
0.502071, rather than estimating new exponents from noisy simulation means.

Numeric arrays, exact comparisons, and a tiny illustrative weight table are in
`results/weight_coverage_toy/simulation.json` (335 KB). Figures:

- `results/weight_coverage_toy/weighted_mse.png`
- `results/weight_coverage_toy/classification_error.png`

Reproduce in a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$VENV_LOCAL/generative-learning/bin/python" \
  experiments/simulate_weight_coverage.py --output results/weight_coverage_toy_repeat
```

## A precise connection to a categorical Boltzmann model

A Boltzmann joint distribution can represent the same dictionary. Use two
categorical blocks, one selecting rank `r` and the other label `k`, with energy

```text
E(r,k) = -a_r - b_k - J_rk.
```

Equivalently the blocks are one-hot vectors with interaction `-vᵀJh`. They are
constrained to exactly one active unit each. This is a categorical, constrained
one-hot RBM construction; the unconstrained standard binary RBM has a different
state space. Hinton describes the binary energy and likelihood gradient in
section 2 and categorical/softmax units in section 13.1 of
[A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf).

Choose `b_k=0`, `a_r=log p_r`, and `J_rk=beta 1[k=ell_r]`, with `beta>0`.
Every row has partition sum `exp(beta)+K-1`, so the normalized joint is exactly

```text
P_beta(r,k) = p_r exp(beta 1[k=ell_r]) / [exp(beta)+K-1].
```

Its rank marginal is `p_r`; its conditional label distribution is a softmax.
As `beta` tends to infinity this becomes the deterministic dictionary. Centering
each row of `J/beta` gives the true linear weights `W*`. The removed row mean
is the same constant `1/K`, so it does not alter this joint distribution.

The mapping concerns representable probabilities, not an assertion that RBM
maximum likelihood or contrastive divergence performs the Bayesian row update.
For conditional maximum likelihood, observed `(r,y)` gives gradient
`d log P(y|r)/d J_jk = 1[j=r](1[k=y]-P(k|r))`: supervised data explicitly
update the weights. Joint likelihood includes the model's negative-phase
expectation as well. Generating negative-phase samples is not equivalent to
querying fresh oracle labels.

Two details prevent overinterpreting this link:

- With finite `beta`, sampled labels are noisy. Even known true rows have
  classification error `(K-1)/(exp(beta)+K-1)` on new noisy labels. One observation
  no longer identifies a row; posterior inference depends on label counts.
  The noiseless coverage identities therefore do not carry over unchanged.
- Substituting a partially learned `beta M` into the energy generally creates
  unequal row partitions. Keeping `a_r=log p_r` then changes the rank marginal.
  For arbitrary `J`, fixing `a_r=log p_r-log sum_k exp(J_rk)` preserves that
  marginal when `b=0`. Also, a softmax of posterior-mean couplings is generally
  different from the posterior mean of the conditional probabilities.

The linear learner supplies explicit true weights and exact overlap/sample laws.
The categorical energy supplies a probabilistic representation of its data model,
with nonlinear normalization. Neither construction predicts the measured Qwen
fine-tuning curves or equates generation with a universally faster learning rule.
