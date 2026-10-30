# Which inputs can change which weights?

A two-weight linear example supplies the mechanism sought in the original question.
It is conditional Gaussian regression with actual closed-loop state acquisition.
The result is about identifiability and experimental input design, not a universal
difference between generative and discriminative architectures.

## Probabilistic model and the missing direction

Let the input be `x=(u,e)`, where `u` is requested motion and `e` is tracking error.
The true regression weights are `w*=(1,-1)`. Teacher labels are

```text
y = xᵀw* + noise,        noise ~ N(0, sigma²),       sigma = 0.25.
w ~ N(0,I).
```

This is a single linear layer, no hidden units or activation. Its conditional
energy is `(y-xᵀw)²/(2 sigma²)`. For the accumulated design matrix `X`, the exact
posterior is Gaussian:

```text
Sigma_n = [I + XᵀX/sigma²]⁻¹,
mu_n    = Sigma_n Xᵀy/sigma².
```

In the demonstration condition, inputs have `e=0` by design. Thus

```text
XᵀX = [[sum u², 0], [0, 0]],
(mu_n)_e = 0,                     (Sigma_n)_ee = 1.
```

No number of these examples changes the corrective coefficient or reduces its
posterior uncertainty. A supervised squared-loss update gives the same structural
fact: its second gradient component is `(prediction-y)*e`, exactly zero.
Ordinary supervision does change the feedforward coefficient. The obstruction is
a missing input direction, rather than recognition preventing data from reaching
weights.

The true corrective coefficient is -1, so squared weight error is at least1 for
this zero-mean posterior learner. Under independent standardized test inputs,
`E_x[(xᵀ(mu-w*))²] = ||mu-w*||²`; the same floor appears in prediction error on
those broader inputs. On the demonstration subspace alone, this weight discrepancy
need not impair predictions.

## Acquisition and exact numerical experiment

Three policies receive the same requested motions and independent measurement
noise arrays across 256 paired repetitions:

- Expert-style demonstrations keep error zero by executing `a=u-e`.
- The learner executes `a=muᵀx+xi` before observing the current teacher label,
  with `xi~N(0,0.5²)` providing persistent exploration.
- A random exploration control executes `a=xi`.

The simulator updates `e_next=e+a-u`; trajectories reset to zero every16steps.
Actions are continuous in this linear experiment, unlike the bounded integer
actions in the LLM task. Every visited input receives a noisy teacher label,
and the posterior updates by the exact rank-one Gaussian formula. At every saved
budget, posterior means and covariances are independently checked against the
batch sufficient-statistic formula above. All8,448 checks passed.

After1,024 labels, averages across the256 repetitions are:

| Acquisition | Squared weight error | Learned corrective coefficient | Smallest input-Gram eigenvalue |
| --- | ---: | ---: | ---: |
| Expert demonstrations | 1.000061 | 0 | 0 |
| Learner exploration + feedback | 0.000340 | -1.001265 | 243.26 |
| Random exploration + feedback | 0.000068 | -1.000087 | 1,020.94 |

The random policy creates even more variation in error and fits the weights better
per acquired label. It need not be a better controller or a cheaper/suitable policy
in a real environment. These results establish that exploration plus informative
feedback opens an unobserved weight direction. They do not establish that adaptive
self-generation uniquely supplies that information.

The Gaussian likelihood is a model of the supervised regression observations.
This is not a claim that the learner is Bayes-optimal over every physical fact of
the control task; a learner given the exact dynamics, objective and sufficient
symbolic reasoning could derive the ideal controller directly. The experiment
isolates what the specified data-driven update learns from each input design.

## Learning rates and relationship to RBMs

The relevant quantity is the spectrum of `XᵀX`, equivalently the Gaussian Fisher
information. An eigenvalue that stays zero leaves a parameter direction unidentified.
Under stable persistent excitation, both eigenvalues can grow proportionally to
sample count, allowing the usual inverse-sample posterior-variance rate. Removing
an error floor is not evidence for two distinct power-law exponents.

A sensitivity run adds the same Gaussian process disturbance of standard
deviation0.05 to all three policies after each action. Expert states now have a
small nonzero error, making both weights identifiable. At1,024labels, expert
squared weight error is0.024403 instead of its previous floor near1; learner
exploration reaches0.000336 and random exploration0.000068. The corresponding
smallest input-Gram eigenvalues are2.39,245.73 and1,020.94. All8,448 posterior
checks pass again. The teacher-style distribution now supplies the corrective
signal weakly, rather than not at all. This illustrates different information
per sample; it does not imply different asymptotic exponents when every direction
is persistently excited. Data and plots are in
`results/linear_feedback_process_noise/`. Use `--process-std 0.05` to reproduce.

This conditional Gaussian model supplies probabilistic grounding and explicit
weights directly. It is not an RBM training result. The separate categorical RBM
representation remains in [WEIGHT_COVERAGE_TOY.md](WEIGHT_COVERAGE_TOY.md); the more
useful common mechanism is whether the acquired examples expose the parameters
that control evaluation behavior. No claim is made that these two coefficients
are identifiable with raw transformer parameter directions.

Run artifacts: `results/linear_feedback/summary.json`. Reproduce in a fresh folder:

```bash
OPENBLAS_NUM_THREADS=1 "$VENV_LOCAL/generative-learning/bin/python" \
  experiments/linear_feedback.py --output results/linear_feedback_repeat
```

Figures:

- `results/linear_feedback/weight_mse.png`
- `results/linear_feedback/corrective_weight.png`
- `results/linear_feedback/minimum_gram_eigenvalue.png`
