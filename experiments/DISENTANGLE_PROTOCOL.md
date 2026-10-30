# Wake-sleep practice that binds foreign factors to the learner's own concepts (DISENTANGLE)

Frozen 24 October 2026, 00:20 UK, after pilots and a learning-rate grid on tuning seed 100, before any run on
confirmation seed 0. Spec: `tex/README.md` (comparison protocol of 23 October). Code:
`experiments/practice_disentangle.py`, `experiments/practice_disentangle_budget.py`.

**Hunch (user).** A wake-sleep encoder-decoder that starts misaligned to foreign samples rectifies itself through its
own samples and so increasingly makes sense of the foreign ones, disentangling them.

**World.** Latent `z ∈ {±1}^K` drawn from a prior π, observation `x = A z + σ ε` with unit-norm Gaussian columns,
K = 6, D = 12, σ = 0.5. Priors: `factorial` (uniform), `chain` (Markov, same-sign probability 0.85), `deep`
(two uniform ±1 causes h, `p(z_i = +1 | h) = sigmoid(3 b_i·h)`, b_i Gaussian).

**Learner.** Same prior π (its own concepts). Decoder `N(W z + c, σ²I)`, factorial logistic recognizer with logits
`R x + b`. Start: second-order matched but rotated, `W0 = A Σ^½ Q Σ^-½` with Haar-random Q, `c0 = Aμ − W0 μ`;
the recognizer is fitted to W0 by 20,000 sleep steps. Every method starts from this learner.

**Budget.** 20,000 steps of two batches of 32 each; 81 evaluations on 4,000 held-out foreign samples. N ∈ {64, 256,
1024} stored foreign samples, primary N = 256. 16 worlds per cell (seed 0).

**Methods and the settings chosen on the tuning grid (one setting per method, all priors):**

| method | what it does | setting |
|---|---|---|
| `wake_sleep` (practice) | wake on stored data + sleep on own samples, local delta rules | lr 0.02, sleep ×10 |
| `wake_attend` (practice, in context) | no recognition weights: wake samples a latent by attention over the last 1,024 own samples (keys: generator output before noise) | lr 0.02 |
| `wake_frozen` | two wake batches on stored data, recognizer frozen | lr 0.02 |
| `wake_fresh` (fresh-sample baseline) | sleep batch replaced by a fresh foreign batch, recognizer frozen | lr 0.02 |
| `binary_ae` | backprop on expected reconstruction through the stochastic ±1 code | lr 0.06, recognizer ×10 |
| `binary_ae+assign` | final binary-AE decoder relabeled by the signed permutation whose code moments best match π's | — |
| `linear_ae` | linear autoencoder (native second-order objective) | lr 0.02 |
| `fastica`, `fastica+assign` | one-shot noise-corrected FastICA, without and with the moment relabeling | — |
| `vae` | ELBO with factorial q and the exact prior density (enumerated) | lr 0.02, recognizer ×3 |
| `exact_em` | gradient EM with the exact posterior (2^K enumeration, prior density) | lr 0.06 |
| `wake_sleep_stream` | reference: wake-sleep with a fresh foreign batch for every wake update | lr 0.02, sleep ×10 |

Prior-agnostic competitors: `wake_frozen`, `wake_fresh`, `binary_ae`, `linear_ae`, `fastica`. The two `+assign`
variants use π only through its first two moments (latent statistics, no rendering or recognition). `vae` and
`exact_em` use π's density.

**Outcome.** KL(true ‖ model) in nats, Monte Carlo on the held-out set. Practice is scored at its final step;
every competitor at its best checkpoint (minimum over the 81 evaluations), or its one-shot value.

**Pre-registered claims** (paired over the 16 worlds; 95% CI from 10,000 bootstrap resamples of the per-world
log-ratio):

- C1. On the chain and deep priors at N = 256, `wake_sleep` has lower KL than each prior-agnostic competitor and
  each `+assign` variant: mean log-ratio CI below 0, and lower in at least 14 of 16 worlds.
- C2 (fresh-sample baseline). `wake_sleep` has lower KL than `wake_fresh` for every prior and every N.
- C3 (budget split, `practice_disentangle_budget.py`, sleep ×10, 16 worlds). Spending half the extra updates on own
  samples (f = 0.5) beats spending all of them on fresh foreign samples (f = 0) for every prior and N.
- C4. `wake_attend` satisfies C1's criterion against the same competitors.

**Descriptive, no directional claim.** `wake_sleep` vs `vae` and `exact_em`; how the best fraction f moves with N;
symmetry-adjusted recognition accuracy of foreign samples; reconstruction error of `binary_ae` vs `wake_sleep`;
the factorial prior (where FastICA is the known one-shot solution).

## Outcome on seed 0 (24 October, 01:15 UK)

C1 failed (against `binary_ae+assign`: chain log-ratio −0.40 [−0.73, −0.08] but 11/16 wins; deep −0.18 [−0.45, +0.10]).
C2 failed at N = 64 for the factorial and deep priors (ties); passed at N = 256 and 1024. C3 passed. C4 failed against
the same competitor as C1. Every claim and comparison: `/var/tmp/jbauer/generative_practice_disentangle/confirm/claims.json`.

Post hoc on the same worlds: practice disentangles but binds the factors to the right own concepts in only 3/16 chain
worlds (nearest signed permutation of the decoder); a swapped labeling is stable for every gradient learner.

## Amendment 1 (24 October 2026, 01:55 UK, before any run on seed 1): binding moves

**New method.** `wake_sleep_bind` is practice plus, every `bind_every` steps, greedy relabeling moves on the learner's
own units (pairwise swaps and single sign flips of W's columns, R's rows and b). A move is accepted while it lowers the
squared distance between the first two moments of the recognized foreign codes (signs of the recognizer's mean codes
on the stored samples) and those of 4,096 latents sampled from the learner's own prior. The same moves are given to the
competitors: `binary_ae_bind`, `vae_bind`. Intervals chosen on tuning seed 100 (one per method, all priors):
`wake_sleep_bind` 1000, `vae_bind` 200, `binary_ae_bind` 1000. Other settings as above.

**Run.** Seed 1 (new worlds), 16 worlds, N = 256, priors chain, deep and factorial. Methods: `wake_sleep_bind`,
`wake_sleep`, `binary_ae_bind`, `binary_ae` (with `+assign`), `vae_bind`, `vae`, `exact_em`, `wake_fresh`.

**Claims.**
- B1. On the chain and deep priors, `wake_sleep_bind` (final) has lower KL than `binary_ae_bind` (best checkpoint)
  and than `binary_ae+assign`: mean log-ratio CI below 0 and lower in at least 14 of 16 worlds.
- B2. On the chain prior, `wake_sleep_bind` has lower KL than `wake_sleep`: CI below 0.
- Descriptive: `wake_sleep_bind` vs `vae_bind`, `vae` and `exact_em`; binding rate (nearest signed permutation in the
  prior's symmetry group) for every method.

## Outcome of Amendment 1 (24 October, 02:15 UK)

B1 passed on the deep prior (vs `binary_ae_bind` −0.69 [−0.90, −0.49], vs `binary_ae+assign` −0.82 [−1.11, −0.57], 15/16
each) and failed on the chain prior (−0.35 [−0.66, −0.03], 10/16; −0.39 [−0.65, −0.17], 12/16). B2 passed (chain
0.92 → 0.75, −0.29 [−0.49, −0.12], 14/16). Descriptive: on the deep prior `wake_sleep_bind` beats `vae` and `exact_em`
(CIs below 0); on the chain prior `vae_bind` is better (+0.33 [+0.07, +0.62]). Correctly bound chain worlds: 4/16 with
binding moves, 1/16 without. All comparisons: `/var/tmp/jbauer/generative_practice_disentangle/bindconfirm/claims.json`.
