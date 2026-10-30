# Why a known generator can support different rehearsal curves

The task is a finite generative model. Draw a latent label z with probability
p_z proportional to z^(-s), then deterministically render x=(a*z+b) mod 257.
The model receives a and b. Because 257 is prime and a is nonzero modulo 257,
the latent label is identifiable from x. A decoder with unrestricted computation
can invert the map without any examples. Its Bayes error is zero at N=0.

The LLM comparison concerns an approximate inverse procedure. A calculation
trace can make forward rendering reliable. Storing the resulting pairs in a
prompt lets a much shorter inverse call retrieve an answer. This turns available
information into a usable representation; it does not add information about the
world. The generated label is known by construction, since z is selected before
rendering x. The learner never needs a judge to decide whether a guessed inverse
label is correct. Incorrect renderings still damage the resulting memory.

## An exact coverage reference

Consider a restricted recognizer that retrieves an answer whenever the matching
pair has been compiled, and fails otherwise. This is a diagnostic reference,
not a claim that an LLM is exactly such a recognizer. It does not even exploit
the prior to guess an unseen label; a prior-only uninformed guess would attain
accuracy p_1. The unrestricted arithmetic decoder remains perfect.

If N correct distinct pairs cover the N most probable labels, the reference
error is

$$R_{\mathrm{ranked}}(N)=\sum_{z>N}p_z.$$

If the teacher supplies N independent samples from p, the expected error is

$$\mathbb E R_{\mathrm{IID}}(N)=\sum_z p_z(1-p_z)^N.$$

The latter follows because label z remains unseen with probability (1-p_z)^N.
These are exact finite-support formulas, allowing repeats in the IID stream.
Ranked generation is a designed curriculum; the model does not discover the
schedule. A teacher using the same curriculum reaches the same reference curve.

For infinite support and p_z=C*z^(-s), s>1, the integral tail gives

$$R_{\mathrm{ranked}}(N)\sim\frac{C}{s-1}N^{-(s-1)}.$$

For IID acquisition, replace (1-p_z)^N by exp(-N*p_z) in the tail. Substituting
u=N*C*z^(-s) in the corresponding integral yields

$$\mathbb E R_{\mathrm{IID}}(N)\sim
\frac{C^{1/s}}{s}\Gamma(1-1/s)N^{-(s-1)/s}.$$

At s=1.5, the exponents are 1/2 and 1/3. In the actual 256-label experiment,
the finite-support reference slopes are different:

| Fit window N | Ranked | IID |
|---|---:|---:|
| 4–32 | 0.617 | 0.360 |
| 8–64 | 0.723 | 0.378 |
| 16–128 | 0.936 | 0.404 |

The fixed primary window is 8–64. The steepening ranked slope at larger N
comes partly from approaching complete coverage of finite support. An empirical
fit over these windows cannot establish an asymptotic infinite-support exponent.

## Failure modes and controls

With independent rendering errors epsilon_z, even a perfect lookup mechanism
has an additional error contribution from incorrectly compiled labels. Ignoring
collisions between incorrect codes and other stored codes, the expected miss
mass after ranked rehearsal is

$$R(N)=\sum_{z>N}p_z+\sum_{z\leq N}p_z\epsilon_z.$$

A persistent error rate can therefore create a nonzero floor. Incorrect codes
can also collide with correct stored codes and create contradictory labels.
Neither effect is repaired in the experiment. Forward accuracy and probability
mass covered by correct codes are diagnosed only after inference.

The matched teacher-ranked control isolates the quality of the generated codes.
The generated-IID control isolates generation from curriculum coverage. If its
codes are correct, it is the same content as teacher-IID. Longer-direct inference
tests whether allocating calculation tokens at query time supplies an equally
useful alternative to compilation. Its token allowance is a control on available
work; API token counts are not measured FLOPs.

This experiment permits test queries to revisit compiled states. Such overlap
is the intended amortization mechanism, and the IID teacher has the same
opportunity. It does not test generalization to exclusively unseen latent labels,
learning an unknown encoder from a few observations, or benefits of authorship
when the final prompt is identical. Those would require different experiments.
