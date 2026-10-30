# Linear Gaussian self-generation and access to existing knowledge

This is a possibility proof for closing an inference gap. It is not yet a model
of an advantage from refreshing an autoregressive policy. The generator is fixed,
so fresh samples from the current and initial generator have identical laws.

Suppose past experience has supplied a generative matrix A. Its knowledge is an
explicit initial condition, not something recovered for free from unlabeled data.

```
z ~ N(0, I)
x | z ~ N(Az, sigma² I)
recognition mean: Bx
C = A Aᵀ + sigma² I
B* = Aᵀ C⁻¹
```

Both mappings are linear. The exact posterior mean is B*x, and its irreducible
expected squared error is tr(I - Aᵀ C⁻¹ A). For any recognition matrix B,

```
E ||Bx - z||² = tr(I - Aᵀ C⁻¹ A) + tr((B-B*) C (B-B*)ᵀ).
```

The second term is an explicit weighted distance between the recognition weights
and the optimal inverse already implied by the generator. This is our elicitation
gap: no newly observed facts are needed to reduce it.

Generate z internally, generate x from it, and fit B to recover z. The population
gradient of half the squared loss and the finite-step mean dynamics are

```
gradient_B = B C - Aᵀ
B_(t+1) - B* = (B_t - B*) (I - eta C).
```

In contrast, drawing a target from the recognizer itself with mean Bx gives
E[(Bx - z_self) xᵀ | B] = 0. Finite-batch noise can move the weights but supplies
no systematic first-order correction toward B*. The asymmetry in the successful
condition is explicit: the generative map carries useful information that the
recognition map does not yet express. This is the principle of the
[wake–sleep recognition update](https://www.cs.toronto.edu/~hinton/absps/ws.pdf).

For a fixed positive-definite C the population dynamics are geometric in each
eigenmode, and fixed-rate stochastic updates leave a noise floor. No universal
power-law exponent follows. A broad spectrum can produce a finite intermediate
range, but engineering that range would not establish an LLM mechanism.

`simulate_elicitation_toy.py` uses actual 8×8 matrices, 32 independent SGD runs,
batch size 64, 512 updates, and eta 0.08. It verifies the posterior normal equations and
the exact risk decomposition at all checkpoints. It also reverses the generator's
latent semantics as a negative control: A and -A have the same input marginal,
but their correct latent inverses have opposite signs. Dreaming can therefore
consolidate an incorrect model as well as a useful one.

The simulation demonstrates a possible mechanism and its limit. It does not
compare matched on-policy and off-policy LLM training, and it is not a statistical
replication of the LLM pilot's arithmetic or permutation task.

```
OPENBLAS_NUM_THREADS=1 "$VENV_LOCAL/generative-learning/bin/python" experiments/simulate_elicitation_toy.py
```
