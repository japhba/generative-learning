# Specificity from a restricted recognizer

This is a constructed weight-space example, not an inferred LLM mechanism. It
makes the reader–source interaction explicit while holding source correctness at
100%. It implements a sleep phase with fixed generators; no wake update or
emergence of generator–recognizer alignment is demonstrated.

## Task and two equally valid sources

The input is a positive magnitude m and a binary context c, drawn independently.
Output a signed latent number z such that |z|=m. Both signs are accepted for every
input; c does not affect which answers are valid. For (m,c)=(3,1), both +3 and −3
are correct. No answer is a universal constant.

Generator A chooses z=m. Generator B chooses z=(−1)^c m. Both see the very same
input sequence and produce equally correct labels. Their joint distributions can
be generated in the forward direction: A samples positive z and independent c;
B samples signed z and sets c=1[z<0]. Both render m=|z| and produce the same
observed distribution of (m,c).

Reader A has the feature phi_A=m; reader B has phi_B=(−1)^c m. Each has just one
trainable scalar, w, and predicts z_hat=w*phi_r. These are their fixed available
representations. Both start at w=0.25. Their task performance is identical at a
given w: normalized reconstruction error is (|w|−1)^2.

## One shared learning rule

For every saved input/label pair from either source, minimize

    L_r(w; m,c,z) = 0.5 * ((w*phi_r(m,c) - z)/m)^2.

This is also a Gaussian recognition negative log likelihood, up to constants,
with a fixed variance proportional to m^2. Since phi_r^2=m^2, SGD is simply

    w <- w + eta * ((phi_r/m)*(z/m) - w).

There is no source label in this rule, correctness bonus, verifier, filtering,
or source-specific learning rate. eta=0.1 in the simulation. The generator's
sampled latent is the sleep training target, as in the original
[wake–sleep algorithm](https://www.cs.toronto.edu/~hinton/absps/ws.pdf).

With the reader's own generator, (phi_r/m)*(z/m)=1 on every example. Therefore

    w_N = 1 - (1-w_0)*(1-eta)^N
    error_N = (1-w_0)^2 * (1-eta)^(2*N).

With the other generator the same correlation is (−1)^c, equally likely ±1.
Its mean is zero, so E[w_N]=(1-eta)^N*w_0: the best restricted least-squares
recognizer averages incompatible latent branches toward zero. With fixed step
size there is also residual sampling variance eta/(2-eta). The simulation reports
actual per-run reconstruction error, not error of the mean weight.

Thus the two readers have a symmetric own-source advantage even though both
source sets consist entirely of valid solutions on the same inputs. The key
quantity is a feature–target correlation, computed by the same update, rather
than an assigned utility or a strategy-logit bonus. A shuffled source tag on
identical saved text would change nothing.

## What the control and limitations tell us

Give both readers the two features [m,(−1)^c m], keep source-blind SGD, and start
them identically at [0.25,0.25]. Now they can fit either generator. The simulated
error goes to zero for either source, removing reader specificity. This rich
control halves its step size because the feature-vector squared norm doubles;
within the rich control both sources still use the identical learning rule.

The restricted feature maps and their alignment with the generators are assumed,
not learned. A sign-invariant reconstruction-only objective would also remove
the need to imitate a particular latent branch. The mechanism therefore needs a
limited, branch-sensitive recognizer and a mismatch between fitting latent labels
and accepting any valid solution. This is an existence proof for the proposed
idea, not evidence that these assumptions describe Qwen or Gemma.

The script checks the exact native learning curve against 4,096 sampled runs,
verifies both sources' correctness for every example, checks symmetry of the
2×2 restricted matrix, and verifies convergence with both features. This model
has exponential relaxation, not a different power-law exponent. Frozen LLM ICL
would require an analogous temporary inference-state update rather than literal
changes to this scalar weight.
