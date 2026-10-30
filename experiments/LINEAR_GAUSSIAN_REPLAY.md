# Linear Gaussian replay: calibration without subspace discovery

This is an analytic construction with deterministic numerical checks. It is not
an explanation established for LLM in-context learning, and the curves are
indexed by population gradient updates, not by sample count.

Let the teacher generate **x** = **W**_T **z**, with standard Gaussian **z**.
The student has a d-by-k matrix **W**, k < d, recognizes with **W**^T, and
reconstructs with the tied map **W W**^T. Only observations are supplied to
learning; neither condition receives sampled latent codes or correctness labels.

Use the same reconstruction loss for either source:

    L_C(W) = 0.5 E ||x - W W^T x||^2
           = 0.5 tr[(I-W W^T) C (I-W W^T)].

The teacher covariance is C_T = W_T W_T^T. The own source is generated offline
from G = W_0, so C_S = W_0 W_0^T stays fixed while the student learns. All
generated observations are treated as constants in differentiation. The same
initialization, loss, optimizer, learning rate, and update count apply to both
sources. Held-out risk is L_{C_T}(W), integrated exactly over fresh teacher data.

The tied reconstruction gradient is

    grad L_C(W) = [(W W^T-I) C + C (W W^T-I)] W.

For W_0 = U diag(s_0) V^T, population own-source dynamics preserve U and V,
and each singular value follows gradient flow

    ds_j/dt = 2 s_{0j}^2 s_j (1-s_j^2).

Every initially nonzero singular value tends to one. Its singular vectors do not
rotate. Even individual stochastic updates from own samples cannot leave the
initial column space: both the sample x and every column of W start within that
space, and the gradient is made from linear combinations of those vectors.

If generations instead come from the current student before each update, holding
the generated sample fixed during that update gives

    dW/dt = 2 W (W^T W) (I-W^T W).

This also preserves the singular vectors in population gradient flow. It is a
different protocol from frozen offline generation and is not what the plot runs.

Teacher risk separates into two orthogonal contributions:

    R_T(W) = 0.5 ||(I-U U^T) W_T||_F^2
           + 0.5 ||(I-diag(s)^2) U^T W_T||_F^2.

Own replay can remove the second term, a gain-calibration error. It leaves the
first term, missing teacher directions, intact. Once W^T W = I, own samples
already reconstruct perfectly and every own-source reconstruction gradient is
zero, regardless of how poorly the subspace agrees with the teacher. If k = d
and W is invertible, orthonormalization makes the reconstruction map the identity
on all inputs; this is why a bottleneck k < d is needed for this task to be
informative about learned representations.

The fixed examples use d=2, k=1, a unit teacher at angle zero, student angle 30
degrees, and initial student scale either 0.5 or 1. Their exact teacher risk is

    R_T(s,theta) = 0.5 [sin(theta)^2 + cos(theta)^2 (1-s^2)^2].

The scale-0.5 student's risk starts at 0.3359375 and tends to 0.125 under own
replay. The scale-1 student's risk stays at 0.125. Teacher replay can change the
angle, and converges toward zero risk in both examples. These cases illustrate
the derivation; they do not establish a universal finite-sample source ordering.

Run experiments/linear_gaussian_replay.py to reproduce the curves. It uses exact
covariances, eta=0.02 and 400 steps; checks the full matrix gradient against
finite differences; verifies the risk decomposition and unchanged own-source
subspace; and records the complete weight/risk trajectory in
results/task_search/linear_gaussian_replay/analysis.json.

Alternative objectives matter:

- If sampled latents are supplied, a sleep recognition loss with frozen generator
  G and trainable recognizer W^T is 0.5 ||W^T G-I||_F^2. At G=W, its partial
  gradient is W(W^T W-I), again a calibration update that has no new teacher
  direction. With the offline generator held fixed, the recognizer tends to G's
  pseudoinverse, rather than enforcing unit singular values of its own weight
  matrix. Keeping latent targets is an extra assumption; the reconstruction
  example above does not require them. For example, take scalar teacher
  generation x_T=z, frozen self generation x_S=0.5z, and recognizer w*x initially
  at w=0.5. Sleep regression drives w toward 2. Teacher latent risk
  0.5 E(w*x_T-z)^2 starts at 0.125, reaches zero at w=1, then tends to 0.5.
  This latent-coordinate risk assumes the teacher's latent axes are fixed.
- Exact maximum likelihood on one's own current distribution has zero expected
  score gradient under the usual regularity conditions. Add observation noise
  for a nonsingular Gaussian model before applying that statement to k < d.
- Comparing W directly with W_T, or requiring recovery of z_T, needs a convention
  or supervision fixing latent coordinates. The isotropic Gaussian prior makes
  W_T and W_T Q observationally indistinguishable for orthogonal Q. Reconstruction
  risk avoids this coordinate ambiguity but measures only a rank-k projection
  task, not density estimation or general reasoning competence.
- If W_S W_S^T = W_T W_T^T, Gaussian observations from the two sources have
  identical distributions. A learner receiving only those observations, with
  identical protocols and no source tags, cannot have a systematic source
  advantage conditional on these fixed generators. Visible latent codes or
  different trajectories would change what information is observed.

The missing ingredient for selective reinforcement of useful rollouts is a
content-dependent assessment anchored to something beyond self consistency:
initial observations, task constraints, or a computation that recognition can
perform more effectively once the trajectory is exposed. Nonlinearities or
depth alone do not provide that anchor. Both source conditions must receive the
same assessment mechanism and information.

The sleep analogy is to generated hidden states teaching recognition weights in
Hinton et al. (1995), https://www.cs.toronto.edu/~hinton/absps/ws.pdf. A tied
linear autoencoder is not the full alternating wake-sleep algorithm.
