# Online learning when the singular values are already correct

This is a new analytic control, distinct from the previous reconstruction-loss
example. It fixes the student's singular values at the teacher's values and
trains only the left singular-vector orientation. The objective is covariance
estimation, so correctness of the prescribed singular values has its ordinary
generative meaning. No latent labels or judgments are supplied to either arm.

In two dimensions, let R(theta) be the rotation matrix and set

    W_T = diag(sqrt(lambda1), sqrt(lambda2))
    W_S(theta) = R(theta) diag(sqrt(lambda1), sqrt(lambda2))
    C(theta) = W_S(theta) W_S(theta)^T.

The right singular-vector matrix can be the identity because standard Gaussian
latents make it observationally unidentifiable. Distinct covariance eigenvalues
make the left singular-vector axes identifiable, up to their sign symmetries.

At each online update, draw a fresh z from N(0,I). The teacher arm supplies
x=W_T z; the own arm supplies x=W_S(theta_current) z. In both arms, recognize
q=W_S(theta_current)^T x and apply the same SGD rule to

    ell(theta;x) = 0.5 ||C(theta)-x x^T||_F^2.

Only the generating matrix differs. The observation is held constant while
differentiating, even in the own arm. All singular values remain fixed by the
angle parameterization. Both arms share the same Gaussian innovations at each
step to allow a paired comparison.

The true held-out excess risk, subtracting the theta-independent minimum
expected loss on fresh teacher samples, is exactly

    R_T(theta) = 0.5 ||C(theta)-C_T||_F^2
               = (lambda1-lambda2)^2 sin(theta)^2.

For y=R(theta)^T x and delta=lambda1-lambda2, the sample gradient is

    d ell/d theta = -2 delta y1 y2
                  = -2 delta q1 q2 / (sigma1 sigma2).

Thus the gradient can be computed from the two recognition activations. On own
data, y_i=sigma_i z_i, so the angle update is

    theta_{n+1} = theta_n + eta c z1 z2,
    c = 2 delta sigma1 sigma2.

This is an unwrapped zero-drift random walk. Its expected increment is exactly
zero conditional on the current student, and its variance per step is eta^2 c^2.
It is not a claim that every realized rotation is zero or that every random
rotation worsens risk. For initially poor orientations, mixing toward a random
orientation can lower expected risk without identifying the teacher.

For teacher data instead,

    E[d ell/d theta | theta] = delta^2 sin(2 theta),

which provides an alignment signal. At a fraction p of teacher samples mixed
with current-student samples, the expected angular gradient is simply p times
this expression. Own samples add no first-order alignment drift in this model.

The exact expected own risk after N constant-step updates follows from the
characteristic function of a product of two independent standard normals:

    rho = (1+4 eta^2 c^2)^(-1/2)
    E[R_T(theta_N)] = delta^2/2 [1-cos(2 theta_0) rho^N].

For an initial error of 30 degrees, expected risk increases toward the random
orientation value. With initial error above 45 degrees, it decreases toward that
same value. Neither case has a teacher-directed angular drift.

The numerical cohort is fixed in advance: lambda=(2,1), theta_0=30 degrees,
eta=0.02, 512 updates, 4096 independent paired runs, seed 20260912. The full
cohort is completed before analyzing risks. Expected initial risk is 0.25 and
expected own risk at N=512 is 0.490364490491529. Teacher SGD has residual noise
because the step size is constant; its nonzero late risk is not an approximation
error of the model class. A decreasing step size is a separate protocol.

The script checks the sample gradient by finite differences, verifies the risk
identity and fixed spectrum, and compares the completed own curves and angular
variance with the exact random-walk predictions. Complete angle trajectories,
per-update mean risks and Monte Carlo standard errors are saved. Plot bands are
pointwise 95% normal Monte Carlo intervals for the mean, not prediction intervals
over individual learners. No power-law exponent is fitted.

For the earlier rank-deficient linear reconstruction model, online sampling
cannot escape the current column space: every own observation and each ordinary
gradient update lie in that space. If columns are already orthonormal, each own
reconstruction gradient vanishes exactly. Unequal singular values permit the
within-span eigenvector fluctuations illustrated here. Rotations within a
degenerate equal-eigenvalue eigenspace do not change the observed distribution.

The related Oja online PCA rule makes the source of subspace rotation explicit:
for an orthonormal U and input covariance C, its population tangent field is
(I-U U^T) C U. A self covariance U Lambda U^T makes this field zero, whereas
teacher covariance can make it nonzero. This subspace statement does not rule
out within-span eigenvector rotations when the spectrum is unequal.

Reference: Oja (1982), A Simplified Neuron Model as a Principal Component
Analyzer, https://neurophysics.ucsd.edu/courses/physics_171/Oja_1982.pdf.
