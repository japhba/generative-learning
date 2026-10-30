# Finite-data wake–sleep: what can improve, and what need not

The model has a uniform prior on four binary causes h and eight conditionally independent binary observations. Context c is the first six observations; y is the last two. A single sigmoid belief-network layer specifies their likelihood. The generator parameters are fitted on the same finite observations available to every method, then frozen. Evaluation concerns p(y | c), so no latent label alignment is required.

## Why the fully expressive baseline catches up

Write a_h = W_context^T h + b_context. The learned generator's posterior satisfies

    log p_N(h | c) = c^T a_h / 2 - sum_j log(2 cosh(a_hj / 2)) + constant(c).

Thus its logits are affine in c, with one output per latent state. Denote their centered coefficients by B_N. They can be compiled directly from the learned generator, even though its parameters need not equal the truth.

On finite real contexts, wake minimizes the empirical mean of KL(q_B(h | c) || p_N(h | c)). Its minimum is zero at B_N. If the observed augmented context design has full column rank, matching all observed posterior logits identifies the centered affine coefficients and hence their values on every context. Fully expressive forward fitting has the same zero-loss solution. This is an attainable solution argument, not a global convergence guarantee for every optimizer or initialization. In the experiment, wake actually recovers the compiled predictor numerically.

Once recognition reaches this solution, its prediction error is the learned generator's conditional estimation error. Sleep cannot provide additional independent information about the true generator. Under a regular local estimation approximation, parameter fluctuations are order N^(-1/2), while conditional KL is quadratic in the fluctuations. This gives an order N^(-1) expected estimation contribution for both methods. It is a conditional asymptotic argument; latent nonidentifiability, local optima, finite N and MAP bias can affect measured finite-range powers.

At this solution, every individual real-context wake gradient is zero. In contrast, an individual dream has cross-entropy gradient q_N(h | c) minus its one-hot sampled cause. Its expectation is zero but its variance is generally positive. Finite stochastic dream batches can therefore leave extra optimization error even when the generator is already perfectly transferred to recognition.

## Why a factorized recognizer changes the question

The mean-field family uses four affine fields:

    q_R(h | c) ∝ exp(h^T (R^T c + b)).

Its latent bits are independent given c. The log-cosh term in the generative posterior generally contains interactions between causes, so this q cannot match the entire learned posterior. Its affine coefficients in state space are also restricted: they are products of a 7-by-4 field matrix and the 4-by-16 state matrix. This makes a weight-space restriction explicit rather than creating an artificial initial neglect bias.

Let u = R^T c + b and m_q = E_q[h] = tanh(u). One sampled dream (h,c) gives

    gradient_u negative_log_q(h | c) = m_q(c) - h.

Taking its generative expectation replaces h with E_pN[h | c]. Thus generation teaches conditional latent moments that the inverse model may not reproduce. Multiplication by c gives the corresponding weight update. This expected update need not vanish: the dream-generating joint p_N(h,c) and the inverse conditional q_R(h | c) are different distributions.

By comparison, the wake field gradient is

    Cov_q(h, log q(h | c) - log p_N(h | c)).

With restricted recognition, these objectives can choose different compromises. The sleep treatment adds forward supervision to wake; it does not merely estimate the same gradient with a different batch. The forward-posterior control extracts soft targets from the learned generator on the existing real contexts, showing that generative information can be transferred without generating new contexts. At large N, learned-model and empirical context distributions become closer, so this control is particularly informative.

In the follow-up, wake obtains a smaller reverse-KL objective while giving worse observable predictions. Sleep and forward teaching trade some reverse-KL optimality for better predictive use of the learned generator. Finite gradients and local optima are still possible; the experiment does not certify global optima.

The earlier local same-curvature argument concerned a neighborhood where q equals p. That point is unavailable in the restricted family. Equality of forward and reverse Hessians at an unattainable point therefore does not rule out different fitting compromises here, and still does not establish different asymptotic data exponents.

## What the observed separation means

A useful decomposition is conceptual rather than an exact additive identity:

    test excess loss = generator estimation effects + recognition approximation effects + optimization effects + interactions.

The fully expressive experiment mostly isolates generator estimation once recognition catches up. In the restricted family, approximation and objective choices leave persistent errors. Reducing those errors improves the learning curve but can change an apparent floor rather than the asymptotic power of N. A fit that ignores nonzero floors can give different effective slopes without establishing different statistical exponents.

The mechanism is not automatic for every probabilistic generative model. For example, a standard RBM's hidden conditional factorizes with affine sigmoid fields. A recognizer with those same fields can represent that conditional exactly. To reproduce this particular restriction there, one needs an additional limitation, partial observation, or a richer inference problem.

For an LLM match, the forward model must contain useful information that an approximate inverse can fail to express. Train or pretrain that forward model from information shared by the methods, generate cause–observation pairs without filtering, and train inverse prediction. Evaluate observable held-out outcomes, account separately for real data and generated practice, and retain posterior teaching or an approximation to it as a control. Simply fitting a predictor to samples from that exact same conditional is a different experiment and retains the zero expected self-score argument.
