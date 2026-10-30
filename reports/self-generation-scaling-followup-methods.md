# A scaling improvement without different exponents

This reduction belongs to r5s, which uses successful-only updates normalized by the number accepted. It is not the s4g rule that also fitted all-failure groups.

## Expected update

For input x, let p(x) be current correct-answer probability and p_s(x) source correctness. Two independent source candidates supply a correct target with probability

    q_s(x) = 1 - (1 - p_s(x))².

With one correct answer, an accepted group updates along grad log p(x). For IID batches of B questions, accepted input examples have density proportional to the original input density times q_s(x). Averaging the accepted examples, and taking zero update for an entirely rejected batch, gives exactly

    E[update] / eta = [1 - (1-Q_s)^B] E[q_s(x) grad log p(x)] / Q_s,
    Q_s = E[q_s(x)].

The bracket is the probability that the whole batch contains at least one accepted example. At the observed B=32 and acceptance around 0.74 or higher, its departure from 1 is negligible. The distribution of an accepted example is independent of the accepted count conditional on that count being positive, which justifies this normalization identity.

For current proposals, q_s=p(2-p); the numerator is grad E[2p-p²/2]. For frozen proposals, q_s=q_0; the numerator is the gradient of the weighted correct-label log likelihood E[q_0 log p]. The scalar normalization changes time but not the direction at each parameter value. Both objectives favor correct answers, although parameter sharing can cause interference between inputs.

## A single layer with linear energies

Let inputs be one-hot vectors e_j with j uniform among m modes. Each mode emits a binary answer h through the normalized conditional model

    P_w(h=1 | e_j) = exp(w_j)/(1+exp(w_j)),
    P_w(h=0 | e_j) = 1/(1+exp(w_j)).

The energy is linear in the weights; the normalization is the Bernoulli/Boltzmann partition function. The correct answer is h=1 for every mode. There is no hidden layer or nonlinear feature transformation. Logistic output probabilities are required by the probabilistic interpretation.

There is also an explicit RBM embedding. Let the visible vector be binary x in {0,1}^m, with one binary hidden answer h, no biases, and energy E_w(x,h)=-h w·x. Its joint partition function is

    Z(w) = 2^m + product_j(1+exp(w_j)).

Conditioning this normalized RBM on a one-hot practice input x=e_j gives exactly the gates above: P(h=1|e_j)=sigmoid(w_j). The practice distribution supplies the one-hot inputs uniformly; it is not required to equal the RBM's visible marginal. Training here optimizes the checker-filtered conditional policy. It is not contrastive divergence or joint-density maximum likelihood, and improvement in conditional answers does not establish improved joint density estimation.

In population time t=eta times optimizer updates, neglecting the negligible all-rejected-batch factor, write p_j=sigmoid(w_j) and epsilon_j=1-p_j. Then

    d w_j/dt = q_s,j epsilon_j / sum_k q_s,k.

Current sampling eventually has all q_j approaching 1. Therefore each epsilon_j is asymptotic to m/t, and mean error is m/t. Frozen sampling keeps q_0,j fixed and positive, giving

    mean error_current ~ m/t,
    mean error_frozen ~ Q_0 sum_j(1/q_0,j)/t,
    Q_0 = mean_j q_0,j.

Both exponents are 1. At equal learning rate and batch size, the asymptotic ratio of required candidate samples is

    gain = mean(q_0) mean(1/q_0) >= 1.

The inequality is Cauchy–Schwarz. Equality holds when the modes have identical initial acceptance. Heterogeneous difficulty creates the improvement. The numerical example p_0=(0.2,0.75,0.95) predicts gain 1.235972431 before either population integration or finite simulation. Its unfitted error amplitudes are 3 and 3.707917293. This is a mechanism demonstration, not a prediction of the LLM's coefficient.

The one-mode population update is simply d w/dt=1-p in both arms. Removing the acceptance-rate normalization would instead introduce a trivial yield-dependent drift coefficient even in that homogeneous case. This is why r5s normalizes by accepted examples.

## Interpreting the weights and the sample axis

All modal weights grow without bound while success approaches 1. Their cosine with the equally positive direction measures whether initially weak modes catch up with stronger ones. It is not overlap with a finite ground-truth maximum-likelihood parameter; perfect deterministic classification has its optimum at infinite logits.

The population overlap has an additional exact reduction. Set R_j=exp(w_j) and use common internal time dS=dt/sum(q_s). For current sampling,

    dR_j/dS = R_j²(R_j+2)/(R_j+1)³,
    Phi(R) = R + (5/4)log R - 1/(2R) - (1/4)log(R+2),
    Phi(R_j(S)) - Phi(R_j(0)) = S.

Therefore the differences between the transformed modes remain fixed. Since t is asymptotic to mS, the variance across logit weights satisfies

    Var_j(w_j) ~ m² Var_j(Phi(R_j(0))) / t².

The mean weight is asymptotic to log(t/m). Writing rho for its cosine with the all-positive equal-weight direction,

    1-rho_current ~ m² Var(Phi(R_0)) / [2 t² log²(t/m)].

For frozen sampling, R_j+log R_j grows exactly as q0_j t/sum(q0) plus its initial constant. Hence Var(w_j) approaches Var(log q0_j), while its mean grows as log(t/sum(q0))+mean(log q0). The frozen overlap therefore closes only as a constant divided by log²(t). Both approaches still have inverse-time error. These population formulas use their initial weights only and are checked against the stored trajectories without amplitude fits. Finite-batch noise can dominate the faster population alignment tail; the displayed finite simulations are not asserted to obey that population overlap asymptote.

Candidate count C=2 B times updates is proportional to t at fixed B and eta. Accepted count is a different axis: frozen sampling produces fewer accepted examples per optimizer update. Matching accepted examples therefore also changes the number of updates, so that comparison alone does not isolate an intrinsic per-example mechanism. Report both axes and the accepted class distribution.

The LLM harness deliberately computes all 32 training rows, with zero gradient weight on rejected rows. This matches forward/backward work. An optimized implementation could skip rejected rows; a counterfactual token-work proxy is proposal tokens plus three times accepted training tokens. Report this alongside the actual full-batch proxy (proposal tokens plus three times all training tokens), and do not claim wall-clock equivalence from candidate counts alone.

## Limits

The toy has independent modes. The LLM shares weights, has many within-class difficulty levels, and can generalize to inputs whose initial probability was very low. A larger LLM gain is compatible with these differences, but needs measurement rather than being attributed to the toy formula. Neither construction demonstrates improvement without correctness feedback or an advantage over direct uniform gold-label supervision.
