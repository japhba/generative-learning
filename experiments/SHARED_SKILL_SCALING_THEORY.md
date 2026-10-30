# Different exponents from a shared skill and a broad difficulty tail

Constructed on 8 September 2026 after viewing the LLM results. This is an analytic mechanism demonstration, checked by numerical integration. It is not a fitted LLM model, an established quantitative explanation of r5s, or a claim of literature novelty.

The earlier r5s toy had finitely many independent trainable mode weights and predicted a coefficient improvement with the same inverse-time exponent. Here one trainable scalar helps across inputs, and problem difficulty is unbounded. With the same verified-only training rule, these assumptions permit different exponents.

## Model and sampling rule

Draw the difficulty d independently on every question from an exponential distribution:

    density(d) = a exp(-a d), d >= 0.

The learner has one shared parameter theta, initially zero. It samples a binary answer h, with correct answer h=1 and

    p_theta(d) = P(h=1 | d) = sigmoid(theta - d).

Improving theta helps every problem; larger d makes a problem harder. This is a conditional Boltzmann model with a linear logit. There is no hidden feature transformation. It also has a normalized joint binary/continuous Boltzmann embedding:

    Energy(d,h) = a d + h(d - theta),
    Z(theta) = 1/a + exp(theta)/(a+1).

The practice distribution is the specified exponential distribution, not the joint model's theta-dependent visible marginal. The trained objective is conditional verified self-training, not joint-density likelihood or contrastive divergence. The scalar is a shared correctness coordinate; it is not a proposal that a single global digit bias explains the LLM.

For each of B IID difficulties, draw two independent answers. Keep a correct sampled answer when at least one exists and otherwise reject the question. Average the log-likelihood gradient over accepted questions; if none is accepted, take no update. The two sources are either the current theta or the frozen initial theta=0. Both students update theta.

Define epsilon_theta(d)=1-p_theta(d), q_s(d)=p_s(d)(2-p_s(d)), and Q_s=E_d[q_s(d)]. Since the gradient of log p_theta is epsilon_theta, the exact expected update at a fixed theta is

    E[Delta theta]/eta = c_B(Q_s) E[q_s epsilon_theta]/Q_s,
    c_B(Q) = 1-(1-Q)^B.

All derivations below use its population gradient flow, with t=eta times updates. At fixed B and eta, the candidate budget is 2 B t/eta, so time and candidate exponents coincide in this limit. The numerical integration does not execute that many SGD steps. Finite-step stochastic trajectories are a separate question.

## The error as a function of the shared weight

For 0<a<1, the mean probability of a wrong answer is

    F(theta) = E[epsilon_theta(d)]
             = a exp(-a theta) integral_{exp(-theta)}^infinity u^(-a)/(1+u) du
             ~ C_a exp(-a theta),
    C_a = a pi/sin(pi a).

Here u=exp(d-theta). The greedy decision is wrong exactly when d>theta, so its population error is

    G(theta) = exp(-a theta), theta >= 0.

Sampled-answer and greedy errors therefore have the same leading exponent in the broad-tail regime. This separation is not only confidence sharpening on a fixed set of already-correct decisions.

## Frozen generation

The frozen acceptance q0(d) decays approximately as 2 exp(-d) on hard inputs. In fact q0(d) exp(d) is bounded, so dominated convergence gives

    d theta/dt ~ K_f exp(-theta),
    K_f = c_B(Q0) E[q0(d) exp(d)]/Q0.

Consequently exp(theta)~K_f t, and

    F_frozen(t) ~ C_a (K_f t)^(-a).

The frozen generator keeps supplying essentially the same, relatively easy, accepted distribution. As the student learns those questions, their residual gradients shrink exponentially in theta.

## Current generation

Now q_theta=1-epsilon_theta^2, so the gradient numerator is

    H(theta) = E[epsilon_theta - epsilon_theta^3].

Evaluating the same tail integral as for F gives

    H(theta)/F(theta) -> a(3-a)/2.

Also Q_theta approaches 1, and therefore c_B(Q_theta) approaches 1. Since dF/dtheta is asymptotic to -a F,

    dF/dt ~ -[a^2(3-a)/2] F^2,
    F_current(t) ~ 2/[a^2(3-a) t].

Thus, for 0<a<1,

    frozen exponent = a,
    current exponent = 1.

No temperature change, task ranking, new output head, or growing candidate count is required. The questions remain IID. The generator changes which questions supply accepted examples.

The mechanistic difference is visible in the weight:

    theta_frozen ~ log t,
    theta_current ~ (log t)/a.

At the moving difficulty level d approximately equal to theta, current success is about one half, and two candidates have acceptance about three quarters. Such questions still provide substantial gradients. Their probability mass scales as exp(-a theta). The initial frozen generator rarely supplies correct answers at this moving frontier. Shared improvement in theta lets success on easier questions help harder questions before they receive much direct accepted practice.

For a=1/2 the concrete laws are

    F_current(t) ~ 3.2/t,
    F_frozen(t) ~ 0.7857026329/sqrt(t), for B=32 and theta0=0.

These amplitudes are derived, not fitted. At a=1/2 there is also an exact finite-theta check:

    F(theta) = atan(exp(theta/2)) exp(-theta/2),
    H(theta) = 5 F(theta)/8 - 3/[8(1+exp(theta))] - 1/[4(1+exp(theta))^2].

Both match the independent difficulty quadrature to about machine precision.

## When the exponent separation disappears

The assumptions matter:

- If a>1, the difficulty distribution has a finite E[exp(d)]. Sampled-answer error is then proportional to exp(-theta), and both methods have exponent 1. Greedy error has exponent a in both methods.
- At a=1, both leading power exponents are 1, with logarithmic corrections: frozen sampled error is proportional to log(t)/t, while current sampled error is proportional to 1/t.
- If difficulty has a fixed finite upper bound, both sampled-answer errors eventually have exponent 1, and greedy error eventually reaches zero. A large bound can still allow a long intermediate range with different effective slopes.
- The argument assumes a shared trainable correctness direction and an unbounded tail of difficulties. It does not apply automatically to independent per-problem weights or to an arbitrary LLM.
- Correctness feedback remains essential to the specified update. Unfiltered samples have a different expected update.

The construction shows that verified on-policy generation can change a scaling exponent. It does not establish that the r5s LLM has an exponential difficulty tail, that one shared parameter captures its learning, or that its observed exponent gap persists indefinitely. The existing fixed-answer bias control cannot explain the LLM's performance; a real shared correctness direction would have to improve appropriate answers across different questions.

The literal toy task is binary success versus failure with h=1 defined as correct. It assumes how a shared ability changes success probabilities; it does not implement or derive an arithmetic algorithm. The actual LLM task counts a target digit in a four-digit list, with balanced answers 0/1/2. The difficulty-tail assumption has not been measured for that task.

A subsequent descriptive check used the already exposed LLM monitoring cohort. The one-skill model predicts a constant log-odds increase across questions and exact preservation of their difficulty ordering. At update 512, the LLM's correlation between initial and current probability ranks is 0.466–0.506 across the three current-source seeds, versus 0.848–0.862 for frozen-source students; the literal toy would give 1. All 1023 probability pairs have finite logits at these checkpoints. Rankings also change within each answer class. Thus the one-parameter model is not a literal description of these trajectories. This does not rule out multiple shared skills, but it prevents importing the toy's numerical exponents without further modeling. The check and all settings are retained in audit_shared_skill_compatibility.py and results/scaling_r5s/shared_skill_theory/llm_compatibility.json.

## Numerical checks and relationship to prior work

The script experiments/shared_skill_scaling.py evaluates exact population integrals with composite Gauss-Legendre quadrature and integrates dt/dtheta=1/drift. It checks a=0.5, a=0.75, a=1.5, and a=0.5 with difficulty capped at 12. A second quadrature order and theta spacing check numerical resolution. All settings are retained in results/scaling_r5s/shared_skill_theory/.

Over population times 1e8 through 1e12, the measured current/frozen sampled-error exponents are 1.000000/0.500006, 1.000107/0.749377, 1.000009/0.999991, and 1.000001/0.999993, respectively. The two broad-tail cases also recover the derived error amplitudes and greedy-error exponents. These are numerical checks of an analytic model, not independent LLM experiments or finite-sample SGD evidence.

Related research gives broader reasons to investigate spectrum, task difficulty and representation sharing. [Bordelon, Canatar and Pehlevan (2020)](https://proceedings.mlr.press/v119/bordelon20a.html) derive learning curves from task and kernel spectra. [Bordelon, Atanasov and Pehlevan (2025)](https://arxiv.org/abs/2409.17858) show that feature learning can improve training-time exponents for hard tasks in their solvable model, while leaving easier-task exponents unchanged. Those papers analyze different comparisons and do not prove the verified current-versus-frozen construction above. Both r5s LLM students update their weights; the frozen object is the candidate generator.
