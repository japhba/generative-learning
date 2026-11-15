"""Numerical checks of tex/notes/model_4_reversible_practice.tex (derivations only there). Nothing here is a new model.
1. Mean flows of the two-level toy (teacher x = (uv, uv, u, v); weights mu_1 = mu_2 = a shallow, mu_3 = p, mu_4 = q deep), started at
   a = p = eps, q = -rho eps: time until the shallow weight / both deep weights reach 1/2 under practice (note: 4 log(1/eps), 6 log(1/eps)) and
   until any weight reaches 1/2 under expected-Hamming reconstruction with the correct root code (note: at least 1/eps - 2).
2. The version with a learned Boolean block (c = its parity component, started at 0): practice time until min(p, -q, -c) = 1/2 (note:
   Theta(1/eps)), reconstruction time until any weight reaches 1/2 (note: at least (eps^-2 - 4) / 2).
3. The two-level toy with single-sample updates instead of mean flow: practice = eta (V - 1/4)(m_i - mu_i) on own samples; reconstruction =
   the score-function step on the Hamming distance to a teacher string with the correct roots, -eta (d - 1)(m_i - mu_i). Steps until the exact
   valid share J exceeds 0.9, and when the shallow agreement mu_1 mu_2 and the deep term (mu_1 + mu_2) mu_3 mu_4 / 2 first exceed 1/2.
4. The same for the learned block (four logits beta_ij of the hidden unit w given (h3, h4), started at 0; eligibility of the active logit w - f_ij).
Usage: python experiments/model4_reversible_practice_check.py (about two minutes)"""

import numpy as np

RHO = 0.5
FLOWS = {"practice, two-level": lambda a, p, q: (0.25 * (1 - a * a) ** 2 * (a + p * q), 0.5 * (1 - p * p) ** 2 * a * q, 0.5 * (1 - q * q) ** 2 * a * p),
         "reconstruction, two-level": lambda a, p, q: (0.5 * (1 - a * a) ** 2 * p * q, (1 - p * p) ** 2 * a * q, (1 - q * q) ** 2 * a * p),
         "practice, learned block": lambda a, p, q, c: (0.25 * (1 - a * a) ** 2 * (a + p * q * c), 0.5 * (1 - p * p) ** 2 * a * q * c, 0.5 * (1 - q * q) ** 2 * a * p * c, 0.125 * (1 - c * c) ** 2 * a * p * q),
         "reconstruction, learned block": lambda a, p, q, c: (0.5 * (1 - a * a) ** 2 * p * q * c, (1 - p * p) ** 2 * a * q * c, (1 - q * q) ** 2 * a * p * c, 0.25 * (1 - c * c) ** 2 * a * p * q)}


def first_time(flow, x, stop, rel=0.002, t_max=1e9):   # Runge-Kutta with a step that moves the largest weight by at most `rel` of itself
    f, t = (lambda y: np.array(FLOWS[flow](*y))), 0.0
    while not stop(x) and t < t_max:
        dt = min(rel * np.abs(x).max() / np.abs(f(x)).max(), 0.05 * max(t, 1.0)); k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3); x, t = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + dt
    return t


def single_samples(rule, eps, eta, runs=512, steps=40000, seed=0):
    rng = np.random.default_rng(seed); th = np.arctanh(np.tile([eps, eps, eps, -RHO * eps], (runs, 1))); when = np.full((3, runs), np.inf)
    for t in range(steps):
        mu = np.tanh(th); J = (1 + mu[:, 0] * mu[:, 1] + (mu[:, 0] + mu[:, 1]) * mu[:, 2] * mu[:, 3]) / 4
        for k, hit in enumerate((J > 0.9, mu[:, 0] * mu[:, 1] > 0.5, (mu[:, 0] + mu[:, 1]) * mu[:, 2] * mu[:, 3] / 2 > 0.5)): when[k, hit & np.isinf(when[k])] = t
        u, v = rng.choice([-1, 1], size=(2, runs)); m = np.where(rng.random((runs, 4)) < (1 + mu) / 2, 1, -1); w = m[:, 2] * u * m[:, 3] * v; x1, x2 = m[:, 0] * w, m[:, 1] * w
        factor = ((x1 == x2) & (x2 * u * v == 1)) - 0.25 if rule == "practice" else -((x1 != u * v).astype(float) + (x2 != u * v) - 1)
        th += eta * factor[:, None] * (m - mu)
    return when


def single_samples_block(rule, eps, eta, runs=512, steps=60000, seed=0):
    rng = np.random.default_rng(seed); th = np.arctanh(np.tile([eps, eps, eps, -RHO * eps], (runs, 1))); beta = np.zeros((runs, 2, 2)); when = np.full(runs, np.inf); r = np.arange(runs)
    for t in range(steps):
        mu, f = np.tanh(th), np.tanh(beta); c = (f[:, 1, 1] + f[:, 0, 0] - f[:, 0, 1] - f[:, 1, 0]) / 4; J = (1 + mu[:, 0] * mu[:, 1] + (mu[:, 0] + mu[:, 1]) * mu[:, 2] * mu[:, 3] * c) / 4
        when[(J > 0.9) & np.isinf(when)] = t; u, v = rng.choice([-1, 1], size=(2, runs)); m = np.where(rng.random((runs, 4)) < (1 + mu) / 2, 1, -1); i, j = (m[:, 2] * u + 1) // 2, (m[:, 3] * v + 1) // 2
        w = np.where(rng.random(runs) < (1 + f[r, i, j]) / 2, 1, -1); x1, x2 = m[:, 0] * w, m[:, 1] * w
        factor = ((x1 == x2) & (x2 == u * v)) - 0.25 if rule == "practice" else -((x1 != u * v).astype(float) + (x2 != u * v) - 1)
        th += eta * factor[:, None] * (m - mu); beta[r, i, j] += eta * factor * (w - f[r, i, j])
    return when


if __name__ == "__main__":
    print("1. two-level mean flow, times in units of log(1/eps) (practice) and of 1/eps (reconstruction)")
    for eps in (1e-2, 1e-4, 1e-8, 1e-16):
        x0 = np.array([eps, eps, -RHO * eps]); L = np.log(1 / eps); ts = first_time("practice, two-level", x0, lambda x: x[0] >= 0.5); td = first_time("practice, two-level", x0, lambda x: min(x[1], x[2]) >= 0.5)
        tr = first_time("reconstruction, two-level", x0, lambda x: np.abs(x).max() >= 0.5) if eps >= 1e-8 else np.nan
        print(f"   eps {eps:.0e}: shallow {ts / L:.2f} log(1/eps), deep {td / L:.2f} log(1/eps); reconstruction {tr * eps:.2f} / eps (note's bound {1 - 2 * eps:.2f} / eps)")
    print("2. learned block, mean flow")
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
        x0 = np.array([eps, eps, -RHO * eps, 0.0]); tp = first_time("practice, learned block", x0, lambda x: min(x[1], -x[2], -x[3]) >= 0.5); tr = first_time("reconstruction, learned block", x0, lambda x: np.abs(x).max() >= 0.5)
        print(f"   eps {eps:.0e}: practice {tp:.3g} = {tp * eps:.2f} / eps; reconstruction {tr:.3g} = {tr * eps * eps:.2f} / eps^2 (note's bound {(1 - 4 * eps * eps) / 2:.2f} / eps^2)")
    print("3. two-level toy, single-sample updates (512 runs, medians in steps; mean flow would give 4 log(1/eps) / eta and 6 log(1/eps) / eta)")
    for eps in (1e-2, 1e-4):
        for eta in (0.05, 0.01):
            for rule in ("practice", "reconstruction"):
                w = single_samples(rule, eps, eta); med = [np.median(x) for x in w]
                print(f"   eps {eps:.0e} eta {eta}: {rule:14s} J > 0.9 after {med[0]:7.0f} ({np.isinf(w[0]).sum()} of 512 never), shallow agreement after {med[1]:7.0f}, deep term after {med[2]:7.0f}; "
                      f"shallow first in {(w[1] < w[2]).mean():.2f} of runs; mean flow {4 * np.log(1 / eps) / eta:.0f} / {6 * np.log(1 / eps) / eta:.0f}")

    print("4. learned block, single-sample updates (512 runs, steps until J > 0.9; mean flow: practice about 8.7 / (eps eta), reconstruction about 3.3 / (eps^2 eta))")
    for eps in (1e-2, 1e-3):
        for rule in ("practice", "reconstruction"):
            w = single_samples_block(rule, eps, 0.05); print(f"   eps {eps:.0e} eta 0.05: {rule:14s} median {np.median(w):7.0f} steps, quartiles {np.quantile(w, 0.25):.0f} to {np.quantile(w, 0.75):.0f} ({np.isinf(w).sum()} of 512 never in 60000)")
