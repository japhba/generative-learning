"""Smallest recognition-parametrised model (RPM): one binary latent z = +-1 (p(z) uniform), two views
x_j = z mu + b_j nu + sigma eps_j (b_j = +-1 independent per view: nuisance), readout f(z | x) = sigmoid(z w.x) shared by
the views, F_j(z) = batch mean of f over view j. log W = log sum_z p(z) prod_j f(z | x_j) / F_j(z).
Learning uses the local rule delta_j = q(+) - f_j(+) [1 + r_+ - <r>_{f_j}], dw = mean_n sum_j delta_j x_j (checked against
finite differences). True single-view posterior: logit = 2 mu.x / sigma^2, so w* = 2 mu / sigma^2.
Also prints how many prior samples a plain Monte Carlo estimate of p(x) needs (chi^2 of posterior vs prior).
Usage: python rpm_two_view_toy.py -> reports/rpm-two-view.png"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
sig = lambda a: 1 / (1 + np.exp(-a))


def data(N, mu, nu, sigma, views=2):
    z = rng.choice([-1, 1], N)
    return np.stack([z[:, None] * mu + rng.choice([-1, 1], N)[:, None] * nu + sigma * rng.standard_normal((N, len(mu))) for _ in range(views)], 1)


def step(w, X):
    """log W per item and the local-rule gradient of its batch mean (p(z) = 1/2 fixed)."""
    fp = sig(X @ w); f = np.stack([fp, 1 - fp], -1)                                   # N, V, 2 (z = +1, -1)
    F = f.mean(0)                                                                     # V, 2
    u = 0.5 * np.prod(f / F, 1); W = u.sum(-1); q = u / W[:, None]                    # posterior over z, N, 2
    r = q.mean(0)[None] / F                                                           # V, 2 (Q bar / F bar per view)
    delta = q[:, None, 0] - fp * (1 + r[:, 0] - (f * r).sum(-1))                     # N, V
    return np.log(W), (delta[..., None] * X).sum(1).mean(0)


def check_gradient():
    X = data(500, np.array([1.0, 0.0]), np.array([0.0, 3.0]), 1.0); w = np.array([0.7, -0.4]); _, g = step(w, X); e = 1e-6
    fd = np.array([(step(w + e * d, X)[0].mean() - step(w - e * d, X)[0].mean()) / (2 * e) for d in np.eye(2)])
    print(f"local rule vs finite differences: max |diff| = {np.abs(g - fd).max():.1e}")


def train(X, w0, steps, lr):
    w = w0.copy(); T = np.unique(np.round(np.geomspace(1, steps, 60)).astype(int)); out = {"t": [0], "w": [w.copy()], "obj": [step(w, X)[0].mean()]}
    for t in range(1, steps + 1):
        _, g = step(w, X); w = w + lr * g
        if t in T: out["t"].append(t); out["w"].append(w.copy()); out["obj"].append(step(w, X)[0].mean())
    return {k: np.array(v) for k, v in out.items()}


def prior_sampling_table():
    """Samples S for 10 % relative error of p_hat(x) = mean_s p(x | z_s), z_s ~ p(z): S = 100 chi^2(p(z|x) || p(z)), median over x."""
    print("plain Monte Carlo from the prior, samples for 10 % relative error (median over x):")
    for D, s in [(1, 1.0), (1, 0.1), (10, 1.0), (10, 0.1), (100, 1.0)]:
        x = rng.normal(0, np.sqrt(1 + s ** 2), (20000, D)); lN = lambda x, v: -x ** 2 / (2 * v) - 0.5 * np.log(2 * np.pi * v)
        l1chi = (lN(x, 1 + s ** 2 / 2) - np.log(2 * s * np.sqrt(np.pi)) - 2 * lN(x, 1 + s ** 2)).sum(1)   # log(1 + chi^2)
        I = D / 2 * np.log(1 + 1 / s ** 2)
        print(f"  Gaussian z in R^{D:<3d} sigma {s:<4}  I(x;z) {I:6.2f} nats  S ~ {100 * np.expm1(np.median(l1chi)):.2g}")
    for L, n in [(3, 8 * 2 ** 7), (4, 8 * 2 ** 15)]: print(f"  RHM depth {L} symbolic leaves (parse unique)  I = log {n} = {np.log(n):.2f} nats  S ~ {100 * (n - 1):.2g}")


def main():
    check_gradient(); prior_sampling_table()
    fig, AX = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # (a) mean update of the scalar weight against w, no nuisance: zero crossing at w* = 2 mu / sigma^2
    ax = AX[0, 0]; grid = np.linspace(0.01, 6, 120)
    for s, c in [(0.7, "#2a6fdb"), (1.0, "#eb6834"), (1.5, "#8a4fd8")]:
        X = data(100000, np.array([1.0]), np.array([0.0]), s); g = np.array([step(np.array([w]), X)[1][0] for w in grid])
        ax.plot(grid, g, color=c, label=f"noise sigma = {s}"); ax.axvline(2 / s ** 2, color=c, ls=":", lw=1)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xlabel("recognition weight w"); ax.set_ylabel("mean update dw")
    ax.set_title(r"(a) 1-D: update vanishes at $w^* = 2\mu/\sigma^2$ (dotted)", loc="left", fontsize="medium"); ax.legend()

    # (b, c, d) two dimensions with nuisance: shared direction mu = (1, 0), per-view background nu = (0, 3), sigma = 1
    mu, nu = np.array([1.0, 0.0]), np.array([0.0, 3.0]); X2 = data(20000, mu, nu, 1.0); X1 = X2[:, :1]; w0 = np.array([0.01, 0.05])
    run2, run1 = train(X2, w0, 3000, 0.5), train(X1, w0, 3000, 0.5); wstar = 2 * mu
    ax = AX[0, 1]
    ax.plot(run2["t"][1:], run2["w"][1:, 0], color="#2a6fdb", label="two views: weight on the shared signal mu")
    ax.plot(run2["t"][1:], run2["w"][1:, 1], color="#eb6834", label="two views: weight on the background nu")
    ax.plot(run1["t"][1:], run1["w"][1:, 0], color="#2a6fdb", ls="--", label="one view only: weight on mu")
    ax.axhline(2, color="0.4", ls=":", lw=1, label="true posterior w* = (2, 0)"); ax.set_xscale("log"); ax.set_xlabel("updates")
    ax.set_title("(b) weights go to the true posterior, background never grows", loc="left", fontsize="medium"); ax.legend()

    ax = AX[1, 0]; Xs = data(200000, mu, nu, 1.0)
    best = step(wstar, Xs)[0].mean()
    ax.plot(run2["t"][1:], run2["obj"][1:], color="#2a6fdb", label="two views"); ax.plot(run1["t"][1:], run1["obj"][1:], color="#2a6fdb", ls="--", label="one view")
    ax.axhline(best, color="0.4", ls=":", lw=1, label=f"at w*: I(x1; x2) ~ {best:.3f} nats"); ax.set_xscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("mean log W [nats]")
    ax.set_title("(c) objective = shared information between the views", loc="left", fontsize="medium"); ax.legend()

    ax = AX[1, 1]; s = np.linspace(-4, 4, 200); cols = plt.cm.Blues(np.linspace(0.35, 1, 5))
    for c, T in zip(cols, [10, 20, 40, 100, 3000]):
        i = np.searchsorted(run2["t"], T); ax.plot(s, sig(run2["w"][i, 0] * s), color=c, label=f"readout after {run2['t'][i]} updates")
    ax.plot(s, sig(2 * s), color="k", ls=":", label="true posterior p(z = +1 | x)"); ax.set_xlabel(r"$\mathbf{x}\cdot\hat{\boldsymbol{\mu}}$"); ax.set_ylabel("f(z = +1 | x)")
    ax.set_title("(d) the readout sharpens onto the true posterior", loc="left", fontsize="medium"); ax.legend()
    out = Path.home() / "generative-learning/reports/rpm-two-view.png"; fig.savefig(out, dpi=110); print(out)
    print("final w two views", run2["w"][-1], "one view", run1["w"][-1])


if __name__ == "__main__": main()
