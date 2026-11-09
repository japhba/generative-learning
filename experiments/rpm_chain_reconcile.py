"""RPM on a chain: where observations are reconciled with the prior.
Latent z_t = +-1 with a sticky Markov prior p(z_t = z_(t-1)) = s (true s = 0.9), observations x_t = z_t mu + sigma eps
(mu = 1, sigma = 1), readout f(+ | x) = sigmoid(w x + b) shared over t, F_t = batch mean of f at position t.
Posterior at t = own vote f_t / F_t times the context C_t = P(z_t | all other x) carried by the prior (forward-backward):
logit q_t = [logit f_t - logit F_t] + logit C_t. Truth: w* = 2 mu / sigma^2 = 2, b* = 0, s* = 0.9.
Learning: wake rule on the transition logits, local readout rule delta_t = q_t(+) - f_t(+) [1 + r_+ - <r>_f] (checked
against finite differences). Usage: python rpm_chain_reconcile.py -> reports/rpm-chain.png"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(1)
sig = lambda a: 1 / (1 + np.exp(-a))
logit = lambda p: np.log(p) - np.log1p(-p)


def sample(N, T, s, mu=1.0, sigma=1.0):
    z = np.empty((N, T)); z[:, 0] = rng.choice([1, -1], N)
    for t in range(1, T): z[:, t] = np.where(rng.random(N) < s, z[:, t - 1], -z[:, t - 1])
    return z, z * mu + sigma * rng.standard_normal((N, T))


def readout(w, b, X, F=None):
    fp = sig(w * X + b); f = np.stack([fp, 1 - fp], -1)                          # N, T, 2 (z = +1, -1)
    return f, (f.mean(0) if F is None else F)


def forward_backward(A, e):
    """Scaled forward-backward with evidence ratios e = f / F (N, T, 2), A[a, b] = p(z_t = b | z_(t-1) = a), p(z_1) = 1/2."""
    N, T, _ = e.shape; pred = np.empty_like(e); al = np.empty_like(e); be = np.ones_like(e); c = np.empty((N, T))
    for t in range(T):
        pred[:, t] = 0.5 if t == 0 else al[:, t - 1] @ A
        u = pred[:, t] * e[:, t]; c[:, t] = u.sum(-1); al[:, t] = u / c[:, t, None]
    for t in reversed(range(T - 1)): be[:, t] = (e[:, t + 1] * be[:, t + 1]) @ A.T / c[:, t + 1, None]
    q = al * be; ctx = pred * be; ctx /= ctx.sum(-1, keepdims=True)
    xi = al[:, :-1, :, None] * A[None, None] * (e[:, 1:] * be[:, 1:])[:, :, None, :] / c[:, 1:, None, None]
    return np.log(c).sum(1), q, ctx, xi


def objective_and_grad(theta, X):
    w, b, G = theta[0], theta[1], theta[2:].reshape(2, 2); A = np.exp(G) / np.exp(G).sum(1, keepdims=True)
    f, F = readout(w, b, X); lw, q, ctx, xi = forward_backward(A, f / F)
    r = q.mean(0) / F                                                              # T, 2
    delta = q[..., 0] - f[..., 0] * (1 + r[None, :, 0] - (f * r[None]).sum(-1))    # N, T
    gG = (xi.sum(1) - q[:, :-1, :, None].sum(1) * A[None]).mean(0)
    T = X.shape[1]
    return lw.mean() / T, np.concatenate([[(delta * X).sum(1).mean(), delta.sum(1).mean()], gG.ravel()]) / T


def check_gradient():
    _, X = sample(300, 8, 0.9); th = np.array([0.8, 0.3, 0.4, -0.2, 0.1, 0.5]); _, g = objective_and_grad(th, X); e = 1e-6
    fd = np.array([(objective_and_grad(th + e * d, X)[0] - objective_and_grad(th - e * d, X)[0]) / (2 * e) for d in np.eye(6)])
    print(f"local rules vs finite differences: max |diff| = {np.abs(g - fd).max():.1e}")


def train(X, steps=3000, lr=2.0, w0=0.3):
    th = np.array([w0, 0.0, 0.0, 0.0, 0.0, 0.0]); T = np.unique(np.round(np.geomspace(1, steps, 80)).astype(int)); out = []
    for t in range(1, steps + 1):
        _, g = objective_and_grad(th, X); th = th + lr * g
        if t in T: A = np.exp(th[2:].reshape(2, 2)); A /= A.sum(1, keepdims=True); out.append((t, th[0], th[1], A[0, 0], A[1, 1]))
    return np.array(out)


def example_sequence():
    """True z: + for t = 1..14, - for t = 15..24; x_8 is a noise outlier."""
    z = np.r_[np.ones(14), -np.ones(10)]; x = z + 0.45 * rng.standard_normal(24); x[7] = -1.0; return z, x


def decompose(x, s, w=2.0, b=0.0):
    A = np.array([[s, 1 - s], [1 - s, s]]); f, F = readout(w, b, x[None], F=np.full((len(x), 2), 0.5))
    _, q, ctx, _ = forward_backward(A, f / F); return f[0, :, 0], ctx[0, :, 0], q[0, :, 0]


def main():
    check_gradient()
    z, x = example_sequence(); t = np.arange(1, 25); own, ctx, post = decompose(x, 0.9)
    print("t   x      own vote f(+)  context P(z=+ | other x)  posterior q(+)   log-odds: own + context = posterior")
    for i in [6, 7, 8, 13, 14, 15, 16]:
        print(f"{t[i]:<3d} {x[i]:+.2f}  {own[i]:.3f}          {ctx[i]:.3f}                     {post[i]:.3f}            "
              f"{logit(own[i]):+.2f} {logit(ctx[i]):+.2f} = {logit(post[i]):+.2f}")
    fig, AX = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = AX[0, 0]
    ax.step(t, (z + 1) / 2, where="mid", color="0.7", lw=4, label="true z (1 = +, 0 = -)")
    ax.plot(t, own, "o-", color="#eb6834", label="own vote f(+ | x_t)"); ax.plot(t, ctx, "s-", color="#8a4fd8", label="context P(z_t = + | other x)")
    ax.plot(t, post, "D-", color="#2a6fdb", label="posterior q(z_t = + | x)"); ax.axvline(8, color="0.5", ls=":", lw=1)
    ax.set_xlabel("t"); ax.set_ylabel("probability of z = +"); ax.legend(loc="center left")
    ax.set_title("(a) prior s = 0.9: the outlier at t = 8 is overruled, the real switch at t = 15 is kept", loc="left", fontsize="medium")

    ax = AX[0, 1]; wdt = 0.4
    ax.bar(t - wdt / 2, logit(own), wdt, color="#eb6834", label="own vote: log f/F ratio"); ax.bar(t + wdt / 2, logit(ctx), wdt, color="#8a4fd8", label="context: logit P(z_t = + | other x)")
    ax.plot(t, logit(post), "D", color="#2a6fdb", label="posterior log-odds = sum of the two"); ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("t"); ax.set_ylabel("log-odds for z = +"); ax.legend(loc="lower left")
    ax.set_title("(b) the reconciliation is a sum of log-odds", loc="left", fontsize="medium")

    ax = AX[1, 0]
    for s, c in [(0.5, "#bbbbbb"), (0.9, "#2a6fdb"), (0.99, "#0b2e6b")]: ax.plot(t, decompose(x, s)[2], "o-", color=c, label=f"posterior, prior stickiness s = {s}")
    ax.step(t, (z + 1) / 2, where="mid", color="0.7", lw=4, alpha=0.5, zorder=0); ax.set_xlabel("t"); ax.set_ylabel("q(z_t = + | x)"); ax.legend(loc="center left")
    ax.set_title("(c) the prior's strength sets who wins: s = 0.5 trusts each x, s = 0.99 smooths", loc="left", fontsize="medium")

    ax = AX[1, 1]; _, X = sample(1000, 20, 0.9); run = train(X); _, X1 = sample(20000, 1, 0.9); run1 = train(X1)
    ax.plot(run[:, 0], run[:, 1], color="#2a6fdb", label="readout weight w (truth 2)"); ax.plot(run[:, 0], run[:, 2], color="#2a6fdb", ls="--", label="readout bias b (truth 0)")
    ax.plot(run[:, 0], (run[:, 3] + run[:, 4]) / 2, color="#8a4fd8", label="prior stickiness s (truth 0.9)")
    ax.plot(run1[:, 0], run1[:, 1], color="#eb6834", ls=":", label="w with single observations (no context)")
    for y in [2, 0.9]: ax.axhline(y, color="0.5", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("updates"); ax.legend(loc="center left")
    ax.set_title("(d) learning from w = 0.3, s = 0.5: prior and readout bootstrap each other", loc="left", fontsize="medium")
    out = Path.home() / "generative-learning/reports/rpm-chain.png"; fig.savefig(out, dpi=110); print(out)
    print("final w b s_++ s_--:", run[-1, 1:], " single observations w:", run1[-1, 1])


if __name__ == "__main__": main()
