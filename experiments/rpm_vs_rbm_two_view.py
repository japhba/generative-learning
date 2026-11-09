"""RPM vs RBM on the two-view data of rpm_two_view_toy.py: x_j = z mu + b_j nu + sigma eps_j, mu = (1, 0) shared,
nu = (0, 3) per-view background, sigma = 1. RBM: Gaussian-Bernoulli on the concatenated x = (x1, x2) in R^4, hidden
h in {-1, +1}^K, p(x, h) ~ exp(-|x|^2 / 2 + h.W x + b.h), so p(x | h) = N(W^T h, I), p(h) ~ exp(b.h + |W^T h|^2 / 2)
and p(h_k = +1 | x) = sigmoid(2 (w_k.x + b_k)). Learning = clamped minus free Hebb, exact over the 2^K hidden states:
dW = <h x^T>_data - <h x^T>_model, with <h x^T>_model = <h h^T>_model W (for K = 1 the free phase is weight decay W).
Small weights: dW ~ (C - I) W, Hebb on the within-input covariance C (eigenvalues 9 per background, 2 shared signal);
the RPM's small-weight rule is Hebb on the cross-view covariance mu mu^T (background eigenvalue 0).
Usage: python rpm_vs_rbm_two_view.py -> reports/rpm-vs-rbm.png"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rpm_two_view_toy import data, step

rng = np.random.default_rng(3)


def logsumexp(a, axis=None):
    m = a.max(axis, keepdims=True); return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)

mu, nu = np.array([1.0, 0.0]), np.array([0.0, 3.0])
S = np.r_[mu, mu] / np.sqrt(2); N1 = np.r_[nu, 0, 0] / 3; N2 = np.r_[0, 0, nu] / 3          # shared signal, background of view 1, view 2


def rbm(W, b, X):
    """Exact mean log-likelihood and clamped-minus-free gradients of a Gaussian-Bernoulli RBM."""
    H = np.array(list(itertools.product([1, -1], repeat=W.shape[0]))); M = H @ W                 # 2^K hidden states, their means W^T h
    lph = H @ b + 0.5 * (M ** 2).sum(1); lph -= logsumexp(lph); ph = np.exp(lph)
    ll = logsumexp(lph[None] - 0.5 * ((X[:, None] - M[None]) ** 2).sum(-1), 1) - X.shape[1] / 2 * np.log(2 * np.pi)
    pos = np.tanh(X @ W.T + b)                                                                     # <h | x>, clamped phase
    return ll.mean(), pos.T @ X / len(X) - np.einsum("s,sk,sd->kd", ph, H, M), pos.mean(0) - ph @ H


def check_gradient(X):
    W, b = 0.3 * rng.standard_normal((2, 4)), np.array([0.1, -0.2]); _, gW, gb = rbm(W, b, X); e = 1e-6; th = np.r_[W.ravel(), b]
    f = lambda t: rbm(t[:8].reshape(2, 4), t[8:], X)[0]
    fd = np.array([(f(th + e * d) - f(th - e * d)) / (2 * e) for d in np.eye(10)])
    print(f"RBM clamped-minus-free rule vs finite differences: max |diff| = {np.abs(np.r_[gW.ravel(), gb] - fd).max():.1e}")


def train_rbm(X, K, seed, steps=6000, lr=0.02):
    g = np.random.default_rng(seed); W, b = 0.01 * g.standard_normal((K, 4)), np.zeros(K); T = np.unique(np.round(np.geomspace(1, steps, 80)).astype(int)); out = []
    for t in range(1, steps + 1):
        ll, gW, gb = rbm(W, b, X); W += lr * gW; b += lr * gb
        if t in T: out.append((t, ll, W @ S, W @ N1, W @ N2))
    return out, W


def train_rpm(X, steps=6000, lr=0.5):
    w = np.array([0.01, 0.05]); T = np.unique(np.round(np.geomspace(1, steps, 80)).astype(int)); out = []
    for t in range(1, steps + 1):
        lw, g = step(w, X); w = w + lr * g
        if t in T: out.append((t, lw.mean(), *w))
    return np.array(out)


def main():
    X2 = data(20000, mu, nu, 1.0); X = X2.reshape(len(X2), 4); check_gradient(X[:2000])
    C = X.T @ X / len(X); print("small-weight RBM growth rates, eigenvalues of C - I:", np.round(np.linalg.eigvalsh(C - np.eye(4))[::-1], 2))
    print("small-weight RPM growth rates, eigenvalues of the cross-view covariance:", np.round(np.linalg.eigvalsh(X2[:, 0].T @ X2[:, 1] / len(X2)), 2))
    r1, W1 = train_rbm(X, 1, 0); r3, W3 = train_rbm(X, 3, 0); rp = train_rpm(X2)
    ends = [train_rbm(X, 3, s)[0][-1][1] for s in range(1, 8)]
    print("RBM K=3 final log-likelihood over seeds 0..7:", np.round([r3[-1][1]] + ends, 3), "(-7.56 = one unit per cause; else two units share a background)")
    print("RBM K=1 final W:\n", W1.round(2), "\nRBM K=3 final W:\n", W3.round(2), "\nRPM final w:", rp[-1, 2:].round(2))
    print(f"final mean log-likelihood RBM K=1 {r1[-1][1]:.3f}, K=3 {r3[-1][1]:.3f};  RPM mean log W {rp[-1, 1]:.3f}")

    fig, AX = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    cS, cN = "#2a6fdb", "#eb6834"
    for ax, (run, K) in zip(AX[0], [(r1, 1), (r3, 3)]):
        t = [o[0] for o in run]; P = np.array([np.stack(o[2:], 1) for o in run])                     # time, unit, (S, N1, N2)
        for k in range(K):
            ax.plot(t, np.abs(P[:, k, 0]), color=cS, label="on the shared signal" if k == 0 else None)
            ax.plot(t, np.abs(P[:, k, 1:]).max(-1), color=cN, ls="--", label="on a background direction" if k == 0 else None)
        ax.axhline(np.sqrt(2), color=cS, ls=":", lw=1); ax.axhline(3, color=cN, ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("|weight projection| per hidden unit"); ax.legend(loc="upper left")
        ax.set_title(f"({'ab'[K > 1]}) RBM, {K} hidden unit{'s' * (K > 1)}: " + ("takes the background" if K == 1 else "all on backgrounds, signal after a plateau"), loc="left")
    ax = AX[1, 0]
    ax.plot(rp[:, 0], rp[:, 2], color=cS, label="weight on the shared signal mu"); ax.plot(rp[:, 0], np.abs(rp[:, 3]), color=cN, ls="--", label="weight on the background nu")
    ax.axhline(2, color=cS, ls=":", lw=1); ax.set_xscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("RPM readout weight"); ax.legend(loc="upper left")
    ax.set_title("(c) RPM, same data: signal only, background never grows", loc="left")
    ax = AX[1, 1]; base = r1[0][1]
    for run, ls, lab in [(r1, "-", "RBM, 1 hidden unit: log p(x) gain"), (r3, "--", "RBM, 3 hidden units: log p(x) gain")]:
        ax.plot([o[0] for o in run], [o[1] - base for o in run], color="0.3", ls=ls, label=lab)
    ax.plot(rp[:, 0], rp[:, 1], color=cS, label="RPM: mean log W (shared information)")
    ax.set_xscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("nats per item"); ax.legend(loc="upper left")
    ax.set_title("(d) learning curves", loc="left")
    out = Path.home() / "generative-learning/reports/rpm-vs-rbm.png"; fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__": main()
