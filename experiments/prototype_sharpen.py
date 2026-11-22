"""Simplest model in which own examples improve generation although the reader is only the learner's own model
(user, 10 Nov 2026: simple x and z, directions and detected latents, no composite model).
x in {-1, +1}^D, exactly one of K latents on. World: latent k shows its prototype s_k; every pixel is independently
replaced (probability rho) by the pixel of a background pattern n_b (b uniform over B patterns, one per example).
Learner: K mean vectors m_k in [-1, 1]^D (its directions), p(x | k) = prod_d (1 + x_d m_kd) / 2, p(k) = 1 / K.
Reading = the exact posterior q(k | x) of this model. Unnormalised code u_k(x) = p(k) prod_d p(x_d | k) / p(x_d),
W = sum_k u_k = p(x) / prod_d p(x_d) (eligibility: one cause explains the pixels better than independent pixels).
One rule, m_k <- mean of x weighted by the code, with the code q or u and x from the world or from the learner:
  watch       q, world examples (plain EM), run to its fixed point first
  world_elig  u, world examples
  own_elig    u, own examples
  own_plain   q, own examples (changes nothing: zero signal)
Every expectation is exact (all 2^D states enumerated), so there is no sampling noise.
Prediction for one full own_elig step when the other pixels identify the latent: in log-odds l = log((1 + m) / (1 - m)),
l_k <- 2 l_k - l_bar, with l_bar the log-odds of the average direction m_bar = mean_k m_k.
Usage: python prototype_sharpen.py -> reports/prototype-sharpen.png"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from itertools import permutations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

D, K, B, RHOS, SEEDS, EPS = 12, 4, 2, [0.1, 0.2, 0.3, 0.4, 0.5], 40, 1e-12
X = np.array(list(product([-1.0, 1.0], repeat=D))); PERMS = np.array(list(permutations(range(K))))
ARMS = ["watch", "world_elig", "own_elig", "own_plain"]; FIG = Path(__file__).resolve().parents[1] / "reports/prototype-sharpen.png"


def logp(m): return np.log((1 + X[:, None, :] * m[None]) / 2 + EPS).sum(-1)                           # states, K
def own(m): return np.exp(logp(m)).mean(1)                                                             # the learner's distribution over states
def code(m, elig):
    lp = logp(m); lu = lp - np.log(K) - (np.log((1 + X * m.mean(0)) / 2 + EPS).sum(-1)[:, None] if elig else np.log(np.exp(lp).mean(1) + EPS)[:, None])
    return np.exp(lu)                                                                                  # u_k(x), or q(k | x) without eligibility


def step(m, P, elig, eta):
    w = P[:, None] * code(m, elig); new = (w.T @ X) / w.sum(0)[:, None]
    return np.clip((1 - eta) * m + eta * new, -1, 1)


def score(m, s, Pk):
    perm = PERMS[np.einsum("pkd,kd->p", m[PERMS], s).argmax()]; m = m[perm]; right = (1 + m * s) / 2; Pw = Pk.sum(1)
    return {"right": right.mean(), "valid": right.prod(1).mean(), "cos": np.mean((m * s).sum(1) / (np.linalg.norm(m, axis=1) * np.sqrt(D))),
            "length": np.mean(np.linalg.norm(m, axis=1)) / np.sqrt(D), "common": np.linalg.norm(m.mean(0)) / np.sqrt(D),
            "read": Pk[np.arange(len(X)), logp(m).argmax(1)].sum(), "kl": (Pw * (np.log(Pw + EPS) - np.log(own(m) + EPS))).sum()}


def run(rho, seed):
    rng = np.random.default_rng(seed); s, n = rng.choice([-1.0, 1.0], (K, D)), rng.choice([-1.0, 1.0], (B, D))
    Pk = np.stack([np.mean([np.prod((1 - rho) * (X == s[k]) + rho * (X == n[b]), 1) for b in range(B)], 0) for k in range(K)], 1) / K   # states, K
    Pw = Pk.sum(1); m = np.clip(0.2 * rng.standard_normal((K, D)), -0.9, 0.9)
    for _ in range(400): m = step(m, Pw, False, 1.0)                                                   # watching to its fixed point
    out = {"watch": score(m, s, Pk)}; mc = np.clip(m, -1 + 1e-9, 1 - 1e-9); l, lbar = np.log((1 + mc) / (1 - mc)), np.log((1 + mc.mean(0)) / (1 - mc.mean(0)))
    out["formula_error"] = np.abs(step(m, own(m), True, 1.0) - np.tanh((2 * l - lbar) / 2)).max()
    out["watch_drift"] = np.abs(step(m, Pw, False, 1.0) - m).max()
    for arm in ARMS[1:]:
        ma = m.copy()
        for _ in range(200): ma = step(ma, Pw if arm == "world_elig" else own(ma), arm != "own_plain", 0.5)
        out[arm] = score(ma, s, Pk)
    return out


def main():
    res = {rho: [run(rho, seed) for seed in range(SEEDS)] for rho in tqdm(RHOS, desc="replacement probability")}
    get = lambda arm, k: np.array([[r[arm][k] for r in res[rho]] for rho in RHOS])                     # rho, worlds
    for k, label in [("right", "share of pixels drawn as in the prototype"), ("valid", "share of products equal to the prototype"), ("cos", "cosine between direction m_k and prototype s_k"),
                     ("length", "length of m_k (1 = corner of the cube)"), ("common", "length of the average direction m_bar"), ("read", "reading of world examples"), ("kl", "KL(world || learner), nats")]:
        print(label)
        for arm in ARMS: print(f"  {arm:11s}", "  ".join(f"rho {rho}: {get(arm, k)[i].mean():6.3f}" for i, rho in enumerate(RHOS)))
    print("worlds (of", SEEDS, ") where own_elig ends above watching / above world_elig, in pixels right:", "  ".join(f"rho {rho}: {(get('own_elig', 'right')[i] > get('watch', 'right')[i] + 1e-9).sum()} / {(get('own_elig', 'right')[i] > get('world_elig', 'right')[i] + 1e-9).sum()}" for i, rho in enumerate(RHOS)))
    print("max |one own_elig step - formula|:", "  ".join(f"rho {rho}: median {np.median([r['formula_error'] for r in res[rho]]):.1e} max {max(r['formula_error'] for r in res[rho]):.1e}" for rho in RHOS))
    print("watching at its fixed point, max change of one more EM step:", f"{max(r['watch_drift'] for rho in RHOS for r in res[rho]):.1e}")

    col = {"watch": "#eb6834", "world_elig": "#b8860b", "own_elig": "#2a6fdb", "own_plain": "0.5"}
    name = {"watch": "after watching (EM fixed point)", "world_elig": "then world examples with eligibility", "own_elig": "then own examples with eligibility", "own_plain": "then own examples, no eligibility"}
    fig, AX = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, k, yl in [(AX[0], "right", "share of pixels drawn as in the prototype"), (AX[1], "cos", "cosine between direction and prototype"), (AX[2], "kl", "KL(world || learner), nats")]:
        for arm in ARMS:
            y = get(arm, k); ax.plot(RHOS, y.mean(1), "o-", color=col[arm], label=name[arm], ls="--" if arm == "own_plain" else "-")
            for i, rho in enumerate(RHOS): ax.plot(np.full(SEEDS, rho), y[i], ".", color=col[arm], alpha=0.15)
        ax.set_xlabel("share of pixels replaced by background, rho"); ax.set_ylabel(yl)
    AX[0].legend(); AX[0].set_title("(a) writing", loc="left"); AX[1].set_title("(b) direction", loc="left"); AX[2].set_title("(c) distance to the world's distribution", loc="left")
    fig.savefig(FIG, dpi=150); print(FIG)


if __name__ == "__main__":
    main()
