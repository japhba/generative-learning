"""Wake-sleep alignment of a misaligned linear decoder to foreign samples, with data-only competitors.

World: z ~ prior over {-1,+1}^K (factorial, chain, or deep), x = A z + sigma eps.
Learner: same prior, decoder x = W z + c, factorial logistic recognizer q(z|x) with logits R x + b.
Start: second-order matched but rotated decoder (W0 Sigma W0^T = A Sigma A^T), recognizer fitted to W0 by sleep.
"""

import argparse
from datetime import datetime
from itertools import product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange


def configs(K): return np.array(list(product([-1, 1], repeat=K)), dtype=float)


def log_prior(prior, Z, rng_params):
    """Log pmf of every row of Z (shape M x K) under the named prior."""
    K = Z.shape[1]
    if prior == "factorial": return np.full(len(Z), -K * np.log(2))
    if prior == "chain":
        rho = rng_params["rho"]
        same = (Z[:, 1:] == Z[:, :-1])
        return -np.log(2) + np.where(same, np.log(rho), np.log(1 - rho)).sum(1)
    if prior == "deep":
        B, beta = rng_params["B"], rng_params["beta"]
        H = configs(B.shape[1])
        logits = beta * H @ B.T  # (2^H, K): p(z_i=+1|h) = sigmoid(2 logit)
        lp = -np.logaddexp(0, -2 * logits[:, None, :] * Z[None, :, :]).sum(2)  # (2^H, M)
        return np.logaddexp.reduce(lp, axis=0) - B.shape[1] * np.log(2)
    raise ValueError(prior)


class World:
    def __init__(self, K, D, sigma, prior, rng, rho=0.85, H=2, beta=1.5, mixing="gaussian"):
        self.K, self.D, self.sigma, self.prior = K, D, sigma, prior
        if mixing == "bars":  # K = 2n bars on an n x n image, rendered additively
            n = K // 2; assert D == n * n
            A = np.zeros((D, K))
            for i in range(n): A[[i * n + j for j in range(n)], i] = 1; A[[j * n + i for j in range(n)], n + i] = 1
        else: A = rng.standard_normal((D, K))
        self.A = A / np.linalg.norm(A, axis=0)
        self.params = {"rho": rho}
        if prior == "deep": self.params.update(B=rng.standard_normal((K, H)), beta=beta)
        self.Z = configs(K); self.logp = log_prior(prior, self.Z, self.params); self.p = np.exp(self.logp)
        self.mu = self.p @ self.Z; self.Sigma = (self.Z - self.mu).T @ ((self.Z - self.mu) * self.p[:, None])

    def sample_z(self, n, rng): return self.Z[rng.choice(len(self.Z), size=n, p=self.p)]

    def sample_x(self, n, rng):
        z = self.sample_z(n, rng)
        return z @ self.A.T + self.sigma * rng.standard_normal((n, self.D)), z


def mixture_logpdf(X, W, c, world):
    """log sum_z p(z) N(x; W z + c, sigma^2 I) for rows of X."""
    means = world.Z @ W.T + c  # (M, D)
    d2 = ((X[:, None, :] - means[None]) ** 2).sum(2)
    return np.logaddexp.reduce(world.logp[None] - d2 / (2 * world.sigma ** 2), axis=1) - world.D / 2 * np.log(2 * np.pi * world.sigma ** 2)


def posterior(X, W, c, world):
    means = world.Z @ W.T + c
    lj = world.logp[None] - ((X[:, None, :] - means[None]) ** 2).sum(2) / (2 * world.sigma ** 2)
    return np.exp(lj - np.logaddexp.reduce(lj, axis=1, keepdims=True))  # (N, M)


class Learner:
    def __init__(self, W, c, R, b): self.W, self.c, self.R, self.b = W.copy(), c.copy(), R.copy(), b.copy(); self.mem = None

    def logits(self, X): return X @ self.R.T + self.b

    def attention(self, X):  # in-context recognition: attend over remembered own dreams (keys x~, values z)
        Xm, Zm, tau2 = self.mem
        lw = -((X[:, None, :] - Xm[None]) ** 2).sum(2) / (2 * tau2)
        return np.exp(lw - np.logaddexp.reduce(lw, axis=1, keepdims=True))

    def mean_code(self, X): return self.attention(X) @ self.mem[1] if self.mem is not None else np.tanh(self.logits(X))

    def sample_q(self, X, rng):
        if self.mem is not None:  # importance resampling of one remembered dream per foreign sample
            w = self.attention(X); idx = (w.cumsum(1) < rng.random((len(X), 1))).sum(1)
            return self.mem[1][np.minimum(idx, len(self.mem[1]) - 1)]
        a = self.logits(X)
        return np.where(rng.random(a.shape) < 1 / (1 + np.exp(-2 * a)), 1.0, -1.0)

    def dream(self, world, rng, n):  # write n fresh own samples into the context memory (FIFO)
        Z = world.sample_z(n, rng); X = Z @ self.W.T + self.c + self.key_noise * world.sigma * rng.standard_normal((n, world.D))
        Xm, Zm, tau2 = self.mem
        self.mem = (np.concatenate([Xm[n:], X]), np.concatenate([Zm[n:], Z]), tau2)

    def wake(self, X, rng, lr):
        Z = self.sample_q(X, rng); E = X - Z @ self.W.T - self.c
        self.W += lr * E.T @ Z / len(X); self.c += lr * E.mean(0)

    def sleep(self, world, rng, lr, n):
        Z = world.sample_z(n, rng); X = Z @ self.W.T + self.c + world.sigma * rng.standard_normal((n, world.D))
        G = Z - np.tanh(self.logits(X))
        self.R += lr * G.T @ X / n; self.b += lr * G.mean(0)


def initial_learner(world, rng, steps, lr, batch, init="rotated", init_scale=1.0):
    """Misaligned decoder with a recognizer fitted to it by sleep.
    rotated: second-order matched, random rotation in whitened prior coordinates.
    swapped: disentangled but units 0 and 2 exchanged (a relabeling that is not a symmetry of a structured prior).
    random: a generic random decoder, Gaussian entries with column norm about init_scale (no second-order learning)."""
    vals, vecs = np.linalg.eigh(world.Sigma)
    half, ihalf = vecs @ np.diag(np.sqrt(vals)) @ vecs.T, vecs @ np.diag(1 / np.sqrt(vals)) @ vecs.T
    Q, _ = np.linalg.qr(rng.standard_normal((world.K, world.K)))
    if init == "swapped": Q = np.eye(world.K)[:, [2, 1, 0] + list(range(3, world.K))]; half = ihalf = np.eye(world.K)
    W = world.A @ half @ Q @ ihalf if init != "random" else init_scale * rng.standard_normal((world.D, world.K)) / np.sqrt(world.D)
    c = world.A @ world.mu - W @ world.mu
    L = Learner(W, c, np.zeros((world.K, world.D)), np.zeros(world.K))
    L.R = np.linalg.pinv(W) / world.sigma ** 2 * 0.25
    for _ in range(steps): L.sleep(world, rng, lr, batch)
    return L


def evaluate(L, world, Xtest, Ztest, true_logpdf):
    kl = float(np.mean(true_logpdf - mixture_logpdf(Xtest, L.W, L.c, world)))
    m = L.mean_code(Xtest)  # recognition of foreign samples, scored against the true latent order (prior breaks symmetry)
    recon_lin = float(np.mean(((Xtest - (m @ L.W.T + L.c)) ** 2).sum(1)))
    cos = (L.W / np.linalg.norm(L.W, axis=0)).T @ world.A  # (K learner, K true)
    align = float(np.mean(np.abs(cos).max(1)))
    acc_direct = float(np.mean(np.sign(m) == Ztest))
    match = np.abs(cos).argmax(1); sgn = np.sign(cos[np.arange(world.K), match])
    acc_matched = float(np.mean(np.sign(m) * sgn == Ztest[:, match]))
    # bit accuracy up to the prior's own symmetries (relabelings that leave the learner's prior unchanged)
    sym = {"chain": [Ztest, -Ztest, Ztest[:, ::-1], -Ztest[:, ::-1]], "deep": [Ztest, -Ztest]}.get(world.prior)
    acc_sym = acc_matched if sym is None else float(max(np.mean(np.sign(m) == Zs) for Zs in sym))
    # recognition of the learner's own samples (latent known by construction), fixed seed so checkpoints are comparable
    r = np.random.default_rng(12345); Zo = world.sample_z(2000, r); Xo = Zo @ L.W.T + L.c + world.sigma * r.standard_normal((2000, world.D))
    acc_own = float(np.mean(np.sign(L.mean_code(Xo)) == Zo))
    return dict(kl=kl, recon=recon_lin, align=align, acc=acc_direct, acc_matched=acc_matched, acc_sym=acc_sym, acc_own=acc_own)


def fastica(X, K, rng, sigma, iters=500):
    Xc = X - X.mean(0); C = Xc.T @ Xc / len(X); vals, vecs = np.linalg.eigh(C)
    top = vecs[:, -K:] / np.sqrt(vals[-K:]); Y = Xc @ top  # whitened (N, K)
    U, _ = np.linalg.qr(rng.standard_normal((K, K)))
    for _ in range(iters):
        G = np.tanh(Y @ U.T); U_new = G.T @ Y / len(Y) - np.diag((1 - G ** 2).mean(0)) @ U
        s, v = np.linalg.eigh(U_new @ U_new.T); U = v @ np.diag(1 / np.sqrt(s)) @ v.T @ U_new
    W = vecs[:, -K:] @ np.diag(np.sqrt(np.maximum(vals[-K:] - sigma ** 2, 1e-6))) @ U.T  # noise-corrected PCA scale, ICA rotation
    return W, X.mean(0)


def assign(W, c, Xdata, world):
    """Relabel latent units by the signed permutation whose code moments best match the learner's prior moments."""
    from itertools import permutations
    S = np.sign((Xdata - c) @ np.linalg.pinv(W).T); mu, C = S.mean(0), np.cov(S.T)
    signs = configs(world.K); best = (np.inf, None)
    for perm in permutations(range(world.K)):
        p = list(perm); Cp, mp = C[np.ix_(p, p)], mu[p]
        cost = (((signs[:, :, None] * signs[:, None, :]) * Cp - world.Sigma) ** 2).sum((1, 2)) + ((signs * mp - world.mu) ** 2).sum(1)
        i = cost.argmin()
        if cost[i] < best[0]: best = (cost[i], (p, signs[i]))
    p, sg = best[1]; Pm = np.zeros((world.K, world.K)); Pm[np.arange(world.K), p] = sg  # s' = Pm s
    return W @ Pm.T, c


PLANS = {  # two sub-updates per step, so every method gets the same number of batches
    "wake_sleep": [("wake", "stored"), ("sleep", None)],
    "wake_frozen": [("wake", "stored"), ("wake", "stored")],
    "wake_fresh": [("wake", "stored"), ("wake", "fresh")],
    "exact_em": [("em", "stored"), ("em", "stored")],
    "em_fresh": [("em", "stored"), ("em", "fresh")],
    "vae": [("vae", "stored"), ("vae", "stored")],
    "vae_fresh": [("vae", "stored"), ("vae", "fresh")],
    "binary_ae": [("bae", "stored"), ("bae", "stored")],
    "linear_ae": [("lae", "stored"), ("lae", "stored")],
    "wake_attend": [("wake", "stored"), ("dream", None)],  # in-context: no recognition weights, attention over own dreams
    "wake_sleep_stream": [("wake", "fresh"), ("sleep", None)],  # infinite-data reference: every wake sample is new
    "wake_sleep_bind": [("wake", "stored"), ("sleep", None)],  # plus periodic binding moves (bind_moves)
    "binary_ae_bind": [("bae", "stored"), ("bae", "stored")],
    "vae_bind": [("vae", "stored"), ("vae", "stored")],
    "vae_stream": [("vae", "fresh"), ("vae", "fresh")],
    "em_stream": [("em", "fresh"), ("em", "fresh")],
}


def em_step(L, X, world, lr):
    """Gradient EM with the exact posterior (enumerates 2^K configurations, uses the prior pmf)."""
    B = len(X); P = posterior(X, L.W, L.c, world); Ez = P @ world.Z; E = X - Ez @ L.W.T - L.c
    cov = np.einsum("nm,mi,mj->ij", P, world.Z, world.Z) / B - (Ez.T @ Ez) / B
    L.W += lr * (E.T @ Ez / B - L.W @ cov); L.c += lr * E.mean(0)


def ae_step(L, X, world, lr, r_scale, kind):
    """Backprop on the expected reconstruction under the factorial code (kind: lae, bae, vae)."""
    B = len(X); a = L.logits(X)
    if kind == "lae": m, var = a, np.zeros_like(a)
    else: m = np.clip(np.tanh(a), -1 + 1e-7, 1 - 1e-7); var = 1 - m ** 2
    E = X - m @ L.W.T - L.c
    gW = E.T @ m / B - L.W * var.mean(0)  # d/dW of -0.5 E_q||x - W z - c||^2
    gm = (E @ L.W + (m * (L.W ** 2).sum(0) if kind != "lae" else 0)) / world.sigma ** 2
    if kind == "vae":  # d/dm of H(q) + E_q log p(z), prior pmf enumerated
        logf = np.log((1 + world.Z[None] * m[:, None, :]) / 2)  # (B, M, K)
        tot = logf.sum(2, keepdims=True)
        dq = 0.5 * world.Z[None] * np.exp(tot - logf)  # dq(z)/dm_i
        gm = gm - np.arctanh(m) + np.einsum("bmk,m->bk", dq, world.logp)
    ga = gm * (1 - m ** 2) if kind != "lae" else gm
    L.W += lr * gW; L.c += lr * E.mean(0)
    L.R += lr * r_scale * ga.T @ X / B; L.b += lr * r_scale * ga.mean(0)


def bind_moves(L, Xdata, world, rng, n_own=4096):
    """Relabel own units by greedy swaps and sign flips while that brings the second moments of the recognized foreign
    codes closer to those of the learner's own dreamed latents (estimated from samples of its prior)."""
    Zo = world.sample_z(n_own, rng); mu, M2 = Zo.mean(0), Zo.T @ Zo / n_own
    S = np.sign(L.mean_code(Xdata)); s_mu, s_M2 = S.mean(0), S.T @ S / len(S); K = world.K; moves = 0
    def cost(P): return ((P @ s_M2 @ P.T - M2) ** 2).sum() + ((P @ s_mu - mu) ** 2).sum()
    P = np.eye(K)
    while True:
        cands = [np.eye(K)[[j if k == i else i if k == j else k for k in range(K)]] for i in range(K) for j in range(i + 1, K)]
        cands += [np.diag(np.where(np.arange(K) == i, -1.0, 1.0)) for i in range(K)]
        best = min(cands, key=lambda C: cost(C @ P))
        if cost(best @ P) >= cost(P) - 1e-9: break
        P = best @ P; moves += 1
    L.W, L.R, L.b = L.W @ P.T, P @ L.R, P @ L.b  # unit i of the relabeled learner is unit P^T i of the old one
    return moves


def run_method(method, world, L0, Xdata, rng, args, evalset):
    L = Learner(L0.W, L0.c, L0.R, L0.b); curve = []; B = args.batch
    if method == "wake_attend":
        Z = world.sample_z(args.memory, rng)
        L.key_noise = 0.0 if args.attend_keys == "mean" else 1.0  # mean: store the generator's output before noise (exact importance weights)
        L.mem = (Z @ L.W.T + L.c + L.key_noise * world.sigma * rng.standard_normal((args.memory, world.D)), Z, (1 + L.key_noise) * world.sigma ** 2)
    every = max(1, args.steps // args.points)
    for t in range(args.steps + 1):
        if t % every == 0: curve.append(dict(step=t, **evaluate(L, world, *evalset)))
        if t == args.steps: break
        for rule, source in PLANS[method]:
            X = Xdata[rng.integers(0, len(Xdata), B)] if source == "stored" else world.sample_x(B, rng)[0] if source == "fresh" else None
            if rule == "wake": L.wake(X, rng, args.lr)
            elif rule == "sleep": L.sleep(world, rng, args.lr * args.sleep_lr_scale, B)
            elif rule == "dream": L.dream(world, rng, B)
            elif rule == "em": em_step(L, X, world, args.lr * args.em_lr_scale)
            else: ae_step(L, X, world, args.lr, args.ae_lr_scale, rule)
        if method.endswith("_bind") and (t + 1) % args.bind_every == 0: bind_moves(L, Xdata, world, rng)
    return curve, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--prior", default="factorial")
    ap.add_argument("--N", type=int, default=256); ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--points", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ae_lr_scale", type=float, default=0.05); ap.add_argument("--sleep_lr_scale", type=float, default=1.0)
    ap.add_argument("--memory", type=int, default=1024); ap.add_argument("--attend_keys", default="mean"); ap.add_argument("--init", default="rotated"); ap.add_argument("--bind_every", type=int, default=200); ap.add_argument("--em_lr_scale", type=float, default=1.0)
    ap.add_argument("--mixing", default="gaussian"); ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--H", type=int, default=2); ap.add_argument("--beta", type=float, default=1.5); ap.add_argument("--tag", default="")
    ap.add_argument("--init_sleep", type=int, default=20000); ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--methods", default="wake_sleep,wake_frozen,wake_fresh,exact_em,binary_ae,linear_ae")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="/var/tmp/jbauer/generative_practice_disentangle")
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / f"{args.prior}_{args.mixing}_K{args.K}_D{args.D}_N{args.N}_{args.init}{args.tag}_{stamp}"; out.mkdir(parents=True)
    methods = args.methods.split(","); results = {m: [] for m in methods}; static = []; weights = {"init": [], "true": []}
    for w in trange(args.worlds, desc="worlds"):
        rng = np.random.default_rng([args.seed, w])
        world = World(args.K, args.D, args.sigma, args.prior, rng, rho=args.rho, H=args.H, beta=args.beta, mixing=args.mixing)
        Xdata, _ = world.sample_x(args.N, rng); Xtest, Ztest = world.sample_x(4000, rng)
        true_lp = mixture_logpdf(Xtest, world.A, np.zeros(args.D), world); evalset = (Xtest, Ztest, true_lp)
        L0 = initial_learner(world, rng, args.init_sleep, 0.02, args.batch, args.init, args.init_scale)  # same start for every method and learning rate
        weights["init"].append(dict(W=L0.W.tolist(), R=L0.R.tolist())); weights["true"].append(dict(W=world.A.tolist()))
        Wi, ci = fastica(Xdata, args.K, rng, args.sigma); Li = Learner(Wi, ci, L0.R, L0.b)
        Lt = Learner(world.A, np.zeros(args.D), L0.R, L0.b)
        Wa, ca = assign(Wi, ci, Xdata, world)
        static.append(dict(fastica=evaluate(Li, world, *evalset)["kl"], init=evaluate(L0, world, *evalset)["kl"], fastica_assigned=evaluate(Learner(Wa, ca, L0.R, L0.b), world, *evalset)["kl"]))
        for m in tqdm(methods, desc="methods", leave=False):
            curve, L = run_method(m, world, L0, Xdata, np.random.default_rng([args.seed, w, list(PLANS).index(m)]), args, evalset)
            results[m].append(curve); weights.setdefault(m, []).append(dict(W=L.W.tolist(), R=L.R.tolist(), c=L.c.tolist(), b=L.b.tolist()))
            Wa, ca = assign(L.W, L.c, Xdata, world); static[-1][f"{m}_assigned"] = evaluate(Learner(Wa, ca, L0.R, L0.b), world, *evalset)["kl"]
    json.dump(dict(args=vars(args), results=results, static=static, weights=weights), open(out / "results.json", "w"))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for m in methods:
        steps = [p["step"] for p in results[m][0]]
        for ax, key in zip(axes, ["kl", "align", "acc", "recon"]):
            vals = np.array([[p[key] for p in run] for run in results[m]])
            ax.plot(steps, vals.mean(0), label=m)
    axes[0].axhline(np.mean([s["fastica"] for s in static]), color="k", ls=":", label="FastICA (one shot)")
    for ax, t in zip(axes, ["KL(true || model) [nats]", "column alignment |cos|", "foreign recognition bit accuracy", "linear recon error"]): ax.set_title(t); ax.set_xlabel("update")
    axes[0].set_yscale("log"); axes[0].legend()
    fig.suptitle(f"{args.prior} prior, K={args.K}, D={args.D}, N={args.N}, sigma={args.sigma}"); fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    summary = {m: {k: float(np.mean([run[-1][k] for run in results[m]])) for k in ["kl", "align", "acc", "acc_matched", "acc_sym", "recon"]} for m in methods}
    for m in methods: summary[m]["best_kl"] = float(np.mean([min(p["kl"] for p in run) for run in results[m]]))
    for key in ["fastica", "init", "fastica_assigned"] + [f"{m}_assigned" for m in methods]:
        if key in static[0]: summary[f"{key}_kl"] = float(np.mean([s[key] for s in static]))
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps(summary, indent=1)); print(out / "curves.png")


if __name__ == "__main__": main()
