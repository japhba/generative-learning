"""Does a quality lead emerge? Reading vs writing in a deep sigmoid belief net learned by wake-sleep (README, "Own
examples: quality lead vs information lead"; 7 Nov 2026).

World: top h_L uniform in {-1,+1}^{K_L}; p(h_l,i = +1 | h_{l+1}) = sigmoid(2 (V_l h_{l+1})_i), V_l Gaussian with entries
of variance beta^2 / K_{l+1} (several parents per child; beta sets how deterministic the productions are); x = A h_1 +
sigma eps with unit-norm columns of A. Layer sizes widen towards the data (e.g. 2 -> 4 -> 8 -> D). Small enough to
enumerate every latent configuration, so the world's exact posterior and both KLs are exact.

Learner: the same architecture with its own generator (W, c, V_l, a_l; top fixed uniform) and a bottom-up factorial
recognizer (R_l, b_l), generic small random start, trained by the wake-sleep delta rules on fresh world samples.

Measured along training, per layer l, as the fraction of samples whose whole layer-l code is right (and per bit):
  reading  the learner reads world samples: its bottom-up mean-field code m_l, matched one-to-one (signed permutation,
           fitted on half the samples) to the world's true h_l, accuracy on the other half. No free linear readout, so a
           random reader stays near chance. Reference: the world's exact posterior means. (A fitted logistic probe is kept
           as read_probe; it credits any reader whose features are linearly informative, e.g. random projections of x.)
  writing  the learner's own samples, parsed strictly in the world's terms: h_1 = the world's maximum-likelihood code for
           the sample (no prior, so the parse does not correct the sample), then h_{l+1} = the parent configuration under
           which h_l is most likely. Layer l (below the top) is valid if h_l is the world's most likely child of some
           parent configuration. Reference: the same parse of world samples. (The exact posterior is too charitable a
           reader for this: its prior repairs invalid samples, so even a random generator scores near the reference.)
Also KL(world || learner) (coverage) and KL(learner || world) (validity of own samples, overall)."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import permutations, product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

OUT = Path("/var/tmp/jbauer/generative_practice_sbn/lead")


def configs(K): return np.array(list(product([-1.0, 1.0], repeat=K)))
def lsig(u): return -np.logaddexp(0, -u)  # log sigmoid
def bern(rng, u): return np.where(rng.random(u.shape) < 1 / (1 + np.exp(-2 * u)), 1.0, -1.0)  # +-1 with p(+1) = sigmoid(2u)


class SBN:
    """Generator: sizes K[0] (bottom, h_1) ... K[-1] (top); V[l]: K[l] x K[l+1], a[l]; W: D x K[0], c. Top uniform."""
    def __init__(self, K, D, sigma, V, a, W, c): self.K, self.D, self.sigma, self.V, self.a, self.W, self.c = K, D, sigma, V, a, W, c

    def sample(self, n, rng):
        h = [None] * len(self.K); h[-1] = np.where(rng.random((n, self.K[-1])) < 0.5, 1.0, -1.0)
        for l in range(len(self.K) - 2, -1, -1): h[l] = bern(rng, h[l + 1] @ self.V[l].T + self.a[l])
        return h, h[0] @ self.W.T + self.c + self.sigma * rng.standard_normal((n, self.D))

    def tables(self):
        """Enumerated latent configurations per layer and the log prior of every joint configuration, shape (C_top, ..., C_bottom)."""
        Hs = [configs(k) for k in self.K]; lp = np.full(len(Hs[-1]), -self.K[-1] * np.log(2))
        for l in range(len(self.K) - 2, -1, -1):
            lc = lsig(2 * (Hs[l + 1] @ self.V[l].T + self.a[l])[:, None, :] * Hs[l][None]).sum(2)  # log p(h_l | h_{l+1}), (C_{l+1}, C_l)
            lp = lp[..., None] + lc.reshape((1,) * (lp.ndim - 1) + lc.shape)
        return Hs, lp

    def loglik(self, X, H1):
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ (H1 @ self.W.T + self.c).T + ((H1 @ self.W.T + self.c) ** 2).sum(1)[None]
        return -d2 / (2 * self.sigma ** 2) - self.D / 2 * np.log(2 * np.pi * self.sigma ** 2)  # (N, C_1)


def logpdf(G, X, chunk=500):
    Hs, lp = G.tables(); lp1 = np.logaddexp.reduce(lp.reshape(-1, lp.shape[-1]), axis=0)  # log p(h_1)
    return np.concatenate([np.logaddexp.reduce(G.loglik(X[i:i + chunk], Hs[0]) + lp1[None], axis=1) for i in range(0, len(X), chunk)])


def posterior_stats(G, X, chunk=200):
    """World's exact posterior on each row of X: per-layer marginal means E[h_l | x], and per-layer expected fraction (and
    probability of all) of units of h_l (l < top) equal to sign(V_l h_{l+1} + a_l)."""
    Hs, lp = G.tables(); L = len(G.K)
    agree = [(np.sign(Hs[l + 1] @ G.V[l].T + G.a[l])[:, None, :] == Hs[l][None]) for l in range(L - 1)]  # (C_{l+1}, C_l, K_l)
    means, frac, allv = [[] for _ in range(L)], [[] for _ in range(L - 1)], [[] for _ in range(L - 1)]
    for i in range(0, len(X), chunk):
        lj = lp[None] + G.loglik(X[i:i + chunk], Hs[0]).reshape((-1,) + (1,) * (L - 1) + (len(Hs[0]),))
        P = np.exp(lj - np.logaddexp.reduce(lj.reshape(len(lj), -1), axis=1).reshape((-1,) + (1,) * L))  # (n, C_top, ..., C_1)
        for l in range(L):  # marginal over layer l: axis index of layer l is L - l (axis 0 = samples)
            m = P.sum(axis=tuple(a for a in range(1, L + 1) if a != L - l)); means[l].append(m @ Hs[l])
        for l in range(L - 1):  # pairwise marginal over (h_{l+1}, h_l): axes L - l - 1, L - l
            pm = P.sum(axis=tuple(a for a in range(1, L + 1) if a not in (L - l - 1, L - l)))
            frac[l].append(np.einsum("npq,pq->n", pm, agree[l].mean(2))); allv[l].append(np.einsum("npq,pq->n", pm, agree[l].all(2).astype(float)))
    return [np.concatenate(m) for m in means], [np.concatenate(f) for f in frac], [np.concatenate(a) for a in allv]


def strict_parse(G, X):
    """Per layer below the top: valid (whole code is a most likely child of some parent) and per-bit agreement with that parent's most likely child."""
    Hs, L = [configs(k) for k in G.K], len(G.K)
    idx = G.loglik(X, Hs[0]).argmax(1); valid, bits = [], []
    for l in range(L - 1):
        lc = lsig(2 * (Hs[l + 1] @ G.V[l].T + G.a[l])[:, None, :] * Hs[l][None]).sum(2)  # (C_{l+1}, C_l)
        star = lc.argmax(1); par = lc[:, idx].argmax(0)  # most likely child of each parent; parsed parent of each sample
        valid.append(float(np.isin(idx, star).mean())); bits.append(float((Hs[l][idx] == Hs[l][star[par]]).mean())); idx = par
    return valid, bits


PERMS = {}


def matched(F, Y):
    """Signed one-to-one match of feature columns to targets (best total |correlation| on the first half, by enumeration);
    accuracy per bit and whole code on the second half."""
    n, K = len(F) // 2, Y.shape[1]; P = PERMS.setdefault(K, np.array(list(permutations(range(F.shape[1]), K))))
    Fc, Yc = F[:n] - F[:n].mean(0), Y[:n] - Y[:n].mean(0)
    C = (Fc.T @ Yc) / np.maximum(np.outer(np.linalg.norm(Fc, axis=0), np.linalg.norm(Yc, axis=0)), 1e-12)
    p = P[np.abs(C)[P, np.arange(K)].sum(1).argmax()]; sgn = np.sign(C[p, np.arange(K)]); sgn[sgn == 0] = 1
    ok = np.where(F[n:, p] * sgn >= 0, 1.0, -1.0) == Y[n:]; return float(ok.mean()), float(ok.all(1).mean())


def probe(F, Y, ridge=1e-3, iters=25):
    """Logistic readout of +-1 targets Y from features F: fit on the first half, accuracy (per bit and whole code) on the second."""
    n = len(F) // 2; F1 = np.hstack([F, np.ones((len(F), 1))]); pred = []
    for y in ((Y + 1) / 2).T:
        beta = np.zeros(F1.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-F1[:n] @ beta)); g = F1[:n].T @ (y[:n] - p) / n - ridge * beta
            beta += np.linalg.solve((F1[:n] * (p * (1 - p))[:, None]).T @ F1[:n] / n + ridge * np.eye(len(beta)), g)
        pred.append(np.where(F1[n:] @ beta > 0, 1.0, -1.0))
    ok = np.array(pred).T == Y[n:]; return float(ok.mean()), float(ok.all(1).mean())


class Learner:
    def __init__(self, K, D, sigma, rng, scale):
        self.G = SBN(K, D, sigma, [scale * rng.standard_normal((K[l], K[l + 1])) for l in range(len(K) - 1)], [np.zeros(K[l]) for l in range(len(K) - 1)],
                     scale * rng.standard_normal((D, K[0])) / np.sqrt(D), np.zeros(D))
        self.R = [np.zeros((K[0], D))] + [np.zeros((K[l + 1], K[l])) for l in range(len(K) - 1)]; self.b = [np.zeros(k) for k in K]

    def read(self, X, rng=None):
        """Bottom-up pass: sampled codes if rng is given, else mean-field means."""
        out, inp = [], X
        for R, b in zip(self.R, self.b):
            u = inp @ R.T + b; h = bern(rng, u) if rng is not None else np.tanh(u); out.append(h); inp = h
        return out

    def wake(self, X, rng, lr):
        G, h = self.G, self.read(X, rng); E = X - h[0] @ G.W.T - G.c; G.W += lr * E.T @ h[0] / len(X); G.c += lr * E.mean(0)
        for l in range(len(G.K) - 1):
            E = h[l] - np.tanh(h[l + 1] @ G.V[l].T + G.a[l]); G.V[l] += lr * E.T @ h[l + 1] / len(X); G.a[l] += lr * E.mean(0)

    def sleep(self, rng, lr, n):
        h, X = self.G.sample(n, rng); inp = [X] + h[:-1]
        for l in range(len(self.R)):
            E = h[l] - np.tanh(inp[l] @ self.R[l].T + self.b[l]); self.R[l] += lr * E.T @ inp[l] / n; self.b[l] += lr * E.mean(0)


def make_world(K, D, sigma, beta, rng):
    A = rng.standard_normal((D, K[0])); A /= np.linalg.norm(A, axis=0)
    return SBN(K, D, sigma, [beta * rng.standard_normal((K[l], K[l + 1])) / np.sqrt(K[l + 1]) for l in range(len(K) - 1)], [np.zeros(K[l]) for l in range(len(K) - 1)], A, np.zeros(D))


def run(task):
    name, K, D, w, args = task
    rng = np.random.default_rng([args["seed"], w, D, *K]); world = make_world(K, D, args["sigma"], args["beta"], rng)
    ht, Xt = world.sample(args["n_eval"], rng); lp_true = logpdf(world, Xt)
    wmeans = posterior_stats(world, Xt)[0]; wv, wb = strict_parse(world, Xt)
    ref = dict(read=[matched(m, h) for m, h in zip(wmeans, ht)], read_probe=[probe(m, h) for m, h in zip(wmeans, ht)], write=list(zip(wb, wv)))
    crng = np.random.default_rng([args["seed"], w, 5]); Rr = Learner(K, D, args["sigma"], crng, 1.0)  # chance: random reader, random writer
    Rr.R = [crng.standard_normal(R.shape) / np.sqrt(R.shape[1]) * 2 for R in Rr.R]; mr = Rr.read(Xt)
    H1 = configs(K[0])[crng.integers(0, 2 ** K[0], args["n_eval"])]; cv, cb = strict_parse(world, H1 @ world.W.T + args["sigma"] * crng.standard_normal((args["n_eval"], D)))
    chance = dict(read=[matched(mr[l], ht[l]) for l in range(len(K))], read_probe=[probe(mr[l], ht[l]) for l in range(len(K))], write=list(zip(cb, cv)))
    L = Learner(K, D, args["sigma"], np.random.default_rng([args["seed"], w, 1]), args["init_scale"]); r = np.random.default_rng([args["seed"], w, 2])
    evals = sorted(set(np.unique(np.round(np.geomspace(1, args["steps"], args["n_points"])).astype(int)).tolist()) | {0}); curve = []
    for t in range(args["steps"] + 1):
        if t in evals:
            m = L.read(Xt); _, Xo = L.G.sample(args["n_eval"], np.random.default_rng([args["seed"], w, 3, t])); ov, ob = strict_parse(world, Xo)
            curve.append(dict(step=t, kl=float(np.mean(lp_true - logpdf(L.G, Xt))), kl_rev=float(np.mean(logpdf(L.G, Xo) - logpdf(world, Xo))),
                              read=[matched(m[l], ht[l]) for l in range(len(K))], read_probe=[probe(m[l], ht[l]) for l in range(len(K))], write=list(zip(ob, ov))))
        if t == args["steps"]: break
        L.wake(world.sample(args["batch"], r)[1], r, args["lr"]); L.sleep(r, args["lr"] * args["sleep_lr_scale"], args["batch"])
    return dict(name=name, K=K, D=D, world=w, ref=ref, chance=chance, curve=curve)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conds", default="expand:2-4-8:8,expand:2-4-8:32,expand:2-4-8:128,flat:4-4-4:32", help="name:K_top-...-K_bottom:D, comma separated")
    ap.add_argument("--worlds", type=int, default=6); ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--n_points", type=int, default=24); ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--init_scale", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--tag", default="")
    args = ap.parse_args(); a = vars(args)
    tasks = []
    for c in args.conds.split(","):
        name, ks, d = c.split(":"); K = [int(k) for k in ks.split("-")][::-1]  # stored bottom (h_1) first
        tasks += [(f"{name}_D{d}", K, int(d), w, a) for w in range(args.worlds)]
    out = OUT / f"run{args.tag}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    with ProcessPoolExecutor(args.workers) as ex: res = list(tqdm(ex.map(run, tasks), total=len(tasks), desc="runs"))
    json.dump(dict(args=a, results=res), open(out / "results.json", "w"))
    for name in dict.fromkeys(r["name"] for r in res):
        rs = [r for r in res if r["name"] == name]; L = len(rs[0]["K"])
        print(f"== {name}  whole-code refs: read {[round(np.mean([r['ref']['read'][l][1] for r in rs]), 3) for l in range(L)]}  write {[round(np.mean([r['ref']['write'][l][1] for r in rs]), 3) for l in range(L - 1)]}")
        for i in range(0, len(rs[0]["curve"]), 2):
            c = [r["curve"][i] for r in rs]
            print(f"  step {c[0]['step']:6d} KL {np.mean([x['kl'] for x in c]):6.3f} rev {np.mean([x['kl_rev'] for x in c]):6.3f}  read " + " ".join(f"{np.mean([x['read'][l][1] for x in c]):.3f}" for l in range(L))
                  + "  write " + " ".join(f"{np.mean([x['write'][l][1] for x in c]):.3f}" for l in range(L - 1)))
    print(out)


if __name__ == "__main__": main()
