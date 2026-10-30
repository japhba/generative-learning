"""Deep wake-sleep on the random hierarchy model: do the layers fall into place one after another, and only with practice?"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def sample_cat(p, rng): return np.minimum((p.cumsum(-1) < rng.random(p.shape[:-1] + (1,))).sum(-1), p.shape[-1] - 1)
def onehot(a, K): return np.eye(K)[a]


class RHM:
    """Random hierarchy model (Cagnetta et al. 2024): a root class is rewritten level by level into s-tuples, each symbol having
    m synonymous productions; rules differ between levels and are shared across positions. Level k = 0 (root) .. L (leaves)."""
    def __init__(self, v, nc, m, s, L, rng):
        self.v, self.nc, self.m, self.s, self.L = v, nc, m, s, L
        tuples = np.array(np.unravel_index(np.arange(v ** s), (v,) * s)).T
        self.rules = [tuples[rng.choice(v ** s, (nc if k == 0 else v) * m, replace=False)].reshape(-1, m, s) for k in range(L)]
        self.parent = []  # tuple index -> symbol producing it (-1 if none), per level
        for rule in self.rules:
            p = -np.ones(v ** s, int); p[np.ravel_multi_index(rule.reshape(-1, s).T, (v,) * s)] = np.repeat(np.arange(len(rule)), m); self.parent.append(p)

    def sample(self, n, rng):
        a = rng.integers(self.nc, size=(n, 1)); tree = [a]
        for k in range(self.L): a = self.rules[k][a, rng.integers(self.m, size=a.shape)].reshape(n, -1); tree.append(a)
        return tree

    def parses(self, x):
        """Per sequence, whether it parses up to height h = 1 (just above the leaves) .. L (root): (n, L) booleans."""
        ok, out, a = np.ones(len(x), bool), [], x
        for k in reversed(range(self.L)):
            p = self.parent[k][np.ravel_multi_index(a.reshape(len(a), -1, self.s).transpose(2, 0, 1), (self.v,) * self.s)]
            ok = ok & (p >= 0).all(1); out.append(ok); a = np.maximum(p, 0)
        return np.stack(out, 1)

    def valid(self, x):
        """Fraction of sequences that parse up to height h = 1 (just above the leaves) .. L (root)."""
        return list(self.parses(x).mean(0))

    def entropy(self): return np.log(self.nc) + (self.s ** self.L - 1) // (self.s - 1) * np.log(self.m)


class Learner:
    """Tree Helmholtz machine. Hidden level k has K[k] states at s^k nodes. Generation: G[k][j] are the logits of child j's state
    given its parent's; recognition: R[k][j] is child j's contribution to the parent's logits, plus a bias r[k].
    rule='delta': local delta rules. rule='hebb': both pathways are normalised co-occurrence counts with forgetting rate lr."""
    def __init__(self, K, v, s, rule, rng, init):
        self.s, self.L, self.K, self.rule = s, len(K), list(K) + [v], rule
        if rule == "delta":
            self.g0 = np.zeros(K[0])
            self.G = [init * rng.standard_normal((s, self.K[k], self.K[k + 1])) for k in range(self.L)]
            self.R = [init * rng.standard_normal((s, self.K[k + 1], self.K[k])) for k in range(self.L)]
            self.r = [np.zeros(self.K[k]) for k in range(self.L)]
        else:  # counts, positive with multiplicative noise to break the symmetry
            self.cg0 = np.ones(K[0]) / K[0]
            self.Cg = [np.exp(init * rng.standard_normal((s, self.K[k], self.K[k + 1]))) / self.K[k] / self.K[k + 1] for k in range(self.L)]
            self.Cr = [np.exp(init * rng.standard_normal((s, self.K[k], self.K[k + 1]))) / self.K[k] / self.K[k + 1] for k in range(self.L)]
            self.cu = [np.ones(self.K[k]) / self.K[k] for k in range(self.L)]
            self.sync()

    def sync(self):
        """Hebbian weights from counts: log conditional frequencies (divisive normalisation)."""
        self.g0 = np.log(self.cg0)
        self.G = [np.log(C / C.sum(-1, keepdims=True)) for C in self.Cg]
        self.R = [np.log(C / C.sum(-1, keepdims=True)).transpose(0, 2, 1) for C in self.Cr]
        self.r = [np.log(c / c.sum()) for c in self.cu]

    def tied(self):
        """Recognition read off the generator: the exact bottom-up (naive Bayes) inverse, as with symmetric weights."""
        P = [softmax(G) for G in self.G]; pi = [softmax(self.g0)]
        for k in range(self.L - 1): pi.append(np.mean([pi[k] @ P[k][j] for j in range(self.s)], 0))
        return [np.log(p).transpose(0, 2, 1) for p in P], [np.log(p) for p in pi]

    def posterior(self, x, rng):
        """Exact posterior sample of all hidden states (inside pass, then top-down sampling), as in Monte Carlo EM."""
        P = [softmax(G) for G in self.G]; beta = [None] * self.L + [onehot(x, self.K[-1])]
        for k in reversed(range(self.L)):
            b = beta[k + 1].reshape(len(x), -1, self.s, self.K[k + 1]); msg = 1.0
            for j in range(self.s): msg = msg * (b[:, :, j] @ P[k][j].T)
            beta[k] = msg / msg.sum(-1, keepdims=True)
        norm = lambda q: q / q.sum(-1, keepdims=True)
        codes = [sample_cat(norm(softmax(self.g0) * beta[0]), rng)]
        for k in range(self.L - 1):
            b = beta[k + 1].reshape(len(x), -1, self.s, self.K[k + 1])
            codes.append(np.stack([sample_cat(norm(P[k][j][codes[k]] * b[:, :, j]), rng) for j in range(self.s)], -1).reshape(len(x), -1))
        return codes + [x]

    def recognize(self, x, rng, tied=False):
        if tied == "exact": return self.posterior(x, rng)
        R, r = self.tied() if tied else (self.R, self.r)
        codes = [None] * self.L + [x]
        for k in reversed(range(self.L)):
            ch = codes[k + 1].reshape(len(x), -1, self.s)
            codes[k] = sample_cat(softmax(sum(R[k][j][ch[..., j]] for j in range(self.s)) + r[k]), rng)
        return codes

    def dream(self, n, rng):
        codes = [sample_cat(softmax(np.broadcast_to(self.g0, (n, 1, self.K[0]))), rng)]
        for k in range(self.L): codes.append(np.stack([sample_cat(softmax(self.G[k][j][codes[k]]), rng) for j in range(self.s)], -1).reshape(n, -1))
        return codes

    def wake(self, x, rng, lr, tied=False):
        codes = self.recognize(x, rng, tied)
        if self.rule == "delta": self.g0 += lr * (onehot(codes[0][:, 0], self.K[0]).mean(0) - softmax(self.g0))
        else: self.cg0 += lr * (onehot(codes[0][:, 0], self.K[0]).mean(0) - self.cg0)
        for k in range(self.L):
            u = codes[k].reshape(-1); ch = codes[k + 1].reshape(-1, self.s); U = onehot(u, self.K[k])
            for j in range(self.s):
                C = onehot(ch[:, j], self.K[k + 1])
                if self.rule == "delta": self.G[k][j] += lr * U.T @ (C - softmax(self.G[k][j][u])) / len(u)
                else: self.Cg[k][j] += lr * (U.T @ C / len(u) - self.Cg[k][j])
        if self.rule == "hebb": self.sync()

    def sleep(self, n, rng, lr):
        codes = self.dream(n, rng)
        for k in range(self.L):
            u = codes[k].reshape(-1); ch = codes[k + 1].reshape(-1, self.s); U = onehot(u, self.K[k])
            if self.rule == "delta":
                err = U - softmax(sum(self.R[k][j][ch[:, j]] for j in range(self.s)) + self.r[k])
                for j in range(self.s): self.R[k][j] += lr * onehot(ch[:, j], self.K[k + 1]).T @ err / len(u)
                self.r[k] += lr * err.mean(0)
            else:
                for j in range(self.s): self.Cr[k][j] += lr * (U.T @ onehot(ch[:, j], self.K[k + 1]) / len(u) - self.Cr[k][j])
                self.cu[k] += lr * (U.mean(0) - self.cu[k])
        if self.rule == "hebb": self.sync()

    def loglik(self, x):
        """Exact log-likelihood by the inside algorithm."""
        beta, logz = onehot(x, self.K[-1]), np.zeros(len(x))
        for k in reversed(range(self.L)):
            b = beta.reshape(len(x), -1, self.s, self.K[k + 1]); msg = 1.0
            for j in range(self.s): msg = msg * (b[:, :, j] @ softmax(self.G[k][j]).T)
            z = msg.sum(-1); logz += np.log(z).sum(1); beta = msg / z[..., None]
        return logz + np.log(beta[:, 0] @ softmax(self.g0))


def nmi(u, a):
    """I(u; a) / H(a): how much of the true symbol a the learner's code u carries."""
    J = np.zeros((u.max() + 1, a.max() + 1)); np.add.at(J, (u, a), 1); J /= J.sum()
    pu, pa = J.sum(1, keepdims=True), J.sum(0, keepdims=True); m = J > 0
    return float((J[m] * np.log(J[m] / (pu @ pa)[m])).sum() / -(pa[pa > 0] * np.log(pa[pa > 0])).sum())


def mutual_info(u, a):
    J = np.zeros((u.max() + 1, a.max() + 1)); np.add.at(J, (u, a), 1); J /= J.sum()
    pu, pa = J.sum(1, keepdims=True), J.sum(0, keepdims=True); m = J > 0
    return float((J[m] * np.log(J[m] / (pu @ pa)[m])).sum())


def sibling_info(c, s, rng):
    """Mutual information between the first two siblings' codes, minus the same with one sibling shuffled across strings
    (removes the plug-in estimator's upward bias of about (K-1)^2 / 2n)."""
    a, b = c[:, 0::s], c[:, 1::s]
    return mutual_info(a.ravel(), b.ravel()) - mutual_info(a.ravel(), b[rng.permutation(len(b))].ravel())


def evaluate(world, ln, rng, n, tied):
    """kl: KL(true || learner). valid, nmi: per layer h = 1..L, generation validity and recognition of the true symbol.
    drive: per layer, mutual information (nats, bias-corrected) between sibling codes one level below on lectures, which bounds what it can learn."""
    tree = world.sample(n, rng); codes = ln.recognize(tree[-1], rng, tied)
    return dict(kl=float(-world.entropy() - ln.loglik(tree[-1]).mean()), valid=world.valid(ln.dream(n, rng)[-1]),
                nmi=[nmi(codes[k].ravel(), tree[k].ravel()) for k in reversed(range(world.L))],
                drive=[sibling_info(codes[k + 1], world.s, rng) for k in reversed(range(world.L))])


METHODS = {"practice": dict(sleep=True, tied=False), "watch": dict(sleep=False, tied=False), "watch_tied": dict(sleep=False, tied=True),
           "watch_exact": dict(sleep=False, tied="exact"), "watch_more": dict(sleep=False, tied=False, fresh=True)}


def run(args, seed, method):
    rng = np.random.default_rng(seed); world = RHM(args.v, args.nc, args.m, args.s, args.L, rng)
    lectures = world.sample(args.N, rng)[-1] if args.N else None
    K = [args.nc * args.m * args.kappa] + [args.v * args.m * args.kappa] * (args.L - 1)
    rng = np.random.default_rng(1000 + seed); ln = Learner(K, args.v, args.s, args.rule, rng, args.init)
    for t in tqdm(range(args.branch_at), leave=False, desc=f"pre-practice seed {seed}"):
        x = lectures[rng.integers(args.N, size=args.batch)] if args.N else world.sample(args.batch, rng)[-1]
        ln.wake(x, rng, args.lr); ln.sleep(args.batch, rng, args.lr * args.sleep_scale)
    rng = np.random.default_rng(2000 + seed); spec = METHODS[method]
    curve = []
    for t in tqdm(range(args.updates + 1), leave=False, desc=f"{method} seed {seed}"):
        if t % args.every == 0: curve.append(dict(t=t, **evaluate(world, ln, rng, args.eval_n, spec["tied"])))
        x = lectures[rng.integers(args.N, size=args.batch)] if args.N else world.sample(args.batch, rng)[-1]
        ln.wake(x, rng, args.lr, spec["tied"])
        if spec["sleep"]: ln.sleep(args.batch, rng, args.lr * args.sleep_scale)
        if spec.get("fresh"): ln.wake(world.sample(args.batch, rng)[-1], rng, args.lr)  # a fresh lecture in place of each dream
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=8); ap.add_argument("--nc", type=int, default=8); ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--s", type=int, default=2); ap.add_argument("--L", type=int, default=3); ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--N", type=int, default=0, help="stored lectures (0 = a fresh lecture every time)")
    ap.add_argument("--rule", default="delta", choices=["delta", "hebb"]); ap.add_argument("--init", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.5); ap.add_argument("--sleep_scale", type=float, default=1.0)
    ap.add_argument("--updates", type=int, default=20000); ap.add_argument("--every", type=int, default=250); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_n", type=int, default=2000); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--branch_at", type=int, default=0, help="practise this many updates first, then branch into the methods")
    ap.add_argument("--methods", default="practice,watch,watch_tied,watch_exact"); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path("/var/tmp/jbauer/generative_practice_rhm") / f"{args.tag}{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True)
    results = {m: [run(args, seed, m) for seed in range(args.seeds)] for m in tqdm(args.methods.split(","))}
    json.dump(dict(args=vars(args), results=results), open(out / "results.json", "w"))
    for m, curves in results.items():
        last = [c[-1] for c in curves]
        print(m, f"kl={np.mean([c['kl'] for c in last]):.2f}", "valid", np.round(np.mean([c["valid"] for c in last], 0), 2), "nmi", np.round(np.mean([c["nmi"] for c in last], 0), 2))
    print(out)


if __name__ == "__main__": main()
