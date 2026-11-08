"""Recognition that learns bottom-up from world strings, by its own local rule, and verifies the learner's own strings.

World: RHM (practice_rhm.RHM). Generator: the tree Helmholtz machine's generator (practice_rhm.Learner tables, delta
rule), K states per hidden level. Recognition: winner-take-all (competitive) Hebbian layers on the same tree. A unit a at
tree level k has a factorised prototype P_k[a, j, :], a distribution over the codes of child slot j; a node's code is the
unit that best explains its children's codes, score_a = sum_j log P_k[a, j, child_j] - theta_k[a] (hard winner; at T > 0
the winner is sampled from softmax(score / T), which at T = 1 is the posterior of a mixture model: online stochastic EM),
computed bottom-up from the leaves. Learning on world strings only: the winner's prototype moves towards its children's
codes (instar rule, floored at eps); optionally (--eta_theta > 0) a homeostatic threshold theta rises for units that win
more than their share (intrinsic plasticity; off by default: log-likelihood scores already recruit fresh units for
patterns no trained unit explains). The world never passes through the generator to get here.
A node is verified if its children's codes have probability at least tau under its unit's prototype (geometric mean over
slots; an untrained, flat unit verifies nothing), and its children are verified: recognition computes the string's
one-level-up codes and checks each pair.

Arms (all see the same number of world strings per update unless stated):
  watch_wta           generator: delta rule on world strings with the WTA codes as latents (watching, recognition ahead)
  watch_wta_practice  + own strings with their generating codes, each node's update signed by verification:
                      (verified - running mean) * delta rule, a local three-factor rule
  practice_wta        own verified strings only; the world reaches the generator only through recognition
  watch_wta_2x        watching with twice the world strings (fresh examples in place of practice)
  watch_wta_soft      watching, WTA at temperature 1 (sampled winners, close to online EM)
Measured every `every` updates: exact KL (inside algorithm), validity of own strings per height, information of the WTA
code about the true symbol per height, and verification per height on world strings: share of intact nodes verified minus
share of nodes broken at that height (one child subtree regrown from a symbol that makes the pair invalid) verified.
Usage: python practice_rhm_wta.py --L 3 --updates 20000 --tag NAME"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from practice_rhm import RHM, Learner, nmi, onehot, sample_cat, softmax


class WTA:
    def __init__(self, K, v, s, rng, eta, eta_theta, temp, tau, eps=1e-3):
        self.L, self.s, self.K, self.eta, self.eta_theta, self.temp, self.tau, self.eps = len(K), s, list(K) + [v], eta, eta_theta, temp, tau, eps
        self.P = []
        for k in range(self.L):
            P = 1 + 0.05 * rng.random((self.K[k], s, self.K[k + 1])); self.P.append(P / P.sum(-1, keepdims=True))
        self.theta = [np.zeros(self.K[k]) for k in range(self.L)]

    def read(self, x, rng=None):
        """Bottom-up codes per tree level (0 = root .. L = leaves) and per node whether it is verified."""
        n = len(x); codes, ok = [None] * self.L + [x], [None] * self.L + [np.ones(x.shape, bool)]
        for k in reversed(range(self.L)):
            ch = codes[k + 1].reshape(n, -1, self.s)
            lP = np.log(self.P[k]); S = sum(lP[:, j, :][:, ch[..., j]] for j in range(self.s)).transpose(1, 2, 0)  # (n, nodes, K_k)
            a = sample_cat(softmax((S - self.theta[k]) / self.temp), rng) if self.temp > 0 and rng is not None else (S - self.theta[k]).argmax(-1)
            best = np.take_along_axis(S + self.theta[k], a[..., None], -1)[..., 0]
            codes[k] = a; ok[k] = (best >= self.s * np.log(self.tau)) & ok[k + 1].reshape(n, -1, self.s).all(-1)
        return codes, ok

    def learn(self, x, rng):
        codes, _ = self.read(x, rng)
        for k in range(self.L):
            a = codes[k].ravel(); ch = codes[k + 1].reshape(-1, self.s); cnt = np.bincount(a, minlength=self.K[k]); won = cnt > 0
            for j in range(self.s):
                M = np.zeros((self.K[k], self.K[k + 1])); np.add.at(M, (a, ch[:, j]), 1)
                self.P[k][won, j] += self.eta * (M[won] / cnt[won, None] - self.P[k][won, j])
            self.P[k] = (1 - self.eps) * self.P[k] + self.eps / self.K[k + 1]
            self.theta[k] += self.eta_theta * (cnt / len(a) * self.K[k] - 1)
        return codes


def fit(ln, codes, lr, w=None):
    """Generator delta rule towards given codes (parent -> children) at every level; node weights w[k] (n, nodes) if given."""
    n = len(codes[-1]); ww = lambda k: np.ones(codes[k].size) if w is None else w[k].reshape(-1)
    ln.g0 += lr * (ww(0)[:, None] * (onehot(codes[0][:, 0], ln.K[0]) - softmax(ln.g0))).mean(0)
    for k in range(ln.L):
        u = codes[k].reshape(-1); ch = codes[k + 1].reshape(-1, ln.s); U = onehot(u, ln.K[k]) * ww(k)[:, None]
        for j in range(ln.s): ln.G[k][j] += lr * U.T @ (onehot(ch[:, j], ln.K[k + 1]) - softmax(ln.G[k][j][u])) / len(u)


def subtree(rhm, a, k, rng):
    a = a[:, None]
    for kk in range(k, rhm.L): a = rhm.rules[kk][a, rng.integers(rhm.m, size=a.shape)].reshape(len(a), -1)
    return a


def corrupt(rhm, tree, h, rng):
    """World strings with one height-h node broken (a child subtree regrown from a symbol making its pair invalid)."""
    L, s, v = rhm.L, rhm.s, rhm.v; X = tree[-1].copy(); n = len(X); ar = np.arange(n); k = L - h
    j = rng.integers(s ** k, size=n); c = rng.integers(s, size=n); kids = tree[k + 1][ar[:, None], j[:, None] * s + np.arange(s)]
    cand = np.repeat(kids[:, None], v, 1); cand[ar, :, c] = np.arange(v)
    bad = rhm.parent[k][np.ravel_multi_index(cand.transpose(2, 0, 1), (v,) * s)] < 0
    b = (rng.random((n, v)) * bad).argmax(1); w = s ** (h - 1)
    X[ar[:, None], (j * s + c)[:, None] * w + np.arange(w)] = subtree(rhm, b, k + 1, rng)
    return X, bad.any(1), j


def evaluate(world, ln, wta, rng, n):
    tree = world.sample(n, rng); codes, ok = wta.read(tree[-1]); L = world.L; ar = np.arange(n); ver = []
    for h in range(1, L + 1):
        k = L - h; Xb, has, j = corrupt(world, tree, h, rng); _, okb = wta.read(Xb)
        ver.append(float(ok[k][ar, j][has].mean() - okb[k][ar, j][has].mean()))
    return dict(kl=float(-world.entropy() - ln.loglik(tree[-1]).mean()), valid=[float(z) for z in world.valid(ln.dream(n, rng)[-1])],
                nmi=[nmi(codes[k].ravel(), tree[k].ravel()) for k in reversed(range(L))], verify=ver,
                verified_world=[float(ok[k].mean()) for k in reversed(range(L))])


def run(job):
    args, arm, seed = job
    rng = np.random.default_rng(seed); world = RHM(args.v, args.nc, args.m, args.s, args.L, rng)
    K = [args.nc * args.m * args.kappa] + [args.v * args.m * args.kappa] * (args.L - 1)
    ln = Learner(K, args.v, args.s, "delta", np.random.default_rng(1000 + seed), args.init); rng = np.random.default_rng(2000 + seed)
    wta = WTA(K, args.v, args.s, np.random.default_rng(3000 + seed), args.eta, args.eta_theta, 1.0 if arm == "watch_wta_soft" else 0.0, args.tau)
    base = np.zeros(args.L); curve = []
    for t in tqdm(range(args.updates + 1), desc=f"{arm} seed {seed}", leave=False, mininterval=10):
        if t % args.every == 0: curve.append(dict(t=t, **evaluate(world, ln, wta, rng, args.eval_n)))
        for _ in range(2 if arm == "watch_wta_2x" else 1):
            codes = wta.learn(world.sample(args.batch, rng)[-1], rng)
            if arm != "practice_wta": fit(ln, codes, args.lr)
        if arm in ("watch_wta_practice", "practice_wta"):
            own = ln.dream(args.batch, rng); _, ok = wta.read(own[-1])
            rate = np.array([ok[k].mean() for k in range(args.L)]); fit(ln, own, args.lr, [ok[k] - base[k] for k in range(args.L)] + [None]); base += 0.01 * (rate - base)
    return dict(arm=arm, seed=seed, curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=8); ap.add_argument("--nc", type=int, default=8); ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--s", type=int, default=2); ap.add_argument("--L", type=int, default=3); ap.add_argument("--kappa", type=int, default=2)
    ap.add_argument("--init", type=float, default=0.5); ap.add_argument("--lr", type=float, default=0.5); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.1); ap.add_argument("--eta_theta", type=float, default=0.0); ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--updates", type=int, default=20000); ap.add_argument("--every", type=int, default=250); ap.add_argument("--eval_n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--arms", default="watch_wta,watch_wta_practice,practice_wta,watch_wta_2x,watch_wta_soft")
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path("/var/tmp/jbauer/generative_practice_rhm") / f"wta_{args.tag}L{args.L}_{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True)
    jobs = [(args, arm, seed) for arm in args.arms.split(",") for seed in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex: res = list(tqdm(ex.map(run, jobs), total=len(jobs), desc="runs"))
    json.dump(dict(args=vars(args), results=res), open(out / "results.json", "w")); print(out)


if __name__ == "__main__": main()
