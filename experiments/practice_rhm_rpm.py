"""Recognition-parametrised model (RPM; Walker, Soulat, Yu & Sahani 2023) on the RHM against the same latent tree with a
learned emission. Does a learner that never models the observations learn to read and to structure faster?

World: RHM (practice_rhm.RHM, same rules per seed as the other studies) whose leaf symbols are rendered as vectors
x_j = mu_a + nu_b + sigma * eps in R^D:
  symbolic  D = v, mu_a = 4 e_a, sigma 0.5, no background (control: the two model classes coincide)
  clean     D = 32, mu_a ~ N(0, I), sigma 2.5 (the exact single-leaf posterior carries 40 % of the symbol's information)
  nuisance  clean + an independent background b per leaf (B = 8, nu_b ~ N(0, 4 I)): variance in x that no other leaf shares
Every learner has the latent tree p(Z): root K0 states, K hidden states per node (factorised child tables, as in
practice_rhm), K_leaf states per leaf latent; all are trained by exact maximum likelihood (inside algorithm, autograd, Adam)
and differ only in how a leaf's latent meets its observation:
  rpm      P(X, Z) = p(Z) prod_j p0(x_j) f(z_j | x_j) / F_j(z_j), f = softmax(W x + b) shared across positions, F_j = batch mean
           of f at position j (p0 = the batch); objective log W(X) = log sum_Z p(Z) prod_j f / F. No model of x.
  gen      p(x_j | z_j) = N(mu_z, s^2 I), K_leaf = v; objective log p(X)
  gen_big  the same with K_leaf = v * B (correctly specified under the nuisance)
  rpm_big  rpm with K_leaf = v * B leaf states (matched to gen_big)
Same start and step scale: each leaf state's vector is the same random rendered leaf; rpm's readout is parametrised in gen's
units, f = softmax((U x + c) / s0^2) with U = mu, c = -|mu|^2 / 2 and s0^2 = gen's initial s^2 (per-dimension variance of x),
so at t = 0 both have the same leaf posteriors and an Adam step moves their logits alike. Adam learning rates --lrs are
all run (an earlier sweep over 0.003 - 1 at depth 3 put every arm's best within 0.03 - 0.3); the plot picks the best per arm.
Measured on fresh world strings at log-spaced updates, exact posteriors by inside-outside:
  read[h]     I(z; a) / H(a): posterior marginal of the latent at height h (0 = leaves .. L = root) against the true symbol
  read_alone  the same at the leaves from a leaf's own evidence times its prior marginal (no structure)
  kl          the latent prior read out in world symbols: p(a) = sum_Z p(Z) prod_j M[z_j, a_j], M[z, a] = p(a | z) under the
              posterior; pi_h follows from kl as in practice_rhm_lag_plot
  obj         the training objective (log W or log p(X)) on the evaluation strings (learning curve)
  ideal       the same reading measures for the world's own model (production states, true emission), once per run
Usage: python practice_rhm_rpm.py --L 3 --updates 20000 --lrs 0.03,0.1,0.3 --tag NAME"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from practice_rhm import RHM

DT = torch.float64


def renderer(cond, v, D, B, sigma, rng):
    if cond == "symbolic": return dict(mu=4.0 * np.eye(v), nu=np.zeros((1, v)), sigma=0.5)
    return dict(mu=rng.standard_normal((v, D)), nu=2.0 * rng.standard_normal((B, D)) if cond == "nuisance" else np.zeros((1, D)), sigma=sigma)


def render(leaves, R, rng):
    b = rng.integers(len(R["nu"]), size=leaves.shape)
    return R["mu"][leaves] + R["nu"][b] + R["sigma"] * rng.standard_normal(leaves.shape + (R["mu"].shape[1],))


def gauss_le(X, mu, s2):
    """log N(x; mu_z, s2 I) for every state z: X (n, J, D), mu (Z, D) -> (n, J, Z)."""
    d2 = (X ** 2).sum(-1, keepdim=True) - 2 * X @ mu.T + (mu ** 2).sum(-1)
    return -d2 / (2 * s2) - X.shape[-1] / 2 * torch.log(2 * torch.pi * s2)


def inside(p0, PG, le, s, track=False):
    """log sum_Z p(Z) prod_leaves e(z_leaf). p0 (K0,), PG[k] (s, K_k, K_k+1) child-given-parent probabilities, le (n, s^L, K_L)
    leaf log-evidence. With track, also unit tensors per level (0 = root .. L = leaves) whose gradients are the posterior marginals."""
    n = le.shape[0]; mx = le.max(-1, keepdim=True).values; beta = torch.exp(le - mx); logz = mx[..., 0].sum(1); ones = []
    if track: o = torch.ones_like(beta, requires_grad=True); ones.append(o); beta = beta * o
    for k in reversed(range(len(PG))):
        b = beta.reshape(n, -1, s, beta.shape[-1]); msg = 1
        for j in range(s): msg = msg * (b[:, :, j] @ PG[k][j].T)
        if track: o = torch.ones_like(msg, requires_grad=True); ones.append(o); msg = msg * o
        z = msg.sum(-1); logz = logz + torch.log(z).sum(1); beta = msg / z[..., None]
    return logz + torch.log(beta[:, 0] @ p0), ones[::-1]


def posterior(p0, PG, le, s):
    with torch.enable_grad():
        lw, ones = inside(p0, PG, le.detach(), s, track=True); return lw.detach(), [g.detach() for g in torch.autograd.grad(lw.sum(), ones)]


def prior_marginals(p0, PG, s):
    m = p0[None]
    for P in PG: m = torch.stack([m @ P[j] for j in range(s)], 1).reshape(-1, P.shape[-1])
    return m  # (s^L, K_L), leaf positions in tree order


def info(q, a):
    """I(z; a) / H(a) for posterior marginals q (n, nodes, Z) against true symbols a (n, nodes)."""
    q = q.reshape(-1, q.shape[-1]).numpy(); a = a.reshape(-1); J = np.zeros((q.shape[1], a.max() + 1)); np.add.at(J.T, a, q); J /= J.sum()
    pz, pa = J.sum(1, keepdims=True), J.sum(0, keepdims=True); m = J > 0
    return float((J[m] * np.log(J[m] / (pz @ pa)[m])).sum() / -(pa[pa > 0] * np.log(pa[pa > 0])).sum())


class Model:
    def __init__(self, arm, K, KL, s, init, rng, x0, var):
        t = lambda a: torch.tensor(a, dtype=DT, requires_grad=True); Ks = list(K) + [KL]; self.arm, self.s = arm, s
        self.g0 = t(np.zeros(K[0])); self.G = [t(init * rng.standard_normal((s, Ks[k], Ks[k + 1]))) for k in range(len(K))]
        if arm.startswith("rpm"): self.U, self.c, self.var = t(x0), t(-(x0 ** 2).sum(1) / 2), var; self.params = [self.g0, *self.G, self.U, self.c]
        else: self.mu, self.ls2 = t(x0), t(np.log(var)); self.params = [self.g0, *self.G, self.mu, self.ls2]

    def prior(self): return torch.softmax(self.g0, -1), [torch.softmax(G, -1) for G in self.G]

    def evidence(self, X):
        if self.arm.startswith("rpm"): lf = torch.log_softmax((X @ self.U.T + self.c) / self.var, -1); return lf - torch.log(lf.exp().mean(0))
        return gauss_le(X, self.mu, self.ls2.exp())

    def objective(self, X): p0, PG = self.prior(); return inside(p0, PG, self.evidence(X), self.s)[0]


def true_model(world, R):
    """The world as a factorised tree over production states (symbol, production): exact tables and leaf log-evidence."""
    v, m, s, L, nc = world.v, world.m, world.s, world.L, world.nc; PG = []
    for k in range(L):
        rule = world.rules[k]; npar = len(rule) * m; leafy = k == L - 1; P = np.zeros((s, npar, v if leafy else v * m))
        for a in range(len(rule)):
            for p in range(m):
                for j in range(s):
                    b = rule[a, p, j]
                    if leafy: P[j, a * m + p, b] = 1
                    else: P[j, a * m + p, b * m:(b + 1) * m] = 1 / m
        PG.append(torch.tensor(P, dtype=DT))
    mu, nu, sig = torch.tensor(R["mu"], dtype=DT), torch.tensor(R["nu"], dtype=DT), R["sigma"]
    def le(X):
        g = torch.stack([gauss_le(X, mu + nu[b], torch.tensor(sig ** 2, dtype=DT)) for b in range(len(nu))], 0)
        return torch.logsumexp(g, 0) - np.log(len(nu))
    return torch.full((nc * m,), 1.0 / (nc * m), dtype=DT), PG, le


def reading(p0, PG, le, tree, s, L):
    """Reading measures for a model with prior (p0, PG) and leaf log-evidence le on world strings with true tree."""
    lw, post = posterior(p0, PG, le, s); leaves = tree[-1]
    alone = torch.softmax(le + torch.log(prior_marginals(p0, PG, s))[None], -1)
    return lw, post, dict(read=[info(post[L - h], tree[L - h]) for h in range(L + 1)], read_alone=info(alone, leaves))


def evaluate(world, R, model, rng, n):
    tree = world.sample(n, rng); X = torch.tensor(render(tree[-1], R, rng), dtype=DT); L, s, v = world.L, world.s, world.v
    with torch.no_grad():
        p0, PG = model.prior(); le = model.evidence(X); lw, post, rec = reading(p0, PG, le, tree, s, L)
        q = post[L].reshape(-1, post[L].shape[-1]).numpy(); M = np.full((q.shape[1], v), 1e-9); np.add.at(M.T, tree[-1].reshape(-1), q)
        lM = torch.tensor(np.log(M / M.sum(1, keepdims=True)), dtype=DT)
        nll = -inside(p0, PG, lM.T[tree[-1]], s)[0]
    return dict(obj=float(lw.mean()), kl=float(nll.mean() - world.entropy()), **rec)


def run(job):
    args, cond, arm, seed, lr = job
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed); world = RHM(args.v, args.nc, args.m, args.s, args.L, rng)
    R = renderer(cond, args.v, args.D, args.B, args.sigma, np.random.default_rng(5000 + seed))
    K = [args.nc * args.m * args.kappa] + [args.v * args.m * args.kappa] * (args.L - 1); KL = args.v * (args.B if arm.endswith("_big") else 1)
    r0 = np.random.default_rng(4000 + seed); pool = render(world.sample(args.v * args.B, r0)[-1][:, 0], R, r0)
    var = render(world.sample(1000, r0)[-1], R, r0).reshape(-1, R["mu"].shape[1]).var(0).mean()
    model = Model(arm, K, KL, args.s, args.init, np.random.default_rng(1000 + seed), pool[:KL], var)
    opt = torch.optim.Adam(model.params, lr=lr); rng = np.random.default_rng(2000 + seed)
    tree = world.sample(args.eval_n, np.random.default_rng(6000 + seed)); X = torch.tensor(render(tree[-1], R, np.random.default_rng(7000 + seed)), dtype=DT)
    p0, PG, tle = true_model(world, R)
    with torch.no_grad(): _, _, ideal = reading(p0, PG, tle(X), tree, args.s, args.L)
    times = set(np.unique(np.round(np.logspace(0, np.log10(args.updates), args.n_eval)).astype(int)).tolist()) | {0}; curve = []
    for t in tqdm(range(args.updates + 1), desc=f"{cond} {arm} {seed} lr {lr:g}", leave=False, mininterval=30):
        if t in times: curve.append(dict(t=t, **evaluate(world, R, model, rng, args.eval_n)))
        X = torch.tensor(render(world.sample(args.batch, rng)[-1], R, rng), dtype=DT)
        loss = -model.objective(X).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return dict(cond=cond, arm=arm, seed=seed, lr=lr, ideal=ideal, curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=8); ap.add_argument("--nc", type=int, default=8); ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--s", type=int, default=2); ap.add_argument("--L", type=int, default=3); ap.add_argument("--kappa", type=int, default=2)
    ap.add_argument("--D", type=int, default=32); ap.add_argument("--B", type=int, default=8); ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--init", type=float, default=0.5); ap.add_argument("--lrs", default="0.03,0.1,0.3"); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--updates", type=int, default=20000); ap.add_argument("--n_eval", type=int, default=60); ap.add_argument("--eval_n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--conds", default="symbolic,clean,nuisance"); ap.add_argument("--arms", default="rpm,gen,gen_big")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path("/var/tmp/jbauer/generative_practice_rhm") / f"rpm_{args.tag}L{args.L}_{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True)
    jobs = [(args, c, a, sd, float(lr)) for lr in args.lrs.split(",") for c in args.conds.split(",") for a in args.arms.split(",") for sd in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex: res = list(tqdm(ex.map(run, jobs), total=len(jobs), desc="runs"))
    json.dump(dict(args=vars(args), results=res), open(out / "results.json", "w")); print(out)


if __name__ == "__main__": main()
