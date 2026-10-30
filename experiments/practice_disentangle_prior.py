"""Initialization assumption 1: does the learner need to know its prior (its own concepts) in advance?

The learner's prior is a pairwise model over its latents, p(z) proportional to exp(h.z + z^T J z / 2), learned during wake by
a Hebbian/anti-Hebbian rule on its recognised codes (dh = eta_p (z - E_p z), dJ = eta_p (z z^T - E_p z z^T), the
Boltzmann-machine rule), and dreamt from during sleep. Conditions: prior fixed at the world's (the original toy), learned
fast from the world's, learned fast or slowly from independent latents (no concepts). Every learner, and the competitors,
are also scored with a pairwise prior fitted afterwards to their codes, so binding to a fixed prior cannot be the only
thing that separates them."""

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, ae_step, evaluate, initial_learner, mixture_logpdf


def pairwise(world, h, J):
    """The world's latent space seen through a pairwise prior with biases h and couplings J."""
    w = copy.copy(world); lp = world.Z @ h + 0.5 * np.einsum("mi,ij,mj->m", world.Z, J, world.Z)
    w.logp = lp - np.logaddexp.reduce(lp); w.p = np.exp(w.logp); w.mu = w.p @ w.Z; w.Sigma = (w.Z - w.mu).T @ ((w.Z - w.mu) * w.p[:, None]); return w


def fit_pairwise(world, Z, iters=300, lr=0.5):
    """Pairwise prior whose first and second moments match the codes Z (maximum likelihood by moment matching)."""
    m1, m2 = Z.mean(0), Z.T @ Z / len(Z); h, J = np.zeros(world.K), np.zeros((world.K, world.K)); off = 1 - np.eye(world.K)
    for _ in range(iters):
        p = pairwise(world, h, J); h += lr * (m1 - p.mu); J += lr * off * (m2 - (world.Z * p.p[:, None]).T @ world.Z)
    return h, J


def top_layer(world, V, a):
    """The latents' prior as a top layer: H uniform binary causes g, p(z_i = +1 | g) = sigmoid(2 (V g + a)_i)."""
    G = np.array(np.meshgrid(*[[-1.0, 1.0]] * V.shape[1])).reshape(V.shape[1], -1).T  # all top configurations
    U = G @ V.T + a; lp = -np.logaddexp(0, -2 * U[:, None, :] * world.Z[None]).sum(2)  # (top configs, z configs)
    w = copy.copy(world); w.logp = np.logaddexp.reduce(lp, axis=0) - np.log(len(G)); w.p = np.exp(w.logp)
    w.mu = w.p @ w.Z; w.Sigma = (w.Z - w.mu).T @ ((w.Z - w.mu) * w.p[:, None]); return w


def run(world, L0, Xdata, rng, args, cond, evalset):
    """cond: prior 'pairwise' (exact negative phase), 'pairwise_dream' (Hebbian in wake, anti-Hebbian on dreams) or 'top'
    (a top layer of the Helmholtz machine, delta rules in wake, its own recognizer trained in sleep)."""
    L = Learner(L0.W, L0.c, L0.R, L0.b); K = world.K; off = 1 - np.eye(K); kind = cond.get("prior", "pairwise")
    h, J = (np.zeros(K), np.zeros((K, K))) if cond["start"] == "independent" else fit_pairwise(world, world.sample_z(20000, rng))
    V, a = 0.1 * rng.standard_normal((K, args.H)), np.zeros(K); R2, b2 = np.zeros((args.H, K)), np.zeros(args.H)
    curve = []; every = max(1, args.steps // args.points)
    for t in range(args.steps + 1):
        own = top_layer(world, V, a) if kind == "top" else pairwise(world, h, J)
        if t % every == 0: curve.append(dict(step=t, **evaluate(L, own, *evalset)))
        if t == args.steps: break
        X = Xdata[rng.integers(0, len(Xdata), args.batch)] if args.N else world.sample_x(args.batch, rng)[0]
        Z = L.sample_q(X, rng); E = X - Z @ L.W.T - L.c; L.W += args.lr * E.T @ Z / len(X); L.c += args.lr * E.mean(0)  # wake
        if kind == "pairwise" and cond["eta_p"]: h += cond["eta_p"] * (Z.mean(0) - own.mu); J += cond["eta_p"] * off * (Z.T @ Z / len(Z) - (world.Z * own.p[:, None]).T @ world.Z)
        if kind == "biases": h += cond["eta_p"] * (Z.mean(0) - own.mu)  # independent latents, learned frequencies only
        if kind == "pairwise_dream":  # Hebbian on recognised codes now, anti-Hebbian on dreamed codes below
            h += cond["eta_p"] * Z.mean(0); J += cond["eta_p"] * off * (Z.T @ Z / len(Z))
            Zd = own.sample_z(args.batch, rng); h -= cond["eta_p"] * Zd.mean(0); J -= cond["eta_p"] * off * (Zd.T @ Zd / len(Zd))
        if kind == "top":  # wake: infer the top causes of the codes, delta rule on the top layer's generative weights
            Gq = np.where(rng.random((len(Z), args.H)) < 1 / (1 + np.exp(-2 * (Z @ R2.T + b2))), 1.0, -1.0); Ez = Z - np.tanh(Gq @ V.T + a)
            V += cond["eta_p"] * Ez.T @ Gq / len(Z); a += cond["eta_p"] * Ez.mean(0)
            Gd = np.where(rng.random((args.batch, args.H)) < 0.5, 1.0, -1.0); Zd = np.where(rng.random((args.batch, K)) < 1 / (1 + np.exp(-2 * (Gd @ V.T + a))), 1.0, -1.0)
            Eg = Gd - np.tanh(Zd @ R2.T + b2); R2 += args.lr * args.sleep_lr_scale * Eg.T @ Zd / len(Zd); b2 += args.lr * args.sleep_lr_scale * Eg.mean(0)  # sleep, top recognizer
        L.sleep(own, rng, args.lr * args.sleep_lr_scale, args.batch)
    return curve, L, (h, J)


def refit_kl(L, world, Xdata, evalset, rng):
    """KL of a learner's decoder when its prior is refitted to its own codes on the stored data (a prior-learning shortcut)."""
    h, J = fit_pairwise(world, np.sign(np.tanh(L.logits(Xdata))) if L.mem is None else L.sample_q(Xdata, rng))
    return evaluate(L, pairwise(world, h, J), *evalset)["kl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="chain"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12); ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--N", type=int, default=256); ap.add_argument("--init", default="rotated"); ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--points", type=int, default=40); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--sleep_lr_scale", type=float, default=10.0)
    ap.add_argument("--eta_fast", type=float, default=0.02); ap.add_argument("--eta_slow", type=float, default=0.0005)
    ap.add_argument("--H", type=int, default=4, help="top-layer units for the 'top_layer' prior")
    ap.add_argument("--conds", default="fixed_true,learned_from_true,learned_from_independent,slow_from_independent,dream_hebb_from_independent,top_layer")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    conds = {"fixed_true": dict(start="true", eta_p=0.0), "learned_from_true": dict(start="true", eta_p=args.eta_fast),
             "learned_from_independent": dict(start="independent", eta_p=args.eta_fast), "slow_from_independent": dict(start="independent", eta_p=args.eta_slow),
             "dream_hebb_from_independent": dict(start="independent", eta_p=args.eta_fast, prior="pairwise_dream"),
             "top_layer": dict(start="independent", eta_p=args.lr, prior="top"), "biases_only": dict(start="independent", eta_p=args.eta_fast, prior="biases")}
    conds = {c: conds[c] for c in args.conds.split(",")}
    out = Path("/var/tmp/jbauer/generative_practice_disentangle/prior") / f"{args.prior}_N{args.N}_{args.init}{args.tag}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True); results = {c: [] for c in conds}; static = []
    for w in tqdm(range(args.worlds), desc="worlds"):
        rng = np.random.default_rng([args.seed, w]); world = World(args.K, args.D, args.sigma, args.prior, rng)
        Xdata, _ = world.sample_x(max(args.N, 1024), rng); Xdata = Xdata[:args.N] if args.N else Xdata; Xt, Zt = world.sample_x(4000, rng)
        evalset = (Xt, Zt, mixture_logpdf(Xt, world.A, np.zeros(args.D), world)); L0 = initial_learner(world, rng, 20000, 0.02, args.batch, args.init)
        st = {}
        for c, cond in tqdm(conds.items(), leave=False, desc="conditions"):
            curve, L, (h, J) = run(world, L0, Xdata, np.random.default_rng([args.seed, w, list(conds).index(c)]), args, cond, evalset)
            results[c].append(curve); st[f"{c}_refit_kl"] = refit_kl(L, world, Xdata, evalset, rng)
        B = Learner(L0.W, L0.c, L0.R, L0.b); r = np.random.default_rng([args.seed, w, 99])  # binary autoencoder, prior-agnostic, protocol settings
        for _ in range(args.steps):
            for _ in range(2): ae_step(B, Xdata[r.integers(0, len(Xdata), args.batch)], world, 0.06, 10.0, "bae")
        st["binary_ae_kl"] = evaluate(B, world, *evalset)["kl"]; st["binary_ae_refit_kl"] = refit_kl(B, world, Xdata, evalset, rng); static.append(st)
    json.dump(dict(args=vars(args), results=results, static=static), open(out / "results.json", "w"))
    for c in conds:
        last = [cv[-1] for cv in results[c]]
        print(f"{c:26s} KL {np.mean([x['kl'] for x in last]):.3f}  refit {np.mean([s[f'{c}_refit_kl'] for s in static]):.3f}  purity {np.mean([x['align'] for x in last]):.3f}  acc_sym {np.mean([x['acc_sym'] for x in last]):.3f}")
    print(f"{'binary autoencoder':26s} KL {np.mean([s['binary_ae_kl'] for s in static]):.3f}  refit {np.mean([s['binary_ae_refit_kl'] for s in static]):.3f}")
    print(out)


if __name__ == "__main__": main()
