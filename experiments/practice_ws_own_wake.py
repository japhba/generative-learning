"""Is a wake step on own samples cleaner than one on foreign samples? (user question, 7 Nov 2026)

The wake update on a sample x, with the code drawn from the recognizer q, has expectation
    gbar(x) = grad_W log p_W(x)  -  grad_W KL(q(z|x) || p_W(z|x))       (q held fixed)
              exact signal (Fisher)   inference bias ("incomprehension")
plus the noise from drawing z ~ q. Both terms are computed exactly by enumerating the 2^K codes, for x from the
learner's own generator, from an independently trained peer, and from the world. Signal and bias are projected on the
world's direction G = E_world grad_W log p_W(x) (ascent on -KL(world || learner)). Then the wake phase's source is
swapped for a few hundred steps of ordinary wake-sleep, and the change in KL(world || learner) is measured. The
source "own_z" keeps the code that generated each own sample instead of reading it (no inference, so no bias).

Model: the one-layer toy of practice_disentangle.py (shared fixed prior, linear-Gaussian decoder, factorial logistic
recognizer), rotated start, trained by wake-sleep on fresh world samples."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, initial_learner, mixture_logpdf, posterior


def copy(L): return Learner(L.W, L.c, L.R, L.b)


def own_x(L, world, n, rng): return world.sample_z(n, rng) @ L.W.T + L.c + world.sigma * rng.standard_normal((n, world.D))


def decompose(L, X, world, Ghat):
    """Per-sample exact signal, wake expectation, amortization gap, posterior entropy, and the projection on Ghat of the
    one-sample wake gradient (mean and variance over z ~ q)."""
    s2, Z, K = world.sigma ** 2, world.Z, world.K
    P = posterior(X, L.W, L.c, world); Ez = P @ Z; Ezz = np.einsum("nm,mi,mj->nij", P, Z, Z)
    a = L.logits(X); m = np.tanh(a); Qzz = m[:, :, None] * m[:, None, :]; Qzz[:, np.arange(K), np.arange(K)] = 1
    Xc = X - L.c
    gstar = (np.einsum("nd,nk->ndk", Xc, Ez) - np.einsum("dj,njk->ndk", L.W, Ezz)) / s2
    gq = (np.einsum("nd,nk->ndk", Xc, m) - np.einsum("dj,njk->ndk", L.W, Qzz)) / s2
    logq = -np.logaddexp(0, -2 * a[:, None, :] * Z[None]).sum(2); q = np.exp(logq)
    lP = np.log(np.maximum(P, 1e-300)); gap = (P * (lP - logq)).sum(1); H = -(P * lP).sum(1)
    proj = ((Xc @ Ghat) @ Z.T - np.einsum("mi,ij,mj->m", Z, L.W.T @ Ghat, Z)[None]) / s2  # <g1(x, z), Ghat> for every code
    pm = (q * proj).sum(1); pv = (q * proj ** 2).sum(1) - pm ** 2
    return dict(gstar=gstar, gq=gq, gap=gap, H=H, pm=pm, pv=pv)


def wake_known(L, X, Z, lr):
    E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / len(X); L.c += lr * E.mean(0)


def kl(L, world, Xt, true_lp): return float(np.mean(true_lp - mixture_logpdf(Xt, L.W, L.c, world)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="chain"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--Ts", default="0,250,1000,4000,16000"); ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=400, help="wake-sleep steps with the wake source swapped"); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--init", default="rotated"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); Ts = [int(t) for t in args.Ts.split(",")]
    out = Path("/var/tmp/jbauer/generative_practice_disentangle/own_wake") / f"{args.prior}_{args.init}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True); rows, cont = [], []
    sources = ["own", "peer", "world"]; record = sorted({int(round(args.steps * f)) for f in [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]})
    for w in tqdm(range(args.worlds), desc="worlds"):
        rng = np.random.default_rng([args.seed, w]); world = World(args.K, args.D, args.sigma, args.prior, rng)
        Xt, _ = world.sample_x(args.n, rng); true_lp = mixture_logpdf(Xt, world.A, np.zeros(args.D), world)
        L = {who: initial_learner(world, np.random.default_rng([args.seed, w, s]), 20000, 0.02, args.batch, args.init) for who, s in [("reader", 1), ("peer", 2)]}
        trng = {who: np.random.default_rng([args.seed, w, s, 1]) for who, s in [("reader", 1), ("peer", 2)]}
        t = 0
        for T in tqdm(Ts, desc="checkpoints", leave=False):
            for who in L:
                for _ in range(T - t): L[who].wake(world.sample_x(args.batch, trng[who])[0], trng[who], args.lr); L[who].sleep(world, trng[who], args.lr * args.sleep_lr_scale, args.batch)
            t = T; R, Pe = L["reader"], L["peer"]
            srng = np.random.default_rng([args.seed, w, T, 7])
            X = {"own": own_x(R, world, args.n, srng), "peer": own_x(Pe, world, args.n, srng), "world": world.sample_x(args.n, srng)[0]}
            G = decompose(R, X["world"], world, np.zeros((args.D, args.K)))["gstar"].mean(0); Ghat = G / np.linalg.norm(G)
            base = dict(world=w, T=T, kl_reader=kl(R, world, Xt, true_lp), kl_peer=kl(Pe, world, Xt, true_lp), G_norm=float(np.linalg.norm(G)))
            for s in sources:
                d = decompose(R, X[s], world, Ghat); gs, b = d["gstar"].mean(0), (d["gq"] - d["gstar"]).mean(0)
                rows.append(dict(base, source=s, gap=float(d["gap"].mean()), H_post=float(d["H"].mean()),
                                 signal_along=float((gs * Ghat).sum()), bias_along=float((b * Ghat).sum()),
                                 signal_norm=float(np.linalg.norm(gs)), bias_norm=float(np.linalg.norm(b)),
                                 bias_rms=float(np.sqrt(((d["gq"] - d["gstar"]) ** 2).sum((1, 2)).mean())),
                                 wake_along=float(d["pm"].mean()), var_x=float(d["pm"].var()), var_z=float(d["pv"].mean()),
                                 snr2=float(d["pm"].mean() ** 2 / (d["pm"].var() + d["pv"].mean()))))
            for s in sources + ["own_z"]:
                for r in range(args.reps):
                    C, crng, traj = copy(R), np.random.default_rng([args.seed, w, T, r, 11]), [base["kl_reader"]]
                    for step in range(1, args.steps + 1):
                        if s == "own_z": Zb = world.sample_z(args.batch, crng); wake_known(C, Zb @ C.W.T + C.c + world.sigma * crng.standard_normal((args.batch, args.D)), Zb, args.lr)
                        else: C.wake({"own": lambda: own_x(C, world, args.batch, crng), "peer": lambda: own_x(Pe, world, args.batch, crng),
                                      "world": lambda: world.sample_x(args.batch, crng)[0]}[s](), crng, args.lr)
                        C.sleep(world, crng, args.lr * args.sleep_lr_scale, args.batch)
                        if step in record: traj.append(kl(C, world, Xt, true_lp))
                    cont.append(dict(world=w, T=T, source=s, rep=r, kl_before=base["kl_reader"], kl_after=traj[-1], traj=traj))
    json.dump(dict(args=vars(args), record=[0] + record, rows=rows, cont=cont), open(out / "results.json", "w"))
    f = lambda xs, k: np.mean([x[k] for x in xs])
    for T in Ts:
        r = [x for x in rows if x["T"] == T]; c = [x for x in cont if x["T"] == T]
        print(f"T={T:6d} KL reader {f(r, 'kl_reader'):.3f} peer {f(r, 'kl_peer'):.3f} |G| {f(r, 'G_norm'):.2f}")
        for s in sources:
            q = [x for x in r if x["source"] == s]
            print(f"   {s:5s} gap {f(q, 'gap'):.3f}  H {f(q, 'H_post'):.3f}  signal {f(q, 'signal_along'):+8.3f}  bias {f(q, 'bias_along'):+8.3f}  "
                  f"|bias| {f(q, 'bias_norm'):.3f} rms {f(q, 'bias_rms'):.2f}  var_x {f(q, 'var_x'):.1f} var_z {f(q, 'var_z'):.1f}  snr2 {f(q, 'snr2'):.2e}")
        print("   dKL after %d steps: " % args.steps + "  ".join(f"{s} {f([x for x in c if x['source'] == s], 'kl_after') - f([x for x in c if x['source'] == s], 'kl_before'):+.3f}" for s in sources + ["own_z"]))
    print(out)


if __name__ == "__main__": main()
