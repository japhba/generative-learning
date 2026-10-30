"""Learning a lesson from own vs foreign examples (README line 40, point 1).

A reader and a peer learn the same world by wake-sleep from independent random starts, for T steps. Then the world's
prior shifts by a lesson, p'(z) proportional to pi(z) exp(b*.z). Writers are told the lesson in their own concept labels
and render the same lesson latents with their own decoders: the reader itself (own), the peer (foreign) and the world
(a perfect teacher). The reader estimates the lesson from how it recognises the examples and is scored by the KL from the
shifted world to its updated model, on held-out samples."""

import argparse
from datetime import datetime
from itertools import permutations
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, initial_learner


def train(world, L, Xdata, rng, args, checkpoints):
    """Wake-sleep on stored foreign samples; copies of the learner at the checkpoint steps."""
    out = {}
    for t in range(max(checkpoints) + 1):
        if t in checkpoints: out[t] = Learner(L.W, L.c, L.R, L.b)
        L.wake(Xdata[rng.integers(0, len(Xdata), args.batch)], rng, args.lr); L.sleep(world, rng, args.lr * args.sleep_lr_scale, args.batch)
    return out


def labels(L, world):
    """Signed permutation P with z_own = P z_world: how a writer told the lesson in world terms puts it into its own units."""
    cos = (L.W / np.linalg.norm(L.W, axis=0)).T @ world.A
    p = max(permutations(range(world.K)), key=lambda p: np.abs(cos[list(p), range(world.K)]).sum())
    P = np.zeros((world.K, world.K)); P[list(p), range(world.K)] = np.sign(cos[list(p), range(world.K)]); return P


def logZ(world, b): return np.logaddexp.reduce(world.logp + world.Z @ b)


def fit_bias(world, target, iters=100):
    """Bias b whose tilted prior pi(z) exp(b.z) has mean `target` (maximum likelihood; Newton with backtracking)."""
    t = np.clip(target, -0.98, 0.98); b = np.zeros(world.K); f = lambda b: b @ t - logZ(world, b)
    for _ in range(iters):
        p = np.exp(world.logp + world.Z @ b - logZ(world, b)); mu = p @ world.Z
        if np.abs(t - mu).max() < 1e-7: break
        C = (world.Z - mu).T @ ((world.Z - mu) * p[:, None]); step = np.linalg.solve(C + 1e-6 * np.eye(world.K), t - mu); s = 1.0
        while f(b + s * step) < f(b) and s > 1e-6: s /= 2
        b = b + s * step
    return b


def learn_lesson(L, X, world, outer=5):
    """Read the examples with the prior tilted by the current estimate, refit the tilt to the mean code, repeat."""
    b = np.zeros(world.K)
    for _ in range(outer): b = fit_bias(world, np.tanh(L.logits(X) + b).mean(0))
    return b


def model_logpdf(X, L, world, b):
    """Reader's model of the shifted world: its own prior tilted by b (own labels), its own decoder."""
    lp = world.logp + world.Z @ b - logZ(world, b); means = world.Z @ L.W.T + L.c
    return np.logaddexp.reduce(lp[None] - ((X[:, None, :] - means[None]) ** 2).sum(2) / (2 * world.sigma ** 2), axis=1) - world.D / 2 * np.log(2 * np.pi * world.sigma ** 2)


def best_bias(X, L, world, iters=30):
    """The best lesson this reader can learn: EM on b with its exact posterior, on unlimited samples of the shifted world."""
    b = np.zeros(world.K); means = world.Z @ L.W.T + L.c; d2 = ((X[:, None, :] - means[None]) ** 2).sum(2) / (2 * world.sigma ** 2)
    for _ in range(iters):
        lj = world.logp + world.Z @ b - d2; post = np.exp(lj - np.logaddexp.reduce(lj, axis=1, keepdims=True))
        b = fit_bias(world, (post @ world.Z).mean(0))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="chain"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--N", type=int, default=1024); ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--Ts", default="0,500,2000,8000,20000"); ap.add_argument("--n_ex", default="16,64,256,1024")
    ap.add_argument("--lesson", type=float, default=1.0, help="size of the bias on each of the two concepts the lesson shifts")
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--init", default="rotated"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    args = ap.parse_args(); Ts = [int(t) for t in args.Ts.split(",")]; ns = [int(n) for n in args.n_ex.split(",")]
    out = Path("/var/tmp/jbauer/generative_practice_disentangle/own") / f"{args.prior}_{args.init}{args.tag}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True); rows = []
    for w in tqdm(range(args.worlds), desc="worlds"):
        rng = np.random.default_rng([args.seed, w]); world = World(args.K, args.D, args.sigma, args.prior, rng); Xdata, _ = world.sample_x(args.N, rng)
        learners = {who: train(world, initial_learner(world, np.random.default_rng([args.seed, w, s]), 20000, 0.02, args.batch, args.init), Xdata,
                               np.random.default_rng([args.seed, w, s, 1]), args, set(Ts)) for who, s in [("reader", 1), ("peer", 2)]}
        bstar = np.zeros(args.K); bstar[rng.choice(args.K, 2, replace=False)] = args.lesson * rng.choice([-1, 1], 2)  # the lesson, world labels
        lp_new = world.logp + world.Z @ bstar - logZ(world, bstar)
        Zt = world.Z[rng.choice(len(world.Z), 4000, p=np.exp(lp_new))]; Xt = Zt @ world.A.T + args.sigma * rng.standard_normal((4000, args.D))
        true = np.logaddexp.reduce(lp_new[None] - ((Xt[:, None, :] - (world.Z @ world.A.T)[None]) ** 2).sum(2) / (2 * args.sigma ** 2), axis=1) - args.D / 2 * np.log(2 * np.pi * args.sigma ** 2)
        for T in Ts:
            R, Pe = learners["reader"][T], learners["peer"][T]; PR, PP = labels(R, world), labels(Pe, world)
            kl = lambda b: float(np.mean(true - model_logpdf(Xt, R, world, b)))
            kl_none, kl_oracle, kl_best = kl(np.zeros(args.K)), kl(PR @ bstar), kl(best_bias(Xt, R, world))
            align = lambda L: float(np.mean(np.abs((L.W / np.linalg.norm(L.W, axis=0)).T @ world.A).max(1)))
            for n in ns:
                Z = world.Z[rng.choice(len(world.Z), n, p=np.exp(lp_new))]; E = args.sigma * rng.standard_normal((n, args.D))
                writers = {"own": (Z @ PR.T) @ R.W.T + R.c + E, "peer": (Z @ PP.T) @ Pe.W.T + Pe.c + E, "world": Z @ world.A.T + E}
                for who, X in writers.items():
                    b = learn_lesson(R, X, world); k = kl(b)
                    rows.append(dict(world=w, T=T, n=n, writer=who, kl=k, kl_none=kl_none, kl_oracle=kl_oracle, kl_best=kl_best, gain=kl_none - k,
                                     learned=(kl_none - k) / (kl_none - kl_best),
                                     read=float(np.mean(np.sign(np.tanh(R.logits(X))) == Z @ PR.T)), b_err=float(np.linalg.norm(b - PR @ bstar)),
                                     align_reader=align(R), align_peer=align(Pe), same_labels=bool(np.allclose(np.abs(PR), np.abs(PP)))))
    json.dump(dict(args=vars(args), rows=rows), open(out / "results.json", "w"))
    for T in Ts:
        for n in ns:
            r = [x for x in rows if x["T"] == T and x["n"] == n]
            print(f"T={T:6d} n={n:5d} best gain {np.mean([x['kl_none'] - x['kl_best'] for x in r]):.3f}  " + "  ".join(f"{who}: gain {np.mean([x['gain'] for x in r if x['writer'] == who]):+.3f} learned {np.median([x['learned'] for x in r if x['writer'] == who]):+.2f} read {np.mean([x['read'] for x in r if x['writer'] == who]):.2f}" for who in ["own", "peer", "world"]))
    print(out)


if __name__ == "__main__": main()
