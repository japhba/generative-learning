"""Capacity gap: can a stronger teacher make a worse student? (README line 40, point 1: inferring the latents of foreign
examples is hard.)

World: K_w independent binary features of decaying strength s_k, x = sum_k s_k a_k z_k + sigma eps. Teachers: the best
K_t-feature model of the world (exact EM on world samples), K_t = K_w being the world itself. Students with K_s < K_w
features learn from fresh teacher samples, by wake-sleep (factorial recognizer trained on own dreams) or by exact EM,
and are scored by KL(world || student). With exact inference the student's optimum is the projection of the teacher,
which can only get better as the teacher approaches the world; any interior optimum in K_t comes from reading."""

import argparse
from datetime import datetime
from functools import partial
import json
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, em_step, initial_learner, mixture_logpdf, posterior


def fit_teacher(world, Kt, X, iters=60):
    """Best Kt-feature model of the world samples X: batch EM, started from the Kt strongest world features."""
    own = World(Kt, world.D, world.sigma, "factorial", np.random.default_rng(0)); W = world.A[:, :Kt].copy(); c = np.zeros(world.D)
    for _ in range(iters):
        P = posterior(X, W, c, own); Ez = P @ own.Z; Ezz = np.einsum("nm,mi,mj->ij", P, own.Z, own.Z)
        Za = np.c_[Ez, np.ones(len(X))]; M = np.block([[Ezz, Ez.sum(0)[:, None]], [Ez.sum(0)[None], np.array([[len(X)]])]])
        Wc = np.linalg.solve(M, Za.T @ X).T; W, c = Wc[:, :Kt], Wc[:, Kt]
    return own, W, c


def run(teacher, own, L0, method, rng, args, evalset, eval_steps):
    (tw, tW, tc), L, curve = teacher, Learner(L0.W, L0.c, L0.R, L0.b), []
    for t in range(args.steps + 1):
        if t in eval_steps: curve.append(dict(step=t, kl=float(np.mean(evalset[1] - mixture_logpdf(evalset[0], L.W, L.c, own)))))
        if t == args.steps: break
        X = tw.sample_z(args.batch, rng) @ tW.T + tc + tw.sigma * rng.standard_normal((args.batch, tw.D))
        if method == "wake_sleep": L.wake(X, rng, args.lr); L.sleep(own, rng, args.lr * args.sleep_lr_scale, args.batch)
        else: em_step(L, X, own, args.lr * args.em_lr_scale)
    return curve


def one(job, args, methods, eval_steps):
    """All teachers and methods for one world and one student capacity."""
    w, Ks = job; rng = np.random.default_rng([args.seed, w]); world = World(args.Kw, args.D, args.sigma, "factorial", rng)
    world.A = world.A * (args.s0 * np.arange(1, args.Kw + 1) ** -args.decay)  # strongest feature first
    Xt = world.sample_x(4000, rng)[0]; true = mixture_logpdf(Xt, world.A, np.zeros(args.D), world); Xfit = world.sample_x(args.n_fit, rng)[0]
    teachers = {Kt: fit_teacher(world, Kt, Xfit) if Kt < args.Kw else (world, world.A, np.zeros(args.D)) for Kt in range(1, args.Kw + 1)}
    own = World(Ks, args.D, args.sigma, "factorial", np.random.default_rng([args.seed, w, Ks])); L0 = initial_learner(own, np.random.default_rng([args.seed, w, Ks, 1]), 5000, 0.02, args.batch, "random")
    kl_best = float(np.mean(true - mixture_logpdf(Xt, teachers[Ks][1], teachers[Ks][2], own)))  # the best Ks-feature model of the world: the exact student's target
    rows = []
    for Kt, teacher in teachers.items():
        kl_teacher = float(np.mean(true - mixture_logpdf(Xt, teacher[1], teacher[2], teacher[0])))
        for m in methods:
            curve = run(teacher, own, L0, m, np.random.default_rng([args.seed, w, Ks, Kt, methods.index(m)]), args, (Xt, true), eval_steps)
            rows.append(dict(world=w, Ks=Ks, Kt=Kt, method=m, kl_teacher=kl_teacher, kl_best=kl_best, curve=curve))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Kw", type=int, default=8); ap.add_argument("--D", type=int, default=16); ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--s0", type=float, default=1.5); ap.add_argument("--decay", type=float, default=0.5, help="feature strengths s_k = s0 k^-decay")
    ap.add_argument("--Ks", default="2,3,4"); ap.add_argument("--methods", default="wake_sleep,em"); ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--em_lr_scale", type=float, default=3.0)
    ap.add_argument("--n_fit", type=int, default=5000); ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    args = ap.parse_args(); Kss, methods = [int(k) for k in args.Ks.split(",")], args.methods.split(",")
    eval_steps = sorted({0, args.steps} | {int(s) for s in np.unique(np.round(np.logspace(1, np.log10(args.steps), 25)))})
    out = Path("/var/tmp/jbauer/generative_practice_disentangle/capacity") / f"Kw{args.Kw}_decay{args.decay}{args.tag}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True); jobs = [(w, Ks) for w in range(args.worlds) for Ks in Kss]
    with Pool(min(len(jobs), args.procs)) as pool: rows = [r for rs in tqdm(pool.imap(partial(one, args=args, methods=methods, eval_steps=eval_steps), jobs), total=len(jobs), desc="world x student") for r in rs]
    json.dump(dict(args=vars(args), rows=rows), open(out / "results.json", "w"))
    for Ks in Kss:
        for m in methods:
            r = [x for x in rows if x["Ks"] == Ks and x["method"] == m]
            print(f"Ks={Ks} {m:10s} best {np.mean([x['kl_best'] for x in r if x['Kt'] == 1]):.3f}  " + "  ".join(f"Kt={Kt}: {np.mean([x['curve'][-1]['kl'] for x in r if x['Kt'] == Kt]):.3f}" for Kt in range(1, args.Kw + 1)))
    print("teacher KL " + "  ".join(f"Kt={Kt}: {np.mean([x['kl_teacher'] for x in rows if x['Kt'] == Kt]):.3f}" for Kt in range(1, args.Kw + 1)))
    print(out)


if __name__ == "__main__": main()
