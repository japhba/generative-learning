"""More supervision, worse concepts? The learner's own prior (its concepts, a chain with correlation rho_own) may differ
from the structure of the foreign data (a chain with rho_world). Fresh foreign data stream in, so nothing can overfit.
We track how well the learner fits the data (KL) and how well its latents keep their meaning (alignment of each own
unit with one world factor), for practice (wake-sleep), watching only (recognizer frozen) and exact EM."""

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, em_step, evaluate, initial_learner, log_prior, mixture_logpdf


def with_prior(world, rho):
    """The same world (same mixing A) seen through a chain prior with correlation rho: the learner's own concepts."""
    w = copy.copy(world); w.params = dict(world.params, rho=rho); w.logp = log_prior("chain", w.Z, w.params); w.p = np.exp(w.logp)
    w.mu = w.p @ w.Z; w.Sigma = (w.Z - w.mu).T @ ((w.Z - w.mu) * w.p[:, None]); return w


def start(own, rng, kind, batch):
    if kind == "random": return initial_learner(own, rng, 20000, 0.02, batch, "random")
    L = Learner(own.A, np.zeros(own.D), np.linalg.pinv(own.A) / own.sigma ** 2 * 0.25, np.zeros(own.K))  # true concepts
    for _ in range(20000): L.sleep(own, rng, 0.02, batch)
    return L


def run(data, own, L0, method, rng, args, evalset, eval_steps):
    L = Learner(L0.W, L0.c, L0.R, L0.b); curve = []; B = args.batch
    for t in range(args.steps + 1):
        if t in eval_steps: curve.append(dict(step=t, **evaluate(L, own, *evalset)))
        if t == args.steps: break
        if method == "practice": L.wake(data.sample_x(B, rng)[0], rng, args.lr); L.sleep(own, rng, args.lr * args.sleep_lr_scale, B)
        elif method == "watch": L.wake(data.sample_x(B, rng)[0], rng, args.lr); L.wake(data.sample_x(B, rng)[0], rng, args.lr)
        elif method == "em": em_step(L, data.sample_x(B, rng)[0], own, args.lr * args.em_lr_scale); em_step(L, data.sample_x(B, rng)[0], own, args.lr * args.em_lr_scale)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho_world", type=float, default=0.5); ap.add_argument("--rho_own", type=float, default=0.85)
    ap.add_argument("--start", default="true", choices=["true", "random"]); ap.add_argument("--methods", default="practice,watch,em")
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12); ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--em_lr_scale", type=float, default=3.0)
    ap.add_argument("--worlds", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    args = ap.parse_args(); methods = args.methods.split(",")
    eval_steps = sorted({0, args.steps} | {int(s) for s in np.unique(np.round(np.logspace(1, np.log10(args.steps), 30)))})
    out = Path("/var/tmp/jbauer/generative_practice_disentangle/conflict") / f"rw{args.rho_world}_ro{args.rho_own}_{args.start}_lr{args.lr}{args.tag}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True); results = {m: [] for m in methods}
    for w in tqdm(range(args.worlds), desc="worlds"):
        rng = np.random.default_rng([args.seed, w]); data = World(args.K, args.D, args.sigma, "chain", rng, rho=args.rho_world); own = with_prior(data, args.rho_own)
        Xt, Zt = data.sample_x(4000, rng); evalset = (Xt, Zt, mixture_logpdf(Xt, data.A, np.zeros(args.D), data))
        L0 = start(own, np.random.default_rng([args.seed, w, 1]), args.start, args.batch)
        for m in tqdm(methods, leave=False, desc="methods"): results[m].append(run(data, own, L0, m, np.random.default_rng([args.seed, w, 2, methods.index(m)]), args, evalset, eval_steps))
    json.dump(dict(args=vars(args), results=results), open(out / "results.json", "w"))
    for m in methods:
        c = results[m]; f = lambda i, k: np.mean([run[i][k] for run in c])
        print(f"{m:9s} KL {f(0, 'kl'):.3f} -> {f(-1, 'kl'):.3f} (min {np.mean([min(p['kl'] for p in run) for run in c]):.3f})   align {f(0, 'align'):.3f} -> {f(-1, 'align'):.3f} (max {np.mean([max(p['align'] for p in run) for run in c]):.3f})   acc_sym {f(0, 'acc_sym'):.3f} -> {f(-1, 'acc_sym'):.3f}")
    print(out)


if __name__ == "__main__": main()
