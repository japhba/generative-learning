"""Own wake steps when recognition is ahead of generation (user question, 7 Nov 2026).

The own wake step W += lr (x_own - W zhat - c) zhat^T, with zhat read from x_own by the recognizer (not the code that
generated x_own), follows -grad_W KL(q(z|x) || p_W(z|x)) on own samples: it moves the generator towards the one the
recognizer inverts. It can only help if the recognizer knows something the generator does not. Here that lead is put in
by hand: a learner trained to convergence by wake-sleep keeps its recognizer while its generator is replaced by a
damaged one (rotated or random start). Arms, all from that state:
  ws            world wake + sleep (ordinary wake-sleep)
  world_keep    world wake, recognizer kept (sleep off)
  world16_keep  wake on 16 stored world samples, recognizer kept
  own_keep      own wake, recognizer kept: no world samples at all
  own_sleep     own wake + sleep (the recognizer is refitted to the damaged generator as it goes)
  own_fitted    own wake with a recognizer fitted to the damaged generator first (no lead), sleep off"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, initial_learner, mixture_logpdf
from practice_ws_own_wake import copy, kl, own_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="chain"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--worlds", type=int, default=8); ap.add_argument("--T0", type=int, default=16000)
    ap.add_argument("--damage", default="rotated"); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); arms = ["ws", "world_keep", "world16_keep", "own_keep", "own_sleep", "own_fitted"]
    record = [0, 25, 50, 100, 200, 400, 1000, 2000, 4000]; record = [r for r in record if r <= args.steps]
    out = Path("/var/tmp/jbauer/generative_practice_disentangle/own_distill") / f"{args.prior}_{args.damage}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True); rows = []
    for w in tqdm(range(args.worlds), desc="worlds"):
        rng = np.random.default_rng([args.seed, w]); world = World(args.K, args.D, args.sigma, args.prior, rng)
        Xt, _ = world.sample_x(4000, rng); true_lp = mixture_logpdf(Xt, world.A, np.zeros(args.D), world); X16 = world.sample_x(16, rng)[0]
        L, trng = initial_learner(world, np.random.default_rng([args.seed, w, 1]), 20000, 0.02, args.batch, "rotated"), np.random.default_rng([args.seed, w, 1, 1])
        for _ in range(args.T0): L.wake(world.sample_x(args.batch, trng)[0], trng, args.lr); L.sleep(world, trng, args.lr * args.sleep_lr_scale, args.batch)
        D0 = initial_learner(world, np.random.default_rng([args.seed, w, 3]), 20000, 0.02, args.batch, args.damage)  # damaged generator + its own fitted recognizer
        base = dict(world=w, kl_trained=kl(L, world, Xt, true_lp))
        for arm in tqdm(arms, desc="arms", leave=False):
            for r in range(args.reps):
                C = Learner(D0.W, D0.c, D0.R, D0.b) if arm == "own_fitted" else Learner(D0.W, D0.c, L.R, L.b)
                crng, traj = np.random.default_rng([args.seed, w, r, 11]), [kl(C, world, Xt, true_lp)]
                for step in range(1, args.steps + 1):
                    X = own_x(C, world, args.batch, crng) if arm.startswith("own") else X16[crng.integers(0, 16, args.batch)] if arm == "world16_keep" else world.sample_x(args.batch, crng)[0]
                    C.wake(X, crng, args.lr)
                    if arm in ("ws", "own_sleep"): C.sleep(world, crng, args.lr * args.sleep_lr_scale, args.batch)
                    if step in record: traj.append(kl(C, world, Xt, true_lp))
                rows.append(dict(base, arm=arm, rep=r, traj=traj))
    json.dump(dict(args=vars(args), record=record, rows=rows), open(out / "results.json", "w"))
    print(f"trained learner KL {np.mean([x['kl_trained'] for x in rows]):.3f}; KL(world || generator) at steps {record}")
    for arm in arms:
        tr = np.array([x["traj"] for x in rows if x["arm"] == arm]); print(f"  {arm:13s} " + " ".join(f"{v:6.3f}" for v in tr.mean(0)))
    print(out)


if __name__ == "__main__": main()
