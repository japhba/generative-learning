"""More supervision on top of the same practice: does watching more foreign samples per own sample help or hurt wake-sleep?
The number of sleep batches is fixed; `ratio` wake batches on foreign samples are done per sleep batch. With N = 0 every
foreign batch is fresh, so any harm from more watching cannot be overfitting."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, evaluate, initial_learner, mixture_logpdf


def run(world, L0, Xdata, rng, ratio, args, evalset):
    L = Learner(L0.W, L0.c, L0.R, L0.b); curve = []; every = max(1, args.steps // args.points); owed = 0.0; seen = 0
    for t in range(args.steps + 1):
        if t % every == 0: curve.append(dict(step=t, foreign=seen, **evaluate(L, world, *evalset)))
        if t == args.steps: break
        owed += ratio
        while owed >= 1:
            X = world.sample_x(args.batch, rng)[0] if args.N == 0 else Xdata[rng.integers(0, len(Xdata), args.batch)]
            L.wake(X, rng, args.lr); owed -= 1; seen += args.batch
        L.sleep(world, rng, args.lr * args.sleep_lr_scale, args.batch)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="chain"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--N", type=int, default=0, help="stored foreign samples; 0 = fresh every batch")
    ap.add_argument("--ratios", default="0.25,0.5,1,2,4,8,16"); ap.add_argument("--init", default="rotated"); ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--points", type=int, default=40); ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    args = ap.parse_args(); ratios = [float(r) for r in args.ratios.split(",")]
    out = Path("/var/tmp/jbauer/generative_practice_disentangle/supervision") / f"{args.prior}_N{args.N}_{args.init}{args.tag}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True); results = {str(r): [] for r in ratios}
    for w in tqdm(range(args.worlds), desc="worlds"):
        rng = np.random.default_rng([args.seed, w]); world = World(args.K, args.D, args.sigma, args.prior, rng)
        Xdata, _ = world.sample_x(max(args.N, 1), rng); Xtest, Ztest = world.sample_x(4000, rng)
        evalset = (Xtest, Ztest, mixture_logpdf(Xtest, world.A, np.zeros(args.D), world))
        L0 = initial_learner(world, rng, 20000, 0.02, args.batch, args.init, args.init_scale)
        for r in tqdm(ratios, leave=False, desc="ratios"): results[str(r)].append(run(world, L0, Xdata, np.random.default_rng([args.seed, w, int(r * 100)]), r, args, evalset))
    json.dump(dict(args=vars(args), results=results), open(out / "results.json", "w"))
    for r in ratios:
        last = [c[-1] for c in results[str(r)]]
        print(f"ratio {r:5g}: kl {np.mean([c['kl'] for c in last]):.3f}  acc_sym {np.mean([c['acc_sym'] for c in last]):.3f}  bound {np.mean([c['acc_sym'] > 0.9 for c in last]):.2f}")
    print(out)


if __name__ == "__main__": main()
