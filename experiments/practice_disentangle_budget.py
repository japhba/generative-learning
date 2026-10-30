"""Fresh foreign samples versus own generation steps for the wake-sleep disentangling learner.

split: every step does one wake update on stored data (if N > 0) and one extra update, which is a sleep
       update with probability f and a wake update on a fresh foreign batch otherwise.
value: practise for P steps on stored data, freeze recognition, then spend 5000 wake steps on fresh or on stored data.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, evaluate, initial_learner, mixture_logpdf


def split_run(world, L0, Xdata, f, rng, steps, lr, B, evalset, points, ss):
    L = Learner(L0.W, L0.c, L0.R, L0.b); curve = []; every = steps // points
    for t in range(steps + 1):
        if t % every == 0: curve.append(evaluate(L, world, *evalset)["kl"])
        if t == steps: break
        if len(Xdata): L.wake(Xdata[rng.integers(0, len(Xdata), B)], rng, lr)
        if rng.random() < f: L.sleep(world, rng, lr * ss, B)
        else: L.wake(world.sample_x(B, rng)[0], rng, lr)
    return curve


def value_run(world, L0, Xdata, P, rng, lr, B, evalset, ss, after=5000):
    L = Learner(L0.W, L0.c, L0.R, L0.b)
    for _ in range(P): L.wake(Xdata[rng.integers(0, len(Xdata), B)], rng, lr); L.sleep(world, rng, lr * ss, B)
    out = {"practised": evaluate(L, world, *evalset)}
    for source in ["fresh", "stored"]:
        M = Learner(L.W, L.c, L.R, L.b)
        for _ in range(after):
            for _ in range(2): M.wake(world.sample_x(B, rng)[0] if source == "fresh" else Xdata[rng.integers(0, len(Xdata), B)], rng, lr)
        out[source] = evaluate(M, world, *evalset)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="chain"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--sleep_lr_scale", type=float, default=10.0)
    ap.add_argument("--Ns", default="0,64,256,1024"); ap.add_argument("--fs", default="0,0.1,0.25,0.5,0.75,0.9,1")
    ap.add_argument("--Ps", default="0,250,1000,4000,16000"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/var/tmp/jbauer/generative_practice_disentangle/budget")
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / f"{args.prior}_K{args.K}_ss{args.sleep_lr_scale}_seed{args.seed}_{stamp}"; out.mkdir(parents=True)
    Ns, fs, Ps = [int(v) for v in args.Ns.split(",")], [float(v) for v in args.fs.split(",")], [int(v) for v in args.Ps.split(",")]
    split = {N: {f: [] for f in fs} for N in Ns}; value = {P: [] for P in Ps}
    for w in tqdm(range(args.worlds), desc="worlds"):
        rng = np.random.default_rng([args.seed, w])
        world = World(args.K, args.D, args.sigma, args.prior, rng)
        Xall, _ = world.sample_x(max(Ns), rng); Xtest, Ztest = world.sample_x(4000, rng)
        evalset = (Xtest, Ztest, mixture_logpdf(Xtest, world.A, np.zeros(args.D), world))
        L0 = initial_learner(world, rng, 20000, 0.02, args.batch)
        for N in tqdm(Ns, desc="N", leave=False):
            for i, f in enumerate(fs): split[N][f].append(split_run(world, L0, Xall[:N], f, np.random.default_rng([args.seed, w, N, i]), args.steps, args.lr, args.batch, evalset, 40, args.sleep_lr_scale))
        for j, P in enumerate(Ps): value[P].append(value_run(world, L0, Xall[:256], P, np.random.default_rng([args.seed, w, 7, j]), args.lr, args.batch, evalset, args.sleep_lr_scale))
    json.dump(dict(args=vars(args), split={str(N): {str(f): v for f, v in d.items()} for N, d in split.items()}, value={str(P): v for P, v in value.items()}), open(out / "results.json", "w"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for N in Ns:
        final = np.array([[run[-1] for run in split[N][f]] for f in fs])
        axes[0].errorbar(fs, final.mean(1), final.std(1) / np.sqrt(final.shape[1]), marker="o", label=f"N = {N} stored")
    axes[0].set_xlabel("fraction of extra updates spent on own samples (sleep)"); axes[0].set_ylabel("final KL(true || model) [nats]"); axes[0].set_yscale("log"); axes[0].legend(); axes[0].set_title(f"budget split after {args.steps} steps")
    for f in fs:
        curves = np.array(split[256][f]); axes[1].plot(np.linspace(0, args.steps, curves.shape[1]), curves.mean(0), label=f"f = {f}")
    axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("KL [nats]"); axes[1].set_title("learning curves, N = 256"); axes[1].legend()
    for key, lab in [("practised", "after practice"), ("fresh", "+ fresh foreign, recognition frozen"), ("stored", "+ stored foreign, recognition frozen")]:
        vals = np.array([[r[key]["kl"] for r in value[P]] for P in Ps])
        axes[2].errorbar(np.array(Ps) + 1, vals.mean(1), vals.std(1) / np.sqrt(vals.shape[1]), marker="o", label=lab)
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("practice steps P (+1)"); axes[2].set_ylabel("KL [nats]"); axes[2].set_title("value of foreign data after practice (N = 256)"); axes[2].legend()
    fig.suptitle(f"{args.prior} prior, K={args.K}, D={args.D}, sigma={args.sigma}"); fig.tight_layout(); fig.savefig(out / "budget.png", dpi=120)
    print(out / "budget.png")


if __name__ == "__main__": main()
