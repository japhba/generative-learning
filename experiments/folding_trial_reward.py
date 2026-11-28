"""Reward-weighted search against a frozen reader in the original two-fold world."""
import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from folding_em import BINS, PEAKS, senses
from folding_trajectory import Experts, trajectory


WATCHES = (128, 2048)
CHECK = (0, 1, 2, 4, 8, 16, 32, 48)
ARMS = ("Normalized judge", "Ordinary ReLU judge", "Shuffled normalized", "Shuffled ReLU")
METRICS = ("mean_valid", "explore_valid", "width_only_valid", "mean_mae", "peak_mae", "logit_sd", "lower_valid", "upper_valid")


def peak(q):
    return .18 + np.array([0, .32])[None, :, None] + .32/(1+np.exp(-q))


def endpoints(start, b):
    x = start
    for _ in range(10):
        x = np.minimum(x/b, (1-x)/(1-b))
    return x


def score(cell, paths):
    w, sides, n, time = paths.shape
    inp = paths[..., :-1].reshape(w, 1, -1)
    pred = cell.predict(inp).reshape(w, 2, sides, n, time-1)
    return -np.min(np.mean((pred-paths[:, None, ..., 1:])**2, axis=-1), axis=1)/(2*.004**2)


def metrics(mu, sd, initial, true, start, noise):
    # Each side of the writer has precisely half of the population mass.
    a, truth = peak(mu[..., None]), true[..., None]
    target = endpoints(start, truth)
    alternate = endpoints(start, truth[:, ::-1])
    def valid(y):
        return np.minimum(abs(y-target), abs(y-alternate)) <= .05
    mean = endpoints(start, a)
    success = valid(mean)
    perturbed = endpoints(start, peak(mu[..., None]+sd[..., None]*noise))
    width_only = endpoints(start, peak(initial[..., None]+sd[..., None]*noise))
    return np.column_stack([success.mean((1,2)), valid(perturbed).mean((1,2)), valid(width_only).mean((1,2)),
                            abs(target-mean).mean((1,2)), abs(truth-a).mean((1,2)), sd.mean(1), success.mean(2)])


def practice(cells, initial, true, rng):
    w = len(initial)
    start, noise = rng.random((w,2,512)), rng.normal(size=(w,2,512))
    mu = np.tile(initial, (4,1,1))
    sd = np.ones_like(mu)
    records = []
    for t in range(max(CHECK)+1):
        if t in CHECK:
            records.append([metrics(m, d, initial, true, start, noise) for m,d in zip(mu, sd)])
        if t == max(CHECK):
            break
        eps, cue = rng.normal(size=(w,2,32)), rng.random((w,2,32))
        permutation = rng.random((w,2,32)).argsort(axis=-1)
        for i, cell in enumerate((*cells, *cells)):
            q = mu[i, ..., None]+sd[i, ..., None]*eps
            paths = trajectory(cue, peak(q))
            log_r = score(cell, paths)
            if i >= 2:
                log_r = np.take_along_axis(log_r, permutation, axis=-1)
            r = np.exp(log_r-log_r.max(-1, keepdims=True))
            r /= r.sum(-1, keepdims=True)
            m = .5*mu[i]+.5*(r*q).sum(-1)
            # Stable form of the variance of the half old, half weighted Gaussian fit.
            var = .5*(sd[i]**2 + (mu[i]-m)**2)+.5*(r*(q-m[..., None])**2).sum(-1)
            mu[i], sd[i] = m, np.sqrt(np.maximum(var, .003**2))
    return np.array(records)


def run(seed, w):
    rng = np.random.default_rng(seed)
    true = rng.uniform([.27,.58], [.42,.73], size=(w,2))
    cells = [Experts(rng,w,kind) for kind in (True,False)]
    cells[1].penalty_rate = .01
    _, base, q0, _, _ = senses()
    pi_count = np.broadcast_to(2*(base @ q0), (w,len(PEAKS))).copy()
    x, y_mid = (np.indices((BINS,BINS)).reshape(2,-1)+.5)/BINS
    preds = np.minimum(x[:,None]/PEAKS, (1-x[:,None])/(1-PEAKS))
    assignments = np.exp(-.5*(BINS*(preds-y_mid[:,None]))**2)
    assignments /= assignments.sum(1, keepdims=True)
    outcomes, direct, root_read = [], [], []
    # Evaluation and Practice draws never enter the Watching stream.
    assess = np.random.default_rng(seed+1)
    for t in tqdm(range(1,max(WATCHES)+1), desc="watching and branching"):
        roots = rng.integers(2, size=w)
        y = trajectory(rng.random(w), true[np.arange(w),roots])
        for cell in cells:
            cell.observe(y)
        v = np.minimum((BINS*y).astype(int), BINS-1)
        raw = v[:,:-1]*BINS+v[:,1:]
        # Continuous peaks can occupy cells missed by the finite bank at a boundary.
        # Midpoint assignments are defined on the full grid for this separate experiment.
        pi_count += assignments[raw].mean(1)
        if t in WATCHES:
            weights = pi_count.reshape(w, 2, -1)
            initial_peak = (weights*PEAKS.reshape(2,-1)).sum(-1)/weights.sum(-1)
            frac = (initial_peak-np.array([.18,.50]))/.32
            initial = np.log(frac/(1-frac))
            outcomes.append(practice(cells, initial, true, assess))
            test = assess.random((w,2,512))
            matched = endpoints(test,true[...,None])
            other = endpoints(test,true[:,::-1,None])
            written = endpoints(test,np.sort(cells[0].p[:,:,0], axis=1)[...,None])
            direct.append((np.minimum(abs(written-matched),abs(written-other))<=.05).mean((1,2)))
            paths = trajectory(test, true[...,None])
            acc = []
            for cell in cells:
                pred = cell.predict(test.reshape(w,1,-1)).reshape(w,2,2,-1)
                inferred = np.argmin(abs(pred-paths[:,None,...,1]),axis=1)
                raw_acc = (inferred==np.arange(2)[None,:,None]).mean((1,2))
                acc.append(np.maximum(raw_acc,1-raw_acc))
            root_read.append(acc)
    return np.array(outcomes), np.array(direct), np.array(root_read), true


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--seed", type=int, default=281321)
    cli.add_argument("--worlds", type=int, default=64)
    args=cli.parse_args()
    runs, direct, read, true=run(args.seed,args.worlds)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out, scores=runs, direct=direct, root_read=read, true=true, watches=WATCHES,
                        rounds=CHECK, arms=ARMS, metrics=METRICS, seed=args.seed)
