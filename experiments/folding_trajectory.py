"""Compare local regime recognition and long-horizon writing with the same ReLU cell."""
import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


STEPS = (0, 8, 32, 128, 512, 2048)
HORIZONS = (1, 4, 10)
METRICS = ("pair_accuracy", "path_accuracy", "mae_1", "mae_4", "mae_10", "valid_1", "valid_4", "valid_10")


def tent(x, a):
    return np.minimum(x/a, (1-x)/(1-a))


def trajectory(x, a):
    x, a = np.broadcast_arrays(x, a)
    states = [x]
    for _ in range(max(HORIZONS)):
        states.append(tent(states[-1], a))
    return np.stack(states, axis=-1)


class Experts:
    def __init__(self, rng, worlds, structured):
        self.structured = structured
        self.penalty_rate = 0.
        self.penalty = np.zeros((worlds, 2))
        self.count = np.zeros((worlds, 2), dtype=int)
        if structured:
            self.p = rng.uniform(.15, .85, size=(worlds, 2, 1))
        else:
            self.p = rng.normal(0, .3, size=(worlds, 2, 18))
            self.p[:, :, 0] += .5
            self.p[:, :, 10:] = rng.uniform(.05, .95, size=(worlds, 2, 8))
            self.first = np.zeros_like(self.p)
            self.second = np.zeros_like(self.p)
            self.updates = np.zeros((worlds, 2), dtype=int)

    def predict(self, x):
        # x is (world, expert or singleton, example).
        p = self.p
        if self.structured:
            return tent(x, p[:, :, :1])
        h = np.maximum(x[..., None]-p[:, :, None, 10:], 0)
        return p[:, :, :1] + p[:, :, 1:2]*x + np.einsum("weij,wej->wei", h, p[:, :, 2:10])

    def observe(self, s):
        x = s[:, None, :-1]
        err = self.predict(x) - s[:, None, 1:]
        chosen = np.argmin(np.mean(err**2, axis=-1) + self.penalty, axis=1)
        ix = (np.arange(len(s)), chosen)
        self.count[ix] += 1
        self.penalty -= self.penalty_rate/2
        self.penalty[ix] += self.penalty_rate
        p = self.p[ix]
        x = x[:, 0]
        e = err[ix]
        if self.structured:
            deriv = np.where(x < p, -x/p**2, (1-x)/(1-p)**2)
            self.p[ix] = np.clip(p-.07*(e*deriv).mean(-1, keepdims=True), .05, .95)
            return
        delta = x[..., None]-p[:, None, 10:]
        g = np.concatenate((e.mean(-1, keepdims=True), (e*x).mean(-1, keepdims=True),
                            (e[..., None]*np.maximum(delta, 0)).mean(1),
                            -(e[..., None]*(delta > 0)).mean(1)*p[:, 2:10]), axis=1)
        self.updates[ix] += 1
        self.first[ix] = .9*self.first[ix] + .1*g
        self.second[ix] = .999*self.second[ix] + .001*g**2
        t = self.updates[ix][:, None]
        self.p[ix] -= .01*(self.first[ix]/(1-.9**t))/(np.sqrt(self.second[ix]/(1-.999**t))+1e-8)


def evaluate(cell, peaks, paths, roots):
    worlds, panel, length = paths.shape
    grid = np.linspace(0, 1, 201)[None, None, :]
    preds = cell.predict(grid)
    truth = tent(grid, peaks[..., None])
    keep = ((preds-truth)**2).sum((1, 2)) <= ((preds[:, ::-1]-truth)**2).sum((1, 2))
    written_root = roots ^ (~keep[:, None])
    cur = paths[:, :, 0]
    w = np.arange(worlds)[:, None]
    j = np.arange(panel)[None, :]
    teacher_paths = trajectory(cur[:, None, :], peaks[:, :, None])
    pred = cell.predict(paths[:, None, :, :-1].reshape(worlds, 1, -1)).reshape(worlds, 2, panel, length-1)
    err = (pred - paths[:, None, :, 1:])**2
    read = [(err[:, :, :, 0].argmin(1) == written_root).mean(1),
            (err.mean(-1).argmin(1) == written_root).mean(1)]
    mae, valid = [], []
    for h in range(1, length):
        cur = np.clip(cell.predict(cur[:, None])[w, written_root, j], 0, 1)
        if h in HORIZONS:
            mae.append(np.abs(cur-paths[:, :, h]).mean(1))
            valid.append((np.min(np.abs(cur[:, None]-teacher_paths[..., h]), axis=1) <= .05).mean(1))
    return np.stack([*read, *mae, *valid], axis=-1)


def run(worlds, seed, balanced):
    # One independent RNG sequence per world, shared data between architectures.
    seeds = np.random.SeedSequence(seed).spawn(worlds)
    def one_world(s):
        rng = np.random.default_rng(s)
        peaks = rng.uniform([.27, .58], [.42, .73])
        x = rng.random(max(STEPS))
        z = rng.integers(2, size=max(STEPS))
        test_z = rng.integers(2, size=512)
        return peaks, trajectory(x, peaks[z]), trajectory(rng.random(len(test_z)), peaks[test_z]), test_z
    peaks, train, test, roots = map(np.array, zip(*(one_world(s) for s in seeds)))
    rng = np.random.default_rng(seed+1)
    cells = [Experts(rng, worlds, structured) for structured in (True, False)]
    if balanced:
        cells.append(copy.deepcopy(cells[-1]))
        cells[-1].penalty_rate = .01
    result, counts = [], []
    for t in tqdm(range(max(STEPS)+1), desc="watched strings"):
        if t:
            for cell in cells:
                cell.observe(train[:, t-1, :])
        if t in STEPS:
            result.append([evaluate(cell, peaks, test, roots) for cell in cells])
            counts.append([cell.count.copy() for cell in cells])
    return np.array(result), peaks, np.array(counts)


def plot(data, dest):
    scores, peaks = data["scores"], data["peaks"]
    fig, ax = plt.subplot_mosaic([["one", "learning"], ["four", "learning"]], layout="constrained")
    x = np.linspace(0, 1, 501)
    for a in peaks[0]:
        path = trajectory(x, a)
        ax["one"].plot(x, path[:, 1])
        ax["four"].plot(x, path[:, 4])
    ax["one"].set(ylabel="Next state", title="Two teacher regimes")
    ax["four"].set(xlabel="Initial state", ylabel="Fourth state")
    cell = list(data["cells"]).index("balanced ReLU spline")
    for idx, label, color in ((0, "Read a pair", "C0"), (5, "Write one step", "C1"), (7, "Write ten steps", "C3")):
        y = scores[:, cell, :, idx]
        m, ci = y.mean(1), 1.96*y.std(1, ddof=1)/np.sqrt(y.shape[1])
        ax["learning"].plot(data["steps"], m, label=label, color=color)
        ax["learning"].fill_between(data["steps"], m-ci, m+ci, color=color, alpha=.15)
    ax["learning"].set(xlabel="Watched trajectories", ylabel="Success probability", title="Learned ReLU cells")
    ax["learning"].set_xscale("symlog", linthresh=8)
    ax["learning"].set_xticks(data["steps"], [f"{t:,}" for t in data["steps"]])
    ax["learning"].legend()
    fig.savefig(dest.with_suffix(".svg"), transparent=True)
    fig.savefig(dest.with_suffix(".png"), dpi=180, facecolor="white")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--seed", type=int, default=281128)
    cli.add_argument("--worlds", type=int, default=64)
    cli.add_argument("--plot", type=Path)
    cli.add_argument("--balanced", action="store_true")
    args = cli.parse_args()
    if args.plot:
        plot(np.load(args.out), args.plot)
    else:
        scores, peaks, counts = run(args.worlds, args.seed, args.balanced)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        names = ["normalized hinge", "ordinary ReLU spline"]
        if args.balanced:
            names.append("balanced ReLU spline")
        np.savez_compressed(args.out, scores=scores, peaks=peaks, steps=STEPS, metrics=METRICS, seed=args.seed,
                            cells=names, counts=counts)
