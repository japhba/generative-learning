"""Learn a tensor code's sparse checks and compile them from visible innovations."""
import argparse
import copy
import itertools
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


FULL = 0.75
CALIBRATION = 8
ARMS = ("Own", "Fresh teacher", "Twin", "Equal-KL foreign", "Whole-word filter", "Check compiler", "Passive solver")
METRICS = ("visible_kl", "full_valid", "root_erased", "bits_erased", "level1_erased", "rank", "false_checks", "calibrated")
EVENTS = ("complete", "old_pass", "useful", "full_pass", "level1_pass", "level1_old_pass", "level1_useful")
CANDIDATES = tuple(sum(1 << i for i in sub) for sub in itertools.combinations(range(7), 4))
SAT = tuple(sum(1 << j for j, h in enumerate(CANDIDATES) if (x & h).bit_count() % 2 == 0) for x in range(128))


def parity(x):
    return x.bit_count() % 2


def positions(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def combine(rows, mask):
    value = 0
    for i in positions(mask):
        value ^= rows[i]
    return value


def evaluate(rows, u):
    return sum(parity(row & u) << i for i, row in enumerate(rows))


def insert(basis, value):
    while value:
        i = value.bit_length() - 1
        if i not in basis:
            basis[i] = value
            return True
        value ^= basis[i]
    return False


def rank(rows):
    basis = {}
    for row in rows:
        insert(basis, row)
    return len(basis)


class World:
    def __init__(self, depth, rng, independent_lines=False):
        self.depth = depth
        self.active_axes = tuple(range(1 if independent_lines else depth))
        axes = []
        for axis in range(depth):
            if axis in self.active_axes:
                omitted = rng.sample(range(4), 3)
                axes.append([1 << i for i in range(4)] + [15 ^ (1 << i) for i in omitted])
            else:
                axes.append([1 << i for i in range(7)])
        cells = list(itertools.product(range(7), repeat=depth))
        def level(cell):
            return sum(cell[a] >= 4 for a in self.active_axes)
        info = [c for c in cells if level(c) == 0]
        hidden = [c for c in cells if level(c) > 0]
        self.k, self.r = len(info), len(hidden)
        self.umask = (1 << self.k) - 1
        self.shallow = [i for i, c in enumerate(hidden) if level(c) == 1]
        coord = {c: i for i, c in enumerate(info + hidden)}
        self.target = [sum(1 << coord[u] for u in itertools.product(*(positions(axes[a][v]) for a, v in enumerate(c)))) for c in hidden]
        self.lines = []
        self.checks = []
        self.level = {}
        self.true = set()
        for axis in self.active_axes:
            lines = []
            for fixed in itertools.product(range(7), repeat=depth - 1):
                cells_on_line = [coord[fixed[:axis] + (i,) + fixed[axis:]] for i in range(7)]
                lines.append(cells_on_line)
                check_level = 1 + level(fixed[:axis] + (0,) + fixed[axis:])
                for h in CANDIDATES:
                    mask = sum(1 << cells_on_line[j] for j in positions(h))
                    self.level[mask] = check_level
                    if combine(axes[axis], h) == 0:
                        self.true.add(mask)
            self.lines.append(lines)
            self.checks.append([[sum(1 << line[j] for j in positions(h)) for h in CANDIDATES] for line in lines])
        self.true_shallow = [h for h in self.true if self.level[h] == 1]
        assert rank(self.true) == self.r

    def teacher(self, rng):
        u = rng.getrandbits(self.k)
        return u | (evaluate(self.target, u) << self.k)


class Student:
    def __init__(self, world, rng):
        self.world = world
        self.reader = [rng.getrandbits(world.k) for _ in range(world.r)]
        self.order_seed = rng.getrandbits(64)
        self.candidates = [(1 << len(CANDIDATES)) - 1] * len(world.active_axes)
        self.seen = 0
        self.dictionary = []
        self.layers = []
        self.wrong = 0
        self.false = 0

    def watch(self, x):
        if self.seen < CALIBRATION:
            w = self.world
            for axis, lines in enumerate(w.lines):
                for line in lines:
                    view = sum(((x >> p) & 1) << j for j, p in enumerate(line))
                    self.candidates[axis] &= SAT[view]
            self.seen += 1
            if self.seen == CALIBRATION:
                self.dictionary = [checks[j] for candidates, axis in zip(self.candidates, w.checks) for checks in axis for j in positions(candidates)]
                random.Random(self.order_seed).shuffle(self.dictionary)
                self.false = sum(h not in w.true for h in self.dictionary)  # audit only
                assert w.true.issubset(self.dictionary)
        if self.dictionary:
            self.learn(x)

    def project(self, x):
        for h, v in self.layers:
            if parity(h & x):
                x ^= v
        return x

    def sample(self, rng):
        return self.project(rng.getrandbits(self.world.k + self.world.r))

    def install(self, h, v):
        w = self.world
        assert v & w.umask == 0 and parity(h & v)
        assert all(parity(g & v) == 0 for g, _ in self.layers)
        rule = (h & w.umask) ^ combine(self.reader, h >> w.k)
        for i in positions(v >> w.k):
            self.reader[i] ^= rule
        self.layers.append((h, v))
        self.wrong += h not in w.true  # audit only

    def learn(self, x):
        if any(parity(h & x) for h, _ in self.layers):
            return 0, 0, 0
        w = self.world
        u = x & w.umask
        prediction = u | (evaluate(self.reader, u) << w.k)
        v = x ^ prediction
        if v:
            for h in self.dictionary:
                if parity(h & v):
                    self.install(h, v)
                    return 1, 1, int(w.level[h] == 1)
        return 1, 0, 0

    def compile_one(self):
        w = self.world
        for i in range(w.r):
            v = self.project(1 << (w.k + i))
            if v:
                for h in self.dictionary:
                    if parity(h & v):
                        self.install(h, v)
                        return

    def metrics(self):
        w = self.world
        eq = [a == b for a, b in zip(self.reader, w.target)]
        s = w.r - len(self.layers)
        return [math.inf if self.wrong else FULL * s * math.log(2), 2.**-s if not self.wrong else 2.**(len(self.layers) - rank([*w.true, *(h for h, _ in self.layers)])),
                (1 + eq[-1]) / 2, .5 + .5 * sum(eq) / w.r, .5 + .5 * sum(eq[j] for j in w.shallow) / len(w.shallow),
                len(self.layers), self.false, bool(self.dictionary)]


class Passive:
    """Online linear equations, with uniform posterior prediction off their span."""
    def __init__(self, world):
        self.world = world
        self.basis = {}

    def observe(self, x):
        w = self.world
        u, y = x & w.umask, x >> w.k
        while u:
            i = u.bit_length() - 1
            if i not in self.basis:
                self.basis[i] = (u, y)
                return
            a, b = self.basis[i]
            u, y = u ^ a, y ^ b
        assert y == 0

    def metrics(self):
        w = self.world
        frac = 2.**(len(self.basis) - w.k)
        return [FULL * (1 - frac) * w.r * math.log(2), frac + (1-frac)*2.**-w.r, *([.5 + .5 * frac] * 3), np.nan, np.nan, np.nan]


def foreign_basis(w, rng):
    echelon, basis = {}, []
    while len(basis) < w.r:
        v = rng.getrandbits(w.r)
        if insert(echelon, v):
            basis.append(v)
    return basis


def experiment(seed, depth, watch, steps, independent_lines=False, reproject=False):
    rng = random.Random(seed)
    streams = [random.Random(rng.getrandbits(64)) for _ in range(10)]
    w = World(depth, streams[0], independent_lines)
    common, donor = Student(w, streams[1]), Student(w, streams[2])
    passive = Passive(w)
    pre = [common.metrics()]
    for _ in range(watch):
        x = w.teacher(streams[3])
        if streams[3].random() < FULL:
            common.watch(x)
            donor.watch(x)
            passive.observe(x)
        pre.append(common.metrics())
    states = [copy.deepcopy(common) for _ in ARMS[:-1]]
    additional = [copy.deepcopy(common)] if reproject else []
    basis = foreign_basis(w, streams[4])
    curves = [[state.metrics() for state in [*states, passive, *additional]]]
    events = np.zeros((steps, 5, len(EVENTS)), dtype=np.uint8)
    for t in range(steps):
        own, teacher, twin, matched, filtered, _ = states
        s = w.r - len(own.layers)
        xown, xtwin = own.sample(streams[5]), donor.sample(streams[6])
        xt = w.teacher(streams[7])
        xf = w.teacher(streams[4]) ^ (combine(basis, streams[4].getrandbits(s)) << w.k)
        xs = [xown, xt, xtwin, xf, filtered.sample(streams[8])]
        complete = streams[9].random() < FULL
        if complete:
            for i, (state, x) in enumerate(zip(states, xs)):
                old_shallow = all(not parity(h & x) for h, _ in state.layers if w.level[h] == 1)
                valid = x >> w.k == evaluate(w.target, x & w.umask)
                valid_shallow = all(not parity(h & x) for h in w.true_shallow)
                if i == 1 and not state.dictionary:
                    before = len(state.layers)
                    state.watch(x)
                    inc = len(state.layers) - before
                    old, use, low = 1, inc, int(inc and w.level[state.layers[-1][0]] == 1)
                elif i == 4 and any(parity(h & x) for h in state.dictionary):
                    old, use, low = 1, 0, 0
                else:
                    old, use, low = state.learn(x)
                events[t, i] = [1, old, use, valid, valid_shallow, old_shallow, low]
            donor.learn(xtwin)
            passive.observe(xt)
            for recipient in additional:
                recipient.learn(recipient.project(xf))
        compiler = states[-1]
        if compiler.dictionary and len(compiler.layers) < w.r:
            compiler.compile_one()
        curves.append([state.metrics() for state in [*states, passive, *additional]])
    return pre, curves, events


def run(args):
    records = [experiment(args.seed + i, args.depth, args.watch, args.steps, args.independent_lines, args.reproject) for i in tqdm(range(args.worlds), desc="worlds")]
    pre, curves, events = map(np.array, zip(*records))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arms = (*ARMS, "Reprojected foreign") if args.reproject else ARMS
    np.savez_compressed(args.out, pre=pre, curves=curves, events=events, arms=arms, metrics=METRICS, event_names=EVENTS,
                        seed=args.seed, depth=args.depth, watch=args.watch, steps=args.steps, independent_lines=args.independent_lines)


def audit():
    """Enumerate surface probabilities and learning opportunities independently of metrics()."""
    w = World(1, random.Random(7))
    watched = Student(w, random.Random(8))
    for u in (1, 2, 4, 8, 0, 3, 7, 15):
        watched.watch(u | (evaluate(w.target, u) << w.k))
    assert set(watched.dictionary) == w.true
    target = [u | (evaluate(w.target, u) << w.k) for u in range(1 << w.k)]
    seeds = list(range(1 << (w.k + w.r)))
    for source in ("Own", "Fresh teacher"):
        student = Student(w, random.Random(9))
        student.dictionary = watched.dictionary
        student.seen = CALIBRATION
        while True:
            outputs = [student.project(x) for x in seeds]
            prob = np.bincount(outputs, minlength=len(seeds)) / len(seeds)
            truth_kl = FULL * np.mean(-np.log(prob[target] * len(target)))
            metrics = student.metrics()
            assert np.isclose(truth_kl, metrics[0]) and np.isclose(prob[target].sum(), metrics[1])
            for triple in itertools.combinations(range(w.k + w.r), 3):
                bins = [sum(((x >> pos) & 1) << j for j, pos in enumerate(triple)) for x in outputs]
                assert np.all(np.bincount(bins, minlength=8) == len(seeds)//8)
            root_correct = np.mean([parity((student.reader[-1] ^ w.target[-1]) & u) == 0 for u in range(1 << w.k)])
            assert root_correct == metrics[2]
            s = w.r - len(student.layers)
            a = rank([x ^ y for x, y in zip(student.reader, w.target)])
            for stream, predicted in ((outputs, 1-2.**-s), (target, 1-2.**-a)):
                successes = [copy.deepcopy(student).learn(x)[1] for x in stream]
                assert np.mean(successes) == predicted
            stream = outputs if source == "Own" else target
            if not any(student.learn(x)[1] for x in stream):
                break
    print("Exact enumeration agrees on KL, validity, root accuracy, all three-bit marginals and useful-trial rates.")


def plot(main, relu, out):
    data = np.load(main)
    net = np.load(relu)
    pre, curve, watch = data["pre"], data["curves"], int(data["watch"])
    t = np.arange(curve.shape[1]) + watch
    fig, axes = plt.subplot_mosaic([["kl", "kl"], ["root", "relu"]], layout="constrained")

    def band(ax, steps, runs, label, color):
        mean = runs.mean(0)
        ci = 1.96*runs.std(0, ddof=1)/np.sqrt(len(runs))
        line, = ax.plot(steps, mean, label=label, color=color)
        ax.fill_between(steps, mean-ci, mean+ci, color=line.get_color(), alpha=.16)

    for panel, metric, title in (("kl", 0, "Population KL (nats)"), ("root", 2, "Erased-root accuracy")):
        ax = axes[panel]
        ax.plot(np.arange(watch+1), pre[:, :, metric].mean(0), color="0.35")
        for i, color in ((0, "C0"), (1, "C1"), (3, "C2"), (6, "C3")):
            band(ax, t, curve[:, :, i, metric], ARMS[i], color)
        ax.axvline(watch, color="0.5", linestyle=":")
        ax.set(xlabel="Total strings", ylabel=title)
    axes["kl"].legend(loc="upper right", ncols=2)
    for i, name in enumerate(("Local inverse", "Prefix writer")):
        band(axes["relu"], net["times"][1:]*64, net["scores"][:, 1:, i, 0], name, ("C4", "C5")[i])
    axes["relu"].set(xlabel="Observed pairs", ylabel="ReLU test MSE", xscale="log", yscale="log")
    axes["relu"].legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"), transparent=True)
    fig.savefig(out.with_suffix(".png"), dpi=180, facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cli = parser.add_subparsers(dest="command", required=True)
    trial = cli.add_parser("run")
    for name, default in (("seed", 281126), ("worlds", 256), ("depth", 2), ("watch", 16), ("steps", 100)):
        trial.add_argument("--" + name, type=int, default=default)
    trial.add_argument("--out", type=Path, required=True)
    trial.add_argument("--independent-lines", action="store_true")
    trial.add_argument("--reproject", action="store_true")
    cli.add_parser("audit")
    graph = cli.add_parser("plot")
    graph.add_argument("--main", type=Path, required=True)
    graph.add_argument("--relu", type=Path, required=True)
    graph.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "audit":
        audit()
    else:
        plot(args.main, args.relu, args.out)
