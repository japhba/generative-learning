"""Opposite self-supervised directions of a binary difference chain."""
import argparse
import math
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def spectrum(n, degree):
    def kernel(s):
        c = 1 - s / n
        return (math.sqrt(1-c*c) + 2*c*(math.pi-math.acos(c))) / (2*math.pi)

    def coefficient(s):
        return sum((-1)**i * math.comb(degree, i) * math.comb(n-degree, s-i)
                   for i in range(max(0, s-n+degree), min(degree, s)+1))

    return math.ldexp(math.fsum(coefficient(s)*kernel(s) for s in range(n+1)), -n)


def pairs(rng, size, n, perm):
    y = 2*rng.integers(2, size=(size, n)).astype(float)-1
    u = y.copy()
    u[:, 1:] *= y[:, :-1]
    return [y[:, perm], u[:, perm]], [u[:, -1], y[:, -1]]


class ReLU:
    def __init__(self, rng, n, width):
        self.params = [rng.normal(size=(n, width))/np.sqrt(2*n), rng.normal(size=width)/np.sqrt(2),
                       rng.normal(size=width)/np.sqrt(width), np.zeros(1)]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def forward(self, x):
        w, b, a, c = self.params
        h = np.maximum(x @ w + b, 0)
        return h, h @ a + c

    def train(self, x, y, t):
        h, pred = self.forward(x)
        dy = 2*(pred-y)/len(y)
        dh = (dy[:, None]*self.params[2])*(h > 0)
        grads = [x.T @ dh, dh.sum(0), h.T @ dy, dy.sum()]
        for m, v, p, g in zip(self.m, self.v, self.params, grads):
            m *= .9
            m += .1*g
            v *= .999
            v += .001*g*g
            p -= .003*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)


def trial(seed, times):
    rng = np.random.default_rng(seed)
    n = 16
    perm = rng.permutation(n)
    init = rng.integers(2**32)
    nets = [ReLU(np.random.default_rng(init), n, 64) for _ in range(2)]
    inputs, labels = pairs(rng, 2048, n, perm)
    scores = []
    for t in range(max(times)+1):
        if t:
            xs, ys = pairs(rng, 64, n, perm)
            for net, x, y in zip(nets, xs, ys):
                net.train(x, y, t)
        if t in times:
            row = []
            for net, x, y in zip(nets, inputs, labels):
                _, pred = net.forward(x)
                row.append([np.mean((pred-y)**2), np.mean((pred > 0) == (y > 0))])
            scores.append(row)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=281127)
    parser.add_argument("--worlds", type=int, default=32)
    args = parser.parse_args()
    times = [0, 4, 8, 16, 32, 64, 128, 256, 512]
    scores = np.array([trial(args.seed+i, times) for i in tqdm(range(args.worlds), desc="paired networks")])
    sizes = [8, 16, 32]
    rates = np.array([[spectrum(n, d) for d in (2, n)] for n in sizes])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, seed=args.seed, times=times, scores=scores, sizes=sizes, rates=rates)
