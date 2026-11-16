"""Model 5, three levels: strings of four shallow codes (k1 k2 | k3 k4). Level 1: the block of model5_lateral_verdict.py on each half (classes of codes).
Level 2: the 64 valid class pairs are the productions of 16 symbols (4 each); the top has 64 valid pairs of symbols. 262,144 valid strings.
Protocols at an equal number of teacher strings n:
  flat: a table of whole strings (best uniform smoothing).
  first: all n strings watched, then practice. Nothing was read into classes while the strings were shown, so the two halves stay independent.
  between: n/2 strings, practice, then n/2 strings read into the class units and their pairs counted (a table over pairs of named class pairs).
           Practice leaves several units with the same pool; they are read as one (see names).
Writes /var/tmp/jbauer/model5_lateral_verdict/depth.npz.   Usage: python model5_depth.py [--worlds 8] [--procs 28]"""

import argparse
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm

from model5_lateral_verdict import BATCH, OUT, Grower, World, watch

ALPHAS = 10.0 ** np.arange(-12, 0.1, 0.5)


class World3:
    def __init__(self, seed):
        rng = np.random.default_rng(10_000 + seed); self.w = w = World(seed=seed); v, m = w.v, w.m
        self.prods = rng.permutation(len(w.rules)).reshape(v, m)                                                                   # symbol -> its m productions (indices into w.rules)
        while True:
            B = np.zeros((v, v), bool)
            for _ in range(m): B[np.arange(v), rng.permutation(v)] = True
            if B.sum() == v * m: break
        self.top = np.argwhere(B); self.logn = np.log(len(self.top) * m ** 2 * m ** 4)

    def strings(self, rng, n):
        w = self.w; t = self.top[rng.integers(len(self.top), size=n)]; q = self.prods[t, rng.integers(w.m, size=(n, 2))]; c = w.rules[q].reshape(n, 4)
        return w.members[c, rng.integers(w.m, size=(n, 4))]


def names(G, side):                                                                                                                # the unit that reads each code. Several units end up with the same pool; a code is read as the largest pool
    GL, GR, T = G.tables(); P = GL if side == 0 else GR; held = P > 0.25 * P.max(1, keepdims=True); key = {}; out = np.zeros(G.N, int)   # that holds it, and equal pools count as one unit (the first of them)
    for k in range(G.N):
        cand = np.flatnonzero(P[:, k] >= 0.5 * P[:, k].max()); a = cand[held[cand].sum(1).argmax()]; out[k] = key.setdefault(held[a].tobytes(), a)
    return out


def one(job):
    n, seed = job; W = World3(seed); w = W.w; rng = np.random.default_rng(20_000 + seed); pop = W.strings(rng, 20000); out = {}
    x = W.strings(rng, n); cnt = {}
    for row in map(tuple, x): cnt[row] = cnt.get(row, 0) + 1
    c = np.array([cnt.get(row, 0) for row in map(tuple, pop)]); out['flat'] = min(float(-W.logn - np.log((c + a) / (n + a * w.N ** 4)).mean()) for a in ALPHAS)
    for proto, n1 in (('first', n), ('between', n // 2)):
        gs = [Grower(w), Grower(w)]
        for g, cols in zip(gs, ((0, 1), (2, 3))):
            for i in range(0, n1, 256): g.watch(x[i:min(i + 256, n1), cols[0]], x[i:min(i + 256, n1), cols[1]], rng)
            for _ in range(32768): g.practice(BATCH, rng)
        P = [g.pairprob() for g in gs]; halves = np.log(P[0][pop[:, 0], pop[:, 1]] + 1e-300) + np.log(P[1][pop[:, 2], pop[:, 3]] + 1e-300)
        if proto == 'first': out['first'] = float(-W.logn - halves.mean()); continue
        nm = [names(gs[0], 0), names(gs[0], 1), names(gs[1], 0), names(gs[1], 1)]; N = w.N
        def read(s): c = np.stack([nm[i][s[:, i]] for i in range(4)], 1); return c, c[:, 0] * N + c[:, 1], c[:, 2] * N + c[:, 3]
        _, q, qp = read(x[n1:]); C = {}
        for a in zip(q, qp): C[a] = C.get(a, 0) + 1
        c, q, qp = read(pop); cc = np.array([C.get(a, 0) for a in zip(q, qp)])
        pools = -sum(np.log(np.bincount(nm[i], minlength=N)[c[:, i]]) for i in range(4))                                          # a name writes the codes that are read as it, each equally often
        out['between'] = min(float(-W.logn - (np.log((cc + a) / (n - n1 + a * N ** 4)) + pools).mean()) for a in ALPHAS)
        out['named'] = float(np.mean([len(set(nm[i][w.members[k]])) == 1 for i in range(4) for k in range(w.v)]) * np.mean([len(set(nm[i])) == w.v for i in range(4)]))   # classes read by one unit each, all distinct
        out['blocks'] = float(np.mean([g.score()['kl'] for g in gs]))
    return n, seed, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=8); ap.add_argument('--procs', type=int, default=28); args = ap.parse_args()
    ns = 2 ** np.arange(12, 18); jobs = [(int(n), s) for n in ns for s in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for n, s, out in tqdm(p.imap_unordered(one, jobs), total=len(jobs)): res[n, s] = out
    np.savez(OUT / 'depth.npz', n=ns, **{k: np.array([[res[int(n), s][k] for s in range(args.worlds)] for n in ns]) for k in res[int(ns[0]), 0]})
    for n in ns: print(n, {k: round(float(np.median([res[int(n), s][k] for s in range(args.worlds)])), 3) for k in res[int(n), 0]})
