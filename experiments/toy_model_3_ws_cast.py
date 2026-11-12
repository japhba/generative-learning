"""Toy model 3 read as wake-sleep (check that the relabelling reproduces its main effect; world as in toy_model_3_chain.py).
World: S states, M allowed successors each; population = valid chains of n = 9 states.
  wake-sleep generator   = the pair table (which links are allowed), iterated forward     (toy model 3 calls it the recogniser)
  wake-sleep recogniser  = span tables q_2k(c | a, b): midpoint of a span given its ends   (toy model 3 calls them the generator)
  wake   count the links of the teacher chains (every state is visible, so nothing has to be inferred)
  sleep  roll the pair table forward into own chains and count (a, b) -> c for every span: each own chain carries its midpoints
Arms: watch = span tables counted from the teacher chains only; sleep = span tables counted from DREAMS own chains only.
Score: ends (a, b) of a fresh population chain are given; the span tables write the 7 states in between in one top-down pass; share of
chains with all 8 links allowed (a missing table entry counts as a failure).
Usage: python toy_model_3_ws_cast.py -> reports/toy3-ws-cast.png, /var/tmp/jbauer/generative_practice_chain/ws_cast.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

S, M, N_STATES, WORLDS, DREAMS, NTEST = 1024, 2, 9, 4, 2 ** 21, 4000
NS = np.unique(np.round(np.geomspace(64, 2 ** 18, 25)).astype(int))
OUT = Path("/var/tmp/jbauer/generative_practice_chain"); FIG = Path(__file__).resolve().parents[1] / "reports/toy3-ws-cast.png"


def roll(rng, succ, know, num):                                                                             # forward chains through the links marked in know [S, M]; chains that reach a state without a known link are dropped
    x = np.empty((num, N_STATES), int); x[:, 0] = rng.choice(np.flatnonzero(know.any(1)), size=num); alive = np.ones(num, bool)
    for t in range(1, N_STATES): kn = know[x[:, t - 1]]; alive &= kn.any(1); x[:, t] = succ[x[:, t - 1], (kn * rng.random(kn.shape)).argmax(1)]
    return x[alive]


class Table:                                                                                                # counts of (a, b) -> c for spans of 2k links, at every position of the chains
    def __init__(self, x, k):
        i = np.arange(N_STATES - 2 * k); u, cnt = np.unique(((x[:, i] * S + x[:, i + 2 * k]) * S + x[:, i + k]).ravel(), return_counts=True)
        self.ab, self.start = np.unique(u // S, return_index=True); self.c, self.cum, self.end = u % S, np.cumsum(cnt), np.r_[self.start[1:], len(u)]

    def draw(self, rng, a, b):                                                                              # c ~ counts, -1 where the table has no entry (or an end is already missing)
        if not len(self.ab): return np.full(len(a), -1)
        key = a * S + b; g = np.minimum(np.searchsorted(self.ab, key), len(self.ab) - 1); hit = (self.ab[g] == key) & (a >= 0) & (b >= 0)
        base = np.where(self.start[g] > 0, self.cum[self.start[g] - 1], 0); total = self.cum[self.end[g] - 1] - base
        return np.where(hit, self.c[np.minimum(np.searchsorted(self.cum, base + rng.random(len(a)) * total, side="right"), len(self.c) - 1)], -1)


def written(rng, tables, allowed, test):                                                                    # one top-down pass between the ends of each test chain
    x = np.full(test.shape, -1); x[:, [0, -1]] = test[:, [0, -1]]
    for k in (4, 2, 1):
        for i in range(0, N_STATES - 1, 2 * k): x[:, i + k] = tables[k].draw(rng, x[:, i], x[:, i + 2 * k])
    return ((x >= 0).all(1) & allowed[np.maximum(x[:, :-1], 0), np.maximum(x[:, 1:], 0)].all(1)).mean()


def world(seed):
    rng = np.random.default_rng(seed); succ = np.stack([rng.permutation(S) for _ in range(M)], 1); full = np.ones((S, M), bool)
    allowed = np.zeros((S, S), bool); allowed[np.arange(S)[:, None], succ] = True; teacher, test = roll(rng, succ, full, NS[-1]), roll(rng, succ, full, NTEST); rows = []
    for n in tqdm(NS, desc=f"world {seed}", leave=False):
        x = teacher[:n]; know = np.zeros((S, M), bool)
        np.logical_or.at(know, x[:, :-1].ravel(), succ[x[:, :-1].ravel()] == x[:, 1:].ravel()[:, None])    # wake: links seen in the teacher chains
        own = roll(rng, succ, know, DREAMS)                                                                 # sleep: own chains from the pair table
        rows.append([know.mean(), written(rng, {k: Table(x, k) for k in (1, 2, 4)}, allowed, test), written(rng, {k: Table(own, k) for k in (1, 2, 4)}, allowed, test), len(own) / DREAMS])
    return np.array(rows)


def half(y): hit = y >= 0.5; return NS[hit.argmax()] if hit.any() else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(WORLDS) as pool: res = np.array(pool.map(world, range(WORLDS)))                               # [world, N, (links known, watch, sleep, share of dreams completed)]
    np.savez(OUT / "ws_cast.npz", ns=NS, res=res); med = np.median(res, 0)
    print("teacher chains | links known | 8-link span written: watch, sleep | own chains completed")
    for n, row in zip(NS, med): print(f"  {n:7d} | {row[0]:.3f} | {row[1]:.3f}  {row[2]:.3f} | {row[3]:.3f}")
    print(f"teacher chains until half of the 8-link spans are written (median of {WORLDS} worlds): watch {half(med[:, 1])}, sleep {half(med[:, 2])}; half of the links known at {half(med[:, 0])}")
    fig, ax = plt.subplots(constrained_layout=True)
    for j, lab in [(0, "links in the pair table"), (1, "8-link span written: span tables from teacher chains"), (2, "8-link span written: span tables from own chains (sleep)")]:
        ax.plot(NS, med[:, j], label=lab); ax.fill_between(NS, res[:, :, j].min(0), res[:, :, j].max(0), alpha=0.15, edgecolor="none")
    ax.set(xscale="log", xlabel="teacher chains", ylabel="share"); fig.legend(loc="outside lower center"); fig.savefig(FIG, dpi=150, transparent=True); print(FIG)


if __name__ == "__main__": main()
