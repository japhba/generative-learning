"""Toy model 3: a learner made of tables on a chain world (plan agreed with the user, 10-11 Nov 2026).

World: S states, each with m allowed successors (m permutations of the states, no fixed point, no shared arc). The
population is the valid chains of n = 2^L + 1 states: uniform start, uniform allowed step. Every state has to fit both
of its neighbours.

The teacher is the ground truth: it shows chains of the population. The learner is made of tables and counts only:
  recogniser  pairs[a, b]: how often b was seen right after a. A link passes once its pair was seen theta times.
              Checking a chain is one lookup per link, at any length.
  generator   bridge[k]: counts of (a, b) -> c, the midpoint c of a span of 2k links from a to b, k = 1, 2, 4, ..., 2^(L-1).
              Writing is one top-down pass: a node holding (a, b) draws c from its table (a uniform guess if it has no
              entry) and hands (a, c) and (c, b) to its two children.
Its two ways of learning use the same counting rule:
  watch       a teacher chain adds its adjacent pairs to the recogniser and each of its spans (a, c, b) to the generator.
  practise    pick a level k, join two units of k links that the learner holds at a shared state (a -> c -> b), write the
              2k + 1 states in one pass, check every link, and count (a, c, b) at level k if all links pass. No teacher data.
              A unit of 1 link is an accepted pair; a unit of 2k links is a span the level-k table has an entry for.
  rehearse    draw an entry the learner holds, write its span in one pass, judge it: the count goes up by one if every link passes
              and down by one if not. With joins this is the one signed rule of the write-up; `run_tandem` starts the generator
              from random entries and compares it with teacher chains.
  aimed       practice that looks a drawn join up first and writes it only if the table lacks it (`run_aim`, `run_curve`).
The write-up calls reading recognition and writing generation, and describes learning in stages (`run_stages`): a teacher
set (pairs counted = recognition; every span counted = wake), then practice; scores on the teacher set and on the population.

Own-versus-foreign extension, nothing given (`run_reach`): a world of regions. The learner is taught in one region, reads a
link only if it has learned it, and can take in a teacher chain only if the chain touches a state it knows.
Earlier version (`run_map`): the step types are given (the same few steps act everywhere), the world has many states, and
the learner can only file a chain that touches its map, the set of states it has produced or reached.

Scores, on held-out population chains (level l has spans of 2^l links):
  read0       share of population links the recogniser accepts
  read<l>     share of population spans the learner holds as one unit (one lookup)
  write<l>    share of population spans (a, b) for which one pass from (a, b) gives a chain with only allowed links
  allow<l>    share of population spans whose own links are all accepted (what the pair table already permits)
  reach<l>    share of population spans joined by some route of accepted links (the ceiling of practice; where computed)
  own<l>      share of the learner's own samples at level l (ends drawn from its table) with only allowed links

Usage: python toy_model_3_chain.py [--only watch practice ...] -> /var/tmp/jbauer/generative_practice_chain/<name>.json
       python toy_model_3_figures.py -> reports/toy3-*.png"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

OUT = Path("/var/tmp/jbauer/generative_practice_chain")
S, M, L, WORLDS = 1024, 2, 3, 16  # the main world: 1024 states, 2 allowed successors, chains of 9 states, 16 random worlds
N_GRID = [2 ** e for e in range(6, 21)]  # numbers of teacher chains for the practice experiments
M_MAX = 2 ** 23  # practice episodes (joins)


def world(S, m, rng):
    """succ[z, g]: the g-th allowed successor of state z."""
    while True:
        succ = np.stack([rng.permutation(S) for _ in range(m)], 1)
        if (succ != np.arange(S)[:, None]).all() and all((succ[:, g] != succ[:, h]).all() for g in range(m) for h in range(g)): return succ


def chains(succ, N, n, rng, noise=0.0, start=None, p=None):
    """N population chains of n states; with noise, each state is replaced by a random one with that probability (a slip).
    p: the probabilities of the m step types (default: uniform)."""
    S, m = succ.shape; x = np.empty((N, n), np.int64); x[:, 0] = rng.integers(S, size=N) if start is None else start
    for t in range(1, n): x[:, t] = succ[x[:, t - 1], rng.integers(m, size=N) if p is None else rng.choice(m, size=N, p=p)]
    if noise: s = rng.random(x.shape) < noise; x[s] = rng.integers(S, size=s.sum())
    return x


def grid(lo, hi, points): return np.unique(np.round(np.geomspace(lo, hi, points)).astype(np.int64))


class Learner:
    """code[0], cnt[0]: the accepted pairs a S + b and their counts. code[k], cnt[k], k >= 1: sorted (a S + b) S + c and counts."""

    def __init__(self, S, L, theta=1, succ=None, dense=True):
        self.S, self.L, self.theta, self.succ = S, L, theta, succ; self.ks = [2 ** l for l in range(L)]  # succ given: the step types are known
        self.pairs = np.zeros(S * S, np.int64) if succ is None and dense else None  # dense=False (large worlds): the pairs live only in the sorted list code[0], theta = 1
        self.code = {k: np.empty(0, np.int64) for k in [0] + self.ks}; self.cnt = {k: np.empty(0, np.int64) for k in [0] + self.ks}
        self.cum = {k: np.zeros(1, np.int64) for k in [0] + self.ks}; self.acc = np.zeros((S, S), bool) if succ is None and dense else None

    def passes(self, a, b):
        if self.succ is not None: return (self.succ[a] == b[..., None]).any(-1)
        if self.acc is not None: return self.acc[a, b]
        key = a * self.S + b; return self.code[0][np.minimum(np.searchsorted(self.code[0], key), len(self.code[0]) - 1)] == key

    def merge(self, k, keys, w=None):
        """Add the keys to table k, each with count 1, or with the weights w (the table's counts then become real numbers)."""
        if w is None: new, n = np.unique(keys, return_counts=True)
        else: new, inv = np.unique(keys, return_inverse=True); n = np.bincount(inv, w, len(new))
        code, inv = np.unique(np.r_[self.code[k], new], return_inverse=True); cnt = np.bincount(inv, np.r_[self.cnt[k], n], len(code))
        self.code[k], self.cnt[k] = code, cnt.astype(np.int64) if w is None and self.cnt[k].dtype == np.int64 else cnt; self.cum[k] = np.r_[0, np.cumsum(self.cnt[k])]

    def add(self, k, a, c, b): self.merge(k, (a * self.S + b) * self.S + c)

    def file(self, x, gate):
        """The generator's counting rule: (ends, midpoint) of every span of every chain; with the gate, only spans whose links all pass."""
        if gate: bad = np.c_[np.zeros(len(x), np.int64), np.cumsum(~self.passes(x[:, :-1], x[:, 1:]), 1)]
        for k in self.ks:
            i = np.arange(x.shape[1] - 2 * k)
            if len(i) == 0: break
            keep = bad[:, i + 2 * k] == bad[:, i] if gate else np.ones((len(x), len(i)), bool)
            self.add(k, x[:, i][keep], x[:, i + k][keep], x[:, i + 2 * k][keep])

    def watch(self, x, levels=True, gate=False):
        """Teacher chains: count the adjacent pairs; with levels, also every span (through the gate if asked)."""
        if self.pairs is None: self.merge(0, x[:, :-1].ravel() * self.S + x[:, 1:].ravel())
        else:
            self.pairs += np.bincount(x[:, :-1].ravel() * self.S + x[:, 1:].ravel(), minlength=self.S ** 2)
            self.code[0] = np.flatnonzero(self.pairs >= self.theta); self.cnt[0] = self.pairs[self.code[0]]; self.cum[0] = np.r_[0, np.cumsum(self.cnt[0])]
            self.acc = (self.pairs >= self.theta).reshape(self.S, self.S)
        if levels: self.file(x, gate)

    def pick(self, k, lo, hi, rng):
        """A row of table k in [lo, hi), with probability proportional to its count; held is False where the range is empty."""
        cum = self.cum[k]; r = np.searchsorted(cum, cum[lo] + rng.random(len(lo)) * (cum[hi] - cum[lo]), side="right") - 1
        return np.minimum(np.clip(r, lo, np.maximum(hi - 1, lo)), len(self.code[k]) - 1), hi > lo

    def ends(self, k, r): c = self.code[k][r]; return (c // self.S, c % self.S) if k == 0 else (c // self.S ** 2, c // self.S % self.S)

    def mid(self, a, b, k, rng):
        """A node's draw: the midpoint of the span (a, b) of 2k links; a uniform guess if the node has no entry."""
        guess = rng.integers(self.S, size=len(a)); code = self.code[k]
        if len(code) == 0: return guess
        key = (a * self.S + b) * self.S; r, held = self.pick(k, np.searchsorted(code, key), np.searchsorted(code, key + self.S), rng)
        return np.where(held, code[r] % self.S, guess)

    def write(self, a, b, k, rng):
        """One top-down pass from the ends (a, b) of a span of 2k links to its 2k + 1 states."""
        c = self.mid(a, b, k, rng)
        return np.stack([a, c, b], 1) if k == 1 else np.c_[self.write(a, c, k // 2, rng), self.write(c, b, k // 2, rng)[:, 1:]]

    def unit(self, k, a, rng):
        """The end of a held unit of k links that starts at a (count-weighted), and whether the learner holds one."""
        j = 0 if k == 1 else k // 2; code = self.code[j]; w = self.S if j == 0 else self.S ** 2
        if len(code) == 0: return a, np.zeros(len(a), bool)
        r, held = self.pick(j, np.searchsorted(code, a * w), np.searchsorted(code, (a + 1) * w), rng)
        return self.ends(j, r)[1], held

    def practise(self, k, B, rng, slip=0.0, gate=True):
        """B joins at level k: two held units of k links end to end, written in one pass, checked, and counted if every link passes."""
        j = 0 if k == 1 else k // 2
        if len(self.code[j]) == 0 or B == 0: return
        a, c = self.ends(j, np.searchsorted(self.cum[j], rng.random(B) * self.cum[j][-1], side="right") - 1); b, held = self.unit(k, c, rng); a, c, b = a[held], c[held], b[held]
        x = np.stack([a, c, b], 1) if k == 1 else np.c_[self.write(a, c, k // 2, rng), self.write(c, b, k // 2, rng)[:, 1:]]
        if slip: s = rng.random(x[:, 1:-1].shape) < slip; x[:, 1:-1][s] = rng.integers(self.S, size=s.sum())  # the hand slips on an interior state
        good = self.passes(x[:, :-1], x[:, 1:]).all(1) if gate else np.ones(len(x), bool)
        self.add(k, x[good, 0], x[good, k], x[good, -1])

    def total(self, j, a, b):
        """The count of the unit (a, b) in table j: the pair itself (j = 0), or the span summed over its midpoints."""
        code, cum, key = self.code[j], self.cum[j], a * self.S + b
        return cum[np.searchsorted(code, key + 1)] - cum[np.searchsorted(code, key)] if j == 0 else cum[np.searchsorted(code, (key + 1) * self.S)] - cum[np.searchsorted(code, key * self.S)]

    def aimed(self, tokens, rng, draws=2 ** 14):
        """Practice that skips what the learner already holds. A join is drawn as in practise and looked up; it is written and
        checked only if the table lacks it, and only written joins cost tokens (2k + 1 states). A new join is filed with the count
        that blind practice would give it in the long run, count(a, c) count(c, b) / count(c, .). Levels are swept bottom-up; a level
        gets fewer draws when its last draws found little, so the tokens go to whichever levels still have gaps, and practice
        stops by itself when it finds nothing new. Returns the tokens written per level and the number of joins looked up."""
        spent = {k: 0 for k in self.ks}; d = {k: draws for k in self.ks}; looked = idle = 0
        while sum(spent.values()) < tokens and idle < 4:
            found = 0
            for k in self.ks:
                j = 0 if k == 1 else k // 2; w = self.S if j == 0 else self.S ** 2
                if len(self.code[j]) == 0: continue
                a, c = self.ends(j, np.searchsorted(self.cum[j], rng.random(d[k]) * self.cum[j][-1], side="right") - 1); b, held = self.unit(k, c, rng); looked += int(held.sum())
                key = np.unique(((a * self.S + b) * self.S + c)[held]); code = self.code[k]
                if len(code): key = key[code[np.minimum(np.searchsorted(code, key), len(code) - 1)] != key]
                d[k] = int(np.clip(8 * len(key) + 256, 256, draws))
                key = key[rng.permutation(len(key))[: max(0, int((tokens - sum(spent.values())) // (2 * k + 1)))]]
                if len(key) == 0: continue
                a, b, c = key // self.S ** 2, key // self.S % self.S, key % self.S; spent[k] += (2 * k + 1) * len(key); found += len(key)
                x = np.stack([a, c, b], 1) if k == 1 else np.c_[self.write(a, c, k // 2, rng), self.write(c, b, k // 2, rng)[:, 1:]]
                good = self.passes(x[:, :-1], x[:, 1:]).all(1); a, c, b = a[good], c[good], b[good]
                out = self.cum[j][np.searchsorted(self.code[j], (c + 1) * w)] - self.cum[j][np.searchsorted(self.code[j], c * w)]
                self.merge(k, (a * self.S + b) * self.S + c, self.total(j, a, c) * self.total(j, c, b) / out)
            idle = 0 if found else idle + 1
        return spent, looked

    def rehearse(self, k, B, rng, signed=True):
        """B rehearsals at level k: an entry the learner holds is drawn in proportion to its count, the span it stands for is written
        in one pass and judged. The entry's count goes up by one if every link passes and, with signed, down by one if not."""
        if len(self.code[k]) == 0 or B == 0: return
        r = np.searchsorted(self.cum[k], rng.random(B) * self.cum[k][-1], side="right") - 1; c = self.code[k][r]; a, b, c = c // self.S ** 2, c // self.S % self.S, c % self.S
        x = np.stack([a, c, b], 1) if k == 1 else np.c_[self.write(a, c, k // 2, rng), self.write(c, b, k // 2, rng)[:, 1:]]
        good = self.passes(x[:, :-1], x[:, 1:]).all(1); np.add.at(self.cnt[k], r[good], 1)
        if signed: np.subtract.at(self.cnt[k], r[~good], 1); keep = self.cnt[k] > 0; self.code[k], self.cnt[k] = self.code[k][keep], self.cnt[k][keep]
        self.cum[k] = np.r_[0, np.cumsum(self.cnt[k])]

    def rounds(self, B, rng, **kw):
        """B practice episodes, each at a level drawn uniformly; lower levels first, so that higher ones can use their new units."""
        for k, Bk in zip(self.ks, rng.multinomial(B, np.ones(self.L) / self.L)): self.practise(k, Bk, rng, **kw)

    def score(self, test, T, rng, own=2000):
        S = self.S; valid = lambda x: T[x[:, :-1], x[:, 1:]].all(1); out = {"read0": float(self.passes(test[:, :-1], test[:, 1:]).mean()), "wrong0": int((self.acc & ~T).sum())}
        for l, k in enumerate(self.ks, 1):
            a, b = test[:, 0], test[:, 2 * k]; key = (a * S + b) * S; code = self.code[k]
            out[f"read{l}"] = float((np.searchsorted(code, key + S) > np.searchsorted(code, key)).mean()); out[f"write{l}"] = float(valid(self.write(a, b, k, rng)).mean())
            out[f"allow{l}"] = float(self.passes(test[:, :2 * k], test[:, 1:2 * k + 1]).all(1).mean()); out[f"size{l}"] = len(code)
            if len(code): ao, bo = self.ends(k, np.searchsorted(self.cum[k], rng.random(own) * self.cum[k][-1], side="right") - 1); out[f"own{l}"] = float(valid(self.write(ao, bo, k, rng)).mean())
        return out

    def reach(self, test):
        """Share of population spans joined by some route of accepted links: what practice can reach with this pair table."""
        R = self.acc.astype(np.float32); out = {}
        for l, k in enumerate(self.ks, 1): R = (R @ R > 0).astype(np.float32); out[f"reach{l}"] = float(R[test[:, 0], test[:, 2 * k]].mean())
        return out


def setup(seed, S=S, m=M, L=L, test=8000, dense=True):
    rng = np.random.default_rng(seed); succ = world(S, m, rng); T = np.zeros((S, S), bool) if dense else None
    if dense: T[np.arange(S)[:, None], succ] = True
    return rng, succ, T, chains(succ, test, 2 ** L + 1, rng)


# ---------------------------------------------------------------------------------------------- core experiments


def train_score(o, sample, T, rng):
    """The same scores on chains of the teacher's training set: share of their links accepted, share of their spans generated."""
    valid = lambda x: T[x[:, :-1], x[:, 1:]].all(1); out = {"train_read0": float(o.passes(sample[:, :-1], sample[:, 1:]).mean()) if len(sample) else 0.0}
    for l, k in enumerate(o.ks, 1): out[f"train_write{l}"] = float(valid(o.write(sample[:, 0], sample[:, 2 * k], k, rng)).mean()) if len(sample) else 0.0
    return out


def run_watch(seed, N_max=2 ** 22, points=72, S=S, m=M, L=L):
    """Watching only: the same teacher chains fill the pair table and, by the same counting rule, every level of the generator.
    Scores on held-out population chains, and (train_*) on the first 2,000 chains of the teacher set, with a generator of their own."""
    rng, succ, T, test = setup(seed, S, m, L); o = Learner(S, L); out, done = [], 0; rng2 = np.random.default_rng(10 ** 6 + seed); sample = np.empty((0, 2 ** L + 1), np.int64)
    for N in grid(8, N_max, points):
        x = chains(succ, N - done, 2 ** L + 1, rng); o.watch(x); done = N; sample = np.r_[sample, x][:2000]; out.append({"N": int(N), **o.score(test, T, rng), **train_score(o, sample, T, rng2)})
    return out


def run_practice(seed, N, levels=True, M_max=M_MAX, points=64, theta=1, noise=0.0, gate_teacher=False, slip=0.0, gate=True, false_pairs=0, S=S, m=M, L=L):
    """Watch N teacher chains, then practise. levels=False: the teacher chains fill the pair table only."""
    rng, succ, T, test = setup(seed, S, m, L); o = Learner(S, L, theta); o.watch(chains(succ, N, 2 ** L + 1, rng, noise), levels, gate_teacher)
    if false_pairs:  # a checker with false accepts: wrong pairs, each as strong as an average right one
        a, b = rng.integers(S, size=false_pairs), rng.integers(S, size=false_pairs); o.pairs[a * S + b] += int(np.ceil(o.pairs[o.pairs > 0].mean())); o.watch(np.empty((0, 2), np.int64), False)
    out, done = [{"M": 0, **o.score(test, T, rng), **o.reach(test)}], 0
    for Mc in grid(16, M_max, points): o.rounds(Mc - done, rng, slip=slip, gate=gate); done = Mc; out.append({"M": int(Mc), **o.score(test, T, rng)})
    return out


def run_schedule(seed, order, N=2 ** 12, M_tot=2 ** 21, blocks=64):
    """The same N teacher chains and M practice episodes in three orders: watch first, practise first, or alternating blocks."""
    rng, succ, T, test = setup(seed); o = Learner(S, L); out, seen, done = [{"examples": 0, "N": 0, "M": 0, **o.score(test, T, rng)}], 0, 0
    w, p = [("w", N // blocks)] * blocks, [("p", M_tot // blocks)] * blocks
    for kind, B in {"watch_first": w + p, "practise_first": p + w, "alternate": [s for pair in zip(w, p) for s in pair]}[order]:
        if kind == "w": o.watch(chains(succ, B, 2 ** L + 1, rng)); seen += B
        else: o.rounds(B, rng); done += B
        out.append({"examples": seen + done, "N": seen, "M": done, **o.score(test, T, rng)})
    return out


def run_split(seed, tokens, share):
    """A fixed budget of tokens (states shown or written): a share for fresh teacher chains (9 tokens each), the rest for practice."""
    rng, succ, T, test = setup(seed); o = Learner(S, L); N = int(round(share * tokens / (2 ** L + 1))); o.watch(chains(succ, N, 2 ** L + 1, rng))
    Mc = int((tokens - N * (2 ** L + 1)) / np.mean([2 * k + 1 for k in o.ks]))  # a join at level k writes 2k + 1 states
    for B in (np.diff(np.r_[0, grid(16, Mc, 40)]) if Mc >= 16 else []): o.rounds(int(B), rng)
    return {"tokens": tokens, "share": share, "N": N, "M": Mc, **o.score(test, T, rng)}


def run_aim(seed, tokens, share, aim=True, rare=None):
    """As run_split, with practice that skips what the learner already holds (aim=True) or blind practice (aim=False).
    rare: the second step type is taken with this probability and the first with the rest (a world of common and rare steps)."""
    rng, succ, T, _ = setup(seed, test=1); p = None if rare is None else [1 - rare, rare]; n = 2 ** L + 1; test = chains(succ, 8000, n, rng, p=p); o = Learner(S, L)
    N = int(round(share * tokens / n)); o.watch(chains(succ, N, n, rng, p=p)); left = tokens - N * n; row = {"tokens": tokens, "share": share, "N": N}
    if aim: spent, looked = o.aimed(left, rng); row |= {"written": sum(spent.values()), "looked": looked}
    else:
        Mc = int(left / np.mean([2 * k + 1 for k in o.ks]))
        for B in (np.diff(np.r_[0, grid(16, Mc, 40)]) if Mc >= 16 else []): o.rounds(int(B), rng)
    return {**row, **o.score(test, T, rng)}


def run_curve(seed, mode, rare=None, N0=2 ** 12, tokens=2 ** 23, points=44):
    """Share of spans written against tokens. fresh: teacher chains only. blind, aim: N0 teacher chains (their tokens counted),
    then practice only, blind or skipping what the learner holds."""
    rng, succ, T, _ = setup(seed, test=1); p = None if rare is None else [1 - rare, rare]; n = 2 ** L + 1; test = chains(succ, 8000, n, rng, p=p); o = Learner(S, L)
    out, spent, used, looked, lvl = [], 0, 0, 0, {k: 0 for k in o.ks}; cost = np.mean([2 * k + 1 for k in o.ks])
    if mode != "fresh": o.watch(chains(succ, N0, n, rng, p=p)); spent = used = N0 * n; out.append({"tokens": spent, "used": used, "looked": 0, **{f"lvl{l}": 0 for l in range(1, L + 1)}, **o.score(test, T, rng)})
    for Tk in grid(2 ** 12, tokens, points):
        if Tk <= spent: continue
        if mode == "fresh": o.watch(chains(succ, int(Tk - spent) // n, n, rng, p=p)); used = int(Tk)
        elif mode == "blind": o.rounds(int((Tk - spent) / cost), rng); used = int(Tk)
        else:
            sp, lk = o.aimed(int(Tk - spent), rng); used += sum(sp.values()); looked += lk
            for k in o.ks: lvl[k] += sp[k]
        spent = int(Tk); out.append({"tokens": spent, "used": used, "looked": looked, **{f"lvl{l}": lvl[k] for l, k in enumerate(o.ks, 1)}, **o.score(test, T, rng)})
    return out


def reachable(T, ks):
    """R[k][a, c]: c can be reached from a in k links."""
    R = {}; P = T.astype(np.float32)
    for k in ks: R[k] = P > 0; P = (P @ P > 0).astype(np.float32)
    return R


def wrong(o, R):
    """Per level: the share of the table's counts on entries whose midpoint does not lie on a route between their ends, and the number of such entries."""
    row = {}
    for l, k in enumerate(o.ks, 1): c = o.code[k]; a, b, mid = c // o.S ** 2, c // o.S % o.S, c % o.S; bad = ~(R[k][a, mid] & R[k][mid, b]); row[f"wrong{l}"] = float(o.cnt[k][bad].sum() / o.cnt[k].sum()); row[f"nwrong{l}"] = int(bad.sum())
    return row


def own_judged(o, B, rng, drop=0):
    """B chains from the learner's own generator (ends from its top table, one pass), judged link by link. The entries that wrote a
    chain which passes are counted once more. drop=1: an entry loses a count when the state it wrote fits neither of its
    neighbours. drop=2: such an entry is removed."""
    S, k = o.S, o.ks[-1]; r = np.searchsorted(o.cum[k], rng.random(B) * o.cum[k][-1], side="right") - 1; c = o.code[k][r]
    x = np.empty((B, 2 * k + 1), np.int64); x[:, 0], x[:, -1], x[:, k] = c // S ** 2, c // S % S, c % S; nodes = [(k, k, r, np.ones(B, bool))]  # level, position of the midpoint, table row, held
    for kk in o.ks[-2::-1]:
        for i in range(0, 2 * k, 2 * kk):
            key = (x[:, i] * S + x[:, i + 2 * kk]) * S; r, held = o.pick(kk, np.searchsorted(o.code[kk], key), np.searchsorted(o.code[kk], key + S), rng)
            x[:, i + kk] = np.where(held, o.code[kk][r] % S, rng.integers(S, size=B)); nodes.append((kk, i + kk, r, held))
    ok = o.passes(x[:, :-1], x[:, 1:]); good = ok.all(1)
    for kk, p, r, held in nodes:
        np.add.at(o.cnt[kk], r[good & held], 1)
        if drop == 1: np.subtract.at(o.cnt[kk], r[held & ~ok[:, p - 1] & ~ok[:, p]], 1)
        if drop == 2: o.cnt[kk][r[held & ~ok[:, p - 1] & ~ok[:, p]]] = 0
    for kk in o.ks: keep = o.cnt[kk] > 0; o.code[kk], o.cnt[kk] = o.code[kk][keep], o.cnt[kk][keep]; o.cum[kk] = np.r_[0, np.cumsum(o.cnt[kk])]


def run_judge(seed, mode, N0=2 ** 14, noise=0.03, tokens=2 ** 21, points=28):
    """A generator with wrong entries, and four ways to spend more tokens on it. Stage 1: N0 chains of a teacher who slips; the
    recogniser accepts a pair seen twice and is then frozen; the generator takes every span as shown, slips included.
    Stage 2 (the generator only):
      teacher         more of the teacher's chains, taken as shown
      teacher_judged  more of the teacher's chains, a span filed only if all its links pass
      own             chains drawn from the learner's own generator and judged (own_judged)
      own_drop        the same, and an entry loses a count when the state it wrote fits neither of its neighbours
    Every chain shown or written costs its 9 tokens; nothing is looked up first."""
    rng, succ, T, test = setup(seed); n = 2 ** L + 1; o = Learner(S, L, theta=2); o.watch(chains(succ, N0, n, rng, noise)); out, spent = [], 0; R = reachable(T, o.ks)
    out.append({"tokens": spent, **o.score(test, T, rng), **wrong(o, R)})
    for Tk in grid(2 ** 12, tokens, points):
        B = int(Tk - spent) // n
        if mode in ["teacher", "teacher_judged"]: o.file(chains(succ, B, n, rng, noise), gate=mode == "teacher_judged")
        else:
            for b in np.diff(np.r_[0, np.linspace(0, B, 9)[1:].astype(int)]): own_judged(o, int(b), rng, int(mode == "own_drop"))  # in 8 parts, so that the tables are up to date
        spent = int(Tk); out.append({"tokens": spent, **o.score(test, T, rng), **wrong(o, R)})
    return out


def run_spur(seed, mode, rows=64, share=0.2, N0=2 ** 12, M0=2 ** 20, tokens=2 ** 21, points=28):
    """A generator with spurious entries of its own, never seeded by the teacher. The teacher (no slips) supplies the pair table
    only, and practice builds the generator's tables. Its top table also holds `rows` random entries, put there by hand, which
    carry `share` of the table's counts: few rows with large counts, or many rows with small ones. Then further tokens:
      teacher     the teacher's chains, every span filed
      own         own samples, judged; what passes is counted once more
      own_drop    and an entry loses a count when the state it wrote fits neither of its neighbours
      own_delete  and such an entry is removed
    Every chain shown or written costs its 9 tokens."""
    rng, succ, T, test = setup(seed); n = 2 ** L + 1; o = Learner(S, L); o.watch(chains(succ, N0, n, rng), levels=False); out, spent = [], 0; R = reachable(T, o.ks); k = o.ks[-1]
    for B in np.diff(np.r_[0, grid(16, M0, 40)]): o.rounds(int(B), rng)
    a, b, c = rng.integers(S, size=(3, rows)); o.merge(k, np.repeat((a * S + b) * S + c, max(1, round(share / (1 - share) * o.cnt[k].sum() / rows))))
    out.append({"tokens": spent, **o.score(test, T, rng), **wrong(o, R)})
    for Tk in grid(2 ** 10, tokens, points):
        B = int(Tk - spent) // n
        if mode == "teacher": o.file(chains(succ, B, n, rng), gate=False)
        else:
            for b in np.diff(np.r_[0, np.linspace(0, B, 9)[1:].astype(int)]): own_judged(o, int(b), rng, ["own", "own_drop", "own_delete"].index(mode))
        spent = int(Tk); out.append({"tokens": spent, **o.score(test, T, rng), **wrong(o, R)})
    return out


def run_tandem(seed, mode, rows=2 ** 16, N0=2 ** 12, tokens=2 ** 23, points=40):
    """A generator that starts from random entries, and one signed rule. The teacher (no slips) first shows N0 chains, which fill
    the pair table only. Every table of the generator starts with `rows` random entries of count 1. Then, at equal tokens:
      signed   practice. An episode picks a level and either joins two held units into a new entry or rehearses an entry it
               holds; the span is written in one pass and judged; the entry's count goes up by one if it passes, down by one if not
      plus     the same, but a failure changes nothing
      teacher  further teacher chains; every span of a chain is counted
    Every chain shown or written costs its tokens."""
    rng, succ, T, test = setup(seed); n = 2 ** L + 1; o = Learner(S, L); o.watch(chains(succ, N0, n, rng), levels=False); out, spent = [], 0; R = reachable(T, o.ks); cost = np.mean([2 * k + 1 for k in o.ks])
    for k in o.ks: a, b, c = rng.integers(S, size=(3, rows)); o.add(k, a, c, b)
    out.append({"tokens": spent, **o.score(test, T, rng), **wrong(o, R)})
    for Tk in grid(2 ** 10, tokens, points):
        if mode == "teacher": o.file(chains(succ, int(Tk - spent) // n, n, rng), gate=False)
        else:
            for B in np.diff(np.r_[0, np.linspace(0, (Tk - spent) / cost, 9)[1:].astype(int)]):  # in 8 parts, so that the tables are up to date
                for k, Bk in zip(o.ks, rng.multinomial(int(B), np.ones(L) / L)): j = rng.binomial(Bk, 0.5); o.practise(k, j, rng); o.rehearse(k, Bk - j, rng, signed=mode == "signed")
        spent = int(Tk); out.append({"tokens": spent, **o.score(test, T, rng), **wrong(o, R)})
    return out


def run_stages(seed, mode, rows=0, N=2 ** 12, tokens=2 ** 23, points=40, rehearse=0.5):
    """The stages of learning, with scores on the population and on the teacher's training set.
      start    the pair table is empty; every table of the generator holds `rows` random entries of count 1
      stage 1  a teacher set of N chains: the pairs are counted (recognition), and every span is counted (wake)
      stage 2  at equal tokens: signed   practice, each episode a join or (with probability rehearse) a rehearsal, count +1 / -1
                                plus     the same, but a failure changes nothing
                                teacher  fresh teacher chains instead, that is, a larger training set
    x is the number of teacher chains in stage 1 and the number of tokens in stage 2."""
    rng, succ, T, test = setup(seed); n = 2 ** L + 1; o = Learner(S, L); R = reachable(T, o.ks); rng2 = np.random.default_rng(10 ** 6 + seed); cost = np.mean([2 * k + 1 for k in o.ks])
    for k in o.ks[: L if rows else 0]: a, b, c = rng.integers(S, size=(3, rows)); o.add(k, a, c, b)
    train = chains(succ, N, n, rng); sample = train[:2000]; out, done, spent = [], 0, 0
    def score(stage, x): return {"stage": stage, "x": int(x), **o.score(test, T, rng), **(wrong(o, R) if all(len(o.code[k]) for k in o.ks) else {}), **train_score(o, sample, T, rng2)}
    out.append(score(0, 0))
    for Nk in grid(8, N, 14): o.watch(train[done:Nk]); done = int(Nk); out.append(score(1, done))
    for Tk in grid(2 ** 10, tokens, points):
        if mode == "teacher": o.watch(chains(succ, int(Tk - spent) // n, n, rng))
        else:
            for B in np.diff(np.r_[0, np.linspace(0, (Tk - spent) / cost, 9)[1:].astype(int)]):  # in 8 parts, so that the tables are up to date
                for k, Bk in zip(o.ks, rng.multinomial(int(B), np.ones(L) / L)): j = rng.binomial(Bk, 1 - rehearse); o.practise(k, j, rng); o.rehearse(k, Bk - j, rng, signed=mode == "signed")
        spent = int(Tk); out.append(score(2, spent))
    return out


def run_window(seed, N, mode, tokens=2 ** 21):
    """After a teacher set of N chains (recognition and wake, empty start), the same number of further tokens spent in one of three
    ways: fresh (fresh teacher chains), practice (joins, drawn blindly), aim (joins that the learner looks up first and skips
    if it holds them; only written joins cost tokens). Scores on the population before and after."""
    rng, succ, T, test = setup(seed); n = 2 ** L + 1; o = Learner(S, L); o.watch(chains(succ, N, n, rng)); before = o.score(test, T, rng)
    if mode == "fresh": o.watch(chains(succ, tokens // n, n, rng))
    elif mode == "aim": o.aimed(tokens, rng)
    else:
        for B in np.diff(np.r_[0, grid(16, int(tokens / np.mean([2 * k + 1 for k in o.ks])), 40)]): o.rounds(int(B), rng)
    return {"before": before, "after": o.score(test, T, rng)}


def run_bridge(seed, S=64, m=3, L=2, N=2 ** 15, M_max=2 ** 22, points=40):
    """Do practised tables converge to the exact bridge conditional? Small world with many routes; the teacher chains fill the pair
    table only. Target: Chapman-Kolmogorov composition of the learner's own pair counts, level by level."""
    rng, succ, T, test = setup(seed, S, m, L); o = Learner(S, L); o.watch(chains(succ, N, 2 ** L + 1, rng), levels=False)
    C = o.pairs.reshape(S, S).astype(float); target, world_target = {}, {}  # C / C.sum(): how often the learner draws the unit (a, c)
    for tgt, U in [(target, C / C.sum()), (world_target, T / T.sum())]:
        for k in o.ks: P = U[:, :, None] * (U / U.sum(1, keepdims=True))[None]; tgt[k] = P; U = P.sum(1)  # P[a, c, b] = U(a, c) U(b | c); the next level's units are (a, b)
    def tv(k, tgt):
        P = np.zeros(S ** 3); c = o.code[k]; P[(c // S ** 2) * S ** 2 + (c % S) * S + c // S % S] = o.cnt[k]; P = P.reshape(S, S, S) / max(o.cnt[k].sum(), 1); Q = tgt[k]
        pb, qb = P.sum(1, keepdims=True), Q.sum(1, keepdims=True); cond = np.abs(np.divide(P, pb, out=np.zeros_like(P), where=pb > 0) - np.divide(Q, qb, out=np.zeros_like(Q), where=qb > 0)).sum(1)
        return float(0.5 * np.abs(P - Q).sum()), float(0.5 * (cond * qb[:, 0]).sum())  # joint TV; population-weighted TV of p(c | a, b), 1 where the learner has no entry
    out, done = [], 0
    for Mc in grid(64, M_max, points):
        o.rounds(Mc - done, rng); done = Mc; row = {"M": int(Mc)}
        for l, k in enumerate(o.ks, 1): row[f"joint{l}"], row[f"cond{l}"] = tv(k, target); row[f"wjoint{l}"], row[f"wcond{l}"] = tv(k, world_target)
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------- own versus foreign


def run_map(seed, own_share, P_max=2 ** 15, points=56, S=2 ** 14, m=M, L=L, seed_chains=1):
    """The step types are known and act everywhere; the learner can file a chain only if it touches its map (the states it has
    produced or reached). Own chain: starts at a state of the map, applies known steps. Foreign chain: a population chain.
    own_share of the examples are own, the rest foreign, in random order."""
    rng, succ, _, test = setup(seed, S, m, L, test=4000, dense=False); n = 2 ** L + 1; o = Learner(S, L, succ=succ); valid = lambda x: (succ[x[:, :-1]] == x[:, 1:, None]).any(-1).all(1)
    first = chains(succ, seed_chains, n, rng); known = np.zeros(S, bool); known[first] = True; o.file(first, False); out, done, used, seen = [], 0, 0, 0
    def score():
        row = {"P": done, "map": float(known.mean()), "usable": float(known[test].any(1).mean()), "predicted": float(1 - (1 - known.mean()) ** n), "used": used, "foreign_seen": seen}
        for l, k in enumerate(o.ks, 1):
            a, b = test[:, 0], test[:, 2 * k]; key = (a * S + b) * S; row[f"read{l}"] = float((np.searchsorted(o.code[k], key + S) > np.searchsorted(o.code[k], key)).mean()); row[f"write{l}"] = float(valid(o.write(a, b, k, rng)).mean())
        return row
    out.append(score())
    for P in grid(1, P_max, points):
        while done < P:  # small batches (at most 1/32 of the examples so far), so that the map is up to date for every example
            B = int(min(P - done, max(1, done // 32))); own = rng.binomial(B, own_share); x = chains(succ, B - own, n, rng); seen += B - own; x = x[known[x].any(1)]; used += len(x)
            x = np.r_[x, chains(succ, own, n, rng, start=rng.choice(np.flatnonzero(known), own))]; known[x] = True; o.file(x, False); done += B
        out.append(score())
    return out


def regions(K, S_r, m, rng, bridge):
    """K regions of S_r states. Every step stays inside its region, except at a share `bridge` of the states: their first step
    is swapped with that of a state in another region, so that it leads there."""
    succ = np.concatenate([world(S_r, m, rng) + r * S_r for r in range(K)]); z = rng.permutation(K * S_r)[: 2 * round(bridge * K * S_r / 2)].reshape(-1, 2); z = z[z[:, 0] // S_r != z[:, 1] // S_r]
    succ[z[:, 0], 0], succ[z[:, 1], 0] = succ[z[:, 1], 0], succ[z[:, 0], 0]
    return succ


def run_reach(seed, share, bridge=0.0, attach=True, tokens=2 ** 23, points=44, K=16, S_r=S, m=M, L=L, N1=2048):
    """Nothing is granted. The learner reads a link only if it has learned it, and it can take in a teacher chain only if the chain
    touches a state it knows (then the whole chain, outwards from that state). A chain that touches nothing it knows is lost.
    World: K regions. Stage 1: N1 teacher chains that start in region 0, the learner's home. Stage 2: a budget of tokens, a share
    for practice (own examples: joins of held units, as everywhere), the rest for fresh chains of the whole population (foreign).
    attach=False: a learner that takes in every fresh chain. K=1: one mixed world, where home is the whole world."""
    rng = np.random.default_rng(seed); succ = regions(K, S_r, m, rng, bridge); n = 2 ** L + 1; o = Learner(K * S_r, L, dense=False); valid = lambda x: (succ[x[:, :-1]] == x[:, 1:, None]).any(-1).all(1)
    tests = {"home": chains(succ, 4000, n, rng, start=rng.integers(S_r, size=4000)), "pop": chains(succ, 8000, n, rng)}
    first = chains(succ, N1, n, rng, start=rng.integers(S_r, size=N1)); known = np.zeros(K * S_r, bool); known[first] = True; o.watch(first)
    cost = np.mean([2 * k + 1 for k in o.ks]); out, spent, fresh, taken, joins = [], 0, 0, 0, 0  # a join at level k writes 2k + 1 states
    def score():
        row = {"tokens": spent, "fresh": fresh, "taken": taken, "joins": joins, "map": float(known.mean()), "usable": float(known[tests["pop"]].any(1).mean()), "links": float(o.passes(tests["pop"][:, :-1], tests["pop"][:, 1:]).mean())}
        for name, test in tests.items():
            for l, k in enumerate(o.ks, 1): row[f"{name}{l}"] = float(valid(o.write(test[:, 0], test[:, 2 * k], k, rng)).mean())
        return row
    out.append(score())
    for T in grid(2 ** 10, tokens, points):
        while spent < T:  # small batches (at most 1/16 of the tokens so far), so that the map is up to date for every chain
            B = int(min(T - spent, max(256, spent // 16))); x = chains(succ, round((1 - share) * B / n), n, rng); fresh += len(x); x = x[known[x].any(1)] if attach else x; taken += len(x)
            if len(x): known[x] = True; o.watch(x)
            Bo = round(share * B / cost); o.rounds(Bo, rng); joins += Bo; spent += B
        out.append(score())
    return out


# ---------------------------------------------------------------------------------------------- runner

JOBS = {
    "watch": [(run_watch, dict(seed=s)) for s in range(WORLDS)],
    "practice": [(run_practice, dict(seed=s, N=N)) for N in N_GRID for s in range(WORLDS)],
    "pairs_only": [(run_practice, dict(seed=s, N=N, levels=False)) for N in N_GRID for s in range(WORLDS)],
    "schedule": [(run_schedule, dict(seed=s, order=o)) for o in ["watch_first", "practise_first", "alternate"] for s in range(WORLDS)],
    "split": [(run_split, dict(seed=s, tokens=t, share=x)) for t in [2 ** 15, 2 ** 17, 2 ** 19, 2 ** 21] for x in [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0] for s in range(WORLDS)],
    "slip": [(run_practice, dict(seed=s, N=2 ** 12, levels=False, slip=0.1, gate=g, M_max=2 ** 22)) for g in [True, False] for s in range(WORLDS)],
    "false": [(run_practice, dict(seed=s, N=2 ** 12, levels=False, false_pairs=f, M_max=2 ** 22)) for f in [0, 32, 128, 512] for s in range(WORLDS)],
    "noisy": [(run_practice, dict(seed=s, N=N, noise=0.03, M_max=2 ** 22, **arm)) for arm in [dict(theta=1), dict(theta=2), dict(theta=2, levels=False)] for N in [2 ** e for e in range(8, 18)] for s in range(WORLDS)],
    "watch_scale": [(run_watch, dict(seed=s, S=Sx, m=mx, L=Lx, N_max=Nx)) for Sx, mx, Lx, Nx in [(4096, 2, 3, 2 ** 24), (1024, 3, 2, 2 ** 21), (1024, 4, 2, 2 ** 22), (256, 2, 2, 2 ** 17)] for s in range(4)],
    "bridge": [(run_bridge, dict(seed=s)) for s in range(WORLDS)],
    "map": [(run_map, dict(seed=s, own_share=x)) for x in [0.0, 0.5, 1.0] for s in range(WORLDS)],
    "map_big": [(run_map, dict(seed=s, own_share=x, S=2 ** 16, P_max=2 ** 16)) for x in [0.0, 1.0] for s in range(WORLDS)],
    "map_split": [(run_map, dict(seed=s, own_share=x, P_max=P, points=4)) for P in [256, 1024, 4096] for x in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] for s in range(WORLDS)],
    "aim": [(run_aim, dict(seed=s, tokens=t, share=x, aim=a, rare=r)) for r in [None, 0.1] for a in [True, False] for t in [2 ** 19, 2 ** 21, 2 ** 23] for x in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0] for s in range(WORLDS)],
    "aim_curve": [(run_curve, dict(seed=s, mode=mo, rare=r, N0=N0)) for r, N0 in [(None, 2 ** 12), (0.1, 2 ** 13)] for mo in ["fresh", "blind", "aim"] for s in range(WORLDS)],
    "judge": [(run_judge, dict(seed=s, mode=mo)) for mo in ["teacher", "teacher_judged", "own", "own_drop"] for s in range(WORLDS)],
    "window": [(run_window, dict(seed=s, N=N, mode=mo)) for mo in ["fresh", "practice", "aim"] for N in N_GRID for s in range(WORLDS)],
    "stages_n": [(run_stages, dict(seed=s, mode=mo, rows=2 ** 13, N=N, tokens=2 ** 21, points=3)) for mo in ["signed", "teacher"] for N in [2 ** e for e in range(8, 21, 2)] for s in range(WORLDS)],
    "stages": [(run_stages, dict(seed=s, mode="signed", rehearse=0.0, tokens=2 ** 26, points=48)) for s in range(WORLDS)],
    "stages_random": [(run_stages, dict(seed=s, mode=mo, rows=r)) for r in [2 ** 13, 2 ** 16] for mo in ["signed", "plus", "teacher"] for s in range(WORLDS)],
    "tandem": [(run_tandem, dict(seed=s, mode=mo, rows=r)) for r in [2 ** 16, 2 ** 13] for mo in ["signed", "plus", "teacher"] for s in range(WORLDS)],
    "spur": [(run_spur, dict(seed=s, mode=mo, rows=r)) for r in [64, 2 ** 15] for mo in ["teacher", "own", "own_drop", "own_delete"] for s in range(WORLDS)],
    "reach": [(run_reach, dict(seed=s, share=x, bridge=b)) for b in [0.0, 1 / 1024, 1 / 64] for x in [0.0, 0.5, 1.0] for s in range(WORLDS)]
    + [(run_reach, dict(seed=s, share=x, bridge=1 / 64, attach=False)) for x in [0.0, 0.5] for s in range(WORLDS)]
    + [(run_reach, dict(seed=s, share=x, K=1, S_r=16 * S)) for x in [0.0, 0.5, 1.0] for s in range(WORLDS)],
}


def call(job): f, kw = job; return {"args": kw, "out": f(**kw)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0]); p.add_argument("--only", nargs="*", default=list(JOBS)); p.add_argument("--procs", type=int, default=28); args = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    for name in tqdm(args.only, desc="experiments"):
        with Pool(args.procs) as pool: res = list(tqdm(pool.imap(call, JOBS[name]), total=len(JOBS[name]), desc=name, leave=False))
        (OUT / f"{name}.json").write_text(json.dumps({"world": dict(S=S, m=M, L=L, worlds=WORLDS), "runs": res})); print(OUT / f"{name}.json")


if __name__ == "__main__": main()
