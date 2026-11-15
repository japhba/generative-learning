"""Pilot: can own samples judged by the reader beat fresh teacher strings at removing what watching leaves undetermined in the writer?
Smallest cross-coupled RHM: depth 2, 4 tokens, V symbols, M productions per symbol and level, one agreement (the feature of token 2 = that of token 3,
RHO values). Target: uniform on the trees that obey every production and the agreement, scored exactly by enumeration.

Reader: given (knows every production and the agreement); it parses teacher strings and judges own samples, one verdict per production and one for the cut.
Writer: top-down softmax tables, logits start random. Root: production table A2[a]. One spare unit u (U values) drawn from g[production] and handed to
both children; child tables BL[b, u], BR[c, u]. Nothing tells the writer what u means.
Rules (all local to one table row):
  watch      teacher string, cross-entropy step on every row its parse names; u is drawn from the writer's own g (the reader does not name it)
  watch+u    the same, but u is inferred for the teacher string (the writer's exact posterior over u): NOT local, reference only
  own signed    own sample; z += eta v (e_choice - p), v = +1 if the reader accepts the row's choice, -1 if it rejects it
  own fail      the same with v = 0 / -1: nothing happens on acceptance. The children and g answer for the cut as well as for their own choice.
  own fail/try  a rejected choice k loses ETA_OWN / p_k on its own entry (capped at CAP): in expectation an entry loses log-mass in proportion to
                how often it fails when tried, and entries that never fail keep their ratios
  (Dropped: z_k -= eta on rejection, without 1 / p_k. A broken cut is no single entry's fault, valid entries go at half the rate of invalid ones and
  every row spreads over all entries: valid 0.55 -> 0.007, KL 0.73 -> 5.1 / 4.7 nats for U = 2 / 8, 32 worlds. Alternating fresh strings with
  own fail/try: valid 0.50 / 0.56, KL 0.93 / 0.78, worse than fresh strings alone in 32 / 30 of 32 worlds.)
Stage 1: N0 teacher strings with `watch`. Stage 2: T further samples per arm. Usage: python experiments/cross_coupling_evict_pilot.py [report]
-> /var/tmp/jbauer/generative_practice_reading/cross_coupling_evict_pilot.npz, reports/cross-coupling-evict-pilot.png (about a minute on 32 cores)"""

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

V, M, RHO, ETA, ETA_OWN, CAP, N0, T, EVERY, WORLDS, US = 8, 4, 2, 0.2, 0.02, 1.0, 30000, 60000, 1500, 32, (2, 8)
ARMS = {"fresh": ("watch", "watch"), "fresh, u inferred (not local)": ("watch+u", "watch+u"), "own, signed": ("own signed", "own signed"), "own, failures only": ("own fail", "own fail"),
        "own, failures only, per try": ("own fail/try", "own fail/try"), "half fresh, half own (failures only)": ("watch", "own fail")}
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_evict_pilot.npz"); PNG = ROOT / "reports/cross-coupling-evict-pilot.png"


def soft(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class World:
    def __init__(self, rng):
        self.R2, self.R1 = (rng.permutation(V * V)[:V * M].reshape(V, M) for _ in range(2)); self.phi = rng.permutation(np.arange(V) % RHO)
        trees = [(a, p2, tL, tR) for a in range(V) for p2 in self.R2[a] for tL in self.R1[p2 // V] for tR in self.R1[p2 % V] if self.phi[tL % V] == self.phi[tR // V]]
        self.trees = np.array(trees); self.ok2 = np.zeros((V, V * V), bool); self.ok1 = np.zeros((V, V * V), bool)
        self.ok2[np.arange(V)[:, None], self.R2] = True; self.ok1[np.arange(V)[:, None], self.R1] = True


class Writer:
    def __init__(self, U, rng): self.A2, self.g, self.BL, self.BR, self.U = rng.normal(size=(V, V * V)), rng.normal(size=(V * V, U)), rng.normal(size=(V, U, V * V)), rng.normal(size=(V, U, V * V)), U

    def score(self, w):
        a, p2, tL, tR = w.trees.T; b, c = p2 // V, p2 % V
        q = soft(self.A2)[a, p2] / V * (soft(self.g)[p2] * soft(self.BL)[b, :, tL] * soft(self.BR)[c, :, tR]).sum(1)
        qv = q / q.sum(); return q.sum(), -np.log(q * len(q)).mean(), np.exp(-(qv * np.log(qv)).sum()) / len(q)

    def watch(self, w, rng, infer):
        a, p2, tL, tR = w.trees[rng.integers(len(w.trees))]; b, c = p2 // V, p2 % V; self.A2[a] += ETA * ((np.arange(V * V) == p2) - soft(self.A2[a]))
        pg, pL, pR = soft(self.g[p2]), soft(self.BL[b]), soft(self.BR[c])
        if infer: resp = pg * pL[:, tL] * pR[:, tR]; resp = resp / resp.sum(); self.g[p2] += ETA * (resp - pg)
        else: resp = np.zeros(self.U); resp[rng.choice(self.U, p=pg)] = 1
        self.BL[b] += ETA * resp[:, None] * ((np.arange(V * V) == tL) - pL); self.BR[c] += ETA * resp[:, None] * ((np.arange(V * V) == tR) - pR)

    def own(self, w, rng, rule):
        a = rng.integers(V); pa = soft(self.A2[a]); p2 = rng.choice(V * V, p=pa); b, c = p2 // V, p2 % V; pg = soft(self.g[p2]); u = rng.choice(self.U, p=pg)
        pL, pR = soft(self.BL[b, u]), soft(self.BR[c, u]); tL, tR = rng.choice(V * V, p=pL), rng.choice(V * V, p=pR); cut = w.phi[tL % V] == w.phi[tR // V]
        for row, p, k, ok in ((self.A2[a], pa, p2, w.ok2[a, p2]), (self.g[p2], pg, u, cut), (self.BL[b, u], pL, tL, w.ok1[b, tL] and cut), (self.BR[c, u], pR, tR, w.ok1[c, tR] and cut)):
            if rule == "own fail/try": row[k] -= 0 if ok else min(ETA_OWN / p[k], CAP)
            elif ok and rule == "own fail": continue
            else: row += ETA * (1 if ok else -1) * ((np.arange(len(row)) == k) - p)


def step(writer, w, rng, rule):
    if rule.startswith("watch"): writer.watch(w, rng, rule == "watch+u")
    else: writer.own(w, rng, rule)


def run(job):
    seed, U = job; rng = np.random.default_rng(seed); w = World(rng); start = Writer(U, rng); stage1 = [start.score(w)]
    for t in range(N0):
        step(start, w, rng, "watch")
        if (t + 1) % EVERY == 0: stage1.append(start.score(w))
    out = {}
    for arm, rules in ARMS.items():
        wr = Writer(U, rng); wr.A2, wr.g, wr.BL, wr.BR = (x.copy() for x in (start.A2, start.g, start.BL, start.BR)); curve = [wr.score(w)]; r = np.random.default_rng(seed + 1000)
        for t in range(T):
            step(wr, w, r, rules[t % 2])
            if (t + 1) % EVERY == 0: curve.append(wr.score(w))
        out[arm] = curve
    return np.array(stage1), np.array([out[a] for a in ARMS])


def report():
    d = np.load(NPZ); fig, axes = plt.subplots(len(US), 3, sharex=True, constrained_layout=True, figsize=(6.4 * 2.1, 4.8 * 1.3))
    x1, x2 = np.arange(N0 // EVERY + 1) * EVERY, N0 + np.arange(T // EVERY + 1) * EVERY
    for i, U in enumerate(US):
        s1, s2 = d[f"stage1_U{U}"], d[f"arms_U{U}"]
        for j, name in zip((0, 2, 1), ("share of own strings that are valid", "share of the valid strings written", "KL(target || writer), nats")):
            ax = axes[i, (0, 2, 1)[j]]; ax.plot(x1, np.median(s1[:, :, j], 0), color="k", label="watching")
            for k, arm in enumerate(ARMS):
                y = s2[:, k, :, j]; ax.plot(x2, np.median(y, 0), color=f"C{k}", linestyle="--" if "not local" in arm else "-", label=arm); ax.fill_between(x2, *np.quantile(y, [0.25, 0.75], 0), color=f"C{k}", alpha=0.15, linewidth=0)
            ax.axvline(N0, color="0.5", linestyle=":"); ax.set(ylabel=name, title=f"spare unit with {U} values"); ax.set_yscale("log" if j == 1 else "linear")
            if i == len(US) - 1: ax.set_xlabel("samples (teacher strings, then the arm's)")
        print(f"U {U}: after watching valid {np.median(s1[:, -1, 0]):.3f} written {np.median(s1[:, -1, 2]):.2f} KL {np.median(s1[:, -1, 1]):.3f}")
        for k, arm in enumerate(ARMS): print(f"   {arm:38s} valid {np.median(s2[:, k, -1, 0]):.3f} [{np.quantile(s2[:, k, -1, 0], 0.25):.3f}, {np.quantile(s2[:, k, -1, 0], 0.75):.3f}]  written {np.median(s2[:, k, -1, 2]):.2f}  KL {np.median(s2[:, k, -1, 1]):.3f} "
                                          f"[{np.quantile(s2[:, k, -1, 1], 0.25):.3f}, {np.quantile(s2[:, k, -1, 1], 0.75):.3f}]  worlds with lower KL than fresh: {(s2[:, k, -1, 1] < s2[:, 0, -1, 1]).sum()}/{len(s2)}")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False); fig.savefig(PNG, dpi=130); print(PNG)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        jobs = [(s, U) for U in US for s in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez(NPZ, **{f"stage1_U{U}": np.array([r[0] for r, j in zip(res, jobs) if j[1] == U]) for U in US}, **{f"arms_U{U}": np.array([r[1] for r, j in zip(res, jobs) if j[1] == U]) for U in US})
    report()
