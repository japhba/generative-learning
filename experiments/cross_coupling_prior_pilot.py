"""Pilot: does a fixed prior over the writer's choices stop the narrowing that own-sample practice shows in cross_coupling_offline_pilot.py?
Same world (depth-2 cross-coupled RHM, one agreement between tokens 2 and 3) and the same learned cut memory. The writer is re-parametrised: every
choice is made by a latent unit with a fixed uniform prior, and only what a unit's value writes is learned.
  root      k (M values)   D2[a, k]      -> production of the root symbol a
  spare     u (RHO values), handed to both children
  children  j (M / RHO values each)  DL[b, u, j], DR[c, u, j] -> production of the child symbol
There are as many codes (k, u, jL, jR) per root symbol as valid trees if every symbol's productions have balanced features at both tokens ("balanced"
world: the uniform prior is exact). In the original world they are not balanced, so the uniform prior is the wrong one there.
Reader: cut memory (teacher strings only) and one naming layer per latent unit, learned from own samples only (the code a sample was written with):
k from (a, production), u from the two cut tokens, j from (child symbol, its production).
Steps:
  teacher string, unnamed    codes drawn from the prior (stage 1: all values of a unit come to write the same)
  teacher string, named      credited to the codes the naming layers read off it
  teacher string, inferred   the writer's exact posterior over the codes (not local; reference)
  own sample, naming         the naming layers learn the sample's codes
  own sample, accepted       if the reader accepts the whole sample, every row that wrote it takes a step towards what it wrote ("legible": times the
                             naming layers' probability of that row's code, in units of chance)
  own sample, verdict        failure-only step on every rejected row
Scores are exact (enumeration) for the equivalent table writer: valid share, share of the valid strings written, KL(target || writer),
I(u; feature at the cut) / H(feature).
Usage: python experiments/cross_coupling_prior_pilot.py [report] -> /var/tmp/jbauer/generative_practice_reading/cross_coupling_prior_pilot.npz,
reports/cross-coupling-prior-pilot.png (about 6 minutes on 32 cores)"""

import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cross_coupling_cycle_pilot import EYE, score
from cross_coupling_evict_pilot import ETA, M, RHO, V, World, soft

U, J, N0, T, EVERY, WORLDS, CONDS = RHO, M // RHO, 30000, 60000, 1500, 32, ("balanced", "original")
# arm: (rule for teacher strings, one teacher string in every k samples (0: none), rule for own samples)
ARMS = {"fresh": ("unnamed", 1, None), "fresh, codes inferred (not local)": ("inferred", 1, None), "half own: naming": ("named", 2, "naming"), "own only: accepted": (None, 0, "keep"),
        "own only: accepted, verdict": (None, 0, "keep+verdict"), "own only: naming, accepted if legible": (None, 0, "naming+keep legible"),
        "own only: naming, accepted if legible, verdict": (None, 0, "naming+keep legible+verdict")}
PLOTTED = [arm for arm in ARMS if "verdict" not in arm]
METRICS = ("share of own strings that are valid", "share of the valid strings written", "KL(target || writer), nats", "what u says about the cut")
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_prior_pilot.npz"); PNG = ROOT / "reports/cross-coupling-prior-pilot.png"


def rebalance(w, rng):
    free, rows = list(rng.permutation(V * V)), []
    for b in range(V):
        while True:
            r = rng.choice(free, M, replace=False)
            if all((np.bincount(w.phi[t], minlength=RHO) == M // RHO).all() for t in (r // V, r % V)): break
        rows.append(r); free = [f for f in free if f not in r]
    w.R1 = np.array(rows); w.ok1[:] = False; w.ok1[np.arange(V)[:, None], w.R1] = True
    w.trees = np.array([(a, p2, tL, tR) for a in range(V) for p2 in w.R2[a] for tL in w.R1[p2 // V] for tR in w.R1[p2 % V] if w.phi[tL % V] == w.phi[tR // V]])


class Writer:
    def __init__(self, rng): self.D2, self.DL, self.DR = rng.normal(size=(V, M, V * V)), rng.normal(size=(V, U, J, V * V)), rng.normal(size=(V, U, J, V * V))

    def tables(self): return SimpleNamespace(A2=np.log(soft(self.D2).mean(1)), g=np.zeros((V * V, U)), BL=np.log(soft(self.DL).mean(2)), BR=np.log(soft(self.DR).mean(2)))

    def copy(self): new = Writer.__new__(Writer); new.D2, new.DL, new.DR = self.D2.copy(), self.DL.copy(), self.DR.copy(); return new


class Reader:
    def __init__(self): self.seen, self.k, self.u, self.jL, self.jR = np.zeros((V, V), bool), np.zeros((V, V * V, M)), np.zeros((V, V, U)), np.zeros((V, V * V, J)), np.zeros((V, V * V, J))


def teacher(wr, rd, w, rng, how):
    a, p2, tL, tR = w.trees[rng.integers(len(w.trees))]; b, c, x2, x3 = p2 // V, p2 % V, tL % V, tR // V; rd.seen[x2, x3] = True; p2s, pL, pR = soft(wr.D2[a]), soft(wr.DL[b]), soft(wr.DR[c])
    if how == "unnamed": rk, ru, rL, rR = np.eye(M)[rng.integers(M)], np.eye(U)[rng.integers(U)], np.tile(np.eye(J)[rng.integers(J)], (U, 1)), np.tile(np.eye(J)[rng.integers(J)], (U, 1))
    elif how == "named": rk, ru, rL, rR = soft(rd.k[a, p2]), soft(rd.u[x2, x3]), np.tile(soft(rd.jL[b, tL]), (U, 1)), np.tile(soft(rd.jR[c, tR]), (U, 1))
    else: lL, lR = pL[:, :, tL], pR[:, :, tR]; rk, ru, rL, rR = p2s[:, p2] / p2s[:, p2].sum(), lL.sum(1) * lR.sum(1) / (lL.sum(1) * lR.sum(1)).sum(), lL / lL.sum(1, keepdims=True), lR / lR.sum(1, keepdims=True)
    wr.D2[a] += ETA * rk[:, None] * (EYE[p2] - p2s); wr.DL[b] += ETA * (ru[:, None] * rL)[:, :, None] * (EYE[tL] - pL); wr.DR[c] += ETA * (ru[:, None] * rR)[:, :, None] * (EYE[tR] - pR)


def practice(wr, rd, w, rng, how):
    a, k, u, jL, jR = rng.integers(V), rng.integers(M), rng.integers(U), rng.integers(J), rng.integers(J); pk = soft(wr.D2[a, k]); p2 = rng.choice(V * V, p=pk); b, c = p2 // V, p2 % V
    pL, pR = soft(wr.DL[b, u, jL]), soft(wr.DR[c, u, jR]); tL, tR = rng.choice(V * V, p=pL), rng.choice(V * V, p=pR); x2, x3 = tL % V, tR // V; cut = rd.seen[x2, x3]
    names = ((rd.k[a, p2], k), (rd.u[x2, x3], u), (rd.jL[b, tL], jL), (rd.jR[c, tR], jR)); qk, qu, qL, qR = (soft(n)[i] for n, i in names)
    rows = ((wr.D2[a, k], pk, p2, qk * M, w.ok2[a, p2]), (wr.DL[b, u, jL], pL, tL, qu * qL * U * J, w.ok1[b, tL] and cut), (wr.DR[c, u, jR], pR, tR, qu * qR * U * J, w.ok1[c, tR] and cut))
    if "naming" in how:
        for n, i in names: n += ETA * ((np.arange(len(n)) == i) - soft(n))
    for row, p, i, legible, ok in rows:
        if "keep" in how and all(r[4] for r in rows): row += ETA * (legible if "legible" in how else 1) * (EYE[i] - p)
        if "verdict" in how and not ok: row -= ETA * (EYE[i] - p)


def run(job):
    seed, cond = job; rng = np.random.default_rng(seed); w = World(rng); start, rd0 = Writer(rng), Reader()
    if cond == "balanced": rebalance(w, rng)
    for t in range(N0): teacher(start, rd0, w, rng, "unnamed")
    out = []
    for arm, (th, k, oh) in ARMS.items():
        wr, rd, r = start.copy(), Reader(), np.random.default_rng(seed + 1000); rd.seen = rd0.seen.copy(); curve = [score(wr.tables(), w)]
        for t in range(T):
            if k and t % k == 0: teacher(wr, rd, w, r, th)
            else: practice(wr, rd, w, r, oh)
            if (t + 1) % EVERY == 0: curve.append(score(wr.tables(), w))
        out.append(curve)
    return np.array(out)


def report():
    d = np.load(NPZ); x = np.arange(T // EVERY + 1) * EVERY; fig, axes = plt.subplots(len(CONDS), len(METRICS), sharex=True, constrained_layout=True, figsize=(6.4 * 2.8, 4.8 * 1.5))
    for cond, row in zip(CONDS, axes):
        s = d[cond]; print(f"{cond} world: start valid {np.median(s[:, 0, 0, 0]):.3f} written {np.median(s[:, 0, 0, 1]):.2f} KL {np.median(s[:, 0, 0, 2]):.3f}")
        for k, arm in enumerate(ARMS):
            e = s[:, k, -1]; print(f"   {arm:48s} valid {np.median(e[:, 0]):.3f}  written {np.median(e[:, 1]):.2f} [{np.quantile(e[:, 1], 0.25):.2f}, {np.quantile(e[:, 1], 0.75):.2f}]  KL {np.median(e[:, 2]):.3f} "
                                    f"[{np.quantile(e[:, 2], 0.25):.3f}, {np.quantile(e[:, 2], 0.75):.3f}]  u about the cut {np.median(e[:, 3]):.2f}  KL at the end below the start in {(e[:, 2] < s[:, k, 0, 2]).sum()}/{len(e)} worlds")
        for j, ax in enumerate(row):
            for n, arm in enumerate(PLOTTED):
                y = s[:, list(ARMS).index(arm), :, j]; ax.plot(x, np.median(y, 0), color=f"C{n}", linestyle="--" if "not local" in arm else "-", label=arm); ax.fill_between(x, *np.quantile(y, [0.25, 0.75], 0), color=f"C{n}", alpha=0.15, linewidth=0)
            ax.set(ylabel=METRICS[j], title=f"{cond} world"); ax.set_yscale("log" if j == 2 else "linear")
            if cond == CONDS[-1]: ax.set_xlabel("samples after Watching")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False); fig.savefig(PNG, dpi=130); print(PNG)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        jobs = [(s, cond) for cond in CONDS for s in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        np.savez(NPZ, **{cond: np.array([r for r, j in zip(res, jobs) if j[1] == cond]) for cond in CONDS})
    report()
