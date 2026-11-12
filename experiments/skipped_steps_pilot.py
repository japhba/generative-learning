"""Pilot: a depthwise own-sample advantage in toy model 3's world, with a teacher who skips steps (wake-sleep with a hidden cause).
World: S states, two allowed successors each (two permutations); the population takes the first with probability p_a in {P_LO, P_HI}.
A population example is a valid chain of LINKS links shown at a granularity k (a power of two, probabilities PI): only every k-th state.
The omitted states are the hidden cause. The teacher set is an i.i.d. sample of such examples.
Learner (which links are allowed is known from the start: atomic steps are verifiable; what it has to learn is p_a):
  generator   forward chain with p^_a = (count + ALPHA) / (n + 2 ALPHA), then the same omission of states
  recogniser  span tables q(midpoint | ends) for spans of 2, 4, .. links
  sleep       own chains; each carries its omitted states and is counted into the span tables
              dreams = "flat": every allowed successor equally likely;  "own": successors drawn with the current p^
  wake        a teacher example is filled in top-down by the span tables and its links are counted; a shown pair whose span is not in the
              tables is not read and changes nothing
Runs, for a stored teacher set of N examples: practice (own chains, then the stored set is read again) against the same number of tokens
spent on fresh teacher examples without practice. Control: a learner whose span tables are complete from the start (mature in depth,
wrong only in p_a). Budget split: P tokens in all, a share of them own. Score: KL(population || generator) per chain (exact).
Usage: python skipped_steps_pilot.py -> reports/skipped-steps-pilot.png, /var/tmp/jbauer/generative_practice_chain/skipped_steps.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

S, LINKS, P_LO, P_HI, ALPHA, GATE, WORLDS = 1024, 4, 0.2, 0.8, 0.5, 0.9, 8
KS = 2 ** np.arange(int(np.log2(LINKS)) + 1); PI = np.r_[np.ones(len(KS) - 1), len(KS) - 1.0]; PI /= PI.sum()   # half of the examples show only the two ends
HS = [int(h) for h in KS[:-1]]; TOKENS = float((PI * (LINKS / KS + 1)).sum())                               # half-spans of the tables; shown states per teacher example
NS, OWN = [4096, 16384, 65536], 2 ** np.arange(6, 19)                                                       # teacher examples; cumulative own chains
BUDGETS, SHARES = 2 ** np.arange(14, 21, 2), [0, 0.125, 0.25, 0.5, 0.75]
OUT = Path("/var/tmp/jbauer/generative_practice_chain"); FIG = Path(__file__).resolve().parents[1] / "reports/skipped-steps-pilot.png"


def chains(rng, succ, p, num):
    x = np.empty((num, LINKS + 1), int); x[:, 0] = rng.integers(S, size=num)
    for t in range(LINKS): x[:, t + 1] = succ[x[:, t], (rng.random(num) >= p[x[:, t]]).astype(int)]
    return x


class Table:                                                                                                # counts of (a, b) -> c for spans of 2h links, at every position of the chains
    def __init__(self, x, h):
        i = np.arange(LINKS + 1 - 2 * h); u, cnt = np.unique(((x[:, i] * S + x[:, i + 2 * h]) * S + x[:, i + h]).ravel(), return_counts=True)
        self.ab, self.start = np.unique(u // S, return_index=True); self.c, self.cum, self.end = u % S, np.cumsum(cnt), np.r_[self.start[1:], len(u)]

    def draw(self, rng, a, b):                                                                              # c ~ counts, -1 where the table has no entry (or an end is already missing)
        if not len(self.ab): return np.full(len(a), -1)
        key = a * S + b; g = np.minimum(np.searchsorted(self.ab, key), len(self.ab) - 1); hit = (self.ab[g] == key) & (a >= 0) & (b >= 0)
        base = np.where(self.start[g] > 0, self.cum[self.start[g] - 1], 0); total = self.cum[self.end[g] - 1] - base
        return np.where(hit, self.c[np.minimum(np.searchsorted(self.cum, base + rng.random(len(a)) * total, side="right"), len(self.c) - 1)], -1)


def tables(own): return {h: Table(own, h) for h in HS}
def phat(C): return (C[:, 0] + ALPHA) / (C.sum(1) + 2 * ALPHA)
def kl(p, q): return LINKS * (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))).mean()              # every position of a chain is uniform over states (permutations)


def read(rng, succ, tab, x, g, gate=0.0):                                                                   # wake: link counts from the examples the learner can fill in (a granularity counts only if at least `gate` of it is read), share read per granularity
    C, share = np.zeros((S, 2)), []
    for k in KS:
        shown = x[g == k][:, ::k]; path = np.full((shown[:, 1:].size, k + 1), -1); path[:, 0], path[:, k], h = shown[:, :-1].ravel(), shown[:, 1:].ravel(), k // 2
        while h >= 1:                                                                                       # top-down: midpoint of each span whose ends are in place
            for i in range(0, k, 2 * h): path[:, i + h] = tab[h].draw(rng, path[:, i], path[:, i + 2 * h]) if tab else -1
            h //= 2
        ok = (path >= 0).all(1); share.append(ok.mean() if len(ok) else np.nan); ok &= share[-1] >= gate; u, v = path[ok][:, :-1].ravel(), path[ok][:, 1:].ravel(); np.add.at(C, (u, (succ[u, 0] != v).astype(int)), 1)
    return C, np.array(share)


def setup(seed): rng = np.random.default_rng(seed); return rng, np.stack([rng.permutation(S) for _ in range(2)], 1), rng.choice([P_LO, P_HI], size=S), np.full(S, 0.5)


def curves(seed):                                                                                           # fixed teacher set, further tokens spent on practice or on fresh teacher examples
    rng, succ, p, flat = setup(seed); mature = tables(chains(rng, succ, flat, 2 ** 20)); res = {}
    for n in NS:
        extra = int(OWN[-1] * (LINKS + 1) / TOKENS); x = chains(rng, succ, p, n + extra); g = rng.choice(KS, size=n + extra, p=PI)
        C0 = read(rng, succ, None, x[:n], g[:n])[0]; Call = read(rng, succ, mature, x[:n], g[:n])[0]; own = {m: np.empty((0, LINKS + 1), int) for m in ("flat", "own")}; q, rows = phat(C0), []
        for d in OWN:
            row = [d * (LINKS + 1)]
            for mode in ("flat", "own"):
                own[mode] = np.r_[own[mode], chains(rng, succ, flat if mode == "flat" else q, d - len(own[mode]))]   # sleep
                C, share = read(rng, succ, tables(own[mode]), x[:n], g[:n]); row += [kl(p, phat(C)), *share[1:]]       # wake: the stored teacher set read again
                if mode == "own": q = phat(C)
            row.append(kl(p, phat(read(rng, succ, tables(own["flat"]), x[:n], g[:n], gate=GATE)[0])))                # flat dreams, a granularity used only once nearly all of it is read
            m = n + int(d * (LINKS + 1) / TOKENS); fresh = C0 + read(rng, succ, None, x[n:m], g[n:m])[0]               # the same tokens as fresh teacher examples, no practice
            rows.append(row + [kl(p, phat(fresh)), kl(p, phat(Call)), kl(p, phat(read(rng, succ, mature, x[:m], g[:m])[0]))])
        res[n] = np.array(rows); res[n, "base"] = np.array([kl(p, phat(C0)), kl(p, phat(Call)), kl(p, flat)])
    return res


def split(seed):                                                                                            # P tokens in all, a share of them own (flat dreams), the rest teacher examples
    rng, succ, p, flat = setup(seed + 1000); out = np.zeros((len(BUDGETS), len(SHARES)))
    for i, P in enumerate(BUDGETS):
        for j, f in enumerate(SHARES):
            n, d = int((1 - f) * P / TOKENS), int(f * P / (LINKS + 1)); x = chains(rng, succ, p, n); g = rng.choice(KS, size=n, p=PI)
            out[i, j] = kl(p, phat(read(rng, succ, tables(chains(rng, succ, flat, d)) if d else None, x, g)[0]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(WORLDS) as pool: res, sp = pool.map(curves, range(WORLDS)), np.array(pool.map(split, range(WORLDS)))
    curve = {n: np.median([r[n] for r in res], 0) for n in NS}; base = {n: np.median([r[n, "base"] for r in res], 0) for n in NS}; L = len(KS) - 1
    np.savez(OUT / "skipped_steps.npz", split=sp, **{f"n{n}": np.array([r[n] for r in res]) for n in NS}, **{f"base{n}": np.array([r[n, "base"] for r in res]) for n in NS})
    cols = dict(flat=1, own=2 + L, gate=3 + 2 * L, fresh=4 + 2 * L, mature=5 + 2 * L, mature_fresh=6 + 2 * L)
    for n in NS:
        c = curve[n]; print(f"teacher examples {n} ({n * TOKENS:.0f} tokens): KL per chain before any example {base[n][2]:.3f}, no practice {base[n][0]:.4f}, every example read {base[n][1]:.4f}")
        print("   further tokens | practice: flat dreams, own dreams, flat dreams gated | fresh instead | mature learner: practice  fresh | read (flat dreams), granularity " + ", ".join(map(str, KS[1:])))
        for row in c: print(f"   {row[0]:13.0f} | {row[cols['flat']]:.4f}  {row[cols['own']]:.4f}  {row[cols['gate']]:.4f} | {row[cols['fresh']]:.4f} | {row[cols['mature']]:.4f}  {row[cols['mature_fresh']]:.4f} | " + "  ".join(f"{v:.2f}" for v in row[2:2 + L]))
        for key in ("flat", "gate"): print(f"   share of worlds in which practice ({key}) is below fresh, per row: {np.round(np.mean([r[n][:, cols[key]] < r[n][:, cols['fresh']] for r in res], 0), 2)}")
    print("budget split, KL per chain (median); rows = total tokens, columns = own share " + str(SHARES))
    for P, row, win in zip(BUDGETS, np.median(sp, 0), (sp[:, :, 1:] < sp[:, :, :1]).mean(0)): print(f"   {P:8d} | " + "  ".join(f"{v:.4f}" for v in row) + f" | worlds where the split beats all-teacher: {np.round(win, 2)}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True); n = NS[1]; c = curve[n]
    for j, k in enumerate(KS[1:]): ax[0].plot(c[:, 0], c[:, 2 + j], label=f"shown states {k} links apart")
    ax[0].set(xscale="log", xlabel="own tokens (practice)", ylabel="share of teacher examples read", title="what the learner can read"); fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower left")
    for key, col, ls, lab in [("flat", "C0", "-", "practice, then the stored teacher set read again"), ("fresh", "C1", "-", "fresh teacher examples instead"), ("mature", "C0", "--", "control (span tables complete): practice"), ("mature_fresh", "C1", "--", "control: fresh teacher examples")]:
        ax[1].plot(c[:, 0], c[:, cols[key]], color=col, ls=ls, label=lab)
    ax[1].axhline(base[n][0], color="gray", ls=":", label="no practice"); ax[1].set(xscale="log", yscale="log", xlabel="further tokens (own or fresh)", ylabel="KL(population || generator) per chain", title=f"stored teacher set: {n:,} examples"); fig.legend(*ax[1].get_legend_handles_labels(), loc="outside lower center")
    for P, row in zip(BUDGETS, np.median(sp, 0)): ax[2].plot(SHARES, row, "o-", label=f"{P:,} tokens in all")
    ax[2].set(yscale="log", xlabel="share of the tokens that are own", ylabel="KL(population || generator) per chain", title="fixed budget, split between teacher and own"); fig.legend(*ax[2].get_legend_handles_labels(), loc="outside lower right")
    fig.savefig(FIG, dpi=150, transparent=True); print(FIG)


if __name__ == "__main__": main()
