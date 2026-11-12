"""Pilot: a depthwise own-sample advantage in toy model 3's world, with a teacher who skips steps (wake-sleep with a hidden cause), and a
check of how alike teacher and own samples are on the surface.
World: S states, two allowed successors each. succ[:, 1] = succ[pair, 0] for a random pairing of the states, so paired states have the same
two successors; the population's preference r in {R_LO, R_HI} is the same for paired states. Every state is then equally likely at every
position, under the population and under a flat walk alike. Two kinds of population:
  link  the first successor is taken with probability r_x                                   (differs from a flat walk in neighbouring pairs)
  far   the first GAP choices are fair; choice t + GAP repeats choice t with probability r_{x_t}   (differs only from GAP + 1 links apart on)
A population example is a chain of LINKS links shown at a granularity k: only every k-th state. The omitted states are the hidden cause.
Learner (which links are allowed is known from the start: atomic steps are verifiable; what it has to learn is r):
  generator   forward chain with r^ = (count + ALPHA) / (n + 2 ALPHA)
  recogniser  span tables q(midpoint | ends) for spans of 2, 4, .. links, counted from chains whose states are all there: the teacher's
              fully shown examples and the learner's own chains (sleep; "flat": all choices fair, "own": the current r^)
  wake        a shown pair of states is filled in top-down by the span tables and its choices are counted, but only if every route from its
              first state is in the tables (whether a pair is read then does not depend on what it hides)
Runs, for a stored teacher set of N examples: practice (own chains, then the stored set is read again) against the same number of tokens of
fresh teacher examples. Control: span tables complete from the start (learner differs from the teacher only in r). Budget split: P tokens
in all, a share of them own. Score: KL(population || generator) per chain (exact). Surface: total variation between the teacher's and the
learner's own chains restricted to a window of positions (exact, all chains enumerated), largest over the placements of the window.
Usage: python skipped_steps_pilot.py -> reports/skipped-steps-pilot.png, /var/tmp/jbauer/generative_practice_chain/skipped_steps.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

S, LINKS, R_LO, R_HI, ALPHA, WORLDS = 1024, 4, 0.2, 0.8, 0.5, 8
KINDS = dict(link=dict(gap=0, ks=[1, 2, 4], pi=[0.25, 0.25, 0.5]), far=dict(gap=2, ks=[1, 2], pi=[0.25, 0.75]), rare=dict(gap=2, ks=[1, 2], pi=[0.0625, 0.9375]))   # rare: as far, the teacher seldom shows every state
NS, OWN = [4096, 16384, 65536], 2 ** np.arange(6, 19)                                                       # teacher examples; cumulative own chains
BUDGETS, SHARES = 2 ** np.arange(14, 21, 2), [0, 0.125, 0.25, 0.5, 0.75]
WINDOWS = {"single\nstates": [(t,) for t in range(5)], "neighbours": [(t, t + 1) for t in range(4)], "2 apart": [(t, t + 2) for t in range(3)], "3 in a row": [(t, t + 1, t + 2) for t in range(3)],
           "3 apart": [(t, t + 3) for t in range(2)], "4 in a row": [(t, t + 1, t + 2, t + 3) for t in range(2)], "every 2nd\nstate": [(0, 2, 4)], "whole\nchain": [(0, 1, 2, 3, 4)]}
OUT = Path("/var/tmp/jbauer/generative_practice_chain"); FIG = Path(__file__).resolve().parents[1] / "reports/skipped-steps-pilot.png"


def chains(rng, succ, q, gap, num):                                                                         # q: the preference used (population r, flat 0.5 or the learner's r^)
    x, c = np.empty((num, LINKS + 1), int), np.empty((num, LINKS), int); x[:, 0] = rng.integers(S, size=num)
    for t in range(LINKS):
        u = rng.random(num); c[:, t] = u >= q[x[:, t]] if gap == 0 else u < 0.5 if t < gap else np.where(u < q[x[:, t - gap]], c[:, t - gap], 1 - c[:, t - gap])
        x[:, t + 1] = succ[x[:, t], c[:, t]]
    return x


def exact(succ, q, gap):                                                                                    # every chain [S * 2^LINKS, LINKS + 1] with its probability under preference q
    c = (np.arange(2 ** LINKS)[:, None] >> np.arange(LINKS)) & 1; x = np.empty((S, 2 ** LINKS, LINKS + 1), int); x[:, :, 0] = np.arange(S)[:, None]; w = np.full((S, 2 ** LINKS), 1 / S)
    for t in range(LINKS):
        w *= np.where(c[:, t] == 0, q[x[:, :, t]], 1 - q[x[:, :, t]]) if gap == 0 else 0.5 if t < gap else np.where(c[:, t] == c[:, t - gap], q[x[:, :, t - gap]], 1 - q[x[:, :, t - gap]])
        x[:, :, t + 1] = succ[x[:, :, t], c[:, t]]
    return x.reshape(-1, LINKS + 1), w.ravel()


def tv(x, wa, wb, pos): return 0.5 * np.abs(np.bincount(np.unique((x[:, pos] * S ** np.arange(len(pos))).sum(1), return_inverse=True)[1], wa - wb)).sum()
def surface(x, wa, wb): return np.array([max(tv(x, wa, wb, list(pos)) for pos in places) for places in WINDOWS.values()])


def reach(succ, n):                                                                                         # [S, 2^n]: the states n links after each state
    e = np.arange(S)[:, None]
    for _ in range(n): e = succ[e].reshape(S, -1)
    return e


class Table:                                                                                                # counts of (a, b) -> c for spans of 2h links, at every position of the chains
    def __init__(self, x, h):
        i = np.arange(LINKS + 1 - 2 * h); u, cnt = np.unique(((x[:, i] * S + x[:, i + 2 * h]) * S + x[:, i + h]).ravel(), return_counts=True)
        self.ab, self.start = np.unique(u // S, return_index=True); self.c, self.cum, self.end = u % S, np.cumsum(cnt), np.r_[self.start[1:], len(u)]
        self.ends = np.bincount(self.ab // S, minlength=S)                                                  # distinct ends on file per first state

    def draw(self, rng, a, b):                                                                              # c ~ counts, -1 where the table has no entry (or an end is already missing)
        if not len(self.ab): return np.full(len(a), -1)
        key = a * S + b; g = np.minimum(np.searchsorted(self.ab, key), len(self.ab) - 1); hit = (self.ab[g] == key) & (a >= 0) & (b >= 0)
        base = np.where(self.start[g] > 0, self.cum[self.start[g] - 1], 0); total = self.cum[self.end[g] - 1] - base
        return np.where(hit, self.c[np.minimum(np.searchsorted(self.cum, base + rng.random(len(a)) * total, side="right"), len(self.c) - 1)], -1)


def tables(x, hs): return {h: Table(x, h) for h in hs}
def phat(C): return (C[:, 0] + ALPHA) / (C.sum(1) + 2 * ALPHA)
def kl(r, q, gap): return (LINKS - gap) * (r * np.log(r / q) + (1 - r) * np.log((1 - r) / (1 - q))).mean()  # every position of a chain is uniform over states (paired successors)


def eligible(succ, tab, need, k):                                                                           # first states from which every route of k links is in the tables, at every level
    if k == 1: return np.ones(S, bool)
    e = eligible(succ, tab, need, k // 2); return (tab[k // 2].ends == need[k // 2]) & e & e[reach(succ, k // 2)].all(1)


def read(rng, succ, need, tab, x, g, kind):                                                                 # wake: counts [S, 2] of the choices in the pairs the learner reads, share of pairs read per granularity
    gap, C, share = KINDS[kind]["gap"], np.zeros((S, 2)), []
    for k in KINDS[kind]["ks"]:
        y = x[g == k]; path = np.full(y.shape, -1); path[:, ::k] = y[:, ::k]; h = k // 2
        while h >= 1:                                                                                       # top-down: midpoint of each span whose ends are in place
            for i in range(0, LINKS, 2 * h): path[:, i + h] = tab[h].draw(rng, path[:, i], path[:, i + 2 * h])
            h //= 2
        el = eligible(succ, tab, need, k)[y[:, :-1:k]]; known = np.repeat(el, k, axis=1); c = (succ[path[:, :-1], 0] != path[:, 1:]).astype(int); share.append(el.mean())
        for t in range(LINKS - gap): m = known[:, t] & known[:, t + gap]; np.add.at(C, (path[m, t], c[m, t] if gap == 0 else (c[m, t] != c[m, t + gap]).astype(int)), 1)
    return C, np.array(share)


def setup(seed, kind):
    rng = np.random.default_rng(seed); perm, s0 = rng.permutation(S), rng.permutation(S); pair = np.empty(S, int); pair[perm[0::2]], pair[perm[1::2]] = perm[1::2], perm[0::2]
    succ, r = np.stack([s0, s0[pair]], 1), rng.choice([R_LO, R_HI], size=S); r[perm[1::2]] = r[perm[0::2]]; ks, pi = np.array(KINDS[kind]["ks"]), np.array(KINDS[kind]["pi"]); hs = [int(k) // 2 for k in ks[1:]]
    need = {h: (np.diff(np.sort(reach(succ, 2 * h), 1), axis=1) != 0).sum(1) + 1 for h in hs}             # distinct ends 2h links after each state
    return rng, succ, r, np.full(S, 0.5), KINDS[kind]["gap"], ks, pi, hs, need, float((pi * (LINKS / ks + 1)).sum())


def curves(args):                                                                                           # fixed teacher set, further tokens spent on practice or on fresh teacher examples
    seed, kind = args; rng, succ, r, flat, gap, ks, pi, hs, need, tokens = setup(seed, kind); L = len(ks) - 1
    rd = lambda tab, x, g: read(rng, succ, need, tab, x, g, kind); mature = tables(chains(rng, succ, flat, gap, 2 ** 20), hs); X, wr = exact(succ, r, gap); res = {"surface": surface(X, wr, exact(succ, flat, gap)[1])}
    for n in NS:
        extra = int(OWN[-1] * (LINKS + 1) / tokens); x = chains(rng, succ, r, gap, n + extra); g = rng.choice(ks, size=n + extra, p=pi); shown = x[g == 1]; nf = np.cumsum(g == 1)   # shown[:nf[i - 1]]: fully shown among the first i examples
        Cbare, C0, Call = (rd(tab, x[:n], g[:n])[0] for tab in (tables(x[:0], hs), tables(shown[:nf[n - 1]], hs), mature)); own = {mode: x[:0] for mode in ("flat", "own")}; q, rows = phat(C0), []
        for d in OWN:
            row = [d * (LINKS + 1)]
            for mode in ("flat", "own"):
                own[mode] = np.r_[own[mode], chains(rng, succ, flat if mode == "flat" else q, gap, d - len(own[mode]))]   # sleep
                C, share = rd(tables(np.r_[own[mode], shown[:nf[n - 1]]], hs), x[:n], g[:n]); row += [kl(r, phat(C), gap), *share[1:]]   # wake: the stored teacher set read again
                if mode == "own": q = phat(C)
            m = n + int(d * (LINKS + 1) / tokens); C, share = rd(tables(shown[:nf[m - 1]], hs), x[:m], g[:m])                     # the same tokens as fresh teacher examples, no practice
            rows.append(row + [kl(r, phat(C), gap), *share[1:], kl(r, phat(rd(mature, x[:m], g[:m])[0]), gap)])
        res[n] = np.array(rows); res[n, "base"] = np.array([kl(r, phat(Cbare), gap), kl(r, phat(C0), gap), kl(r, phat(Call), gap), kl(r, flat, gap)]); res[n, "surface"] = surface(X, wr, exact(succ, phat(C0), gap)[1])
    return res


def split(args):                                                                                            # P tokens in all, a share of them own (flat dreams), the rest teacher examples
    seed, kind = args; rng, succ, r, flat, gap, ks, pi, hs, need, tokens = setup(seed + 1000, kind); out = np.zeros((len(BUDGETS), len(SHARES)))
    for i, P in enumerate(BUDGETS):
        for j, f in enumerate(SHARES):
            n, d = int((1 - f) * P / tokens), int(f * P / (LINKS + 1)); x = chains(rng, succ, r, gap, n); g = rng.choice(ks, size=n, p=pi)
            out[i, j] = kl(r, phat(read(rng, succ, need, tables(np.r_[chains(rng, succ, flat, gap, d), x[g == 1]], hs), x, g, kind)[0]), gap)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True); jobs = [(seed, kind) for kind in KINDS for seed in range(WORLDS)]
    with Pool(len(jobs)) as pool: res, sp = pool.map(curves, jobs), pool.map(split, jobs)
    res = {kind: res[i * WORLDS:(i + 1) * WORLDS] for i, kind in enumerate(KINDS)}; sp = {kind: np.array(sp[i * WORLDS:(i + 1) * WORLDS]) for i, kind in enumerate(KINDS)}; med = lambda kind, key: np.median([r[key] for r in res[kind]], 0)
    np.savez(OUT / "skipped_steps.npz", **{f"{kind}_split": sp[kind] for kind in KINDS}, **{f"{kind}_surface": np.array([r["surface"] for r in res[kind]]) for kind in KINDS},
             **{f"{kind}_{name}{n}": np.array([r[key] for r in res[kind]]) for kind in KINDS for n in NS for name, key in [("n", n), ("base", (n, "base")), ("surface", (n, "surface"))]})
    names = [w.replace("\n", " ") for w in WINDOWS]
    for kind in KINDS:
        ks = KINDS[kind]["ks"]; L = len(ks) - 1; cols = dict(flat=1, own=2 + L, fresh=3 + 2 * L, mature_fresh=4 + 3 * L); tokens = float((np.array(KINDS[kind]["pi"]) * (LINKS / np.array(ks) + 1)).sum())
        print(f"\n=== population kind '{kind}': granularities {ks} with probabilities {KINDS[kind]['pi']}, {tokens:.2f} tokens per teacher example ===")
        print("surface, total variation teacher vs own chains by window: " + " | ".join(names)); print("   flat walk (own samples before any example):  " + "  ".join(f"{v:.4f}" for v in med(kind, "surface")))
        for n in NS: print(f"   own samples after {n:6d} examples, no practice: " + "  ".join(f"{v:.4f}" for v in med(kind, (n, "surface"))))
        for n in NS:
            c, b = med(kind, n), med(kind, (n, "base")); print(f"teacher examples {n} ({n * tokens:.0f} tokens): KL per chain before any example {b[3]:.3f}, only fully shown examples counted {b[0]:.4f}, no practice {b[1]:.4f}, every example read {b[2]:.4f}")
            print("   further tokens | practice: flat dreams, own dreams | fresh instead | control: fresh | pairs read after practice (flat), granularity " + ", ".join(map(str, ks[1:])) + " | pairs read with fresh instead")
            for row in c: print(f"   {row[0]:13.0f} | {row[cols['flat']]:.4f}  {row[cols['own']]:.4f} | {row[cols['fresh']]:.4f} | {row[cols['mature_fresh']]:.4f} | " + "  ".join(f"{v:.2f}" for v in row[2:2 + L]) + " | " + "  ".join(f"{v:.2f}" for v in row[4 + 2 * L:4 + 3 * L]))
            for key in ("flat", "own"): print(f"   share of worlds in which practice ({key}) is below fresh, per row: {np.round(np.mean([r[n][:, cols[key]] < r[n][:, cols['fresh']] for r in res[kind]], 0), 2)}")
        print("budget split, KL per chain (median); rows = total tokens, columns = own share " + str(SHARES))
        for P, row, win in zip(BUDGETS, np.median(sp[kind], 0), (sp[kind][:, :, 1:] < sp[kind][:, :, :1]).mean(0)): print(f"   {P:8d} | " + "  ".join(f"{v:.4f}" for v in row) + f" | worlds where the split beats all-teacher: {np.round(win, 2)}")

    n = NS[1]; fig, ax = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
    for kind, key, ls, lab in [("link", "surface", "--", "link preference: flat dreams (first pilot)"), ("link", (n, "surface"), ":", f"link preference: own samples after {n:,} examples"), ("far", "surface", "-", "far preference: own samples before any example"), ("far", (n, "surface"), "-.", f"far preference: own samples after {n:,} examples")]:
        ax[0].plot(range(len(WINDOWS)), med(kind, key), "o", ls=ls, label=lab)
    ax[0].set(xticks=range(len(WINDOWS)), xlabel="what a test looks at", ylabel="total variation, teacher vs own chains", title="alike on the surface?"); ax[0].set_xticklabels(list(WINDOWS), rotation=60)
    for a, kind in zip(ax[1:3], ("far", "rare")):
        L = len(KINDS[kind]["ks"]) - 1; cols = dict(flat=1, own=2 + L, fresh=3 + 2 * L, mature_fresh=4 + 3 * L); c, b = med(kind, n), med(kind, (n, "base"))
        for key, col, ls, lab in [("flat", "C0", "-", "practice, flat dreams"), ("own", "C2", "-", "practice, dreams with the current preference"), ("fresh", "C1", "-", "fresh teacher examples instead"), ("mature_fresh", "C1", "--", "control (span tables complete): fresh")]:
            a.plot(c[:, 0], c[:, cols[key]], color=col, ls=ls, label=lab)
        a.axhline(b[2], color="C0", ls="--", label="control: practice (= every example read)"); a.axhline(b[1], color="gray", ls=":", label="no practice")
        a.set(xscale="log", yscale="log", xlabel="further tokens (own or fresh)", ylabel="KL(population || generator) per chain", title=f"{n:,} stored examples, 1 in {round(1 / KINDS[kind]['pi'][0])} fully shown")
    for P, row in zip(BUDGETS, np.median(sp["rare"], 0)): ax[3].plot(SHARES, row, "o-", label=f"{P:,} tokens in all")
    ax[3].set(yscale="log", xlabel="share of the tokens that are own", ylabel="KL(population || generator) per chain", title="fixed budget, 1 in 16 fully shown")
    for a in ax: a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


if __name__ == "__main__": main()
