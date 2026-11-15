"""Pilot: the fixed-prior writer of cross_coupling_prior_pilot.py at depth 3, with nothing handed to the reader.
World: RHM of depth 3 (root a -> two level-2 symbols -> four level-1 symbols -> eight tokens; V symbols per level, M productions per symbol), with an
agreement at every cut: the neighbouring tokens (2,3), (4,5), (6,7) share the token feature phi0, and the level-1 symbols 2 and 3 share the symbol feature
phi1. Balanced: every level-1 symbol has one production per pair of token features, every level-2 symbol's productions have balanced phi1 at both
places. Target: uniform on the valid derivations (8192). A teacher sample is a derivation (symbols visible).

Writer (top-down, one pass): every choice is made by a latent unit with a fixed uniform prior, only what a unit's value writes is learned.
  root      k3 (M values)                                D3[a, k3]                -> production of a
  spares    u3 at the root (RHO^2 values, handed to all descendants), u2 at each level-2 node (RHO values, handed to its two children)
  level 2   k2 (M / RHO values)                          D2[side][b, u3, k2]      -> production of the level-2 symbol
  level 1   j (M / RHO values)                           D1[place][y, u2, u3, j]  -> production of the level-1 symbol
In the balanced world this prior is exact (as many codes as valid derivations, up to codes that must write the same at the two inner level-1 nodes).
Two forms of the writer: "tables" (one row of logits per joint code, as above) and "layers" (one linear layer per node on one-hot groups with a softmax:
the logits are a sum of one weight vector per unit value, D2 = B[b, k2] + Bu[u3], D1 = C[y, j] + Cu2[u2] + Cu3[u3]; a spare unit's weights are shared by
all symbols at that place). Both hold the exact writer.
Reader, all blank at the start:
  checker   memories of what occurred in teacher samples: productions per level, token pairs at a cut, level-1 symbol pairs at the root's cut
  naming    one layer per unit of the writer, learned from own samples only: k from (symbol, production), u2 from its cut's two tokens,
            u3 from the root cut's two tokens plus its two level-1 symbols (two tables, logits added)
Steps:
  teacher sample, unnamed / named / inferred   codes from the prior / read off by the naming layers / the writer's exact posterior (not local)
  own sample, naming                           the naming layers learn the codes the sample was written with
  own sample, accepted if legible              a row whose production and adjacent cuts the checker accepts steps towards what it wrote, times the naming
                                               layers' probability of the codes that address the row (in units of chance, capped at CAP)
  ... per try                                  the same step divided by the row's probability of what it wrote (rate ETA_TRY, factor capped at CAP_TRY): in
                                               expectation every option that is accepted and legible is pushed equally often, however often it is written
Scores are exact (enumeration): valid share, share of the valid derivations written, KL(target || writer), what u2 says about its cut and u3 about the
root's two agreements on own valid samples (u2 also within each value of u3, which does not require one convention for all u3), and the same for the names
the reader gives teacher samples.
Usage: python experiments/cross_coupling_depth3_pilot.py [report] -> /var/tmp/jbauer/generative_practice_reading/cross_coupling_depth3_pilot.npz,
reports/cross-coupling-depth3-pilot-{layers,tables}.png (about 22 minutes on 32 cores)"""

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

V, M, RHO, ETA, CAP, ETA_TRY, CAP_TRY, N0, T, EVERY, WORLDS, CONDS = 8, 4, 2, 0.2, 4.0, 0.05, 20.0, 30000, 150000, 3000, 32, ("layers", "tables")
U3, U2, K3, K2, J, EYE = RHO * RHO, RHO, M, M // RHO, M // RHO, np.eye(V * V)
# arm: (rule for teacher samples, one teacher sample in every k samples (0: none), rule for own samples)
ARMS = {"fresh": ("unnamed", 1, None), "fresh, codes inferred (not local)": ("inferred", 1, None), "half own: naming": ("named", 2, "naming"),
        "half own: naming, accepted if legible": ("named", 2, "naming+keep legible"), "1 in 16 fresh: naming, accepted if legible": ("named", 16, "naming+keep legible"),
        "own only: naming, accepted if legible": (None, 0, "naming+keep legible"), "half own: naming, accepted if legible, per try": ("named", 2, "naming+keep legible per try"),
        "1 in 16 fresh: naming, accepted if legible, per try": ("named", 16, "naming+keep legible per try"), "own only: naming, accepted if legible, per try": (None, 0, "naming+keep legible per try")}
PLOTTED = [arm for arm in ARMS if "legible" not in arm or "per try" in arm]
METRICS = ("share of own samples that are valid", "share of the valid derivations written", "KL(target || writer), nats", "u3 about the root's token agreement", "u3 about the root's symbol agreement",
           "u2 about its cut (left)", "u2 about its cut (right)", "teacher samples: name of u3 about the token agreement", "teacher samples: name of u3 about the symbol agreement",
           "teacher samples: name of u2 about its cut", "u2 about its cut within each value of u3 (left)", "u2 about its cut within each value of u3 (right)")
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_depth3_pilot.npz"); PNG = ROOT / "reports/cross-coupling-depth3-pilot.png"


def soft(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def onehot(i, n): return np.arange(n) == i


def mi(qu, f):
    Jt = np.stack([qu[f == v].sum(0) for v in range(RHO)]); Jt = Jt / Jt.sum(); Jf, Ju = Jt.sum(1, keepdims=True), Jt.sum(0, keepdims=True); nz = Jt > 0
    return (Jt[nz] * np.log(Jt[nz] / (Jf * Ju)[nz])).sum() / -(Jf * np.log(Jf)).sum()


class World:
    def __init__(self, rng):
        self.phi0, self.phi1 = (rng.permutation(np.arange(V) % RHO) for _ in range(2)); self.R3 = rng.permutation(V * V)[:V * M].reshape(V, M)
        self.R2 = self.rules(rng, lambda r: all((np.bincount(self.phi1[t], minlength=RHO) == M // RHO).all() for t in (r // V, r % V)))
        self.R1 = self.rules(rng, lambda r: len(set(zip(self.phi0[r // V], self.phi0[r % V]))) == M)
        trees = [(a, p3, pL, pR, t1, t2, t3, t4) for a in range(V) for p3 in self.R3[a] for pL in self.R2[p3 // V] for pR in self.R2[p3 % V] if self.phi1[pL % V] == self.phi1[pR // V]
                 for t1 in self.R1[pL // V] for t2 in self.R1[pL % V] if self.phi0[t1 % V] == self.phi0[t2 // V] for t3 in self.R1[pR // V] if self.phi0[t2 % V] == self.phi0[t3 // V]
                 for t4 in self.R1[pR % V] if self.phi0[t3 % V] == self.phi0[t4 // V]]
        self.trees = np.array(trees); a, p3, pL, pR, t1, t2, t3, t4 = self.trees.T
        self.f0, self.f1, self.fL, self.fR = self.phi0[t2 % V], self.phi1[pL % V], self.phi0[t1 % V], self.phi0[t3 % V]   # features at the root's cut (token, symbol) and at the two inner cuts

    def rules(self, rng, ok):   # every symbol: M unused pairs that satisfy the balance condition
        free, rows = list(rng.permutation(V * V)), []
        for s in range(V):
            while True:
                r = rng.choice(free, M, replace=False)
                if ok(r): break
            rows.append(r); free = [f for f in free if f not in r]
        return np.array(rows)


class Writer:   # tables
    def __init__(self, rng): self.D3, self.D2, self.D1 = rng.normal(size=(V, K3, V * V)), rng.normal(size=(2, V, U3, K2, V * V)), rng.normal(size=(4, V, U2, U3, J, V * V))

    def copy(self): new = type(self).__new__(type(self)); new.__dict__ = {k: v.copy() for k, v in self.__dict__.items()}; return new

    def full(self): return self.D2, self.D1   # logits for every joint code: (side, symbol, u3, k2, .), (place, symbol, u2, u3, j, .)

    def rows2(self, s, sym): return self.D2[s, sym]

    def rows1(self, i, y): return self.D1[i, y]

    def add2(self, s, sym, G): self.D2[s, sym] += G   # G: one step per joint code, (u3, k2, .)

    def add1(self, i, y, G): self.D1[i, y] += G   # (u2, u3, j, .)


class Layers(Writer):
    def __init__(self, rng):
        self.D3, self.B, self.Bu = rng.normal(size=(V, K3, V * V)), rng.normal(size=(2, V, K2, V * V)), rng.normal(size=(2, U3, V * V))
        self.C, self.Cu2, self.Cu3 = rng.normal(size=(4, V, J, V * V)), rng.normal(size=(4, U2, V * V)), rng.normal(size=(4, U3, V * V))

    def full(self): return self.Bu[:, None, :, None] + self.B[:, :, None], self.Cu2[:, None, :, None, None] + self.Cu3[:, None, None, :, None] + self.C[:, :, None, None]

    def rows2(self, s, sym): return self.Bu[s][:, None] + self.B[s, sym][None]

    def rows1(self, i, y): return self.Cu2[i][:, None, None] + self.Cu3[i][None, :, None] + self.C[i, y][None, None]

    def add2(self, s, sym, G): self.Bu[s] += G.sum(1); self.B[s, sym] += G.sum(0)

    def add1(self, i, y, G): self.Cu2[i] += G.sum((1, 2)); self.Cu3[i] += G.sum((0, 2)); self.C[i, y] += G.sum((0, 1))


class Reader:
    def __init__(self):
        self.prod3, self.prod2, self.prod1, self.cut0, self.cut1 = (np.zeros(s, bool) for s in ((V, V * V),) * 3 + ((V, V),) * 2)
        self.k3, self.k2, self.j, self.u2, self.u3 = np.zeros((V, V * V, K3)), np.zeros((2, V, V * V, K2)), np.zeros((4, V, V * V, J)), np.zeros((2, V, V, U2)), np.zeros((2, V, V, U3))

    def names(self, tree):   # per unit of the writer [k3, u3, k2 left, k2 right, u2 left, u2 right, j x 4]: the rows of logits that are added to read it
        a, p3, pL, pR, *ts = tree; ys = (pL // V, pL % V, pR // V, pR % V)
        return ([self.k3[a, p3]], [self.u3[0, ts[1] % V, ts[2] // V], self.u3[1, ys[1], ys[2]]], [self.k2[0, p3 // V, pL]], [self.k2[1, p3 % V, pR]], [self.u2[0, ts[0] % V, ts[1] // V]],
                [self.u2[1, ts[2] % V, ts[3] // V]], *([self.j[i, ys[i], ts[i]]] for i in range(4)))

    def see(self, tree):
        a, p3, pL, pR, *ts = tree; ys = (pL // V, pL % V, pR // V, pR % V); self.prod3[a, p3] = self.prod2[p3 // V, pL] = self.prod2[p3 % V, pR] = self.cut1[ys[1], ys[2]] = True
        for i in range(4): self.prod1[ys[i], ts[i]] = True
        for i in range(3): self.cut0[ts[i] % V, ts[i + 1] // V] = True

    def verdicts(self, tree):   # one per row that wrote the sample [root, level 2 left / right, level 1 x 4]: its production and the cuts next to what it wrote
        a, p3, pL, pR, *ts = tree; ys = (pL // V, pL % V, pR // V, pR % V); c1 = self.cut1[ys[1], ys[2]]; c0 = [True] + [self.cut0[ts[i] % V, ts[i + 1] // V] for i in range(3)] + [True]
        return [self.prod3[a, p3], self.prod2[p3 // V, pL] and c1, self.prod2[p3 % V, pR] and c1] + [self.prod1[ys[i], ts[i]] and c0[i] and c0[i + 1] for i in range(4)]


def score(wr, rd, w):
    a, p3, pL, pR, *ts = w.trees.T; ys = (pL // V, pL % V, pR // V, pR % V); F2, F1 = wr.full(); A3, B2, C1 = soft(wr.D3).mean(1), soft(F2).mean(3), soft(F1).mean(4)
    leaf = [C1[i][ys[i], :, :, ts[i]].transpose(0, 2, 1) for i in range(4)]   # (derivation, u3, u2)
    q = (A3[a, p3] / V / U3)[:, None, None, None] * (B2[0][p3 // V, :, pL] * B2[1][p3 % V, :, pR])[:, :, None, None] * (leaf[0] * leaf[1] / U2)[:, :, :, None] * (leaf[2] * leaf[3] / U2)[:, :, None, :]
    qt = q.sum((1, 2, 3)); qv = qt / qt.sum(); n3, n2 = soft(rd.u3[0][ts[1] % V, ts[2] // V] + rd.u3[1][ys[1], ys[2]]), [soft(rd.u2[0][ts[0] % V, ts[1] // V]), soft(rd.u2[1][ts[2] % V, ts[3] // V])]
    return (qt.sum(), np.exp(-(qv * np.log(qv)).sum()) / len(qt), -np.log(qt * len(qt)).mean(), mi(q.sum((2, 3)), w.f0), mi(q.sum((2, 3)), w.f1), mi(q.sum((1, 3)), w.fL), mi(q.sum((1, 2)), w.fR),
            mi(n3, w.f0), mi(n3, w.f1), (mi(n2[0], w.fL) + mi(n2[1], w.fR)) / 2, *(sum(x[:, v].sum() / qt.sum() * mi(x[:, v], f) for v in range(U3)) for x, f in ((q.sum(3), w.fL), (q.sum(2), w.fR))))


def teacher(wr, rd, w, rng, how):
    tree = w.trees[rng.integers(len(w.trees))]; rd.see(tree); a, p3, pL, pR, *ts = tree; ys = (pL // V, pL % V, pR // V, pR % V)
    P3, P2, P1 = soft(wr.D3[a]), [soft(wr.rows2(0, p3 // V)), soft(wr.rows2(1, p3 % V))], [soft(wr.rows1(i, ys[i])) for i in range(4)]   # (k3, .), (u3, k2, .), (u2, u3, j, .)
    if how == "inferred":
        l2, l1 = [P2[s][:, :, p] for s, p in enumerate((pL, pR))], [P1[i][:, :, :, ts[i]] for i in range(4)]; m2, m1 = [x.sum(1) for x in l2], [x.sum(2) for x in l1]
        joint = (m2[0] * m2[1])[:, None, None] * (m1[0] * m1[1]).T[:, :, None] * (m1[2] * m1[3]).T[:, None, :]; joint = joint / joint.sum(); side = [joint.sum(2).T, joint.sum(1).T]   # (u2, u3)
        rk3, W2, W1 = P3[:, p3] / P3[:, p3].sum(), [joint.sum((1, 2))[:, None] * l2[s] / m2[s][:, None] for s in range(2)], [side[i // 2][:, :, None] * l1[i] / m1[i][:, :, None] for i in range(4)]
    else:
        sizes = (K3, U3, K2, K2, U2, U2, J, J, J, J); r = [soft(sum(parts)) for parts in rd.names(tree)] if how == "named" else [onehot(rng.integers(n), n) for n in sizes]
        rk3, W2, W1 = r[0], [r[1][:, None] * r[2 + s][None, :] for s in range(2)], [r[4 + i // 2][:, None, None] * r[1][None, :, None] * r[6 + i][None, None, :] for i in range(4)]
    wr.D3[a] += ETA * rk3[:, None] * (EYE[p3] - P3)
    for s, (sym, p) in enumerate(((p3 // V, pL), (p3 % V, pR))): wr.add2(s, sym, ETA * W2[s][:, :, None] * (EYE[p] - P2[s]))
    for i in range(4): wr.add1(i, ys[i], ETA * W1[i][:, :, :, None] * (EYE[ts[i]] - P1[i]))


def practice(wr, rd, w, rng, how):
    a, k3, u3 = (int(rng.integers(n)) for n in (V, K3, U3)); k2, u2, j = rng.integers(K2, size=2).tolist(), rng.integers(U2, size=2).tolist(), rng.integers(J, size=4).tolist()
    probs = [soft(wr.D3[a, k3])]; p3 = rng.choice(V * V, p=probs[0]); probs += [soft(wr.rows2(s, sym)[u3, k2[s]]) for s, sym in enumerate((p3 // V, p3 % V))]; pL, pR = (rng.choice(V * V, p=p) for p in probs[1:])
    ys = (pL // V, pL % V, pR // V, pR % V); probs += [soft(wr.rows1(i, ys[i])[u2[i // 2], u3, j[i]]) for i in range(4)]; ts = [rng.choice(V * V, p=p) for p in probs[3:]]; tree = (a, p3, pL, pR, *ts)
    names, codes = rd.names(tree), (k3, u3, *k2, *u2, *j); q = [soft(sum(parts))[code] * len(parts[0]) for parts, code in zip(names, codes)]   # in units of chance
    legible = [q[0], q[1] * q[2], q[1] * q[3]] + [q[1] * q[4 + i // 2] * q[6 + i] for i in range(4)]
    if "naming" in how:
        for parts, code in zip(names, codes):
            step = ETA * (onehot(code, len(parts[0])) - soft(sum(parts)))
            for part in parts: part += step
    def size(p, leg): return ETA_TRY * min(leg / p, CAP_TRY) if "per try" in how else ETA * (min(leg, CAP) if "legible" in how else 1)

    steps = [size(p[wrote], leg) * (EYE[wrote] - p) if "keep" in how and ok else None for p, wrote, ok, leg in zip(probs, (p3, pL, pR, *ts), rd.verdicts(tree), legible)]
    if steps[0] is not None: wr.D3[a, k3] += steps[0]
    for s, sym in enumerate((p3 // V, p3 % V)):
        if steps[1 + s] is not None: G = np.zeros((U3, K2, V * V)); G[u3, k2[s]] = steps[1 + s]; wr.add2(s, sym, G)
    for i in range(4):
        if steps[3 + i] is not None: G = np.zeros((U2, U3, J, V * V)); G[u2[i // 2], u3, j[i]] = steps[3 + i]; wr.add1(i, ys[i], G)


def exact_writer(w):   # the writer that the fixed prior allows in the balanced world (test of the scores)
    wr = Writer(np.random.default_rng(0)); wr.D3[:], wr.D2[:], wr.D1[:] = -30, -30, -30
    for s in range(V):
        wr.D3[s, np.arange(K3), w.R3[s]] = 0
        for u3 in range(U3):
            f1, f0 = u3 // RHO, u3 % RHO
            for side, place in ((0, w.R2[s] % V), (1, w.R2[s] // V)): wr.D2[side, s, u3, np.arange(K2), w.R2[s][w.phi1[place] == f1]] = 0
            for u2 in range(U2):
                first, second = w.phi0[w.R1[s] // V], w.phi0[w.R1[s] % V]
                for i, pick in enumerate((second == u2, (first == u2) & (second == f0), (first == f0) & (second == u2), first == u2)): wr.D1[i, s, u2, u3, :, w.R1[s][pick]] = 0
    return wr


def run(job):
    seed, cond = job; rng = np.random.default_rng(seed); w = World(rng); start, rd0 = (Layers if cond == "layers" else Writer)(rng), Reader()
    for t in range(N0): teacher(start, rd0, w, rng, "unnamed")
    out = []
    for arm, (th, k, oh) in ARMS.items():
        wr, rd, r = start.copy(), Reader(), np.random.default_rng(seed + 1000); rd.prod3, rd.prod2, rd.prod1, rd.cut0, rd.cut1 = (m.copy() for m in (rd0.prod3, rd0.prod2, rd0.prod1, rd0.cut0, rd0.cut1))
        curve = [score(wr, rd, w)]
        for t in range(T):
            if k and t % k == 0: teacher(wr, rd, w, r, th)
            else: practice(wr, rd, w, r, oh)
            if (t + 1) % EVERY == 0: curve.append(score(wr, rd, w))
        out.append(curve)
    return np.array(out)


def report():
    d = np.load(NPZ); x = np.arange(T // EVERY + 1) * EVERY; shown = (0, 1, 2, 10, 3, 4)
    for cond in CONDS:
        s = d[cond]; fig, axes = plt.subplots(2, 3, sharex=True, constrained_layout=True, figsize=(6.4 * 2.2, 4.8 * 1.6))
        print(f"{cond}: start valid {np.median(s[:, 0, 0, 0]):.3f} written {np.median(s[:, 0, 0, 1]):.2f} KL {np.median(s[:, 0, 0, 2]):.3f}")

        def half(k, j): h = s[:, k, :, j] > 0.5; t = np.where(h.any(1), x[h.argmax(1)], np.inf); return f"{np.median(t):.0f} ({np.isinf(t).sum()} never)"

        for k, arm in enumerate(ARMS):
            e = s[:, k, -1]
            print(f"   {arm:52s} valid {np.median(e[:, 0]):.3f}  written {np.median(e[:, 1]):.2f} [{np.quantile(e[:, 1], 0.25):.2f}, {np.quantile(e[:, 1], 0.75):.2f}]  KL {np.median(e[:, 2]):.3f} "
                  f"[{np.quantile(e[:, 2], 0.25):.3f}, {np.quantile(e[:, 2], 0.75):.3f}], below 0.2 in {(e[:, 2] < 0.2).sum()}, below the start in {(e[:, 2] < s[:, k, 0, 2]).sum()} of {len(e)}\n"
                  f"        at the end, u3 token / symbol {np.median(e[:, 3]):.2f} / {np.median(e[:, 4]):.2f}, u2 {np.median(e[:, 5:7].mean(1)):.2f} (within u3: {np.median(e[:, 10:12].mean(1)):.2f}); "
                  f"teacher samples read: u3 {np.median(e[:, 7]):.2f} / {np.median(e[:, 8]):.2f}, u2 {np.median(e[:, 9]):.2f}\n"
                  f"        half-way after: u2 within u3, left {half(k, 10)}, right {half(k, 11)}; u3 token {half(k, 3)}, u3 symbol {half(k, 4)}")
        for ax, j in zip(axes.flat, shown):
            for n, arm in enumerate(PLOTTED):
                y = s[:, list(ARMS).index(arm), :, j]; ax.plot(x, np.median(y, 0), color=f"C{n}", linestyle="--" if "not local" in arm else "-", label=arm); ax.fill_between(x, *np.quantile(y, [0.25, 0.75], 0), color=f"C{n}", alpha=0.15, linewidth=0)
            ax.set(ylabel=METRICS[j]); ax.set_yscale("log" if j == 2 else "linear")
        for ax in axes[-1]: ax.set_xlabel("samples after Watching")
        handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False); fig.suptitle(f"writer as {cond}")
        png = PNG.with_name(f"{PNG.stem}-{cond}.png"); fig.savefig(png, dpi=130); print(png)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        jobs = [(seed, cond) for cond in CONDS for seed in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        np.savez(NPZ, **{cond: np.array([r for r, j in zip(res, jobs) if j[1] == cond]) for cond in CONDS})
    report()
