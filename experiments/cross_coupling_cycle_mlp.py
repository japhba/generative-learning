"""The loop of cross_coupling_cycle_pilot.py with layers instead of tables: every map is ONE linear layer on one-hot groups followed by a softmax, no
conjunction units for the spare unit. Same world (depth-2 cross-coupled RHM, 4 tokens, one agreement between tokens 2 and 3), same arms, exact scores.

Writer (top-down), units in groups of which one is active:
  symbol a (V)            -> production (V*V), the pair of children (b, c)                    weights A2
  production (V*V)        -> spare unit u (U)                                                 weights g
  [child symbol b (V), u] -> production of the left child (V*V), its two tokens               logits LB[b] + LU[u]
  [child symbol c (V), u] -> production of the right child (V*V)                              logits RB[c] + RU[u]
  so u acts on a child by one weight vector per value, the same for every symbol (in the table version every (symbol, u) had a row of its own)
Reader (bottom-up):
  tokens -> production -> symbol: given (it parses)
  cut check: S[x2, x3], set when a teacher string shows the pair (Hebbian, between the two token groups at the cut)
  naming: [token x2 (V), token x3 (V)] -> u (U), logits N2[x2] + N3[x3], learned from own samples (target: the writer's u)
Rules: delta rule (cross-entropy step) on every layer for teacher strings, with u unnamed / named by the reader / inferred by the writer's posterior;
delta rule on the naming layer for own samples; failure-only three-factor step on the writer's layers for own samples the reader rejects.
Extra scores: total variation between own and teacher strings on single tokens, on pairs of sibling tokens, on the pair at the cut, on whole strings.
Usage: python experiments/cross_coupling_cycle_mlp.py [report] -> /var/tmp/jbauer/generative_practice_reading/cross_coupling_cycle_mlp.npz,
reports/cross-coupling-cycle-mlp.png (a few minutes on 32 cores)"""

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cross_coupling_evict_pilot import ETA, V, World, soft

T, EVERY, WORLDS, CONDS = 60000, 1500, 32, [(U, N0) for N0 in (30000, 300) for U in (2, 8)]
ARMS = {"fresh": ("unnamed", None), "fresh, u inferred (not local)": ("inferred", None), "half own: verdict": ("unnamed", "verdict"), "half own: naming": ("named", "naming"),
        "half own: naming + verdict": ("named", "naming+verdict")}
METRICS = ("share of own strings that are valid", "share of the valid strings written", "KL(target || writer), nats", "what u says about the cut", "own vs teacher, single tokens (TV)",
           "own vs teacher, sibling tokens (TV)", "own vs teacher, tokens at the cut (TV)", "own vs teacher, whole strings (TV)")
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_cycle_mlp.npz"); PNG = ROOT / "reports/cross-coupling-cycle-mlp.png"
EYE = np.eye(V * V)


class Net:
    def __init__(self, U, rng):
        self.U = U; self.A2, self.g = rng.normal(size=(V, V * V)), rng.normal(size=(V * V, U)); self.LB, self.RB = rng.normal(size=(2, V, V * V)); self.LU, self.RU = rng.normal(size=(2, U, V * V))
        self.seen, self.N2, self.N3 = np.zeros((V, V), bool), np.zeros((V, U)), np.zeros((V, U))

    def copy(self):
        n = Net.__new__(Net); n.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}; return n


def score(n, w):
    a, p2, tL, tR = w.trees.T; b, c = p2 // V, p2 % V; pa, pg, pL, pR = soft(n.A2), soft(n.g), soft(n.LB[:, None] + n.LU[None]), soft(n.RB[:, None] + n.RU[None])
    qu = (pa[a, p2] / V)[:, None] * pg[p2] * pL[b, :, tL] * pR[c, :, tR]; q = qu.sum(1); qv = q / q.sum()
    J = np.stack([qu[w.phi[tL % V] == f].sum(0) for f in range(w.phi.max() + 1)]) / q.sum(); Jf, Ju = J.sum(1, keepdims=True), J.sum(0, keepdims=True); nz = J > 0
    full = sum(pa[:, k].mean() * (pL[k // V].T * pg[k]) @ pR[k % V] for k in range(V * V)).reshape(V, V, V, V); tgt = np.zeros((V * V, V * V)); tgt[tL, tR] = 1 / len(a); tgt = tgt.reshape(V, V, V, V)
    tv = lambda axes: max(0.5 * np.abs(full.sum(ax) - tgt.sum(ax)).sum() for ax in axes)
    return (q.sum(), np.exp(-(qv * np.log(qv)).sum()) / len(q), -np.log(q * len(q)).mean(), (J[nz] * np.log(J[nz] / (Jf * Ju)[nz])).sum() / -(Jf * np.log(Jf)).sum(),
            tv([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]), tv([(2, 3), (0, 1)]), tv([(0, 3)]), tv([()]))


def teacher(n, w, rng, how):
    a, p2, tL, tR = w.trees[rng.integers(len(w.trees))]; b, c, x2, x3 = p2 // V, p2 % V, tL % V, tR // V; n.seen[x2, x3] = True
    n.A2[a] += ETA * (EYE[p2] - soft(n.A2[a])); pg, pL, pR = soft(n.g[p2]), soft(n.LB[b] + n.LU), soft(n.RB[c] + n.RU)
    if how == "unnamed": resp = np.eye(n.U)[rng.choice(n.U, p=pg)]
    else:
        resp = soft(n.N2[x2] + n.N3[x3]) if how == "named" else pg * pL[:, tL] * pR[:, tR]; resp = resp / resp.sum(); n.g[p2] += ETA * (resp - pg)
    eL, eR = ETA * resp[:, None] * (EYE[tL] - pL), ETA * resp[:, None] * (EYE[tR] - pR); n.LB[b] += eL.sum(0); n.LU += eL; n.RB[c] += eR.sum(0); n.RU += eR


def own(n, w, rng, how):
    a = rng.integers(V); pa = soft(n.A2[a]); p2 = rng.choice(V * V, p=pa); b, c = p2 // V, p2 % V; pg = soft(n.g[p2]); u = rng.choice(n.U, p=pg)
    pL, pR = soft(n.LB[b] + n.LU[u]), soft(n.RB[c] + n.RU[u]); tL, tR = rng.choice(V * V, p=pL), rng.choice(V * V, p=pR); x2, x3 = tL % V, tR // V
    if "naming" in how: e = ETA * (np.eye(n.U)[u] - soft(n.N2[x2] + n.N3[x3])); n.N2[x2] += e; n.N3[x3] += e
    if "verdict" in how:
        cut = n.seen[x2, x3]
        if not w.ok2[a, p2]: n.A2[a] -= ETA * (EYE[p2] - pa)
        if not cut: n.g[p2] -= ETA * (np.eye(n.U)[u] - pg)
        if not (w.ok1[b, tL] and cut): e = ETA * (EYE[tL] - pL); n.LB[b] -= e; n.LU[u] -= e
        if not (w.ok1[c, tR] and cut): e = ETA * (EYE[tR] - pR); n.RB[c] -= e; n.RU[u] -= e


def run(job):
    seed, U, N0 = job; rng = np.random.default_rng(seed); w = World(rng); start = Net(U, rng)
    for t in range(N0): teacher(start, w, rng, "unnamed")
    out = []
    for arm, (th, oh) in ARMS.items():
        n, r = start.copy(), np.random.default_rng(seed + 1000); curve = [score(n, w)]
        for t in range(T):
            if oh is None or t % 2 == 0: teacher(n, w, r, th)
            else: own(n, w, r, oh)
            if (t + 1) % EVERY == 0: curve.append(score(n, w))
        out.append(curve)
    return np.array(out)


def report():
    d = np.load(NPZ); shown, cols = [c for c in CONDS if c[1] == CONDS[0][1]], (3, 2, 0, 6); x = np.arange(T // EVERY + 1) * EVERY
    fig, axes = plt.subplots(len(shown), len(cols), sharex=True, constrained_layout=True, figsize=(6.4 * 2.7, 4.8 * 1.3))
    for U, N0 in CONDS:
        s = d[f"U{U}_N{N0}"]; e = s[:, 0, 0]
        print(f"U {U}, {N0} teacher strings first: start valid {np.median(e[:, 0]):.3f} written {np.median(e[:, 1]):.2f} KL {np.median(e[:, 2]):.3f}  own vs teacher (TV): single tokens {np.median(e[:, 4]):.3f} "
              f"sibling tokens {np.median(e[:, 5]):.3f} tokens at the cut {np.median(e[:, 6]):.3f} whole strings {np.median(e[:, 7]):.3f}")
        for k, arm in enumerate(ARMS):
            e = s[:, k, -1]; half = (s[:, k, :, 3] > 0.5); t_half = np.where(half.any(1), x[half.argmax(1)], np.inf)
            print(f"   {arm:32s} valid {np.median(e[:, 0]):.3f}  written {np.median(e[:, 1]):.2f}  KL {np.median(e[:, 2]):.3f} [{np.quantile(e[:, 2], 0.25):.3f}, {np.quantile(e[:, 2], 0.75):.3f}]  "
                  f"u about the cut {np.median(e[:, 3]):.2f} (half after {np.median(t_half):.0f})  KL below fresh in {(e[:, 2] < s[:, 0, -1, 2]).sum()}/{len(e)}, never below 0.3 in {(s[:, k, :, 2].min(1) >= 0.3).sum()}  "
                  f"TV single {np.median(e[:, 4]):.3f} sibling {np.median(e[:, 5]):.3f} cut {np.median(e[:, 6]):.3f} string {np.median(e[:, 7]):.3f}")
        if (U, N0) not in shown: continue
        for ax, j in zip(axes[shown.index((U, N0))], cols):
            for k, arm in enumerate(ARMS):
                y = s[:, k, :, j]; ax.plot(x, np.median(y, 0), color=f"C{k}", linestyle="--" if "not local" in arm else "-", label=arm); ax.fill_between(x, *np.quantile(y, [0.25, 0.75], 0), color=f"C{k}", alpha=0.15, linewidth=0)
            ax.set(ylabel=METRICS[j], title=f"spare unit with {U} values"); ax.set_yscale("log" if j == 2 else "linear")
            if (U, N0) == shown[-1]: ax.set_xlabel("further samples")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=len(ARMS), frameon=False); fig.savefig(PNG, dpi=110); print(PNG)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        jobs = [(s, U, N0) for U, N0 in CONDS for s in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        np.savez(NPZ, **{f"U{U}_N{N0}": np.array([r for r, j in zip(res, jobs) if j[1:] == (U, N0)]) for U, N0 in CONDS})
    report()
