"""Pilot: is there a loop between own samples, a learned reader and teacher strings that snowballs? Same world and writer as cross_coupling_evict_pilot.py
(depth-2 cross-coupled RHM, 4 tokens, one agreement between tokens 2 and 3; softmax writer with random logits and a spare unit u handed to both children).

New here: the reader's part at the cut is learned.
  cut memory  seen[x2, x3], the token pairs met at the cut in the teacher strings so far. Verdict on an own sample: the cut is accepted iff its pair was seen.
  naming      N[x2, x3] -> u, softmax logits, learned from own samples only (cross-entropy step towards the u the sample was written with).
  The productions stay given: the reader parses teacher strings and checks own productions as before.
Steps:
  teacher string, u unnamed    the writer draws u from its own g (as in stage 1)
  teacher string, u named      the reader names it, responsibilities = softmax N[x2, x3]; child rows are updated with these weights and g learns them
  teacher string, u inferred   responsibilities = the writer's exact posterior over u (not local; reference)
  own sample, naming           the reader learns to name the sample's u from its cut
  own sample, verdict          failure-only three-factor step (v = 0 / -1) on every row whose choice the reader rejects, with the learned cut memory
Arms: N0 teacher strings (u unnamed), then T samples: all fresh, or fresh and own in alternation. Scores are exact (enumeration):
valid share, share of the valid strings written, KL(target || writer), and how much u says about the feature at the cut, I(u; f) / H(f), on own valid strings.
Usage: python experiments/cross_coupling_cycle_pilot.py [report] -> /var/tmp/jbauer/generative_practice_reading/cross_coupling_cycle_pilot.npz,
reports/cross-coupling-cycle-pilot.png (a few minutes on 32 cores)"""

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cross_coupling_evict_pilot import ETA, V, World, Writer, soft

T, EVERY, WORLDS, CONDS = 60000, 1500, 32, [(U, N0) for N0 in (30000, 300) for U in (2, 8)]
ARMS = {"fresh": ("unnamed", None), "fresh, u inferred (not local)": ("inferred", None), "half own: verdict": ("unnamed", "verdict"), "half own: naming": ("named", "naming"),
        "half own: naming + verdict": ("named", "naming+verdict")}
METRICS = ("share of own strings that are valid", "share of the valid strings written", "KL(target || writer), nats", "what u says about the cut")
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_cycle_pilot.npz"); PNG = ROOT / "reports/cross-coupling-cycle-pilot.png"
EYE = np.eye(V * V)


class Reader:
    def __init__(self, U): self.seen, self.N = np.zeros((V, V), bool), np.zeros((V, V, U))


def score(wr, w):
    a, p2, tL, tR = w.trees.T; b, c = p2 // V, p2 % V; qu = (soft(wr.A2)[a, p2] / V)[:, None] * soft(wr.g)[p2] * soft(wr.BL)[b, :, tL] * soft(wr.BR)[c, :, tR]; q = qu.sum(1); qv = q / q.sum()
    J = np.stack([qu[w.phi[tL % V] == f].sum(0) for f in range(w.phi.max() + 1)]) / q.sum(); Jf, Ju = J.sum(1, keepdims=True), J.sum(0, keepdims=True); nz = J > 0
    return q.sum(), np.exp(-(qv * np.log(qv)).sum()) / len(q), -np.log(q * len(q)).mean(), (J[nz] * np.log(J[nz] / (Jf * Ju)[nz])).sum() / -(Jf * np.log(Jf)).sum()


def teacher(wr, rd, w, rng, how):
    a, p2, tL, tR = w.trees[rng.integers(len(w.trees))]; b, c, x2, x3 = p2 // V, p2 % V, tL % V, tR // V; rd.seen[x2, x3] = True
    wr.A2[a] += ETA * (EYE[p2] - soft(wr.A2[a])); pg, pL, pR = soft(wr.g[p2]), soft(wr.BL[b]), soft(wr.BR[c])
    if how == "unnamed": resp = np.eye(wr.U)[rng.choice(wr.U, p=pg)]
    else:
        resp = soft(rd.N[x2, x3]) if how == "named" else pg * pL[:, tL] * pR[:, tR]; resp = resp / resp.sum(); wr.g[p2] += ETA * (resp - pg)
    wr.BL[b] += ETA * resp[:, None] * (EYE[tL] - pL); wr.BR[c] += ETA * resp[:, None] * (EYE[tR] - pR)


def own(wr, rd, w, rng, how):
    a = rng.integers(V); pa = soft(wr.A2[a]); p2 = rng.choice(V * V, p=pa); b, c = p2 // V, p2 % V; pg = soft(wr.g[p2]); u = rng.choice(wr.U, p=pg)
    pL, pR = soft(wr.BL[b, u]), soft(wr.BR[c, u]); tL, tR = rng.choice(V * V, p=pL), rng.choice(V * V, p=pR); x2, x3 = tL % V, tR // V
    if "naming" in how: rd.N[x2, x3] += ETA * (np.eye(wr.U)[u] - soft(rd.N[x2, x3]))
    if "verdict" in how:
        cut = rd.seen[x2, x3]
        for row, p, k, ok in ((wr.A2[a], pa, p2, w.ok2[a, p2]), (wr.g[p2], pg, u, cut), (wr.BL[b, u], pL, tL, w.ok1[b, tL] and cut), (wr.BR[c, u], pR, tR, w.ok1[c, tR] and cut)):
            if not ok: row -= ETA * ((np.arange(len(row)) == k) - p)


def run(job):
    seed, U, N0 = job; rng = np.random.default_rng(seed); w = World(rng); start, rd0 = Writer(U, rng), Reader(U)
    for t in range(N0): teacher(start, rd0, w, rng, "unnamed")
    out = []
    for arm, (th, oh) in ARMS.items():
        wr, rd, r = Writer(U, rng), Reader(U), np.random.default_rng(seed + 1000); wr.A2, wr.g, wr.BL, wr.BR, rd.seen = start.A2.copy(), start.g.copy(), start.BL.copy(), start.BR.copy(), rd0.seen.copy()
        curve = [score(wr, w)]
        for t in range(T):
            if oh is None or t % 2 == 0: teacher(wr, rd, w, r, th)
            else: own(wr, rd, w, r, oh)
            if (t + 1) % EVERY == 0: curve.append(score(wr, w))
        out.append(curve)
    return np.array(out), rd0.seen.sum() / (w.phi[:, None] == w.phi[None, :]).sum()


def report():
    d = np.load(NPZ); shown, cols = [c for c in CONDS if c[1] == CONDS[0][1]], (3, 2, 0); x = np.arange(T // EVERY + 1) * EVERY
    fig, axes = plt.subplots(len(shown), len(cols), sharex=True, constrained_layout=True, figsize=(6.4 * 2.1, 4.8 * 1.3))
    for U, N0 in CONDS:
        s = d[f"U{U}_N{N0}"]; print(f"U {U}, {N0} teacher strings first (cut memory holds {np.median(d[f'seen_U{U}_N{N0}']):.2f} of the allowed pairs): start valid {np.median(s[:, 0, 0, 0]):.3f} written "
                                    f"{np.median(s[:, 0, 0, 1]):.2f} KL {np.median(s[:, 0, 0, 2]):.3f}")
        for k, arm in enumerate(ARMS):
            e = s[:, k, -1]; half = (s[:, k, :, 3] > 0.5); t_half = np.where(half.any(1), x[half.argmax(1)], np.inf)
            print(f"   {arm:32s} valid {np.median(e[:, 0]):.3f}  written {np.median(e[:, 1]):.2f}  KL {np.median(e[:, 2]):.3f} [{np.quantile(e[:, 2], 0.25):.3f}, {np.quantile(e[:, 2], 0.75):.3f}]  "
                  f"u about the cut {np.median(e[:, 3]):.2f} (half of it after {np.median(t_half):.0f} samples)  KL below fresh in {(e[:, 2] < s[:, 0, -1, 2]).sum()}/{len(e)} worlds, never below 0.3 in {(s[:, k, :, 2].min(1) >= 0.3).sum()}")
        if (U, N0) not in shown: continue
        for ax, j in zip(axes[shown.index((U, N0))], cols):
            for k, arm in enumerate(ARMS):
                y = s[:, k, :, j]; ax.plot(x, np.median(y, 0), color=f"C{k}", linestyle="--" if "not local" in arm else "-", label=arm); ax.fill_between(x, *np.quantile(y, [0.25, 0.75], 0), color=f"C{k}", alpha=0.15, linewidth=0)
            ax.set(ylabel=METRICS[j], title=f"spare unit with {U} values"); ax.set_yscale("log" if j == 2 else "linear")
            if (U, N0) == shown[-1]: ax.set_xlabel("further samples")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=len(ARMS), frameon=False); fig.savefig(PNG, dpi=130); print(PNG)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        jobs = [(s, U, N0) for U, N0 in CONDS for s in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        np.savez(NPZ, **{f"U{U}_N{N0}": np.array([r[0] for r, j in zip(res, jobs) if j[1:] == (U, N0)]) for U, N0 in CONDS}, **{f"seen_U{U}_N{N0}": np.array([r[1] for r, j in zip(res, jobs) if j[1:] == (U, N0)]) for U, N0 in CONDS})
    report()
