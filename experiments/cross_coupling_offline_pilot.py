"""Pilot: does the writer of cross_coupling_cycle_pilot.py improve offline, i.e. on own samples alone, once Watching is over?
Same world, writer, reader and stage 1 (N0 teacher strings, u unnamed) as there; in the naming loop of that pilot the writer changes on teacher strings
only (own samples train the naming layer), so nothing there says whether practice alone moves the writer.

New step on an own sample:
  accepted, re-read   if the reader accepts the whole sample (every production, and the cut pair is in the cut memory), the writer takes a watching
                      step on it as if it were a teacher string: credited to the u it was written with ("u as written", = reinforcing accepted
                      samples) or to the u the naming layer reads off its cut ("u named"). The cut memory never learns from own samples.
  weighted            the same step times a weight, so that in expectation it is the step a teacher string would give (the reader's accepted set
                      is the target here: uniform on the valid trees). u as written: weight = the naming layer's probability of the written u /
                      the writer's probability of what it wrote (one scalar, from the rows that fired). u named: weight = 1 / the writer's
                      probability of the string, summed over u (not local; reference). Weights are divided by their running mean over accepted
                      samples (rate RATE) and capped at CAP.
Steps kept from the cycle pilot: naming (the naming layer learns the sample's u from its cut), verdict (failure-only step on every rejected row).
Arms: all fresh; own only with each rule; and one teacher string (u named) in every 2 or 16 samples. Scores are exact (enumeration): valid share,
share of the valid strings written, KL(target || writer), I(u; feature at the cut) / H(feature).
Usage: python experiments/cross_coupling_offline_pilot.py [report] -> /var/tmp/jbauer/generative_practice_reading/cross_coupling_offline_pilot.npz,
reports/cross-coupling-offline-pilot.png (about 7 minutes on 32 cores)"""

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cross_coupling_cycle_pilot import EYE, Reader, score, teacher
from cross_coupling_evict_pilot import ETA, V, World, Writer, soft

T, EVERY, WORLDS, RATE, CAP, CONDS = 60000, 1500, 32, 0.01, 5.0, [(U, N0) for N0 in (30000, 300) for U in (2, 8)]
# arm: (rule for teacher strings, one teacher string in every k samples (0: none), rule for own samples)
ARMS = {"fresh": ("unnamed", 1, None), "half own: naming + verdict": ("named", 2, "naming+verdict"), "1 in 16 fresh: naming + verdict": ("named", 16, "naming+verdict"),
        "1 in 16 fresh: naming, accepted re-read (u named), verdict": ("named", 16, "naming+keep named+verdict"), "own only: verdict": (None, 0, "verdict"),
        "own only: accepted re-read (u as written)": (None, 0, "keep written"), "own only: naming, accepted re-read (u named)": (None, 0, "naming+keep named"),
        "own only: naming, accepted re-read (u named), verdict": (None, 0, "naming+keep named+verdict"),
        "own only: naming, weighted re-read (u as written)": (None, 0, "naming+keep written+weighted"), "own only: naming, weighted re-read (u as written), verdict": (None, 0, "naming+keep written+weighted+verdict"),
        "own only: naming, weighted re-read (u named; not local)": (None, 0, "naming+keep named+weighted")}
PLOTTED = ("fresh", "half own: naming + verdict", "own only: verdict", "own only: accepted re-read (u as written)", "own only: naming, weighted re-read (u as written)",
           "own only: naming, weighted re-read (u named; not local)")
METRICS = ("share of own strings that are valid", "share of the valid strings written", "KL(target || writer), nats", "what u says about the cut")
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_offline_pilot.npz"); PNG = ROOT / "reports/cross-coupling-offline-pilot.png"


def practice(wr, rd, w, rng, how):
    a = rng.integers(V); pa = soft(wr.A2[a]); p2 = rng.choice(V * V, p=pa); b, c = p2 // V, p2 % V; pg = soft(wr.g[p2]); u = rng.choice(wr.U, p=pg)
    pL, pR = soft(wr.BL[b, u]), soft(wr.BR[c, u]); tL, tR = rng.choice(V * V, p=pL), rng.choice(V * V, p=pR); x2, x3 = tL % V, tR // V; cut = rd.seen[x2, x3]
    oks = (w.ok2[a, p2], cut, w.ok1[b, tL] and cut, w.ok1[c, tR] and cut); resp_named = soft(rd.N[x2, x3]); resp = resp_named if "named" in how else np.eye(wr.U)[u]
    if "naming" in how: rd.N[x2, x3] += ETA * (np.eye(wr.U)[u] - soft(rd.N[x2, x3]))
    if "verdict" in how:
        for row, p, k, ok in zip((wr.A2[a], wr.g[p2], wr.BL[b, u], wr.BR[c, u]), (pa, pg, pL, pR), (p2, u, tL, tR), oks):
            if not ok: row -= ETA * ((np.arange(len(row)) == k) - p)
    if "keep" in how and all(oks):
        step = ETA
        if "weighted" in how:
            raw = 1 / (pa[p2] * (pg * soft(wr.BL[b])[:, tL] * soft(wr.BR[c])[:, tR]).sum()) if "named" in how else resp_named[u] / (pa[p2] * pg[u] * pL[tL] * pR[tR])
            rd.norm = raw if rd.norm is None else rd.norm + RATE * (raw - rd.norm); step = ETA * min(raw / rd.norm, CAP)
        wr.A2[a] += step * (EYE[p2] - pa); wr.g[p2] += step * (resp - pg)
        wr.BL[b] += step * resp[:, None] * (EYE[tL] - soft(wr.BL[b])); wr.BR[c] += step * resp[:, None] * (EYE[tR] - soft(wr.BR[c]))


def run(job):
    seed, U, N0 = job; rng = np.random.default_rng(seed); w = World(rng); start, rd0 = Writer(U, rng), Reader(U)
    for t in range(N0): teacher(start, rd0, w, rng, "unnamed")
    out = []
    for arm, (th, k, oh) in ARMS.items():
        wr, rd, r = Writer(U, rng), Reader(U), np.random.default_rng(seed + 1000); wr.A2, wr.g, wr.BL, wr.BR, rd.seen, rd.norm = start.A2.copy(), start.g.copy(), start.BL.copy(), start.BR.copy(), rd0.seen.copy(), None
        curve = [score(wr, w)]
        for t in range(T):
            if k and t % k == 0: teacher(wr, rd, w, r, th)
            else: practice(wr, rd, w, r, oh)
            if (t + 1) % EVERY == 0: curve.append(score(wr, w))
        out.append(curve)
    return np.array(out)


def report():
    d = np.load(NPZ); shown = [c for c in CONDS if c[1] == CONDS[0][1]]; x = np.arange(T // EVERY + 1) * EVERY
    fig, axes = plt.subplots(len(shown), len(METRICS), sharex=True, constrained_layout=True, figsize=(6.4 * 2.8, 4.8 * 1.5))
    for U, N0 in CONDS:
        s = d[f"U{U}_N{N0}"]; print(f"U {U}, {N0} teacher strings first: start valid {np.median(s[:, 0, 0, 0]):.3f} written {np.median(s[:, 0, 0, 1]):.2f} KL {np.median(s[:, 0, 0, 2]):.3f}")
        for k, arm in enumerate(ARMS):
            e = s[:, k, -1]; best = s[:, k, :, 2].min(1)
            print(f"   {arm:62s} valid {np.median(e[:, 0]):.3f}  written {np.median(e[:, 1]):.2f}  KL {np.median(e[:, 2]):.3f} [{np.quantile(e[:, 2], 0.25):.3f}, {np.quantile(e[:, 2], 0.75):.3f}]  "
                  f"lowest KL on the way {np.median(best):.3f}  u about the cut {np.median(e[:, 3]):.2f}  KL at the end below the start in {(e[:, 2] < s[:, k, 0, 2]).sum()}/{len(e)} worlds, "
                  f"below fresh in {(e[:, 2] < s[:, 0, -1, 2]).sum()}/{len(e)}")
        if (U, N0) not in shown: continue
        for j, ax in enumerate(axes[shown.index((U, N0))]):
            for n, arm in enumerate(PLOTTED):
                y = s[:, list(ARMS).index(arm), :, j]; ax.plot(x, np.median(y, 0), color=f"C{n}", linestyle="--" if "not local" in arm else "-", label=arm); ax.fill_between(x, *np.quantile(y, [0.25, 0.75], 0), color=f"C{n}", alpha=0.15, linewidth=0)
            ax.set(ylabel=METRICS[j], title=f"spare unit with {U} values"); ax.set_yscale("log" if j == 2 else "linear")
            if (U, N0) == shown[-1]: ax.set_xlabel("own samples after Watching (fresh: teacher strings)")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False); fig.savefig(PNG, dpi=130); print(PNG)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        jobs = [(s, U, N0) for U, N0 in CONDS for s in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        np.savez(NPZ, **{f"U{U}_N{N0}": np.array([r for r, j in zip(res, jobs) if j[1:] == (U, N0)]) for U, N0 in CONDS})
    report()
