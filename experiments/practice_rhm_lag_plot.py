"""Figure: in the tree Helmholtz machine on the RHM, the learner's code for the height-(h-1) symbols becomes informative before
its likelihood captures level h, for every way of training recognition (wake-sleep, tied, exact posterior, joint ELBO).
Reading at height h: I(u; a) / H(a), the learner's sampled code u against the true symbol a, as progress from step 0 to 1.
Level-h generation: pi_h = clip((log N_(h-1) - NLL) / (log N_(h-1) - log N_h), 0, 1) from the exact NLL, N_h = number of strings
valid up to height h. Onset: first evaluation at which a seed's progress reaches 1/2.
Usage: python practice_rhm_lag_plot.py -> reports/rhm-helmholtz-lag.png"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/var/tmp/jbauer/generative_practice_rhm"
ARMS = [("practice", "fine_k2_L{L}_practice_", "wake-sleep\n(separate recognition)", "#2a78d6"),
        ("watch_tied", "fine_k2_L{L}_watch_tied_", "tied\n(recognition = generator's transpose)", "#1baf7a"),
        ("watch_exact", "fine_k2_L{L}_watch_exact_", "exact posterior\n(stochastic EM)", "#eb6834"),
        ("elbo", "elbo_k2_L{L}_", "joint ELBO\n(encoder and decoder by one gradient)", "#8a4fd8")]
LEVEL = ["#9ec5f0", "#2a78d6", "#0b3a75", "#06203f"]  # light to dark with height


def load(prefix, key):
    f = sorted(glob.glob(f"{ROOT}/{prefix}2*/results.json"))[-1]; d = json.load(open(f)); a = d["args"]; v, m, s, nc, L = a["v"], a["m"], a["s"], a["nc"], a["L"]
    logN = [s ** L * np.log(v)] + [s ** (L - h) * (np.log(v if h < L else nc) + (s ** h - 1) // (s - 1) * np.log(m)) for h in range(1, L + 1)]
    curves = d["results"][key]; t = np.array([c["t"] for c in curves[0]]); out = []
    for cv in curves:
        nll = np.array([c["kl"] for c in cv]) + logN[-1]
        gen = np.array([np.clip((logN[h - 1] - nll) / (logN[h - 1] - logN[h]), 0, 1) for h in range(1, L + 1)])
        q = np.array([c["nmi"] for c in cv]).T; rd = (q - q[:, :1]) / (1 - q[:, :1])
        out.append((gen, rd))
    return t, out, L


def onset(t, p): i = np.flatnonzero(p >= 0.5); return t[i[0]] if len(i) else np.nan


def main():
    fig = plt.figure(figsize=(19, 8.2), constrained_layout=True); gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.25])
    sc = fig.add_subplot(gs[:, 4]); lags = {}
    for row, L in enumerate([3, 4]):
        for col, (key, prefix, label, colour) in enumerate(ARMS):
            ax = fig.add_subplot(gs[row, col]); t, runs, L = load(prefix.format(L=L), key); sel = t > 0
            for h in range(1, L + 1):
                G = np.array([g[h - 1] for g, _ in runs]); ax.plot(t[sel], np.median(G, 0)[sel], "-", color=LEVEL[h - 1], label=f"likelihood has level {h}")
                if h >= 2:
                    Rd = np.array([r[h - 2] for _, r in runs]); ax.plot(t[sel], np.median(Rd, 0)[sel], "--", color=LEVEL[h - 1], label=f"code reads height {h - 1}")
                    on = [(onset(t, r[h - 2]), onset(t, g[h - 1])) for g, r in runs]
                    print(f"{key:12s} depth {L} level {h}: reads height {h - 1} at {np.nanmedian([x for x, _ in on]):7.0f}, likelihood has level {h} at {np.nanmedian([y for _, y in on]):7.0f}")
                    for x, y in on: sc.scatter(x, y, color=colour, marker="o" if L == 3 else "s", s=40, alpha=0.8, edgecolor="white")
                    lags.setdefault(key, []).extend([y / x for x, y in on if np.isfinite(x) and np.isfinite(y)])
            ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05); ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
            if row == 0: ax.set_title(label, loc="left", fontsize="medium")
            if col == 0: ax.set_ylabel(f"depth {L}: progress\n(median of {len(runs)} seeds)")
            if row == 1: ax.set_xlabel("updates")
            if col == 0 and row == 1: ax.legend(loc="upper left", fontsize="small")
    lo, hi = 400, 60000; xs = np.array([lo, hi]); sc.plot(xs, xs, color="0.4", lw=1); sc.plot(xs, 3 * xs, color="0.6", lw=1, ls="--")
    sc.text(2.5e4, 1.6e4, "same time", ha="center", color="0.4"); sc.text(1.1e4, 4.4e4, "3x later", ha="right", color="0.6")
    sc.set_xscale("log"); sc.set_yscale("log"); sc.set_xlim(lo, hi); sc.set_ylim(lo, hi)
    sc.set_xlabel("code reads height h-1 (updates to half-way)"); sc.set_ylabel("likelihood has level h (updates to half-way)")
    sc.set_title("every seed and level: above the line = reading first", loc="left", fontsize="medium")
    for key, _, label, colour in ARMS: sc.scatter([], [], color=colour, label=f"{label.splitlines()[0]}: lag x{np.median(lags.get(key, [np.nan])):.1f}")
    sc.scatter([], [], color="0.5", marker="o", label="depth 3"); sc.scatter([], [], color="0.5", marker="s", label="depth 4"); sc.legend(loc="lower right", fontsize="small")
    out = Path.home() / "generative-learning/reports/rhm-helmholtz-lag.png"; fig.savefig(out, dpi=130); print(out)
    for key, v in lags.items(): print(key, "lag ratios", np.round(sorted(v), 1))


if __name__ == "__main__": main()
