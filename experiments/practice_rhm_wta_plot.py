"""Figure and onset table for practice_rhm_wta.py, with the wake-sleep and exact-posterior arms of the Helmholtz machine
(practice_rhm.py fine runs, same worlds and generator) as references.
pi_h = clip((log N_(h-1) - NLL) / (log N_(h-1) - log N_h), 0, 1): the share of level h's information the generator's
likelihood has taken up. Verification at height h: share of intact world nodes the WTA recognizer verifies minus share of
nodes broken at that height it verifies. Onset: first evaluation at which a seed's value reaches 1/2; median over seeds.
Usage: python practice_rhm_wta_plot.py -> reports/rhm-wta.png"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/var/tmp/jbauer/generative_practice_rhm"
ARMS = [("practice", "wake-sleep", "#2a78d6", "-"), ("watch_exact", "exact posterior (stochastic EM)", "#eb6834", "-"),
        ("watch_wta_soft", "WTA recognition, soft (T = 1)", "#e8a33a", "-"), ("watch_wta", "WTA recognition, hard", "#1baf7a", "-"),
        ("watch_wta_2x", "WTA, twice the world strings", "#0b6e4f", "--"), ("watch_wta_practice", "WTA + verified own practice", "#8a4fd8", "-"),
        ("practice_wta", "verified own practice only", "#999999", ":")]


def log_counts(a, L):
    v, m, s, nc = a["v"], a["m"], a["s"], a["nc"]
    return [s ** L * np.log(v)] + [s ** (L - h) * (np.log(v if h < L else nc) + (s ** h - 1) // (s - 1) * np.log(m)) for h in range(1, L + 1)]


def load(L):
    runs = {}
    for key in ["practice", "watch_exact"]:
        d = json.load(open(sorted(glob.glob(f"{ROOT}/fine_k2_L{L}_{key}_2*/results.json"))[-1])); runs[key] = (d["args"], [[dict(c, t=c["t"]) for c in cv] for cv in d["results"][key]])
    d = json.load(open(sorted(glob.glob(f"{ROOT}/wta_L{L}_2*/results.json"))[-1]))
    for arm in dict.fromkeys(r["arm"] for r in d["results"]): runs[arm] = (d["args"], [r["curve"] for r in d["results"] if r["arm"] == arm])
    return runs


def onset(t, p): i = np.flatnonzero(p >= 0.5); return t[i[0]] if len(i) else np.nan


def main():
    fig, AX = plt.subplots(2, 5, figsize=(22, 8.6), constrained_layout=True)
    for row, L in zip(AX, [3, 4]):
        runs = load(L); logN = log_counts(runs["watch_wta"][0], L); H = logN[-1]
        print(f"== depth {L}: steps until half-way, median over seeds")
        for key, label, col, ls in ARMS:
            a, curves = runs[key]; t = np.array([c["t"] for c in curves[0]]); sel = t > 0
            kl = np.array([[c["kl"] for c in cv] for cv in curves]); line = []
            for h in range(1, L + 1):
                pi = np.clip((logN[h - 1] - (kl + H)) / (logN[h - 1] - logN[h]), 0, 1)
                row[h - 1].plot(t[sel], np.median(pi, 0)[sel], ls, color=col, label=label)
                on = [onset(t, p) for p in pi]; line.append(f"pi_{h} {np.nanmedian(on) if not np.all(np.isnan(on)) else np.nan:>7.0f}")
            if "verify" in curves[0][0] and key == "watch_wta":
                for h in range(1, L + 1):
                    V = np.array([[c["verify"][h - 1] for c in cv] for cv in curves]); row[h - 1].plot(t[sel], np.median(V, 0)[sel], ":", color="k", lw=1.5, label="WTA recognition verifies level h")
                    print(f"   WTA verifies height {h} at {np.nanmedian([onset(t, v) for v in V]):.0f}")
            print(f"   {key:20s} " + "  ".join(line))
            row[L].plot(t[sel], np.median(kl, 0)[sel], ls, color=col)
        for h in range(1, L + 1):
            ax = row[h - 1]; ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05); ax.axhline(0.5, color="0.8", lw=0.8)
            ax.set_title(f"depth {L}: likelihood has level {h}" + (" (root)" if h == L else ""), loc="left", fontsize="medium"); ax.set_xlabel("updates")
        for h in range(L, 4): row[h].axis("off") if h > L else None
        ax = row[L]; ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"depth {L}: KL(world || generator) [nats]", loc="left", fontsize="medium"); ax.set_xlabel("updates")
        if L == 3: row[4].axis("off")
        row[0].set_ylabel(f"depth {L}: progress (median of 4 seeds)")
    h, l = AX[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncols=4)
    out = Path.home() / "generative-learning/reports/rhm-wta.png"; fig.savefig(out, dpi=130); print(out)


if __name__ == "__main__": main()
