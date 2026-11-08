"""Figure and onset table for practice_rhm_rpm.py: recognition-parametrised tree (no model of x) against the same latent tree
with a Gaussian emission, on the RHM with rendered leaves. Per arm and condition the learning rate with the smallest mean
log-KL over the run (median over seeds) is shown; every run of that rate is used.
pi_h = clip((log N_(h-1) - NLL) / (log N_(h-1) - log N_h), 0, 1) from the KL of the latent prior read out in world symbols.
Onset: first evaluation at which a seed's value reaches 1/2 (reading: 1/2 of the world's own posterior); median over seeds.
Usage: python practice_rhm_rpm_plot.py [TAG] -> reports/rhm-rpm.png"""

import glob
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/var/tmp/jbauer/generative_practice_rhm"
ARMS = [("rpm", "recognition-parametrised (no model of x), 8 leaf states", "#8a4fd8", "-"), ("rpm_big", "recognition-parametrised, 64 leaf states", "#8a4fd8", "--"),
        ("gen", "Gaussian emission, 8 leaf states", "#eb6834", "-"), ("gen_big", "Gaussian emission, 64 leaf states", "#eb6834", "--")]
CONDS = [("symbolic", "symbolic leaves (control)"), ("clean", "rendered leaves, D = 32"), ("nuisance", "rendered + background per leaf")]


def log_counts(a, L):
    v, m, s, nc = a["v"], a["m"], a["s"], a["nc"]
    return [s ** L * np.log(v)] + [s ** (L - h) * (np.log(v if h < L else nc) + (s ** h - 1) // (s - 1) * np.log(m)) for h in range(1, L + 1)]


def load(tag, L):
    runs = {}
    for f in sorted(glob.glob(f"{ROOT}/rpm_{tag}*L{L}_2*/results.json")):
        d = json.load(open(f)); a = d["args"]
        for r in d["results"]: runs.setdefault((r["cond"], r["arm"], r["lr"]), []).append(r)
    return a, runs


def onset(t, p, thr=0.5): i = np.flatnonzero(p >= thr); return t[i[0]] if len(i) else np.nan


def best(runs, cond, arm):
    score = {lr: np.median([np.mean(np.log([c["kl"] for c in r["curve"]])) for r in rs]) for (c, a, lr), rs in runs.items() if c == cond and a == arm}
    lr = min(score, key=score.get); return lr, runs[(cond, arm, lr)]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main_"
    for L in [3, 4]:
        a, runs = load(tag, L)
        if not runs: continue
        logN = log_counts(a, L); H = logN[-1]
        fig, AX = plt.subplots(3, 5, figsize=(23, 12), constrained_layout=True)
        print(f"== depth {L}: updates until half-way, median over seeds (reading: half of the world's own posterior)")
        for row, (cond, ctitle) in zip(AX, CONDS):
            for key, label, col, ls in ARMS:
                if not any(k[:2] == (cond, key) for k in runs): continue
                lr, rs = best(runs, cond, key); t = np.array([c["t"] for c in rs[0]["curve"]]); sel = t > 0
                kl = np.array([[c["kl"] for c in r["curve"]] for r in rs]); rd = np.array([[c["read"] for c in r["curve"]] for r in rs])
                al = np.array([[c["read_alone"] for c in r["curve"]] for r in rs]); ob = np.array([[c["obj"] for c in r["curve"]] for r in rs])
                ideal = np.array([r["ideal"]["read"] for r in rs]).mean(0); ideal_alone = np.mean([r["ideal"]["read_alone"] for r in rs])
                pi = [np.clip((logN[h - 1] - (kl + H)) / (logN[h - 1] - logN[h]), 0, 1) for h in range(1, L + 1)]
                row[0].plot(t[sel], np.median(kl, 0)[sel], ls, color=col, label=label)
                row[1].plot(t[sel], np.median(rd[:, :, 0], 0)[sel], ls, color=col); row[1].plot(t[sel], np.median(al, 0)[sel], ":", color=col, lw=1)
                row[2].plot(t[sel], np.median(rd[:, :, 1], 0)[sel], ls, color=col); row[3].plot(t[sel], np.median(rd[:, :, L], 0)[sel], ls, color=col)
                row[4].plot(t[sel], np.median(ob - ob[:, :1], 0)[sel], ls, color=col)
                on = lambda P, thr=0.5: np.nanmedian([onset(t, p, thr) for p in P]) if not all(np.isnan([onset(t, p, thr) for p in P])) else np.nan
                print(f"  {cond:9s} {key:8s} lr {lr:<6g} " + "  ".join(f"pi_{h} {on(pi[h - 1]):6.0f}" for h in range(1, L + 1))
                      + "  " + "  ".join(f"read_{h} {on(rd[:, :, h] / ideal[h]):6.0f}" for h in range(L + 1))
                      + f"  final kl {np.median(kl[:, -1]):.2f} read_0 {np.median(rd[:, -1, 0]):.2f} alone {np.median(al[:, -1]):.2f}")
            for i, h in [(1, 0), (2, 1), (3, L)]: row[i].axhline(ideal[h], color="0.4", lw=1, ls="--")
            row[1].axhline(ideal_alone, color="0.4", lw=1, ls=":")
            row[0].set_ylabel(f"{ctitle}")
        titles = ["KL(world || latent prior in world symbols) [nats]", "reading the leaf symbols (dotted: leaf alone)", "reading height 1",
                  f"reading the root (height {L})", "objective gain since start [nats / string]"]
        for j, ax in enumerate(AX.T):
            for k, x in enumerate(ax):
                x.set_xscale("log"); x.set_title(titles[j], loc="left", fontsize="medium") if k == 0 else None
                if j == 0: x.set_yscale("log")
                if 1 <= j <= 3: x.set_ylim(-0.03, 1.03)
                if k == 2: x.set_xlabel("updates")
        AX[0, 1].plot([], [], "--", color="0.4", label="world's own posterior"); AX[0, 1].plot([], [], ":", color="0.4", label="world's own, leaf alone")
        AX[0, 1].legend(loc="upper left", fontsize="small"); h, l = AX[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncols=4)
        out = Path.home() / f"generative-learning/reports/rhm-rpm{'' if L == 4 else '-L3'}.png"; fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__": main()
