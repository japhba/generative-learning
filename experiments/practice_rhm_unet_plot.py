"""Figure and onset table for practice_rhm_unet.py: reading vs writing per height, over training.
Progress = (score - score at step 0) / (1 - score at step 0), per world; for the likelihood, the share of level h's information
log N_(h-1) - log N_h (N_h: strings valid up to height h) that the held-out bound has gone below; median and IQR over worlds. Onset: first evaluated
step at which a world's progress reaches 1/2 (median over worlds).
Evaluations after a world's run diverged (held-out loss excess more than twice its best so far, and 0.5 nats above it) are dropped.
Usage: python practice_rhm_unet_plot.py RUN_DIR [P] -> reports/rhm-unet-read-write.png"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


MEASURES = [  # key, height offset, label, colour, line style
    ("writes", 0, "writes: own strings valid up to h", "#2a78d6", "-"),
    ("writes_node", 0, "writes: own height-h nodes valid", "#7fb2ea", "-"),
    ("kl", 0, "likelihood has level h (held-out bound below log N_h)", "#0b3a75", "-"),
    ("fills", 0, "fills: one masked leaf of a world string", "#1baf7a", "--"),
    ("verifies", 0, "verifies: prefers a world string over it broken at h", "#d13b7a", "--"),
    ("reads", -1, "reads the height-(h-1) symbol (probe, held-out synonyms)", "#8a4fd8", ":"),
    ("reads", 0, "reads the height-h symbol", "#8a4fd8", "--"),
]


def log_counts(a, L):
    """log N_h, h = 0..L: number of strings valid up to height h (uniform over them is the best model knowing levels <= h)."""
    v, m, s, nc = a["v"], a["m"], a["s"], a["nc"]
    return [s ** L * np.log(v)] + [s ** (L - h) * (np.log(v if h < L else nc) + (s ** h - 1) // (s - 1) * np.log(m)) for h in range(1, L + 1)]


def stable(r):
    """False from the first evaluation at which the held-out loss excess over the true entropy exceeds twice its best so far
    and by more than 0.5 nats (Adam divergence late in a run); those points are dropped."""
    e = np.array([c["nll"] for c in r["curve"]]) - r["entropy"]; b = np.minimum.accumulate(e); return ~np.logical_or.accumulate((e > 2 * b) & (e - b > 0.5))


def progress(r, key, h):
    if key == "kl":  # share of level h+1's information (log N_h - log N_(h+1)) that the held-out bound has captured
        N = r["logN"]; e = np.array([c["nll"] for c in r["curve"]]); p = np.clip((N[h] - e) / (N[h] - N[h + 1]), 0, 1)
    else: s = np.array([c[key][h] for c in r["curve"]]); p = (s - s[0]) / (1 - s[0])
    p[~stable(r)] = np.nan; return p


def onset(x, p): i = np.flatnonzero(p >= 0.5); return x[i[0]] if len(i) else np.nan


def main():
    run = Path(sys.argv[1]); P = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    data = {}
    for f in sorted(run.glob(f"L*_P{P}.json")):
        d = json.load(open(f)); L = int(f.stem.split("_")[0][1:]); data[L] = d["results"]
        for r in data[L]: r["logN"] = log_counts(d["args"], L)
    Ls = sorted(data); H = max(Ls)
    fig, AX = plt.subplots(len(Ls), H + 1, figsize=(3.6 * (H + 1), 3.4 * len(Ls)), constrained_layout=True, squeeze=False)
    for row, L in zip(AX, Ls):
        rs = data[L]; x = np.array([c["step"] for c in rs[0]["curve"]])
        print(f"== depth {L}: {len(rs)} worlds; onset (step where progress reaches 1/2, median over worlds)")
        for r in rs:
            if not stable(r).all(): print(f"   world {r['world']} diverged after step {x[stable(r)][-1]}; later evaluations dropped")
        for h in range(L):
            ax = row[h]; line = []
            for key, off, label, col, ls in MEASURES:
                if h + off < 0 or (key != "kl" and key not in rs[0]["curve"][0]): continue
                Pm = np.array([progress(r, key, h + off) for r in rs]); m = np.nanmedian(Pm, 0); lo, hi = np.nanpercentile(Pm, [25, 75], axis=0)
                ax.plot(x, m, ls, color=col, label=label); ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
                on = [onset(x, p) for p in Pm]; line.append(f"{key}{'' if off == 0 else '(h-1)'} {np.nanmedian(on):>7.0f} [{np.nanmin(on):.0f}-{np.nanmax(on):.0f}]" if not np.all(np.isnan(on)) else f"{key}{'' if off == 0 else '(h-1)'}  never")
            print(f"   height {h + 1}: " + "; ".join(line))
            ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, x[-1] * 1.3); ax.set_ylim(-0.15, 1.08); ax.axhline(1, color="0.6", lw=0.8, ls=":")
            ax.set_title(f"depth {L}, height {h + 1}" + (" (root)" if h + 1 == L else ""), loc="left", fontsize="medium")
            if h == 0: ax.set_ylabel("progress from step 0 to ideal\n(median, IQR over worlds)")
            if row is AX[-1]: ax.set_xlabel("training steps (batch 128, fresh strings)")
        for h in range(L, H): row[h].axis("off")
        ax = row[H]
        for key, ls, label in [("nll", "-", "held-out bound"), ("train_loss", "--", "training loss")]:
            E = np.array([[np.nan if c[key] is None or not ok else c[key] - r["entropy"] for c, ok in zip(r["curve"], stable(r))] for r in rs]); ax.plot(x, np.nanmedian(E, 0), ls, color="0.2", label=label)
        ax.set_xscale("symlog", linthresh=10); ax.set_yscale("log"); ax.set_xlim(0, x[-1] * 1.3); ax.set_title(f"depth {L}: loss − true entropy [nats]", loc="left", fontsize="medium")
        if row is AX[-1]: ax.set_xlabel("training steps")
        ax.legend(loc="upper right")
    h, l = AX[0, 0].get_legend_handles_labels(); h2, l2 = AX[-1, 1].get_legend_handles_labels()
    fig.legend(h + [a for a, b in zip(h2, l2) if b not in l], l + [b for b in l2 if b not in l], loc="outside lower center", ncols=3)
    out = Path.home() / "generative-learning/reports/rhm-unet-read-write.png"; fig.savefig(out, dpi=130); print(out)


if __name__ == "__main__": main()
