"""Figure and summary for practice_sbn_lead.py: reading vs writing progress per layer.
Progress = (score - chance) / (reference - chance), per world, whole-code scores. Chance: a random reader (random
recognizer weights) and samples rendered from uniformly random bottom codes, both stored by the run.
Bottom row: the same progress against KL(world || model), for the wake-sleep learner along training (lines) and for the
world's own generator perturbed by noise, read with its exact posterior (crosses; practice_sbn_lead_static.py).
Usage: python practice_sbn_lead_plot.py RUN_DIR -> reports/sbn-lead.png"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


COL = ["#2a78d6", "#eb6834", "#1baf7a"]
TITLE = {"expand_D8": "widening 2→4→8, 1 pixel per bottom cause", "expand_D32": "widening 2→4→8, 4 pixels per bottom cause",
         "expand_D128": "widening 2→4→8, 16 pixels per bottom cause", "flat_D32": "flat 4→4→4, 8 pixels per bottom cause"}


MIN_RANGE = 0.15  # a world's score is used only where reference - chance >= this (otherwise progress is mostly noise)


def progress(r, kind, l, curve=None):
    s = np.array([c[kind][l][1] for c in (curve or r["curve"])]); c0 = r["chance"][kind][l]; rng = r["ref"][kind][l][1] - c0
    return (s - c0) / rng if rng >= MIN_RANGE else np.full(len(s), np.nan)


def main():
    run = Path(sys.argv[1]); d = json.load(open(run / "results.json")); res = d["results"]
    for r in res: r["chance"] = {k: [c[1] for c in v] for k, v in r["chance"].items()}
    names = list(dict.fromkeys(r["name"] for r in res))
    static = json.load(open(run / "static.json"))
    fig, AX = plt.subplots(2, len(names), figsize=(4.4 * len(names), 8.6), constrained_layout=True, sharey=True); axs = AX[0]
    for ax, ax2, name in zip(axs, AX[1], names):
        rs = [r for r in res if r["name"] == name]; L = len(rs[0]["K"]); x = np.array([c["step"] for c in rs[0]["curve"]])
        win = (x >= 30) & (x <= 3000)
        print(f"== {name}")
        for l in range(L - 1):
            for kind, ls in [("read", "-"), ("write", "--")]:
                P = np.array([progress(r, kind, l) for r in rs]); ok = ~np.isnan(P[:, 0]); m = np.median(P[ok], 0); lo, hi = np.percentile(P[ok], [25, 75], axis=0)
                ax.plot(x, m, ls, color=COL[l], marker="o", ms=3, label=f"layer {l + 1}, {'reading world samples' if kind == 'read' else 'writing own samples'}")
                ax.fill_between(x, lo, hi, color=COL[l], alpha=0.12, lw=0)
                if ok.sum() < len(rs): print(f"   layer {l + 1} {kind}: {ok.sum()}/{len(rs)} worlds with reference - chance >= {MIN_RANGE}")
            if l < L - 1:
                gap = np.array([progress(r, "read", l)[win].mean() - progress(r, "write", l)[win].mean() for r in rs]); gap = gap[~np.isnan(gap)]
                print(f"   layer {l + 1}: chance read {np.mean([r['chance']['read'][l] for r in rs]):.3f} write {np.mean([r['chance']['write'][l] for r in rs]):.3f}; progress over steps 30-3000, reading - writing: median {np.median(gap):+.2f}, mean {gap.mean():+.2f}, reading ahead in {np.sum(gap > 0)}/{len(gap)} worlds")
            if l < L - 1:  # progress against KL: learner (lines), perturbed world generator (crosses)
                kl = np.array([[c["kl"] for c in r["curve"]] for r in rs]).mean(0); sel = x >= 1
                st = [s for s in static if s["name"] == name]; eps = sorted({s["eps"] for s in st})
                for kind, ls in [("read", "-"), ("write", "--")]:
                    ax2.plot(kl[sel], np.nanmedian(np.array([progress(r, kind, l) for r in rs]), 0)[sel], ls, color=COL[l])
                    sk = [np.median([s["kl"] for s in st if s["eps"] == e]) for e in eps[1:]]
                    sp = [np.nanmedian([progress(r, kind, l, [s for s in st if s["eps"] == e and s["world"] == r["world"]]) for r in rs]) for e in eps[1:]]
                    ax2.plot(sk, sp, ls, color=COL[l], marker="x", ms=7, lw=0.8, alpha=0.7)
        ax2.set_xscale("log"); ax2.invert_xaxis(); ax2.axhline(1, color="0.6", lw=0.8, ls=":"); ax2.set_xlabel("KL(world ‖ model) [nats]")
        ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, x[-1] * 1.3); ax.set_ylim(-0.3, 1.15); ax.axhline(1, color="0.6", lw=0.8, ls=":")
        ax.set_title(TITLE.get(name, name), loc="left", fontsize="medium"); ax.set_xlabel("wake-sleep steps on world samples")
    axs[0].set_ylabel("progress from chance to the world's level\n(whole layer code right; median, IQR over worlds)")
    AX[1, 0].set_ylabel("progress (as above)"); AX[1, 0].plot([], [], "k-", label="wake-sleep learner"); AX[1, 0].plot([], [], "kx--", label="world's generator + noise, exact reader")
    h, l = axs[0].get_legend_handles_labels(); h2, l2 = AX[1, 0].get_legend_handles_labels(); fig.legend(h + h2, l + l2, loc="outside lower center", ncols=3)
    out = Path.home() / "generative-learning/reports/sbn-lead.png"; fig.savefig(out, dpi=130); print(out)


if __name__ == "__main__": main()
