"""Figure for practice_ws_own_distill.py: KL(world || generator) after the generator is damaged, by arm.
Usage: python practice_ws_own_distill_plot.py RUN_DIR [RUN_DIR ...] -> reports/ws-own-distill.png"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ARMS = {"world_keep": ("world samples, recognizer kept", "#1baf7a", "-"), "ws": ("world samples + sleep (plain wake-sleep)", "#1baf7a", "--"),
        "world16_keep": ("16 stored world samples, recognizer kept", "#eda100", "-"), "own_keep": ("own samples, recognizer kept (no world data)", "#2a78d6", "-"),
        "own_sleep": ("own samples + sleep", "#2a78d6", "--"), "own_fitted": ("own samples, recognizer fitted to damaged generator", "#8a8a8a", "-")}


def main():
    runs = [Path(p) for p in sys.argv[1:]]
    fig, axs = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 5.4), constrained_layout=True, squeeze=False)
    for ax, run in zip(axs[0], runs):
        d = json.load(open(run / "results.json")); a, rows, x = d["args"], d["rows"], np.array(d["record"])
        for arm, (name, col, ls) in ARMS.items():
            tr = np.array([[np.mean([r["traj"][i] for r in rows if r["arm"] == arm and r["world"] == w]) for i in range(len(x))] for w in sorted({r["world"] for r in rows})])
            m, se = tr.mean(0), tr.std(0, ddof=1) / np.sqrt(len(tr))
            ax.plot(x, m, ls, color=col, marker="o", label=name); ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0)
        ax.axhline(np.mean([r["kl_trained"] for r in rows]), color="0.3", lw=0.8, ls=":", label="trained learner before damage")
        ax.set_xscale("symlog", linthresh=25); ax.set_xlim(0, x[-1] * 1.3); ax.set_yscale("log"); ax.set_xlabel("steps after the generator was replaced")
        ax.set_title(f"{a['prior']} prior, {a['damage']} replacement", loc="left")
    axs[0, 0].set_ylabel("KL(world ‖ generator) [nats]"); h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncols=4)
    out = Path.home() / "generative-learning/reports/ws-own-distill.png"; fig.savefig(out, dpi=130); print(out)


if __name__ == "__main__": main()
