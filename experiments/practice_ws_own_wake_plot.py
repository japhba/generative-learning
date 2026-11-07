"""Figures for practice_ws_own_wake.py: reading gap, wake-update decomposition and effect of swapping the wake source.
Usage: python practice_ws_own_wake_plot.py RUN_DIR [RUN_DIR ...]  (first run -> reports/ws-own-wake.png, all -> ws-own-wake-all.png)"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

COL = {"own": "#2a78d6", "peer": "#eb6834", "world": "#1baf7a", "own_z": "#8a8a8a"}
NAME = {"own": "own samples, code read", "peer": "peer's samples", "world": "world samples", "own_z": "own samples, generating code kept"}
REPORTS = Path.home() / "generative-learning/reports"


def per_world(rows, T, src, f):
    """Mean of f over the rows of one source and checkpoint, one value per world."""
    ws = sorted({r["world"] for r in rows})
    return np.array([np.mean([f(r) for r in rows if r["T"] == T and r["source"] == src and r["world"] == w]) for w in ws])


def band(ax, x, vals, src, ls="-", label=None):
    m = np.array([v.mean() for v in vals]); se = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in vals])
    ax.plot(x, m, ls, color=COL[src], marker="o", label=label); ax.fill_between(x, m - se, m + se, color=COL[src], alpha=0.15, lw=0)


def row(axs, run):
    d = json.load(open(run / "results.json")); a, rows, cont = d["args"], d["rows"], d["cont"]
    Ts = [int(t) for t in a["Ts"].split(",")]; x = np.arange(len(Ts))
    for s in ["own", "peer", "world"]: band(axs[0], x, [per_world(rows, T, s, lambda r: r["gap"]) for T in Ts], s, label=NAME[s])
    axs[0].set_yscale("log"); axs[0].set_ylabel("reading error KL(exact posterior ‖ recognizer)\n[nats per sample]")
    for s in ["own", "peer", "world"]:
        band(axs[1], x, [per_world(rows, T, s, lambda r: r["signal_along"] / r["G_norm"]) for T in Ts], s, "-")
        band(axs[1], x, [per_world(rows, T, s, lambda r: r["bias_along"] / r["G_norm"]) for T in Ts], s, "--")
    axs[1].axhline(0, color="0.6", lw=0.8); axs[1].set_ylabel("mean wake update along the world's direction\n(÷ exact world gradient)")
    axs[1].plot([], [], "k-", label="exact likelihood gradient"); axs[1].plot([], [], "k--", label="inference bias"); axs[1].legend(loc="best")
    for s in ["own", "peer", "world", "own_z"]: band(axs[2], x, [per_world(cont, T, s, lambda r: r["kl_after"] - r["kl_before"]) for T in Ts], s, label=NAME[s])
    axs[2].axhline(0, color="0.6", lw=0.8); axs[2].set_yscale("symlog", linthresh=0.05)
    axs[2].set_ylabel(f"change in KL(world ‖ learner) after\n{a['steps']} wake-sleep steps with this wake source")
    for ax in axs: ax.set_xticks(x, [f"{t:,}" for t in Ts]); ax.set_xlabel("wake-sleep steps on world samples before the test")
    axs[0].set_title(f"{a['prior']} prior, {a['init']} start: how well each sample is read", loc="left")
    axs[1].set_title("what an average wake step on it does", loc="left"); axs[2].set_title("what training on it does", loc="left")
    return axs


def main():
    runs = [Path(p) for p in sys.argv[1:]]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True); row(axs, runs[0]); axs[0].legend(loc="best")
    h, l = axs[2].get_legend_handles_labels(); axs[2].legend(h, l, loc="best")
    fig.savefig(REPORTS / "ws-own-wake.png", dpi=150); print(REPORTS / "ws-own-wake.png")
    fig, axs = plt.subplots(len(runs), 3, figsize=(17, 4.6 * len(runs)), constrained_layout=True, squeeze=False)
    for i, run in enumerate(runs): row(axs[i], run)
    h, l = axs[0, 2].get_legend_handles_labels(); axs[0, 2].legend(h, l, loc="best")
    fig.savefig(REPORTS / "ws-own-wake-all.png", dpi=120); print(REPORTS / "ws-own-wake-all.png")


if __name__ == "__main__": main()
