"""Schematic for tex/model_4_cross_couplings.html (Figure 3). The RHM as a graph of interactions between symbols, with agreement between neighbouring nodes
that are not siblings at the leaves only (top) and at every level (bottom); the grey lines alone are the plain RHM. Marked: the cut between
the two halves of the root and the agreements that cross it (what a writer that writes the two halves independently has to carry).
Usage: python rhm_cross_coupling_schematic.py -> tex/figures/model4-cross-couplings.svg (transparent), reports/model4-cross-couplings.png"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

L = 4; ROOT = Path(__file__).resolve().parents[1]; SVG, PNG = ROOT / "tex/figures/model4-cross-couplings.svg", ROOT / "reports/model4-cross-couplings.png"
TREE, BOND, CROSS = "0.6", "#2a78d6", "#eb6834"
PANELS = [("agreement at the leaves only", [0], "crossing the cut: one agreement, however tall the tree\na node carries: its symbol and one feature per edge"),
          ("agreement at every level", range(L - 1), "crossing the cut: one agreement at every height below the children\na node carries: its symbol and one feature per edge and level below it")]


def pos(h, i): return (i + 0.5) * 2 ** h, h


def main():
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), constrained_layout=True)
    for ax, (title, coupled, note) in zip(axes, PANELS):
        for h in range(1, L + 1):
            for i in range(2 ** (L - h)):
                for c in (2 * i, 2 * i + 1): ax.plot(*zip(pos(h, i), pos(h - 1, c)), color=TREE, zorder=1)
        for h in coupled:
            for i in range(1, 2 ** (L - h) - 1, 2): ax.plot(*zip(pos(h, i), pos(h, i + 1)), color=CROSS if i + 1 == 2 ** (L - h - 1) else BOND, zorder=2)
        ax.axvline(2 ** (L - 1), color="k", linestyle="--", zorder=0)
        ax.scatter(*zip(*[pos(h, i) for h in range(L + 1) for i in range(2 ** (L - h))]), facecolor="w", edgecolor="k", zorder=3)
        ax.set(title=title, xlabel=note, ylabel="height of the node", xticks=[], yticks=range(L + 1)); ax.spines[["top", "right", "bottom"]].set_visible(False)
    handles = [Line2D([], [], color=TREE, label="production (a parent and its two children)"), Line2D([], [], color=BOND, label="agreement of two neighbours that are not siblings"),
               Line2D([], [], color=CROSS, label="agreement that crosses the cut"), Line2D([], [], color="k", linestyle="--", label="cut between the two halves of the root")]
    fig.legend(handles=handles, loc="outside lower center", ncols=1, frameon=False)
    SVG.parent.mkdir(exist_ok=True); fig.savefig(SVG, transparent=True); fig.savefig(PNG, dpi=150); print(SVG, PNG, sep="\n")


if __name__ == "__main__": main()
