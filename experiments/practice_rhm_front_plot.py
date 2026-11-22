"""Figure: how deep the reader reads and from how deep the writer writes, over learning, in the tree Helmholtz machine on the RHM
(saved runs of practice_rhm.py, depth 4, v 8, m 2; no new simulation). Per level h, median over seeds:
  writer  share of level h's information that the likelihood has: clip((log N_(h-1) - NLL) / (log N_(h-1) - log N_h), 0, 1), N_h = strings valid up to h
  reader  share of the true height-h symbol's entropy that the learner's sampled code carries, I(code; symbol) / H(symbol), as progress from step 0
Left: wake-sleep (reader taught by own samples only). Right: the reader replaced by the writer's exact posterior.
Usage: python practice_rhm_front_plot.py -> reports/rhm-front.png"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from practice_rhm_lag_plot import load, onset

ARMS = [("practice", "fine_k2_L4_practice_", "wake-sleep: reader taught by own samples"), ("watch_exact", "fine_k2_L4_watch_exact_", "reader = the writer's exact posterior")]
FIG = Path(__file__).resolve().parents[1] / "reports/rhm-front.png"


def main():
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True, sharey=True)
    for a, (key, prefix, title) in zip(ax, ARMS):
        t, runs, L = load(prefix, key); sel = t > 0; row = []
        for h in range(1, L):                                                                                # the root's symbol is not identifiable from the strings, so the top level is left out
            G, R = np.array([g[h - 1] for g, _ in runs]), np.array([r[h - 1] for _, r in runs])
            a.plot(t[sel], np.median(G, 0)[sel], "-", color=f"C{h - 1}", label=f"writer has level {h}"); a.plot(t[sel], np.median(R, 0)[sel], "--", color=f"C{h - 1}", label=f"reader reads level {h}")
            row += [np.nanmedian([onset(t, g) for g in G]), np.nanmedian([onset(t, r) for r in R])]
        print(f"{title}: updates to half-way, levels 1..{L - 1} as (writer, reader): " + "  ".join(f"({w:.0f}, {r:.0f})" for w, r in zip(row[0::2], row[1::2])))
        a.set(xscale="log", xlabel="updates", title=title)
    ax[0].set_ylabel("share of the level's information"); ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5)); fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


if __name__ == "__main__": main()
