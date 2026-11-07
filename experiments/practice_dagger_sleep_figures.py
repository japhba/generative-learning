"""Figure for the sleep-phase section of the DAgger toy: learning curves and weights after watching (depth 4), and final
success against depth. Reads the runs of practice_dagger_sleep.py."""

import argparse
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

LABEL = {"watch_more": "watching, as many extra lectures", "own_hindsight": "sleep: imitate own dream for what it proved",
         "babble_hindsight": "sleep, babbling dreams", "own_hindsight_optimal": "sleep + per-state verdict (rest of dream is a shortest route)",
         "own_planner": "DAgger: own planner labels own states", "practice_reinforce": "paired comparison (own checker)"}
STYLE = {"watch_more": ("#8a8a85", "-"), "own_hindsight": ("#eb6834", "-"), "babble_hindsight": ("#eb6834", "--"),
         "own_hindsight_optimal": ("#1baf7a", "-"), "own_planner": ("#2a78d6", "-"), "practice_reinforce": ("#6250d6", "--")}


def load(root, pattern):
    out = {}
    for path in sorted(glob.glob(f"{root}/{pattern}/results.json")):
        r = json.load(open(path)); out.setdefault(r["args"]["D"], dict(args=r["args"], results={}, ref=r["ref"]))["results"].update(r["results"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/var/tmp/jbauer/generative_practice_dagger/sleep"); ap.add_argument("--figs", default="/nfs/nhome/live/jbauer/generative-learning/tex/figures")
    args = ap.parse_args()
    main_run = load(args.root, "b3_D4_eps0.15_main*")[4]; depth = load(args.root, "b3_D*_eps0.15_depth*"); a = main_run["args"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    for m in LABEL:
        c = np.array(main_run["results"][m]); x = np.arange(c.shape[1]) * a["every"]; col, ls = STYLE[m]
        for ax, k in zip(axes[:3], [0, 3, 1]):
            mu, se = c[:, :, k].mean(0), c[:, :, k].std(0) / np.sqrt(len(c))
            ax.plot(x, mu, color=col, ls=ls, label=LABEL[m]); ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.15, lw=0)
    axes[0].axhline(np.mean([r["planner"] for r in main_run["ref"]]), color="k", ls=":", label="planner at test time (reference)")
    for ax, t in zip(axes[:3], ["success (depth 4, slip rate 0.15)", "weight from 'off track' to undo", "weight from 'leads to target' to lemma"]):
        ax.set_xlabel("update after watching"); ax.set_title(t)
    Ds = sorted(D for D in depth if all(m in depth[D]["results"] for m in ["own_hindsight", "own_hindsight_optimal", "own_planner"]))
    for m in ["watch_more", "own_hindsight", "own_hindsight_optimal", "own_planner"]:
        v = [np.array([c[-1][0] for c in depth[D]["results"][m]]) for D in Ds]; col, ls = STYLE[m]
        axes[3].errorbar(Ds, [x.mean() for x in v], [x.std() / np.sqrt(len(x)) for x in v], color=col, ls=ls, marker="o")
    axes[3].plot(Ds, [np.mean([r["planner"] for r in depth[D]["ref"]]) for D in Ds], color="k", ls=":", marker="o")
    axes[3].set_xlabel("derivation depth D (slack 4 steps)"); axes[3].set_title("final success against depth"); axes[3].set_xticks(Ds)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{args.figs}/practice-dagger-sleep.png", dpi=130)
    for D in Ds: print(D, "  ".join(f"{m}={np.mean([c[-1][0] for c in depth[D]['results'][m]]):.3f}" for m in depth[D]["results"]), f"planner={np.mean([r['planner'] for r in depth[D]['ref']]):.3f}")
    print(f"{args.figs}/practice-dagger-sleep.png")


if __name__ == "__main__": main()
