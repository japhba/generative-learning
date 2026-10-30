"""Figures for the watching-vs-rederiving (DAgger) toy: learning curves, depth sweep, slip-rate sweep."""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABEL = {"watch": "watching (teacher forcing on lectures)", "watch_more": "watching as many extra lectures",
         "practice_reinforce": "rederiving, own checker, paired comparison", "dagger_self": "rederiving, own planner labels own states (DAgger)",
         "practice_rft": "rederiving, imitate own checked successes", "practice_unverified": "rederiving, imitate everything (no verdict)",
         "practice_lecture_checker": "rederiving, checker knows only the lectures", "planner": "planner at test time (reference)"}
ORDER = ["watch", "watch_more", "practice_reinforce", "dagger_self", "practice_rft", "practice_unverified", "practice_lecture_checker"]
COLOR = {m: f"C{i}" for i, m in enumerate(ORDER)}


def load(root):
    runs = {}
    for path in glob.glob(f"{root}/*/results.json"):
        r = json.load(open(path)); a = r["args"]; runs[(a["D"], a["eps"])] = r
    return runs


def final(r, m):
    if m == "planner": return np.array([x["planner"] for x in r["ref"]])
    if m == "watch": return np.array([x["watch_best"] for x in r["ref"]])
    return np.array([c[-1][0] for c in r["results"][m]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/var/tmp/jbauer/generative_practice_dagger"); ap.add_argument("--figs", default="/nfs/nhome/live/jbauer/generative-learning/tex/figures")
    args = ap.parse_args(); runs = load(args.root)
    r = runs[(4, 0.15)]; a = r["args"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for m in ORDER:
        c = np.array(r["results"][m]); x = (0 if m == "watch" else a["watch_steps"]) + np.arange(c.shape[1]) * a["every"]
        for ax, k in zip(axes, [0, 3, 1]):
            mu, se = c[:, :, k].mean(0), c[:, :, k].std(0) / np.sqrt(len(c))
            ax.plot(x, mu, color=COLOR[m], label=LABEL[m]); ax.fill_between(x, mu - se, mu + se, color=COLOR[m], alpha=0.15)
    axes[0].axhline(final(r, "planner").mean(), color="k", ls=":", label=LABEL["planner"])
    for ax, t in zip(axes, ["success from axioms to target (with slips)", "weight from 'off track' to undo", "weight from 'leads to target' to lemma choice"]):
        ax.axvline(a["watch_steps"], color="gray"); ax.set_xlabel("update"); ax.set_title(t)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(f"{args.figs}/practice-dagger-learning.png", dpi=130)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    Ds = sorted(D for D, e in runs if e == 0.15); eps = sorted(e for D, e in runs if D == 4)
    for m in ["watch", "watch_more", "practice_reinforce", "dagger_self", "practice_rft", "planner"]:
        c = COLOR.get(m, "k"); ls = ":" if m == "planner" else "-"
        v = [final(runs[(D, 0.15)], m) for D in Ds]
        axes[0].errorbar(Ds, [x.mean() for x in v], [x.std() / np.sqrt(len(x)) for x in v], color=c, ls=ls, marker="o", label=LABEL[m])
        axes[1].plot(Ds, [1 - x.mean() for x in v], color=c, ls=ls, marker="o")
        v = [final(runs[(4, e)], m) for e in eps]
        axes[2].errorbar(eps, [x.mean() for x in v], [x.std() / np.sqrt(len(x)) for x in v], color=c, ls=ls, marker="o")
    axes[0].set_xlabel("derivation depth D (slack 4 steps)"); axes[0].set_ylabel("success"); axes[0].set_title("success against depth (slip rate 0.15)")
    axes[1].set_xlabel("derivation depth D"); axes[1].set_ylabel("failure rate"); axes[1].set_yscale("log"); axes[1].set_title("failure against depth")
    axes[2].set_xlabel("slip rate"); axes[2].set_ylabel("success"); axes[2].set_title("success against slip rate (depth 4)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{args.figs}/practice-dagger-sweeps.png", dpi=130)
    for key in sorted(runs):
        print(key, "  ".join(f"{m}={final(runs[key], m).mean():.3f}" for m in ["watch", "watch_more", "practice_reinforce", "dagger_self", "practice_rft", "practice_unverified", "practice_lecture_checker", "planner"]))
    print(f"{args.figs}/practice-dagger-learning.png"); print(f"{args.figs}/practice-dagger-sweeps.png")


if __name__ == "__main__": main()
