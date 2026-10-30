"""Learning curves for the crossword toy."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

LABEL = {"watch": "watching (replay the lectures)", "fresh": "a fresh lecture per practice grid",
         "practice": "practice, own checker (reward + entropy)", "practice_true": "practice, true checker (reward + entropy)",
         "practice_is": "practice, own checker (imitate accepted, 1/q)", "practice_is_true": "practice, true checker (imitate accepted, 1/q)",
         "practice_rft": "practice, own checker (imitate accepted)"}
PANELS = [("valid", "share of own grids that are valid"), ("novel_valid", "share of own grids that are valid and not a lecture"),
          ("coverage", "distinct valid grids / possible (2,000 samples)"), ("kl", "KL(uniform over valid grids || filler), nats")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--frontier", action="store_true", help="runs are groups 'dir1+dir2+...' per panel; plot validity against coverage")
    args = ap.parse_args()
    if args.frontier: return frontier([[json.load(open(f"{d}/results.json")) for d in g.split("+")] for g in args.runs], args.out)
    runs = [json.load(open(f"{r}/results.json")) for r in args.runs]
    fig, axes = plt.subplots(len(runs), len(PANELS), figsize=(5 * len(PANELS), 4 * len(runs)), squeeze=False)
    for row, r in zip(axes, runs):
        a = r["args"]; methods = list(r["results"][0]["curves"])
        for i, m in enumerate(methods):
            t = np.array([c["t"] for c in r["results"][0]["curves"][m]])
            for ax, (key, title) in zip(row, PANELS):
                get = lambda c, key=key: c[key]
                v = np.array([[get(c) for c in res["curves"][m]] for res in r["results"]])
                for s in v: ax.plot(t, s, color=f"C{i}", alpha=0.2)
                ax.plot(t, v.mean(0), color=f"C{i}", label=LABEL[m]); ax.set_title(title); ax.set_xlabel("update after watching")
        ref = np.mean([res["reference"]["kl"] for res in r["results"]]); row[-1].axhline(ref, color="k", ls=":", label="best this filler can do (fit to all valid grids)")
        row[-1].set_yscale("log"); row[0].set_ylabel(f"N = {a['N']} lectures, checker: {a['checker']}")
    h, l = axes[0, -1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08 if len(runs) > 1 else 0.15, 1, 1)); fig.savefig(args.out, dpi=110); print(args.out)



def frontier(runs, out):
    """Validity against coverage at the end of each run: one marker per learner (mean over seeds), thin markers per seed."""
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5.5), squeeze=False); axes = axes[0]
    for ax, group in zip(axes, runs):
        seen = set()
        for r in group:
            lam = r["args"]["entropy"]
            for m in r["results"][0]["curves"]:
                last = [res["curves"][m][-1] for res in r["results"]]; x, y = [c["coverage"] for c in last], [c["valid"] for c in last]
                name = LABEL[m] + (f", entropy weight {lam:g}" if m == "practice" else ""); key = (m, lam)
                if key in seen: continue
                seen.add(key); c = f"C{list(LABEL).index(m)}"; mk = {0.3: "v", 1.0: "o", 3.0: "s", 10.0: "D"}.get(lam, "o") if m == "practice" else "o"
                ax.scatter(x, y, color=c, marker=mk, alpha=0.25); ax.scatter(np.mean(x), np.mean(y), color=c, marker=mk, s=90, edgecolor="k", label=name)
        ref = [res["reference"] for res in group[0]["results"]]
        ax.scatter(np.mean([c["coverage"] for c in ref]), np.mean([c["valid"] for c in ref]), color="k", marker="*", s=200, label="fit to all valid grids (reference)")
        ax.set_xlabel("coverage: distinct valid grids / possible"); ax.set_ylabel("validity of own grids"); ax.set_title(f"N = {group[0]['args']['N']} lectures")
    axes[0].legend(fontsize="small"); fig.tight_layout(); fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__": main()
