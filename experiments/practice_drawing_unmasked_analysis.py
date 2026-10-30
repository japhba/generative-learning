"""Summarize every unmasked world, including unsuccessful comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import practice_drawing_library as m


LABELS = dict(sleep="Joint learning", wake="Cached-cause replay", prior="Initial-generator dreams",
              frozen_proposal="Untrained recognizer", fixed_cache="No new explanations",
              pretrained_frozen="Train recognizer once")


def analyze(source, exact):
    rows = json.loads((source/"summary.json").read_text())
    n = len(rows)
    assert n == json.loads((source/"protocol.json").read_text())["arguments"]["worlds"]
    conditions = list(rows["0"]["conditions"])
    runs = {c: [np.load(source/f"run_{i}/{c}.npz") for i in range(n)] for c in conditions}
    bootstrap = np.random.default_rng(20103300).integers(n, size=(50000, n))

    def estimate(values):
        values = np.asarray(values)
        means = values[bootstrap].mean(axis=-1)
        return dict(mean=float(values.mean()), median=float(np.median(values)), per_world=values.tolist(),
                    bootstrap95=np.quantile(means, [.025, .975]).tolist(),
                    bootstrap9833=np.quantile(means, [.05/6, 1-.05/6]).tolist())

    endpoints = {c: np.array([r["records"][-1, 2] for r in runs[c]]) for c in conditions}
    em_rows = json.loads((exact/"summary.json").read_text())
    assert len(em_rows) == n
    em = np.array([em_rows[str(i)]["final"][2] for i in range(n)])
    direct = np.array([rows[str(i)]["direct"]["cv_kl"] for i in range(n)])
    result = dict(worlds=n, initial=estimate([r["records"][0, 2] for r in runs["sleep"]]),
                  final={c: estimate(v) for c, v in endpoints.items()}, direct=estimate(direct), exact_em=estimate(em),
                  contrast={}, recognition={}, certificates={}, stopping={}, pretraining={},
                  direct_choices=[rows[str(i)]["direct"]["selected"] for i in range(n)])
    comparisons = endpoints | {"direct": direct, "exact_em": em}
    for c, v in comparisons.items():
        if c == "sleep": continue
        difference = endpoints["sleep"]-v
        result["contrast"][c] = estimate(difference) | dict(
            worlds_improved=int((difference < 0).sum()),
            leave_one_out_means=[float(np.delete(difference, i).mean()) for i in range(n)])
    for c in conditions:
        result["recognition"][c] = dict(
            nll=estimate([r["records"][-1, 5] for r in runs[c]]),
            valid=estimate([r["records"][-1, 6] for r in runs[c]]))
        result["certificates"][c] = dict(
            gain=estimate([r["records"][-1, 4] for r in runs[c]]),
            positive_worlds=int(sum(r["records"][-1, 4] > 0 for r in runs[c])),
            max_bound_drop=float(max(np.diff(r["records"][:, 3]).max() for r in runs[c])),
            min_likelihood_slack=float(min(np.min(-r["records"][:, 1]+r["records"][:, 3]) for r in runs[c])),
            min_initial_upper_slack=float(min(r["records"][0, 1]+rows[str(i)]["upper_ll"] for i, r in enumerate(runs[c]))))
        assert result["certificates"][c]["max_bound_drop"] < 1e-8
        assert result["certificates"][c]["min_likelihood_slack"] > -1e-8
        assert result["certificates"][c]["min_initial_upper_slack"] >= 0
        result["stopping"][c] = [dict(step=int(r["records"][-1, 0]), seconds=float(r["records"][-1, 8]),
                                       status=str(r["status"])) for r in runs[c]]
    result["pretraining"]["updates"] = [int(np.load(source/f"run_{i}/pretraining.npz")["records"][-1, 0]) for i in range(n)]
    (source/"analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    grid = np.arange(0, 1601, 20)
    fig, axes = plt.subplots(2, 1, layout="constrained")
    for c in ["sleep", "pretrained_frozen", "frozen_proposal", "fixed_cache", "wake"]:
        for ax, column in zip(axes, [2, 6]):
            curves = np.array([np.interp(grid, r["records"][:, 0], r["records"][:, column]) for r in runs[c]])
            ax.plot(grid, curves.mean(axis=0), label=LABELS[c])
    axes[0].axhline(result["direct"]["mean"], color="black", linestyle=":", label="CV analytic fits")
    axes[0].axhline(result["exact_em"]["mean"], color="black", linestyle="--", label="Original-data exact EM")
    axes[0].set(ylabel="Population KL (nats)", yscale="log")
    axes[1].set(xlabel="Generator updates; stopped models held fixed", ylabel="Valid inferred program rate")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xlim(0, 1600)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    fig.savefig(m.ROOT/"tex/archive/figures/practice-drawing-unmasked-learning.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, layout="constrained")
    for ax, key in zip(axes, ["pretrained_frozen", "exact_em"]):
        comparator = comparisons[key]
        for i in range(n):
            ax.plot([0, 1], [comparator[i], endpoints["sleep"][i]], alpha=.5)
        ax.plot([0, 1], [comparator.mean(), endpoints["sleep"].mean()], color="black", marker="o", label="Mean")
        label = "Train recognizer once" if key == "pretrained_frozen" else "Original-data exact EM"
        ax.set_xticks([0, 1], [label, "Joint learning"])
        ax.set(ylabel="Population KL (nats)", yscale="log")
        ax.legend()
    fig.savefig(m.ROOT/"tex/archive/figures/practice-drawing-unmasked-controls.png")
    plt.close(fig)

    fig, ax = plt.subplots(layout="constrained")
    for i, r in enumerate(runs["sleep"]):
        ax.plot(r["records"][:, 0], r["records"][:, 4], label=f"World {i}")
    ax.axhline(0, color="black", linestyle=":")
    ax.set(xlabel="Joint-learning updates", ylabel="Certified data-likelihood gain (nats)")
    ax.set_xscale("symlog", linthresh=20)
    fig.savefig(m.ROOT/"tex/archive/figures/practice-drawing-unmasked-certificate.png")
    plt.close(fig)

    # World zero is fixed in advance for illustration, not selected by performance.
    fig, axes = plt.subplots(3, 1, layout="constrained")
    for ax, name, b in zip(axes, ["Teacher motifs (evaluation only)", "Initial motifs", "Learned motifs"],
                           [np.load(source/"run_0/world.npz")["teacher_b"],
                            np.load(source/"run_0/world.npz")["initial_b"], runs["sleep"][0]["b"]]):
        im = ax.imshow(b.reshape(9, 4), vmin=0, vmax=1, aspect="auto")
        ax.set_title(name)
        ax.set_xticks(range(4), ["Forward", "Right", "Left", "Jump"])
        ax.set_yticks([1, 4, 7], ["Motif 1", "Motif 2", "Motif 3"])
    fig.colorbar(im, ax=axes, label="Command probability")
    fig.savefig(m.ROOT/"tex/archive/figures/practice-drawing-unmasked-motifs.png")
    plt.close(fig)
    print(json.dumps({key: result[key] for key in ["initial", "final", "direct", "exact_em", "contrast"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.exact)
