"""Report gradual local learning, fixed explanations, and finite-data certificates."""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import practice_drawing_library as m
import practice_drawing_sampled as s


def analyze(source, original):
    protocol = json.loads((source / "protocol.json").read_text())
    n = protocol["arguments"]["worlds"]
    summaries = json.loads((source / "summary.json").read_text())
    assert len(summaries) == n
    conditions = protocol["arguments"]["conditions"]
    runs = {c: [np.load(source / f"run_{i}/{c}.npz") for i in range(n)] for c in conditions}
    bootstrap = np.random.default_rng(20104299).integers(n, size=(20000, n))

    def estimate(values):
        values = np.asarray(values)
        return dict(mean=float(values.mean()), per_world=values.tolist(),
                    bootstrap95=np.quantile(values[bootstrap].mean(axis=-1), [.025, .975]).tolist())

    result = dict(worlds=n, final={}, recognition={}, certificate={}, comparisons={}, weight_statistics={})
    final = {c: np.array([r["records"][-1, 2] for r in runs[c]]) for c in conditions}
    for c in conditions:
        result["final"][c] = estimate(final[c])
        result["recognition"][c] = estimate([r["records"][-1, 6] for r in runs[c]])
        result["certificate"][c] = estimate([r["records"][-1, 4] for r in runs[c]])
        result["comparisons"][c] = estimate(final["sleep"] - final[c])
        for r in runs[c]:
            assert np.min(r["conditional_gains"]) >= -1e-10
            assert np.max(np.diff(r["records"][:, 3])) < 1e-8
            assert np.min(r["records"][:, 3] - r["records"][:, 1]) >= -1e-8
    result["initial"] = estimate([r["records"][0, 2] for r in runs["sleep"]])
    result["stopping"] = {c: [dict(step=int(r["records"][-1, 0]), status=str(r["status"])) for r in runs[c]]
                          for c in conditions}
    result["certificate_positive_worlds"] = int(sum(r["records"][-1, 4] > 0 for r in runs["sleep"]))
    initial_entropy, final_entropy, initial_match, final_match = [], [], [], []
    permutations = list(itertools.permutations(range(3)))
    for i in range(n):
        world = np.load(original / f"run_{i}/world.npz")
        teacher = world["teacher_b"]
        before, after = world["initial_b"], runs["sleep"][i]["b"]
        for b, entropy, matched in [(before, initial_entropy, initial_match), (after, final_entropy, final_match)]:
            entropy.append(-np.mean(np.sum(b * np.log(b), axis=-1)))
            matched.append(min(np.mean((teacher - b[list(p)]) ** 2) for p in permutations))
    result["weight_statistics"] = dict(initial_command_entropy=estimate(initial_entropy),
                                       final_command_entropy=estimate(final_entropy),
                                       initial_permutation_matched_mse=estimate(initial_match),
                                       final_permutation_matched_mse=estimate(final_match))
    (source / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")

    grid = np.arange(0, protocol["arguments"]["steps"] + 1, 20)
    labels = dict(sleep="Gradual local practice", fixed_cache="No new explanations",
                  fixed_posterior="Frozen initial causal posterior", frozen_proposal="Untrained inverse")
    fig, axes = plt.subplots(2, 1, layout="constrained")
    for c in conditions:
        for ax, column in zip(axes, [2, 6]):
            values = np.array([np.interp(grid, r["records"][:, 0], r["records"][:, column]) for r in runs[c]])
            line, = ax.plot(grid, values.mean(axis=0), label=labels[c])
            interval = np.quantile(values[bootstrap].mean(axis=1), [.025, .975], axis=0)
            ax.fill_between(grid, *interval, color=line.get_color(), alpha=.15)
    inputs = [json.loads((original / f"run_{i}/summary.json").read_text()) for i in range(n)]
    axes[0].axhline(np.mean([r["one_step"]["population_kl"] for r in inputs]),
                   linestyle="--", color="tab:green", label="One exact posterior fit")
    axes[0].axhline(np.mean([r["analytic"]["cv_kl"] for r in inputs]),
                   linestyle=":", color="tab:red", label="Cross-validated direct fits")
    axes[0].set(ylabel="Population KL (nats)", yscale="log")
    axes[1].set(xlabel="Local updates; stopped models held fixed", ylabel="Valid inferred program rate")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xlim(0, grid[-1])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    fig.savefig(m.ROOT / "tex/archive/figures/practice-gradual-local-learning.png")
    plt.close(fig)

    fig, ax = plt.subplots(layout="constrained")
    for i, run in enumerate(runs["sleep"]):
        ax.plot(run["records"][:, 0], run["records"][:, 4], label=f"World {i}")
    ax.axhline(0, linestyle=":", color="black")
    ax.set(xlabel="Local generator updates", ylabel="Certified likelihood gain (nats)")
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlim(0, grid[-1])
    fig.savefig(m.ROOT / "tex/archive/figures/practice-gradual-local-certificate.png")
    plt.close(fig)

    world = np.load(original / "run_0/world.npz")
    learned = runs["sleep"][0]
    permutation = min(permutations, key=lambda p: np.mean((world["teacher_b"] - learned["b"][list(p)]) ** 2))
    fig, axes = plt.subplots(3, 1, layout="constrained")
    for ax, title, b in zip(axes, ["Population motifs (evaluation only)", "Initial weights", "Learned weights, aligned"],
                            [world["teacher_b"], world["initial_b"], learned["b"][list(permutation)]]):
        im = ax.imshow(b.reshape(9, 4), vmin=0, vmax=1, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(4), ["Forward", "Right", "Left", "Jump"])
        ax.set_yticks([1, 4, 7], ["Motif 1", "Motif 2", "Motif 3"])
    fig.colorbar(im, ax=axes, label="Command probability")
    fig.savefig(m.ROOT / "tex/archive/figures/practice-gradual-local-weights.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 8, layout="constrained")
    for row, label, a, b in zip(axes, ["Population", "Initial", "Practised"],
                               [world["teacher_a"], world["initial_a"], learned["a"]],
                               [world["teacher_b"], world["initial_b"], learned["b"]]):
        canvas = m.render(s.decode(m.draw(a, b, np.random.default_rng(20104003), 8)))
        pixels = ((canvas[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).reshape(8, 8, 8)
        for ax, pixels_i in zip(row, pixels):
            ax.imshow(pixels_i, cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(label)
    fig.savefig(m.ROOT / "tex/archive/figures/practice-gradual-local-drawings.png")
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ["worlds", "initial", "final", "certificate_positive_worlds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.original)
