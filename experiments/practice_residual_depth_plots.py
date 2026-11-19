"""Plot the visible data law and exact learning curves of the residual-depth model."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from practice_residual_depth_exact import binary_curve, live_twin_curve, marginal_curves, outcome_probabilities


FIGURES = Path(__file__).resolve().parents[1] / "tex" / "figures"
DATA = Path("/nfs/gatsbystor/jbauer/generative-learning/residual_depth/figure_values.npz")
OWN, FRESH, TWIN, MATCHED = "C0", "C1", "C2", "C4"
GROUP_LABELS = [r"$e$", r"$R$", r"$R^2$", r"$S$", r"$RS$", r"$R^2S$"]


def visible_pair_laws():
    """Rows are first tokens. The product of the two tokens is the emitted move."""
    elements = [(rotation, reflection) for reflection in range(2) for rotation in range(3)]
    products = np.empty((6, 6), dtype=int)
    for left, (i, j) in enumerate(elements):
        for right, (k, l) in enumerate(elements):
            products[left, right] = elements.index(((i + (-1)**j * k) % 3, (j + l) % 2))
    teacher = (products == 1) / 6
    writer = ((2/3) * (products == 0) + (1/3) * (products == 1)) / 6
    laws = np.stack([teacher, writer])
    np.testing.assert_allclose(laws.sum(axis=1), 1/6)
    np.testing.assert_allclose(laws.sum(axis=2), 1/6)
    return products, laws


def budget_laws(total, step):
    """Index by actual teacher count and total count, with every teacher string first."""
    values = np.full((total + 1, total + 1, 2), np.nan)
    # Without a teacher the judge always abstains, so the initial performance persists.
    values[0] = next(binary_curve(0, 1, step))
    for teachers in range(1, total + 1):
        curve = np.array(list(binary_curve(total, teachers, step)))
        values[teachers, teachers:] = curve[teachers:]
    return values


def twin_laws(profiles, step, total):
    curves = np.array([list(marginal_curves(profile, step, total)) for profile in profiles])
    live = np.array(list(live_twin_curve(*profiles, step=step, budget=total))).T
    # The matched swap and the live twin have the same law at the comparison state.
    np.testing.assert_allclose(curves[:, 1, 1], live[:, 1])
    reach = []
    teacher = np.array([1, 1, 1, 0, 0])
    for reader, twin in zip(profiles, profiles[::-1]):
        probs = outcome_probabilities(reader, np.stack([reader, twin, teacher]))
        reach.append(np.stack([np.ones(3), 1 - probs[:, 0], probs[:, 2:].sum(axis=-1)], axis=-1))
    return curves, live, np.array(reach)


def save_figure(fig, directory, name):
    stem = directory / f"residual-depth-{name}"
    fig.savefig(stem.with_suffix(".png"), facecolor="white", dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), transparent=True, bbox_inches="tight", metadata={"Date": None})
    fig.savefig(stem.with_suffix(".pdf"), transparent=True, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def plot_visible_data(laws, directory):
    fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
    for ax, table, title in zip(axes, laws, (r"Teacher  $P(R)=1$", r"Writer  $P(R)=1/3$")):
        image = ax.imshow(table, origin="upper", cmap="Blues", vmin=0, vmax=1/6, interpolation="nearest")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=GROUP_LABELS, yticklabels=GROUP_LABELS,
               title=title, xlabel=r"Second token $v$")
    axes[0].set_ylabel(r"First token $u$")
    fig.suptitle("A pair of tokens encodes a move")
    bar = fig.colorbar(image, ax=axes, orientation="horizontal", label=r"Joint probability $P(u,v)$")
    bar.set_ticks([0, 1/18, 1/9, 1/6], labels=["0", "1/18", "1/9", "1/6"])
    save_figure(fig, directory, "data")


def plot_learning(budget, watch, directory):
    time = np.arange(budget.shape[1])
    fresh = budget[time, time]
    own = budget[np.minimum(time, watch), time]
    fig, axes = plt.subplots(2, 2, sharex=True, layout="constrained")
    for column, title in enumerate(("Population KL", "Teacher root accuracy")):
        ax = axes[0, column]
        ax.plot(time, own[:, column], color=OWN, label="Switch to own")
        ax.plot(time, fresh[:, column], color=FRESH, linestyle="--", label="Keep watching")
        ax.set_title(title)
    axes[0, 0].set_ylabel("Nats")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    differences = fresh[:, 0] - own[:, 0], 100 * (own[:, 1] - fresh[:, 1])
    for ax, diff, title, ylabel in zip(axes[1], differences, ("KL reduction", "Accuracy gain"), ("Nats", "Percentage points")):
        ax.plot(time, diff, color=OWN)
        ax.plot(100, diff[100], "o", color=OWN)
        ax.set(title=title, xlabel="Total strings", ylabel=ylabel)
    for ax in axes.flat:
        ax.axvline(watch, color="0.45", linestyle=":")
        ax.set_xlim(time[0], time[-1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2)
    save_figure(fig, directory, "learning")


def plot_budget(budget, directory):
    total = np.arange(1, budget.shape[1])
    # Each plotted cell uses the nearest attainable integer number of own samples.
    fraction = np.linspace(0, 1, 2 * total[-1] + 1)
    own_count = np.floor(fraction[:, None] * total + 0.5).astype(int)
    teachers = total - own_count
    loss = budget[teachers, total, 0]
    gain = budget[total, total, 0] - loss
    informed_gain = np.ma.masked_where(teachers == 0, gain)
    extent = [total[0] - 0.5, total[-1] + 0.5, -fraction[1]/2, 1 + fraction[1]/2]
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, layout="constrained")
    unavailable = "#d9d9d9"
    for ax, z, title, colors in zip(axes, (loss, informed_gain), ("Population KL", "KL gain over fresh teacher data"),
                                    ("viridis_r", "YlGnBu")):
        cmap = plt.get_cmap(colors).copy()
        cmap.set_bad(unavailable)
        image = ax.imshow(z, extent=extent, origin="lower", aspect="auto",
                          interpolation="nearest", cmap=cmap, vmin=0)
        ax.plot(100, 0.5, marker="o", markeredgecolor="black", markerfacecolor="white", linestyle="none")
        ax.set(title=title, ylabel="Own share", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        fig.colorbar(image, ax=ax, label="Nats")
    axes[-1].set_xlabel("Total strings, teacher strings first")
    handles = [Line2D([], [], marker="o", markeredgecolor="black", markerfacecolor="white", linestyle="none"),
               Patch(facecolor=unavailable)]
    fig.legend(handles, ["50 teacher + 50 own", "No teacher (gain panel)"], loc="outside upper center", ncols=2)
    save_figure(fig, directory, "budget")


def plot_twins(curves, live, reach, directory):
    fig, axes = plt.subplots(2, 2, sharey="row", layout="constrained")
    time = np.arange(curves.shape[1])
    for i, title in enumerate(("Weaker early gate", "Weaker middle gate")):
        ax = axes[0, i]
        for values, label, color, style in zip(reach[i], ("Own", "Live twin", "Fresh teacher"),
                                               (OWN, TWIN, FRESH), ("-", "-.", "--")):
            ax.plot([1, 2, 3], values, marker="o", color=color, linestyle=style, label=label)
        ax.set(title=title, xticks=[1, 2, 3], xticklabels=["Early", "Middle", "Terminal"], ylim=(0, 1.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax = axes[1, i]
        own, matched, fresh = curves[i].T
        ax.axhline(0, color=OWN)
        for values, label, color, style in zip((live[i], matched, fresh), ("Live twin", "KL-matched swap", "Fresh teacher"),
                                               (TWIN, MATCHED, FRESH), ("-.", ":", "--")):
            ax.plot(time, values - own, marker="o", color=color, linestyle=style, label=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Extra strings")
    axes[0, 0].set_ylabel("Chance reader reaches gate")
    axes[1, 0].set_ylabel(r"$\mathcal{L}_{\rm arm}-\mathcal{L}_{\rm own}$ (nats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    lower_handles, lower_labels = axes[1, 0].get_legend_handles_labels()
    handles.append(lower_handles[1])
    labels.append(lower_labels[1])
    fig.legend(handles, labels, loc="outside upper center", ncols=2)
    save_figure(fig, directory, "twins")


def main(directory, data):
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "residual-depth", "pdf.fonttype": 42})
    directory.mkdir(parents=True, exist_ok=True)
    data.parent.mkdir(parents=True, exist_ok=True)
    products, pairs = visible_pair_laws()
    binary_step, watch = 0.01, 50
    budget = budget_laws(150, binary_step)
    profiles = np.array([(1/3, 5/6, 1/3, 1/3, 1/3), (5/6, 1/3, 1/3, 1/3, 1/3)])
    twin_step = 0.1
    curves, live, reach = twin_laws(profiles, twin_step, 6)
    np.savez_compressed(data, group_elements=["e", "R", "R^2", "S", "RS", "R^2S"], group_product=products,
                        pair_sources=["teacher", "writer"], pair_joint=pairs, binary_step=binary_step,
                        binary_watch=watch, binary_total=np.arange(budget.shape[1]), binary_teachers=np.arange(budget.shape[0]),
                        binary_population_kl=budget[:, :, 0], binary_root_accuracy=budget[:, :, 1],
                        twin_initial_profiles=profiles, twin_step=twin_step, twin_extra_strings=np.arange(curves.shape[1]),
                        twin_arms=["own", "KL-matched swap", "fresh teacher"], twin_population_kl=curves,
                        live_twin_population_kl=live, reach_sources=["own", "live twin", "fresh teacher"],
                        initial_gate_reach=reach)
    plot_visible_data(pairs, directory)
    plot_learning(budget, watch, directory)
    plot_budget(budget, directory)
    plot_twins(curves, live, reach, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    main(args.figures, args.data)
