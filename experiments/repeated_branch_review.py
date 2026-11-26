"""Pack per-world results, check the analytic evaluators, and draw the figures in the OverLyX note."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
import numpy as np

from repeated_branch import DATA, MARKS, M, NAMES, PATTERNS, RHO, WATCH, coins, draw, mass, row_fidelity, setup
from repeated_branch_stack import log_prob

FIGS = Path(__file__).resolve().parents[1] / "tex" / "figures"
LABELS = ("Unguided", "Direct reuse", "Local score", "Scalar score", "Pair score")
SUITES = ("main", "watching", "budgets", "probes", "pair_gate", "stack", "lag", "calibration", "challenges")
MAIN_MARKS = ",".join(map(str, MARKS))


def pack(stage, destination):
    arrays = {}
    for suite in SUITES:
        with np.load(stage / f"{suite}.npz") as data:
            arrays.update({f"{suite}/{key}": data[key] for key in data})
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def main_curve(data, method, arm):
    return data[f"main/{method}/{arm}/default/{WATCH}/{MAIN_MARKS}"]


def verify(data):
    for key in data.files:
        x = data[key]
        if np.issubdtype(x.dtype, np.number):
            assert np.isfinite(x).all(), key
    for method in NAMES:
        fork = main_curve(data, method, "own")[0]
        for arm in ("teacher", "twin", "matched", "classic", "sleep", "random", "frozen_reader", "plain_both", "plain_wake"):
            np.testing.assert_array_equal(fork, main_curve(data, method, arm)[0])
        np.testing.assert_array_equal(fork, data[f"watching/{method}/watching/default/0/0,1,2,4,8,16,24,32"][-1])
    for key in data.files:
        if "/no_reference/" in key:
            x = data[key]
            assert np.max(x[..., -2] - x[..., -1]) < 1e-9

    # Independently enumerate root states, a foreign relabeling, and every upper latent in one tree.
    rng = np.random.default_rng(901)
    motif, target, net, _ = setup(901, 1, "scalar", "default")
    g = np.tanh(net.writer)
    product_mass = np.prod((1 + PATTERNS * g) / 2, axis=1)
    np.testing.assert_allclose(mass(g)[0], product_mass)
    perm = rng.permutation(M)
    signed = PATTERNS * motif
    transformed = signed[:, perm] * motif
    donor_mean = (g * motif)[:, perm] * motif
    src = (mass(g) + mass(-g)) / 2
    np.testing.assert_allclose(src, (mass(donor_mean, transformed) + mass(-donor_mean, transformed))/2)
    np.testing.assert_allclose(mass(target, transformed) + mass(-target, transformed), mass(target) + mass(-target))
    h = coins(rng, np.zeros((1, 1, M, M)))
    likelihood = np.prod((1 + PATTERNS[:, :, None] * h[0, 0] * g) / 2, axis=(1, 2))
    brute = likelihood @ ((mass(target) + mass(-target))[0] / 2)
    np.testing.assert_allclose(log_prob(h, g, target)[0, 0], np.log(brute))

    # A sampled raw-string check exercises the lossless parity pooling and teacher readout calculation.
    n = 12000
    leaves, z = draw(rng, np.repeat(target, n, 0), np.repeat(RHO*motif, n, 0))
    mask = coins(rng, np.zeros_like(leaves))
    np.testing.assert_array_equal(mask * (mask * leaves), leaves)
    h = leaves[:, :, 0]
    reader = net.shallow
    field = (leaves * reader[:, None]).sum(-1)
    read = np.where(field == 0, h, np.sign(field))
    np.testing.assert_allclose((h == read).mean(), row_fidelity(reader, RHO*motif, True)[0], atol=.008)
    np.testing.assert_allclose((h * z).mean(0), target[0], atol=.015)
    # Exact Hessian of deep population KL at zero by a separate finite-difference calculation.
    pt = (mass(target) + mass(-target)) / 2
    def cross_entropy(w):
        return -(pt * np.log((mass(np.tanh(w)) + mass(-np.tanh(w))) / 2)).sum()
    for direction, eig in ((motif, -(M-1)*RHO**2),
                           (motif*np.tile([1., -1.], M//2), RHO**2)):
        step = 1e-4 * direction / np.linalg.norm(direction)
        curvature = (cross_entropy(step) + cross_entropy(-step) - 2*cross_entropy(0*step))/1e-8
        np.testing.assert_allclose(curvature, eig, atol=2e-6)


def canvas(rows, cols):
    fig, axes = plt.subplots(rows, cols, layout="constrained", squeeze=False)
    fig.set_size_inches(fig.get_size_inches() * [cols, rows])
    return fig, axes


def save(fig, name):
    svg = FIGS / f"repeated_branch_{name}.svg"
    fig.savefig(svg, transparent=True, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(FIGS / f"repeated_branch_{name}.png", facecolor="white")
    plt.close(fig)


def mark_watch(ax):
    ax.axvline(WATCH, color="0.55", linestyle=":")
    ax.set_xscale("symlog", linthresh=WATCH)
    ax.set_xticks((0, 32, 64, 160, 544), labels=("0", "32", "64", "160", "544"))
    ax.set_xlim(0, 544)
    ax.set_xlabel("Total strings")


def learning(data):
    fig, axes = canvas(2, 2)
    for method, name, color in zip(NAMES, LABELS, (f"C{i}" for i in range(len(NAMES)))):
        for arm, style in (("own", "-"), ("teacher", "--")):
            c = main_curve(data, method, arm).mean(1)
            for ax, col in zip(axes[0], (0, 2)):
                ax.plot(MARKS, c[:, col], color=color, ls=style, label=name if arm == "own" else "_nolegend_")
    for ax in axes[0]:
        ax.set_xscale("symlog", linthresh=16)
        ax.set_xticks((0, 16, 32, 64, 128, 256, 512), labels=("0","16","32","64","128","256","512"))
        ax.set_xlabel("Additional strings after Watching")
    axes[0, 0].set(title="Generation after Watching", ylabel="Population KL (nats per string)")
    axes[0, 1].set(title="Recognition after Watching", ylabel="Teacher-root accuracy", ylim=(.48, 1.02))
    axes[0, 0].legend(loc="upper left", ncols=2)
    axes[0, 1].legend(handles=[Line2D([], [], color="0.25", ls=style, label=name)
                               for style, name in (("-", "Own"), ("--", "Fresh teacher"))], loc="center right")
    t = data["stack/times"]
    ax = axes[1, 0]
    c = data["stack/scalar/own"].mean(1)
    for j, name in enumerate(("Shallow", "Middle", "Root"), start=2):
        ax.plot(t, c[:, j], label=name)
    ax.plot(t, data["stack/scalar/freeze_middle"].mean(1)[:, 4], ":", color="C2", label="Root, middle frozen")
    ax.plot(t, data["stack/scalar/teacher"].mean(1)[:, 4], "--", color="C2", label="Root, fresh teacher")
    ax.set(title="Recognition through three levels", ylabel="Teacher-cause accuracy", ylim=(.48, 1.02))
    ax.legend()
    ax = axes[1, 1]
    for arm, name, style in (("own", "Own", "-"), ("teacher", "Fresh teacher", "--"),
                             ("freeze_middle", "Own, middle frozen", ":")):
        values = data[f"stack/scalar/{arm}"][:, :, 0]
        avg, se = values.mean(1), values.std(1, ddof=1)/np.sqrt(values.shape[1])
        line, = ax.plot(t, avg, ls=style, label=name)
        ax.fill_between(t, avg-1.96*se, avg+1.96*se, color=line.get_color(), alpha=.15)
    ax.set(title="Generation through three levels", yscale="log", ylabel="Population KL (nats per string)")
    ax.legend()
    for ax in axes[1]:
        mark_watch(ax)
    save(fig, "learning")


def budgets(data):
    totals, shares = (32, 64, 128, 256, 512), (0., .25, .50, .75, 1.)
    fig, axes = canvas(1, 3)
    panels = []
    for m in NAMES[2:]:
        panels.append(np.array([[data[f"budgets/{m}/own/default/{t-int(t*f)}/{int(t*f)}"][0,:,0].mean()
                                 for t in totals] for f in shares]))
    norm = SymLogNorm(linthresh=.1, vmin=0, vmax=max(x.max() for x in panels))
    for ax, x, name in zip(axes[0], panels, LABELS[2:]):
        im = ax.imshow(x, origin="lower", norm=norm, cmap="viridis", aspect="auto")
        ax.set(title=name, xlabel="Total strings", xticks=range(len(totals)), xticklabels=totals,
               yticks=range(len(shares)), yticklabels=shares)
        for (i, j), value in np.ndenumerate(x):
            ax.text(j, i, f"{value:.2f}" if value < 1 else f"{value:.1f}", ha="center", va="center",
                    color="white" if norm(value) < .56 else "black")
    axes[0, 0].set_ylabel("Fraction spent on Own")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Mean population KL (nats)", ticks=[0,.1,1,10,100])
    save(fig, "budgets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "study.npz")
    parser.add_argument("--pack", type=Path, help="Local directory of per-suite output files")
    args = parser.parse_args()
    if args.pack:
        pack(args.pack, args.data)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "repeated-branch"})
    with np.load(args.data) as data:
        verify(data)
        learning(data)
        budgets(data)
