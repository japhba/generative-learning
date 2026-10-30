"""Display the independent fixed-norm confirmation and its limitation case."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / "results/bestn_b2k/norm_confirmation/complete.json").read_text())
    for rho, suffix in ((.3, "density"), (.8, "limitation")):
        fig, ax = plt.subplots(layout="constrained")
        for source, label in (("self_sampled", "Self-sampled labels"), ("matched_gaussian", "Covariance-matched Gaussian"),
                              ("isotropic_gaussian", "Isotropic Gaussian")):
            rows = [r for r in result["curves"] if r["rho0"] == rho and r["source"] == source]
            x = np.array([r["step"] * 8 for r in rows])
            mean = np.array([r["density_change"] for r in rows])
            interval = 1.96 * np.array([r["density_change_se"] for r in rows])
            line, = ax.plot(x, mean, label=label)
            ax.fill_between(x, mean - interval, mean + interval, alpha=.15, color=line.get_color())
        ax.axhline(0, linestyle=":", color="gray")
        ax.set_xlabel("Real input samples presented")
        ax.set_ylabel("Change in marginal cross-entropy (nats; lower is better)")
        ax.set_title(f"No feedback, fixed weight norm: initial overlap {rho}")
        ax.legend()
        path = ROOT / f"reports/bestn-norm-{suffix}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)


if __name__ == "__main__": main()
