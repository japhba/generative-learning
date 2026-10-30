"""Plot immutable b2k toy trajectories against their pre-simulation forecasts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bestn_theory import MASS, NODES, sigmoid
from simulate_finite_sample import metrics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/bestn_b2k/toy"
REPORTS = ROOT / "reports"


def sampled_success(weights):
    margins = 1.5 * weights[..., 0, None] + np.linalg.norm(weights, axis=-1)[..., None] * NODES
    return sigmoid(2 * margins) @ MASS


def main():
    completed = json.loads((OUTPUT / "complete.json").read_text())
    predictions = json.loads((OUTPUT / "predictions.json").read_text())["rows"]
    result = next(row for row in completed["results"] if row["case"]["name"] == "rho0.3_n2")
    data = np.load(result["path"])
    values = metrics(data["weights"])
    values["success"] = sampled_success(data["weights"])
    endpoints = []
    for field, label, name, prediction_key in (("error", "Population classification error", "error", "error"),
                                              ("overlap", "Normalized weight overlap", "overlap", "rho"),
                                              ("success", "Sampled response accuracy at temperature 1", "sampled", "success")):
        fig, ax = plt.subplots(layout="constrained")
        for index, arm in enumerate(("current", "frozen")):
            samples = values[field][:, index]
            mean = samples.mean(axis=-1)
            interval = 1.96 * samples.std(axis=-1, ddof=1) / np.sqrt(samples.shape[-1])
            color = f"C{index}"
            ax.plot(data["steps"] * 64, mean, color=color, label=arm.capitalize() + " source: SGD")
            ax.fill_between(data["steps"] * 64, mean - interval, mean + interval, color=color, alpha=.2)
            rows = [p for p in predictions if p["case"] == "rho0.3_n2" and p["arm"] == arm]
            ax.plot([p["step"] * 64 for p in rows], [p[prediction_key] for p in rows], color=color,
                    linestyle="--", label=arm.capitalize() + " source: forecast")
        ax.set_xscale("symlog", linthresh=64)
        ax.set_xlabel("Training candidate rollouts")
        ax.set_ylabel(label)
        ax.set_title("RBM: best-of-2, identical feedback budgets")
        ax.legend()
        path = REPORTS / f"bestn-toy-{name}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)
    for row in completed["results"]:
        w = np.load(row["path"])["weights"]
        errors = metrics(w[-1])["error"]
        paired = errors[0] - errors[1]
        successes = sampled_success(w[-1])
        paired_success = successes[0] - successes[1]
        endpoints.append({"case": row["case"], "current_error": float(errors[0].mean()), "frozen_error": float(errors[1].mean()),
                          "difference": float(paired.mean()), "ci95": (paired.mean() + np.array([-1, 1]) * 1.96 * paired.std(ddof=1) / np.sqrt(len(paired))).tolist(),
                          "current_sampled_success": float(successes[0].mean()), "frozen_sampled_success": float(successes[1].mean()),
                          "sampled_success_difference": float(paired_success.mean()), "sampled_success_ci95":
                          (paired_success.mean() + np.array([-1, 1]) * 1.96 * paired_success.std(ddof=1) / np.sqrt(len(paired_success))).tolist()})
    (OUTPUT / "analysis.json").write_text(json.dumps({"endpoints": endpoints}, indent=2) + "\n")


if __name__ == "__main__": main()
