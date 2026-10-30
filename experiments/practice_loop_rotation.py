"""A reconstruction-flat latent-axis control, including a decisive direct-MLE bypass."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


def rotate(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def draw_angles(rng, theta, kappa, count):
    return theta + (rng.vonmises(0, kappa, count) + 2 * np.pi * rng.integers(4, size=count)) / 4


def main():
    load_dotenv(Path.home() / ".env")
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_loop_rotation"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out / "run_source.py")
    kappa, n, seeds, steps, batch, rate = 2.0, 128, 32, 1200, 256, 0.001
    grid = np.linspace(-np.pi, np.pi, 200000, endpoint=False)
    ratio = np.sum(np.cos(grid) * np.exp(kappa * np.cos(grid))) / np.sum(np.exp(kappa * np.cos(grid)))
    methods = ["practice", "real_mle", "rescaled_real_mle", "population_practice", "reconstruction", "unjudged", "wrong_checker", "copy_moment"]
    saved = np.arange(0, steps + 1, 10)
    curves = np.zeros((seeds, len(methods), len(saved), 3))
    parameters = np.zeros((seeds, 4))
    equivalence_errors, reconstruction_errors = [], []
    for seed in tqdm(range(seeds), desc="Rotation control"):
        rng = np.random.default_rng(73100 + seed)
        true = rng.uniform(-np.pi / 8, np.pi / 8)
        observations = draw_angles(rng, true, kappa, n)
        moment = np.mean(np.exp(4j * observations))
        learned = np.angle(moment) / 4
        initial = true + 0.6
        angles = np.full(len(methods), initial)
        angles[-1] = learned
        parameters[seed] = [true, learned, initial, abs(moment)]
        x = np.stack([np.cos(observations), np.sin(observations)], axis=1)
        for step in range(steps + 1):
            if step % 10 == 0:
                index = step // 10
                curves[seed, :, index, 0] = kappa * ratio * (1 - np.cos(4 * (angles - true)))
                curves[seed, :, index, 1] = kappa * ratio * (1 - np.cos(4 * (angles - learned)))
                curves[seed, :, index, 2] = np.abs(np.angle(np.exp(4j * (angles - true)))) / 4
                for angle in angles:
                    w = rotate(angle)
                    reconstruction_errors.append(float(np.max(np.abs(x @ w @ w.T - x))))
                equivalence_errors.append(float(abs(angles[2] - angles[3])))
            if step == steps:
                break
            for mi in [0, 5, 6]:
                samples = draw_angles(rng, angles[mi], kappa, batch)
                score = 4 * kappa * np.sin(4 * (samples - angles[mi]))
                if mi == 5:
                    reward = np.ones(batch)
                else:
                    checker = learned + (np.pi / 4 if mi == 6 else 0)
                    reward = kappa * np.cos(4 * (samples - checker))
                gradient = np.sum((reward - reward.mean()) * score) / (batch - 1)
                angles[mi] += rate * gradient
            mle_gradient = 4 * kappa * np.mean(np.sin(4 * (observations - angles[1])))
            angles[1] += rate * mle_gradient
            rescaled_gradient = 4 * kappa * np.mean(np.sin(4 * (observations - angles[2]))) * ratio / abs(moment)
            angles[2] += rate * rescaled_gradient
            angles[3] -= rate * 4 * kappa * ratio * np.sin(4 * (angles[3] - learned))
        assert np.isfinite(curves[seed]).all()
    np.savez_compressed(out / "trajectories.npz", curves=curves, steps=saved, parameters=parameters)
    summary = {"uk_created": datetime.now(ZoneInfo("Europe/London")).isoformat(), "parameters": {"kappa": kappa, "observations": n, "seeds": seeds, "updates": steps, "batch": batch, "learning_rate": rate, "first_seed": 73100, "bessel_ratio": ratio}, "methods": methods, "max_reconstruction_absolute_error": max(reconstruction_errors), "max_population_vs_rescaled_mle_angle_difference": max(equivalence_errors), "initial_mean_true_kl": float(curves[:, 0, 0, 0].mean()), "final_mean_true_kl": dict(zip(methods, curves[:, :, -1, 0].mean(axis=0).tolist())), "final_mean_angle_degrees": dict(zip(methods, np.rad2deg(curves[:, :, -1, 2]).mean(axis=0).tolist())), "interpretation": "Positive separation from ordinary reconstruction, but a failure of strict practice necessity: direct original-data MLE has exactly the same population angular dynamics after a scalar time rescaling. The scalar angular score rule is not an all-synapse local Hebbian network."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1)
    for method, label in [("practice", "Judged own trials"), ("real_mle", "Likelihood on original observations"), ("reconstruction", "Original-data reconstruction"), ("wrong_checker", "Incorrect frozen checker")]:
        mi = methods.index(method)
        values = curves[:, mi, :, 0]
        line, = axes[0].plot(saved, values.mean(axis=0), label=label)
        lo, hi = np.quantile(values, [0.1, 0.9], axis=0)
        axes[0].fill_between(saved, lo, hi, color=line.get_color(), alpha=0.15)
        axes[1].plot(saved, np.rad2deg(curves[:, mi, :, 2]).mean(axis=0), color=line.get_color())
    axes[0].axhline(curves[:, -1, -1, 0].mean(), linestyle="--", color="black", label="Direct transfer of learned moment")
    axes[0].set(ylabel="True generation KL", yscale="log")
    axes[0].legend()
    axes[1].set(xlabel="Updates", ylabel="Latent-axis error (degrees)")
    fig.tight_layout()
    figure = root / "tex" / "archive" / "figures" / "practice-loop-rotation.png"
    fig.savefig(figure)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(str(figure))


if __name__ == "__main__":
    main()
