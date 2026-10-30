"""Exact population mode dynamics of a depth-factorized sigmoid generator."""

from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def correlation(coupling):
    exponential = np.exp(4 * coupling)
    return (exponential - 1) / (exponential + 3)


def kl(c, gain):
    m2 = np.tanh(gain / (2 * np.sqrt(3))) ** 2
    return np.sum((1 + 3 * c) / 4 * np.log((1 + 3 * c) / (1 + 3 * m2)) + 3 * (1 - c) / 4 * np.log((1 - c) / (1 - m2)))


def direction(c, gain):
    m = np.tanh(gain / (2 * np.sqrt(3)))
    return np.sqrt(3) * m * (c - m * m) / (1 + 3 * m * m)


def run(c, c_true, depth, initial, dt, max_time):
    gain = np.full(2, initial)
    records = [[0.0, *gain, kl(c, gain), kl(c_true, gain)]]
    def flow(value):
        return depth * value ** (2 - 2 / depth) * direction(c, value)
    for step in tqdm(range(1, int(max_time / dt) + 1), desc=f"Depth {depth} population flow", leave=False):
        k1 = flow(gain)
        k2 = flow(gain + dt * k1 / 2)
        k3 = flow(gain + dt * k2 / 2)
        k4 = flow(gain + dt * k3)
        gain += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        assert np.isfinite(gain).all() and (gain > 0).all()
        if step % 20 == 0:
            records.append([step * dt, *gain, kl(c, gain), kl(c_true, gain)])
            if records[-1][3] < 1e-9:
                break
    return np.array(records)


def main():
    load_dotenv(Path.home() / ".env")
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_depth_modes"
    output.mkdir(parents=True, exist_ok=True)
    states = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
    features = np.column_stack([states[:, start] * states[:, start + 1] + states[:, start] * states[:, start + 2] + states[:, start + 1] * states[:, start + 2] for start in [0, 3]])
    true_coupling = np.array([0.8, 0.35])
    score = features @ true_coupling
    probability = np.exp(score - score.max())
    probability /= probability.sum()
    rng = np.random.default_rng(205381)
    observations = rng.choice(len(states), size=512, p=probability)
    c = features[observations].mean(axis=0) / 3
    assert ((c > 0) & (c < 1)).all(), "The ferromagnetic interior assumption failed."
    fitted_coupling = np.log((1 + 3 * c) / (1 - c)) / 4
    target_gain = 2 * np.sqrt(3) * np.arctanh(np.sqrt(c))
    c_true = correlation(true_coupling)
    errors = []
    for gain in [np.array([0.2, 0.7]), np.array([2.8, 1.4]), np.array([5.3, 3.0])]:
        numerical = np.zeros(2)
        for index in range(2):
            perturbation = np.eye(2)[index] * 1e-5
            numerical[index] = -(kl(c, gain + perturbation) - kl(c, gain - perturbation)) / 2e-5
        errors.append(float(np.max(np.abs(numerical - direction(c, gain)))))
    assert max(errors) < 1e-9
    curves = {str(depth): run(c, c_true, depth, 0.1, 0.02, 400.0) for depth in tqdm([1, 2, 3], desc="Balanced depth controls")}
    np.savez_compressed(output / "trajectories.npz", **curves, true_coupling=true_coupling, fitted_coupling=fitted_coupling, observations=states[observations], target_gain=target_gain)
    fig, ax = plt.subplots()
    for depth, curve in curves.items():
        ax.plot(curve[:, 0], curve[:, 4], label=f"Depth {depth}")
    ax.axhline(kl(c_true, target_gain), linestyle="--", color="black", label="Frozen learned density")
    ax.set(xlabel="Per-layer gradient-flow time", ylabel="KL(true distribution || generator)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "tex" / "archive" / "figures" / "practice-depth-learning.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for depth, linestyle in zip(curves, ["-", "--", ":"]):
        curve = curves[depth]
        for atom in range(2):
            ax.plot(curve[:, 0], curve[:, atom + 1], color=f"C{atom}", linestyle=linestyle, label=f"Depth {depth}, atom {atom + 1}")
    for atom in range(2):
        ax.axhline(target_gain[atom], color=f"C{atom}", linestyle="-.")
    ax.set(xlabel="Per-layer gradient-flow time", ylabel="End-to-end singular gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "tex" / "archive" / "figures" / "practice-depth-modes.png")
    plt.close(fig)
    metadata = {"uk_created": datetime.now(ZoneInfo("Europe/London")).isoformat(), "scope": "Population gradient-flow illustration on supplied aligned atom axes; not a sampled-practice or feature-discovery experiment. Depths share the same initial end matrix and density family; time is not equal FLOPs or initial effective learning rate.", "seed": 205381, "observations": 512, "true_coupling": true_coupling.tolist(), "fitted_coupling": fitted_coupling.tolist(), "learned_target_gain": target_gain.tolist(), "critic_true_kl": float(kl(c_true, target_gain)), "initial_gain": 0.1, "dt": 0.02, "gradient_max_error": max(errors), "stopping": "Learned-target forward KL below 1e-9, or flow time 400.", "runs": {depth: {"final_time": float(curve[-1, 0]), "final_true_kl": float(curve[-1, 4]), "final_learned_kl": float(curve[-1, 3]), "maximum_internal_kl_increase": float(np.diff(curve[:, 3]).max())} for depth, curve in curves.items()}, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    assert all(value["maximum_internal_kl_increase"] < 1e-10 for value in metadata["runs"].values())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(__file__, output / "run_source.py")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
