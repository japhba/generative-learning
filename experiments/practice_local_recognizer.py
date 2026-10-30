"""Fit the learned Ising atoms using only local conditional prediction residuals."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

import practice_density as density


def local_gradient(data, theta, pairs, penalty):
    d = data.shape[1]
    coupling = np.zeros((d, d))
    for value, (i, j) in zip(theta[d:], pairs):
        coupling[i, j] = coupling[j, i] = value
    field = data @ coupling + theta[:d]
    residual = data - np.tanh(field)
    correlations = residual.T @ data / (len(data) * d)
    gradient = np.r_[residual.mean(axis=0) / d, [correlations[i, j] + correlations[j, i] for i, j in pairs]] - penalty * theta
    loss = np.logaddexp(0, -2 * data * field).mean() + penalty * theta @ theta / 2
    return gradient, float(loss)


def main():
    load_dotenv(Path.home() / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "practice_density_sigmoid")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    args = parser.parse_args()
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_local_recognizer"
    assert not output.exists()
    output.mkdir(parents=True)
    source = np.load(args.source / "trajectories.npz")
    states, pairs, _, features, _, _ = density.make_problem(6)
    rng = np.random.default_rng(194000)
    test_theta = rng.normal(0, 0.2, 21)
    grad, _ = local_gradient(source["real_data_0"], test_theta, pairs, 0.001)
    numerical = np.zeros(21)
    for i in range(21):
        plus, minus = test_theta.copy(), test_theta.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        numerical[i] = -(local_gradient(source["real_data_0"], plus, pairs, 0.001)[1] - local_gradient(source["real_data_0"], minus, pairs, 0.001)[1]) / 2e-5
    error = float(np.max(np.abs(numerical - grad)))
    assert error < 1e-8
    records = []
    for seed_index in tqdm(range(args.seeds), desc="Local atom learning"):
        data, reference = source[f"real_data_{seed_index}"], source[f"teacher_{seed_index}"]
        theta, curves = np.zeros(21), []
        for step in range(args.steps + 1):
            gradient, loss = local_gradient(data, theta, pairs, 0.001)
            if step % 100 == 0:
                curves.append({"step": step, "loss": loss, "gradient_norm": float(np.linalg.norm(gradient))})
                if np.linalg.norm(gradient) < 1e-6:
                    break
                if step >= 2000 and len(curves) >= 11:
                    log_step = np.log([r["step"] for r in curves[-10:]])
                    loss_slope = np.polyfit(log_step, np.log([r["loss"] for r in curves[-10:]]), 1)[0]
                    grad_slope = np.polyfit(log_step, np.log([r["gradient_norm"] for r in curves[-10:]]), 1)[0]
                    if max(abs(loss_slope), abs(grad_slope)) < 0.02:
                        break
            if step == args.steps:
                break
            theta += args.learning_rate * gradient
        _, reference_loss = local_gradient(data, reference, pairs, 0.001)
        log_local = features @ theta
        log_local -= density.logsumexp(log_local)
        log_reference = features @ reference
        log_reference -= density.logsumexp(log_reference)
        record = {"seed_index": seed_index, "theta": theta.tolist(), "curves": curves, "loss_excess_vs_newton": loss - reference_loss, "kl_newton_to_local": float(np.exp(log_reference) @ (log_reference - log_local)), "true_kl": float(np.exp(source["log_true"]) @ (source["log_true"] - log_local))}
        assert record["loss_excess_vs_newton"] < 1e-5
        records.append(record)
    (output / "results.json").write_text(json.dumps({"source": str(args.source), "finite_difference_error": error, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records}, indent=2) + "\n")
    (output / "source.py").write_text(Path(__file__).read_text())
    fig, ax = plt.subplots()
    for record in records:
        ax.plot([r["step"] for r in record["curves"]], [r["loss"] for r in record["curves"]], alpha=0.4)
    ax.set(xlabel="Local conditional-residual updates", ylabel="Penalized atom negative log pseudolikelihood")
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "tex" / "archive" / "figures" / "practice-density-local-recognizer.png")
    plt.close(fig)
    print(json.dumps({"output": str(output), "finite_difference_error": error, "maximum_loss_excess": max(r["loss_excess_vs_newton"] for r in records), "maximum_kl_vs_newton": max(r["kl_newton_to_local"] for r in records), "mean_true_kl": np.mean([r["true_kl"] for r in records])}, indent=2))


if __name__ == "__main__":
    main()
