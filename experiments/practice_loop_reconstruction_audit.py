"""Independent, stronger reconstruction controls for the three-atom latent model."""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


STATES = np.array(list(product([-1.0, 1.0], repeat=3)))
NODES, QUAD = np.polynomial.hermite.hermgauss(32)
NOISE, QUAD = np.sqrt(2) * NODES, QUAD / np.sqrt(np.pi)


def density(h):
    logits = 0.5 * (STATES @ h) ** 2
    p = np.exp(logits - logits.max())
    return p / p.sum()


def rec_loss_gradient(h, p, noise, quadrature, recognition):
    z = (STATES @ recognition)[:, None] + noise[None, :]
    logits = z[:, :, None] * h[None, None, :]
    loss = np.logaddexp(logits, -logits) - STATES[:, None, :] * logits
    error = np.tanh(logits) - STATES[:, None, :]
    grad = np.einsum("s,k,ski,sk->i", p, quadrature, error, z)
    return float(np.einsum("s,k,ski->", p, quadrature, loss)), grad, error


def matched_rec(h, p, noisy):
    noise, quadrature = (NOISE, QUAD) if noisy else (np.zeros(1), np.ones(1))
    loss, grad, error = rec_loss_gradient(h, p, noise, quadrature, h)
    grad += np.einsum("s,k,ski,i,sj->j", p, quadrature, error, h, STATES)
    return loss, grad


def pseudolikelihood(h, p):
    j = np.outer(h, h)
    np.fill_diagonal(j, 0)
    field = STATES @ j
    error = np.tanh(field) - STATES
    derivative = error.T @ (p[:, None] * STATES)
    derivative += derivative.T.copy()
    np.fill_diagonal(derivative, 0)
    loss = np.sum(p[:, None] * (np.logaddexp(field, -field) - STATES * field))
    return float(loss), derivative @ h


def measurements(h, p_data, p_true, h_true):
    q = density(h)
    c = STATES.T @ (p_data[:, None] * STATES)
    reconstruction = np.trace(c) - 2 * h @ c @ h + (h @ h) * (h @ c @ h)
    angle = np.arccos(np.clip(abs(h @ h_true) / np.linalg.norm(h) / np.linalg.norm(h_true), 0, 1))
    return [float(np.sum(p_true * np.log(p_true / q))), float(np.rad2deg(angle)), float(reconstruction), float(-np.sum(p_data * np.log(q))), float(np.linalg.norm(h))]


def run():
    load_dotenv(Path.home() / ".env")
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_loop_reconstruction_audit"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out / "run_source.py")
    target = np.array([1.2, 0.7, -0.4])
    truth = density(target)
    methods = ["linear_ae", "deterministic_sigmoid_ae", "stochastic_sigmoid_ae", "frozen_recognition_ae", "masked_reconstruction", "exact_likelihood"]
    n, seeds, steps, rate = 512, 16, 4000, 0.02
    checkpoints = np.arange(0, steps + 1, 20)
    curves = np.zeros((seeds + 1, len(methods), len(checkpoints), 5))
    losses = np.zeros((seeds + 1, len(methods), len(checkpoints)))
    weights = np.zeros((seeds + 1, len(methods), len(checkpoints), 3))
    gradient_checks = []
    for seed in tqdm(range(seeds + 1), desc="Reconstruction alternatives"):
        rng = np.random.default_rng(78400 + seed)
        p_data = truth.copy() if seed == seeds else np.bincount(rng.choice(8, n, p=truth), minlength=8) / n
        covariance = STATES.T @ (p_data[:, None] * STATES)
        _, directions = np.linalg.eigh(covariance)
        initial = directions[:, -1] * np.sign(directions[:, -1] @ target)
        hs = np.tile(initial, (len(methods), 1))
        active = np.ones(len(methods), dtype=bool)
        history = [[] for _ in methods]
        for step in range(steps + 1):
            gradients = np.zeros_like(hs)
            objectives = np.zeros(len(methods))
            for mi, method in enumerate(methods):
                h = hs[mi]
                if method == "linear_ae":
                    objectives[mi] = measurements(h, p_data, truth, target)[2]
                elif method == "deterministic_sigmoid_ae":
                    objectives[mi], gradients[mi] = matched_rec(h, p_data, False)
                elif method == "stochastic_sigmoid_ae":
                    objectives[mi], gradients[mi] = matched_rec(h, p_data, True)
                elif method == "frozen_recognition_ae":
                    objectives[mi], gradients[mi], _ = rec_loss_gradient(h, p_data, NOISE, QUAD, initial)
                elif method == "masked_reconstruction":
                    objectives[mi], gradients[mi] = pseudolikelihood(h, p_data)
                else:
                    q = density(h)
                    objectives[mi] = -np.sum(p_data * np.log(q))
                    gradients[mi] = (STATES.T @ (q[:, None] * STATES) - covariance) @ h
            if step == 0:
                eps = 1e-5
                for noisy in [False, True]:
                    _, analytic = matched_rec(initial, p_data, noisy)
                    numeric = np.array([(matched_rec(initial + eps * np.eye(3)[i], p_data, noisy)[0] - matched_rec(initial - eps * np.eye(3)[i], p_data, noisy)[0]) / (2 * eps) for i in range(3)])
                    gradient_checks.append(float(np.max(np.abs(numeric - analytic))))
                _, analytic = pseudolikelihood(initial, p_data)
                numeric = np.array([(pseudolikelihood(initial + eps * np.eye(3)[i], p_data)[0] - pseudolikelihood(initial - eps * np.eye(3)[i], p_data)[0]) / (2 * eps) for i in range(3)])
                gradient_checks.append(float(np.max(np.abs(numeric - analytic))))
            if step % 20 == 0:
                ci = step // 20
                for mi, h in enumerate(hs):
                    curves[seed, mi, ci] = measurements(h, p_data, truth, target)
                    weights[seed, mi, ci] = h
                    losses[seed, mi, ci] = objectives[mi]
                    history[mi].append(objectives[mi])
                    if ci >= 20:
                        window = np.array(history[mi][-10:])
                        x = np.log(checkpoints[ci - 9:ci + 1])
                        slope = np.polyfit(x, np.log(window), 1)[0]
                        if abs(slope) < 1e-6 and np.linalg.norm(gradients[mi]) < 1e-4:
                            active[mi] = False
            assert np.isfinite(hs).all()
            assert np.max(np.linalg.norm(hs, axis=1)) < 20
            if step == steps:
                break
            hs[active] -= rate * gradients[active]
    np.savez_compressed(out / "trajectories.npz", curves=curves, objectives=losses, weights=weights, steps=checkpoints, truth=truth, target=target)
    summary = {"uk_created": datetime.now(ZoneInfo("Europe/London")).isoformat(), "methods": methods, "parameters": {"n": n, "finite_seeds": seeds, "first_seed": 78400, "updates": steps, "rate": rate, "quadrature_nodes": 32}, "max_gradient_check_error": max(gradient_checks), "measurements": ["true_kl", "angle_degrees", "linear_reconstruction", "empirical_nll", "weight_norm"], "initial_population": curves[-1, 0, 0].tolist(), "finite_final_means": {method: curves[:-1, mi, -1].mean(axis=0).tolist() for mi, method in enumerate(methods)}, "population_final": {method: curves[-1, mi, -1].tolist() for mi, method in enumerate(methods)}, "population_initial_gradients": {"deterministic_sigmoid_ae": matched_rec(weights[-1, 0, 0], truth, False)[1].tolist(), "masked_reconstruction": pseudolikelihood(weights[-1, 0, 0], truth)[1].tolist()}, "scope": "Masked reconstruction means conditional cross entropy of each observed spin given the other observed spins, using the same rank-one Ising weights; it uses no generated samples. All start at the exact finite-data linear tied-AE optimum. Population row uses exact probabilities rather than finite observations. Sigmoid-AE rows are diagnostic fixed-budget fits, not certified global optima."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1)
    labels = ["Linear reconstruction", "Sigmoid reconstruction", "Noisy sigmoid reconstruction", "Fixed encoder reconstruction", "Masked reconstruction", "Exact likelihood"]
    for mi, label in enumerate(labels):
        axes[0].plot(checkpoints, curves[:-1, mi, :, 0].mean(axis=0), label=label)
        axes[1].plot(checkpoints, curves[-1, mi, :, 1])
    axes[0].set(ylabel="Finite-data mean true KL", yscale="log")
    axes[0].legend()
    axes[1].set(xlabel="Gradient updates", ylabel="Population latent angle (degrees)")
    fig.tight_layout()
    figure = root / "tex" / "archive" / "figures" / "practice-loop-reconstruction-controls.png"
    fig.savefig(figure)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
