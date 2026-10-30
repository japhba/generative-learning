"""Finite learned quadratic recognition -> one-hidden-layer inverse practice.

Exploratory mechanism experiment, not a study of an LLM or data scaling law.
The verifier sees finite noisy forward demonstrations and is then frozen.
Practice never accesses the true square map; truth appears only in evaluation.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy.special import expit, ndtr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
KNOTS = np.linspace(0.25, 1.25, 8)
METHODS = ("current", "frozen", "constant", "oracle", "linear")


def hidden(x, linear=False):
    columns = [np.ones(len(x)), x]
    if not linear: columns.extend(np.maximum(x[:, None] - KNOTS, 0).T)
    return np.column_stack(columns) / 3


def prediction(weights, features, floor):
    mean, raw = (features @ weights.T).T
    sigma = floor + np.logaddexp(0, raw)
    return mean, sigma, raw


def measure(weights, features, x, floor):
    mean, sigma, _ = prediction(weights, features, floor)
    low, high = np.sqrt(x - 0.15), np.sqrt(x + 0.15)
    success = ndtr((high - mean) / sigma) - ndtr((low - mean) / sigma)
    success += ndtr((-low - mean) / sigma) - ndtr((-high - mean) / sigma)
    nll = np.log(sigma * np.sqrt(2 * np.pi)) + (x + 0.03**2 + mean**2) / (2 * sigma**2)
    mse = mean**4 + 6 * mean**2 * sigma**2 + 3 * sigma**4 - 2 * x * (mean**2 + sigma**2) + x**2
    return np.array([success.mean(), nll.mean(), mse.mean(), mean.mean(), sigma.mean()])


def supervised_inverse(x, y, floor):
    features = hidden(x)
    start = np.zeros((2, features.shape[1])); start[1, 0] = 3 * np.log(np.expm1(0.7 - floor))
    def objective(vector):
        weights = vector.reshape(start.shape)
        mean, sigma, raw = prediction(weights, features, floor)
        z = (y - mean) / sigma
        loss = np.mean(np.log(sigma * np.sqrt(2 * np.pi)) + z**2 / 2)
        grad_mean = features.T @ ((mean - y) / sigma**2) / len(y)
        grad_raw = features.T @ ((1 - z**2) * expit(raw) / sigma) / len(y)
        return loss, np.stack([grad_mean, grad_raw]).ravel()
    result = minimize(objective, start.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-9, "ftol": 1e-12})
    assert result.success, result.message
    return result.x.reshape(start.shape)


def run(args):
    load_dotenv(Path.home() / ".env")
    destination = Path(os.environ["FAST_CACHE_DIR"]) / "generative_learning_practice_theory_inverse"
    destination.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"knots": KNOTS.tolist(), "methods": list(METHODS), "metrics": ["true_success", "two_branch_nll", "forward_mse", "mean", "sigma"], "sigma_floor": 0.05, "initial_mean": 0.25, "initial_sigma": 0.7, "verifier_width": 0.12, "demonstration_pairs": 12, "demonstration_noise": 0.03, "test_success_tolerance": 0.15, "test_target_noise": 0.03, "description": "Both roots are valid. Initial positive mean is a stated weak symmetry-breaking prior; it is not learned from the symmetric demonstrations."}
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    floor = config["sigma_floor"]
    records, baselines, coefficients, final_weights, demonstrations = [], [], [], [], []
    checkpoints = np.arange(0, args.steps + 1, args.eval_every)
    assert args.steps % args.eval_every == 0
    x_eval = np.linspace(0.251, 1.249, 1024)
    features_eval = {method: hidden(x_eval, method == "linear") for method in METHODS}
    for seed in tqdm(range(args.seeds), desc="inverse seeds"):
        data_rng = np.random.default_rng(1000 + seed)
        magnitude = data_rng.uniform(0.5, np.sqrt(1.25), 12)
        noise = data_rng.normal(0, 0.03, 12)
        y_demo = np.concatenate([magnitude, -magnitude])
        x_demo = np.tile(magnitude**2 + noise, 2)
        design = np.column_stack([np.ones(len(y_demo)), y_demo, y_demo**2])
        verifier = np.linalg.lstsq(design, x_demo, rcond=None)[0]
        coefficients.append(verifier); demonstrations.append(np.stack([x_demo, y_demo]))
        direct = supervised_inverse(x_demo, y_demo, floor)
        root = (-verifier[1] + np.sqrt(verifier[1]**2 + 4 * verifier[2] * (x_eval - verifier[0]))) / (2 * verifier[2])
        baseline = measure(direct, hidden(x_eval), x_eval, floor).tolist()
        baseline += [float(np.mean(np.abs(root**2 - x_eval) < 0.15)), float(np.mean((root**2 - x_eval)**2))]
        baselines.append(baseline)
        for method in tqdm(METHODS, desc="inverse arms", leave=False):
            rng = np.random.default_rng(2000 + seed)
            width = features_eval[method].shape[1]
            initial = np.zeros((2, width)); initial[0, 0] = 3 * 0.25
            initial[1, 0] = 3 * np.log(np.expm1(0.7 - floor))
            weights = initial.copy()
            curve = [measure(weights, features_eval[method], x_eval, floor)]
            weight_history = [weights.copy()]
            for step in range(1, args.steps + 1):
                x = rng.uniform(0.25, 1.25, args.batch)
                features = hidden(x, method == "linear")
                mean, sigma, raw = prediction(weights, features, floor)
                source_mean, source_sigma, _ = prediction(initial if method == "frozen" else weights, features, floor)
                y = source_mean + source_sigma * rng.normal(size=args.batch)
                if method == "constant": reward = np.ones(args.batch)
                else:
                    forward = y**2 if method == "oracle" else verifier[0] + verifier[1] * y + verifier[2] * y**2
                    reward = np.exp(-0.5 * ((forward - x) / 0.12)**2)
                residual = (y - mean) / sigma
                delta_mean = features.T @ (reward * residual / sigma) / args.batch
                delta_raw = features.T @ (reward * (residual**2 - 1) * expit(raw) / sigma) / args.batch
                weights += args.rate * np.stack([delta_mean, delta_raw])
                assert np.isfinite(weights).all()
                if step % args.eval_every == 0:
                    curve.append(measure(weights, features_eval[method], x_eval, floor))
                    weight_history.append(weights.copy())
            records.append(np.array(curve)); final_weights.append(weights)
            np.savez_compressed(destination / f"seed{seed}_{method}.npz", steps=checkpoints, curve=curve, weights=np.array(weight_history), verifier=verifier, x_demo=x_demo, y_demo=y_demo, x_eval=x_eval)
    curves = np.array(records).reshape(args.seeds, len(METHODS), len(checkpoints), 5)
    baselines = np.array(baselines); coefficients = np.array(coefficients)
    summary = {method: {"initial": curves[:, j, 0].mean(0).tolist(), "final_mean": curves[:, j, -1].mean(0).tolist(), "final_std": curves[:, j, -1].std(0).tolist(), "final_by_seed": curves[:, j, -1].tolist()} for j, method in enumerate(METHODS)}
    summary["supervised_inverse"] = {"mean": baselines[:, :5].mean(0).tolist(), "by_seed": baselines[:, :5].tolist()}
    summary["direct_inversion"] = {"true_success": float(baselines[:, 5].mean()), "forward_mse": float(baselines[:, 6].mean()), "by_seed": baselines[:, 5:].tolist()}
    summary["verifier_coefficients"] = coefficients.tolist()
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(destination / "all_curves.npz", steps=checkpoints, curves=curves, baselines=baselines, coefficients=coefficients, demonstrations=demonstrations)
    labels = {"current": "Learned verifier, current samples", "frozen": "Learned verifier, initial samples", "constant": "Constant score", "oracle": "Exact verifier (diagnostic)", "linear": "Linear generator features"}
    figure, axis = plt.subplots()
    for j, method in enumerate(METHODS):
        mean = curves[:, j, :, 0].mean(0); se = curves[:, j, :, 0].std(0) / np.sqrt(args.seeds)
        axis.plot(checkpoints * args.batch, mean, label=labels[method]); axis.fill_between(checkpoints * args.batch, mean - se, mean + se, alpha=0.15)
    axis.axhline(baselines[:, 0].mean(), linestyle=":", color="black", label="Direct inverse fit to demonstrations")
    axis.axhline(baselines[:, 5].mean(), linestyle="--", color="grey", label="Directly invert learned verifier")
    axis.set(xlabel="Candidate generations", ylabel="Held-out valid generation probability", ylim=(0, 1.05))
    axis.legend(); figure.tight_layout(); figure.savefig(ROOT / "reports/practice-theory-inverse-learning.png"); plt.close(figure)
    figure, axes = plt.subplots(2, 1)
    for j, method in enumerate(("current", "frozen", "constant")):
        index = METHODS.index(method)
        axes[0].plot(checkpoints * args.batch, curves[:, index, :, 0].mean(0), label=labels[method])
        axes[1].plot(checkpoints * args.batch, curves[:, index, :, 1].mean(0), label=labels[method])
    axes[0].set(ylabel="Valid generation probability"); axes[0].legend()
    axes[1].set(xlabel="Candidate generations", ylabel="NLL of both valid branches")
    figure.tight_layout(); figure.savefig(ROOT / "reports/practice-theory-inverse-density.png"); plt.close(figure)
    print(json.dumps({"destination": str(destination), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8); parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--batch", type=int, default=512); parser.add_argument("--rate", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=150)
    run(parser.parse_args())
