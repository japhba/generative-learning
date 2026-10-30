"""Branch, symmetry, variance and range controls for practice_theory_inverse."""

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
from scipy.special import expit, ndtr, logsumexp
from tqdm.auto import tqdm

from practice_theory_inverse import hidden, prediction


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("current", "frozen", "linear", "fixed_variance", "variance_only", "symmetric")


def features(x, low, high, linear=False): return hidden(0.25 + (x - low) / (high - low), linear)


def measure(weights, h, x, floor):
    mean, sigma, _ = prediction(weights, h, floor)
    low, high = np.sqrt(x - 0.15), np.sqrt(x + 0.15)
    success = ndtr((high - mean) / sigma) - ndtr((low - mean) / sigma)
    success += ndtr((-low - mean) / sigma) - ndtr((-high - mean) / sigma)
    greedy = np.mean(np.abs(mean**2 - x) < 0.15)
    nll = np.log(sigma * np.sqrt(2 * np.pi)) + (x + 0.03**2 + mean**2) / (2 * sigma**2)
    return np.array([success.mean(), greedy, np.mean((mean**2 - x)**2), nll.mean(), np.mean(np.abs(mean)), sigma.mean()])


def fit_inverse(h, targets, floor):
    start = np.zeros((2, h.shape[1])); start[0, 0] = 3 * targets.mean()
    start[1, 0] = 3 * np.log(np.expm1(0.7 - floor))
    def objective(vector):
        mean, sigma, raw = prediction(vector.reshape(start.shape), h, floor)
        residual = (targets - mean) / sigma
        loss = np.mean(np.log(sigma * np.sqrt(2 * np.pi)) + residual**2 / 2)
        grad = np.stack([h.T @ (-residual / sigma), h.T @ ((1 - residual**2) * expit(raw) / sigma)]) / len(targets)
        return loss, grad.ravel()
    result = minimize(objective, start.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 6000, "gtol": 1e-8, "ftol": 1e-11})
    assert result.success, result.message
    return result.x.reshape(start.shape)


def main(args):
    load_dotenv(Path.home() / ".env")
    destination = Path(os.environ["FAST_CACHE_DIR"]) / "generative_learning_practice_theory_inverse" / "controls" / args.case
    destination.mkdir(parents=True, exist_ok=True)
    low, high = (0.25, 1.25) if args.case == "narrow" else (0.25, 6.25)
    floor = 0.05
    config = vars(args) | {"low": low, "high": high, "sigma_floor": floor, "methods": METHODS, "metrics": ["sampled_success", "greedy_success", "mean_forward_mse", "two_branch_nll", "absolute_mean", "sigma"], "normalised_input": "0.25+(x-low)/(high-low)", "initial_signs": "alternating -1,+1 across seeds", "symmetry_arm": "zero mean, antithetic candidate pairs at identical x", "branch_supervision": "absolute values of the same observed y; choosing either sign is valid"}
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    x_eval = np.linspace(low + 0.001, high - 0.001, 1024)
    checkpoints = np.arange(0, args.steps + 1, args.eval_every)
    assert args.steps % args.eval_every == 0 and args.batch % 2 == 0
    curves, supervised, roots, mixture_nll = [], [], [], []
    for seed in tqdm(range(args.seeds), desc=f"inverse {args.case}"):
        rng = np.random.default_rng(1000 + seed)
        magnitudes = rng.uniform(np.sqrt(low), np.sqrt(high), 12)
        y_demo = np.concatenate([magnitudes, -magnitudes])
        x_demo = np.tile(magnitudes**2 + rng.normal(0, 0.03, 12), 2)
        verifier = np.linalg.lstsq(np.column_stack([np.ones(len(y_demo)), y_demo, y_demo**2]), x_demo, rcond=None)[0]
        branch_metrics, fitted = [], []
        for linear in (False, True):
            weights = fit_inverse(features(x_demo, low, high, linear), np.abs(y_demo), floor)
            h_eval = features(x_eval, low, high, linear)
            branch_metrics.append(measure(weights, h_eval, x_eval, floor)); fitted.append(weights)
        supervised.append(branch_metrics)
        mean, sigma, _ = prediction(fitted[0], features(x_eval, low, high), floor)
        nodes, probabilities = np.polynomial.hermite.hermgauss(32)
        target = np.sqrt(x_eval[:, None]) + np.sqrt(2) * 0.03 * nodes[None]
        logdensities = np.stack([-0.5 * ((target - mean[:, None]) / sigma[:, None])**2, -0.5 * ((target + mean[:, None]) / sigma[:, None])**2])
        logdensity = logsumexp(logdensities, axis=0) - np.log(2 * sigma[:, None] * np.sqrt(2 * np.pi))
        mixture_nll.append(float(np.mean(-logdensity @ probabilities / np.sqrt(np.pi))))
        direct_root = (-verifier[1] + np.sqrt(verifier[1]**2 + 4 * verifier[2] * (x_eval - verifier[0]))) / (2 * verifier[2])
        roots.append(float(np.mean(np.abs(direct_root**2 - x_eval) < 0.15)))
        for method in tqdm(METHODS, desc="controls", leave=False):
            generator = np.random.default_rng(2000 + seed)
            h_eval = features(x_eval, low, high, method == "linear")
            initial = np.zeros((2, h_eval.shape[1])); initial[0, 0] = 3 * 0.25 * (2 * (seed % 2) - 1)
            if method == "symmetric": initial[0] = 0
            initial[1, 0] = 3 * np.log(np.expm1(0.7 - floor))
            weights = initial.copy(); curve = [measure(weights, h_eval, x_eval, floor)]; history = [weights.copy()]
            for step in range(1, args.steps + 1):
                if method == "symmetric":
                    x_half = generator.uniform(low, high, args.batch // 2); epsilon_half = generator.normal(size=args.batch // 2)
                    x = np.tile(x_half, 2); epsilon = np.concatenate([epsilon_half, -epsilon_half])
                else:
                    x = generator.uniform(low, high, args.batch); epsilon = generator.normal(size=args.batch)
                h = features(x, low, high, method == "linear")
                mean, sigma, raw = prediction(weights, h, floor)
                source_mean, source_sigma, _ = prediction(initial if method == "frozen" else weights, h, floor)
                y = source_mean + source_sigma * epsilon
                reward = np.exp(-0.5 * ((verifier[0] + verifier[1] * y + verifier[2] * y**2 - x) / 0.12)**2)
                residual = (y - mean) / sigma
                update = np.stack([h.T @ (reward * residual / sigma), h.T @ (reward * (residual**2 - 1) * expit(raw) / sigma)]) / args.batch
                if method == "fixed_variance": update[1] = 0
                if method == "variance_only": update[0] = 0
                weights += args.rate * update
                assert np.isfinite(weights).all()
                if step % args.eval_every == 0:
                    curve.append(measure(weights, h_eval, x_eval, floor)); history.append(weights.copy())
            curves.append(curve)
            np.savez_compressed(destination / f"seed{seed}_{method}.npz", steps=checkpoints, curve=curve, weights=history, verifier=verifier, x_demo=x_demo, y_demo=y_demo, x_eval=x_eval, branch_sft_weights=fitted[0], branch_linear_sft_weights=fitted[1])
    curves = np.array(curves).reshape(args.seeds, len(METHODS), len(checkpoints), 6)
    supervised = np.array(supervised)
    summary = {method: {"initial": curves[:, j, 0].mean(0).tolist(), "mean": curves[:, j, -1].mean(0).tolist(), "std": curves[:, j, -1].std(0).tolist(), "by_seed": curves[:, j, -1].tolist()} for j, method in enumerate(METHODS)}
    summary["branch_sft"] = {"mean": supervised[:, 0].mean(0).tolist(), "by_seed": supervised[:, 0].tolist()}
    summary["branch_linear_sft"] = {"mean": supervised[:, 1].mean(0).tolist(), "by_seed": supervised[:, 1].tolist()}
    summary["direct_root_success"] = roots; summary["branch_sft_symmetric_mixture_nll"] = mixture_nll
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(destination / "all_curves.npz", steps=checkpoints, curves=curves, supervised=supervised)
    figure, axes = plt.subplots(2, 1)
    labels = {"current": "Current samples", "frozen": "Frozen proposals", "linear": "Linear features", "fixed_variance": "Mean only (fixed variance)", "variance_only": "Variance only (fixed mean)", "symmetric": "Exact symmetric sampling"}
    for j, method in enumerate(METHODS):
        for k in range(2): axes[k].plot(checkpoints * args.batch, curves[:, j, :, k].mean(0), label=labels[method])
    for k in range(2):
        axes[k].axhline(supervised[:, 0, k].mean(), color="black", linestyle=":", label="Branch-resolved SFT")
        axes[k].set(ylabel=("Sample validity", "Greedy mean validity")[k], ylim=(0, 1.05))
    axes[0].legend(); axes[1].set_xlabel("Candidate generations")
    figure.tight_layout(); figure.savefig(ROOT / f"reports/practice-theory-inverse-{args.case}-controls.png"); plt.close(figure)
    print(json.dumps({"destination": str(destination), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--case", choices=["narrow", "wide"], required=True)
    parser.add_argument("--seeds", type=int, default=4); parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--batch", type=int, default=512); parser.add_argument("--rate", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=150)
    main(parser.parse_args())
