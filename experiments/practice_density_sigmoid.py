"""Two binary hidden units compile a learned Ising score into a shallow generator."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_density as density


HIDDEN = np.array(list(itertools.product((-1.0, 1.0), repeat=2)))


def components(weight, bias, prior):
    logits = HIDDEN @ weight.T + bias
    log_prior = ((HIDDEN + 1.0) / 2.0) @ prior - np.logaddexp(0.0, prior).sum()
    return logits, log_prior


def log_density(y, weight, bias, prior):
    logits, log_prior = components(weight, bias, prior)
    return density.generator(y, logits, log_prior)[0]


def sample(rng, weight, bias, prior, batch):
    hidden = 2.0 * (rng.random((batch, 2)) < density.sigmoid(prior)) - 1.0
    probability = density.sigmoid(hidden @ weight.T + bias)
    y = 2.0 * (rng.random(probability.shape) < probability) - 1.0
    return y, hidden


def gradient(y, hidden, weight, bias, prior, coefficient, joint):
    if joint:
        residual = (y + 1.0) / 2.0 - density.sigmoid(hidden @ weight.T + bias)
        weighted = coefficient[:, None] * residual
        return weighted.T @ hidden, weighted.sum(axis=0), coefficient @ ((hidden + 1.0) / 2.0 - density.sigmoid(prior))
    logits, log_prior = components(weight, bias, prior)
    _, responsibility, probability = density.generator(y, logits, log_prior)
    responsibility /= responsibility.sum(axis=1, keepdims=True)
    weighted_hidden = coefficient[:, None] * responsibility
    grad_logits = weighted_hidden.T @ ((y + 1.0) / 2.0) - weighted_hidden.sum(axis=0)[:, None] * probability
    # Group equal-sign terms to preserve exact zero-weight hidden symmetry.
    # A BLAS dot product can leave ~1e-17 symmetry-breaking roundoff here.
    grad_weight = np.stack([grad_logits[HIDDEN[:, j] > 0].sum(axis=0) - grad_logits[HIDDEN[:, j] < 0].sum(axis=0) for j in range(2)], axis=1)
    grad_prior = coefficient @ (responsibility @ ((HIDDEN + 1.0) / 2.0) - density.sigmoid(prior))
    return grad_weight, grad_logits.sum(axis=0), grad_prior


def evaluate(states, weight, bias, prior, log_true, log_teacher, modes):
    logits, log_prior = components(weight, bias, prior)
    return density.evaluate(states, logits, log_prior, log_true, log_teacher, modes)


def audit(states, features, log_true):
    rng = np.random.default_rng(5471)
    weight, bias, prior = rng.normal(0, 0.5, (6, 2)), rng.normal(0, 0.2, 6), rng.normal(0, 0.2, 2)
    energy = features @ rng.normal(0, 0.2, features.shape[1])
    logq = log_density(states, weight, bias, prior)
    coefficients = {"free_energy": np.exp(logq) * (energy - logq), "importance": density.softmax(energy)}
    functions = {"free_energy": lambda logp: np.exp(logp) @ (energy - logp), "importance": lambda logp: density.softmax(energy) @ logp}
    errors = {}
    for method, coefficient in coefficients.items():
        grads = gradient(states, None, weight, bias, prior, coefficient, False)
        analytical = np.concatenate([value.ravel() for value in grads])
        parameter = np.r_[weight.ravel(), bias, prior]
        numerical = np.zeros_like(parameter)
        for index in range(len(parameter)):
            plus, minus = parameter.copy(), parameter.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            lp = log_density(states, plus[:12].reshape(6, 2), plus[12:18], plus[18:])
            lm = log_density(states, minus[:12].reshape(6, 2), minus[12:18], minus[18:])
            numerical[index] = (functions[method](lp) - functions[method](lm)) / 2e-5
        errors[method] = float(np.max(np.abs(analytical - numerical)))
        assert errors[method] < 1e-8
    logits, log_prior = components(weight, bias, prior)
    logq, responsibility, _ = density.generator(states, logits, log_prior)
    coefficients_joint = (np.exp(logq)[:, None] * responsibility * (energy - logq)[:, None]).ravel()
    joint = gradient(np.repeat(states, 4, axis=0), np.tile(HIDDEN, (len(states), 1)), weight, bias, prior, coefficients_joint, True)
    marginal = gradient(states, None, weight, bias, prior, np.exp(logq) * (energy - logq), False)
    errors["joint_marginal_identity"] = float(max(np.max(np.abs(x - y)) for x, y in zip(joint, marginal)))
    assert errors["joint_marginal_identity"] < 1e-14
    # An analytical capacity check, never used for initialization or practice.
    m_squared = (np.exp(4 * 0.8) - 1) / (np.exp(4 * 0.8) + 3)
    ideal = np.zeros((6, 2))
    ideal[:3, 0], ideal[3:, 1] = 2 * np.arctanh(np.sqrt(m_squared)), 2 * np.arctanh(np.sqrt(m_squared))
    errors["exact_true_representation"] = float(np.max(np.abs(log_density(states, ideal, np.zeros(6), np.zeros(2)) - log_true)))
    assert errors["exact_true_representation"] < 1e-12
    return errors


def train(rng, states, pairs, theta, log_true, log_teacher, modes, initial, data, method, args):
    weight, bias, prior = [array.copy() for array in initial]
    records = [evaluate(states, weight, bias, prior, log_true, log_teacher, modes)]
    optimized, eval_steps = [], []
    base_method = method.removeprefix("joint_")
    stop_reason = "maximum budget"
    for step in range(1, args.steps + 1):
        if method == "direct_mle":
            y, hidden = data, None
        else:
            y, hidden = sample(rng, weight, bias, prior, args.batch)
        logq = log_density(y, weight, bias, prior)
        energy = density.atom_features(y, pairs) @ theta
        if base_method == "free_energy":
            reward = energy - logq
            coefficient = (reward - reward.mean()) / (len(y) - 1)
        elif base_method == "importance":
            coefficient = density.softmax(energy - logq)
        elif method == "score":
            coefficient = density.softmax(energy)
        elif method in ["unfiltered", "direct_mle"]:
            coefficient = np.full(len(y), 1.0 / len(y))
        else:
            raise ValueError(method)
        grad_w, grad_b, grad_a = gradient(y, hidden, weight, bias, prior, coefficient, method.startswith("joint_"))
        weight += args.learning_rate * grad_w
        bias += args.learning_rate * grad_b
        prior += args.learning_rate * grad_a
        assert np.isfinite(weight).all() and np.isfinite(bias).all() and np.isfinite(prior).all()
        if step % args.eval_every == 0:
            records.append(evaluate(states, weight, bias, prior, log_true, log_teacher, modes))
            if method == "direct_mle":
                loss = -log_density(data, weight, bias, prior).mean()
            elif base_method == "free_energy":
                loss = records[-1][2]
            elif base_method in ["importance", "unfiltered"]:
                loss = records[-1][1] - np.exp(log_teacher) @ log_teacher
            else:
                state_energy = density.atom_features(states, pairs) @ theta
                loss = state_energy.max() - density.logsumexp(log_density(states, weight, bias, prior) + state_energy)
            optimized.append(float(loss))
            eval_steps.append(step)
            if step >= 800 and len(optimized) >= 16:
                slopes = []
                for start, end in [(-16, -8), (-8, None)]:
                    x = np.log(np.array(eval_steps)[start:end])
                    slopes.extend([np.polyfit(x, np.log(np.array(optimized)[start:end] + 1e-12), 1)[0], np.polyfit(x, np.log(np.array(records[1:])[start:end, 1] + 1e-12), 1)[0]])
                if max(abs(slope) for slope in slopes) < 0.02:
                    stop_reason = "stationary loss and teacher forward KL"
                    break
    stop_step = step
    for _ in range(stop_step + args.eval_every, args.steps + 1, args.eval_every):
        records.append(records[-1].copy())
    return np.array(records), weight, bias, prior, stop_step, stop_reason


def plot(output, figure_dir):
    archive = np.load(output / "trajectories.npz")
    metadata = json.loads((output / "metadata.json").read_text())
    labels = {"unfiltered": "Unfiltered self-imitation", "score": "Positive energy weighting", "free_energy": "Free energy", "importance": "Importance correction", "direct_mle": "Direct MLE on real data", "joint_free_energy": "Free energy, local sampled hidden", "joint_importance": "Importance correction, local sampled hidden"}
    for suffix, methods, metric, ylabel, log_scale in [
        ("learning", ["unfiltered", "score", "free_energy", "importance", "direct_mle"], 0, "KL(true distribution || generator)", True),
        ("local", ["free_energy", "importance", "joint_free_energy", "joint_importance"], 0, "KL(true distribution || generator)", True),
        ("entropy", ["unfiltered", "score", "free_energy", "importance", "direct_mle"], 3, "Generator entropy (nats)", False),
    ]:
        fig, ax = plt.subplots()
        for method in methods:
            values = archive[method][:, :, metric]
            line, = ax.plot(archive["steps"], values.mean(axis=0), label=labels[method])
            low, high = np.quantile(values, [0.1, 0.9], axis=0)
            ax.fill_between(archive["steps"], low, high, alpha=0.2, color=line.get_color())
        reference = np.mean(metadata["teacher_true_kl"]) if metric == 0 else metadata["true_entropy"]
        ax.axhline(reference, color="black", linestyle="--", label="Frozen learned density" if metric == 0 else "True entropy")
        ax.set(xlabel="Practice updates", ylabel=ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"practice-density-sigmoid-{suffix}.png")
        plt.close(fig)
    if metadata["arguments"]["warmup_steps"] > 0:
        fig, ax = plt.subplots()
        for index in range(metadata["arguments"]["seeds"]):
            trajectory = archive[f"warmup_{index}"]
            ax.plot(trajectory[:, 0], trajectory[:, 1])
        ax.axhline(np.mean(metadata["teacher_true_kl"]), color="black", linestyle="--", label="Frozen learned density")
        ax.set(xlabel="Direct-MLE pretraining updates", ylabel="KL(true distribution || generator)")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "practice-density-sigmoid-pretraining.png")
        plt.close(fig)


def interpretability_plots(output, figure_dir):
    archive = np.load(output / "trajectories.npz")
    metadata = json.loads((output / "metadata.json").read_text())
    states, pairs, _, features, _, _ = density.make_problem(6)
    # Deterministically use the lowest confirmation seed, never the prettiest fit.
    seed_index = 0
    teacher = archive[f"teacher_{seed_index}"]
    fitted, history = density.fit_pseudolikelihood(archive[f"real_data_{seed_index}"], pairs, 0.001)
    assert np.max(np.abs(fitted - teacher)) < 1e-10
    np.savez_compressed(output / "recognizer_parameter_trace.npz", history=history)
    fig, ax = plt.subplots()
    shown = set()
    for index, (i, j) in enumerate(pairs):
        category = "Within-block atoms" if i // 3 == j // 3 else "Cross-block atoms"
        ax.plot(history[:, 0], history[:, 3 + 6 + index], color="C0" if i // 3 == j // 3 else "C1", label=category if category not in shown else None)
        shown.add(category)
    ax.axhline(0.8, color="black", linestyle="--", label="True within-block coupling")
    ax.axhline(0.0, color="black", linestyle=":", label="True cross-block coupling")
    ax.set(xlabel="Recognizer Newton updates", ylabel="Learned pair-atom coefficient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-sigmoid-atoms.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for index in range(metadata["arguments"]["seeds"]):
        recognition = archive[f"recognizer_{index}"]
        ax.plot(recognition[:, 0], recognition[:, 1])
    ax.set(xlabel="Recognizer Newton updates", ylabel="Penalized pseudolikelihood loss")
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-sigmoid-recognizer-learning.png")
    plt.close(fig)

    method = "joint_importance"
    weight = archive[f"{method}_weight_{seed_index}"]
    bias = archive[f"{method}_bias_{seed_index}"]
    prior = archive[f"{method}_prior_{seed_index}"]
    magnitude = 2 * np.arctanh(np.sqrt((np.exp(4 * 0.8) - 1) / (np.exp(4 * 0.8) + 3)))
    ideal = np.zeros((6, 2))
    ideal[:3, 0], ideal[3:, 1] = magnitude, magnitude
    candidates = [weight[:, permutation] * np.array(signs)[None, :] for permutation in [(0, 1), (1, 0)] for signs in itertools.product((-1, 1), repeat=2)]
    aligned = min(candidates, key=lambda candidate: np.square(candidate - ideal).sum())
    fig, axes = plt.subplots(1, 2, layout="constrained")
    limit = max(np.max(np.abs(ideal)), np.max(np.abs(aligned)))
    for ax, value, title in zip(axes, [ideal, aligned], ["Analytical true decoder", "Learned decoder, seed " + str(metadata["arguments"]["seed_start"]) ]):
        color = ax.imshow(value, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(title=title, xlabel="Hidden spin", ylabel="Output spin", xticks=[0, 1], xticklabels=["1", "2"], yticks=np.arange(6), yticklabels=np.arange(1, 7))
    fig.colorbar(color, ax=axes, label="Output-logit weight")
    fig.savefig(figure_dir / "practice-density-sigmoid-weights.png")
    plt.close(fig)

    log_teacher = features @ teacher
    log_teacher -= density.logsumexp(log_teacher)
    log_generated = log_density(states, weight, bias, prior)
    correlations = [(states.T * np.exp(logp)) @ states for logp in [archive["log_true"], log_teacher, log_generated]]
    fig, axes = plt.subplots(1, 3, layout="constrained")
    for ax, value, title in zip(axes, correlations, ["True", "Fitted critic", "Generator"]):
        color = ax.imshow(value, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set(title=title, xlabel="Spin", ylabel="Spin", xticks=np.arange(6), xticklabels=np.arange(1, 7), yticks=np.arange(6), yticklabels=np.arange(1, 7))
    fig.colorbar(color, ax=axes, label="Expected spin product")
    fig.savefig(figure_dir / "practice-density-sigmoid-correlations.png")
    plt.close(fig)
    diagnostics = {"selected_seed": metadata["arguments"]["seed_start"], "method": method, "display_alignment": "Minimize decoder Frobenius error over hidden sign flips and column permutations; display only, never used in learning.", "true_decoder": ideal.tolist(), "learned_decoder_aligned": aligned.tolist(), "correlations": {name: value.tolist() for name, value in zip(["true", "teacher", "generator"], correlations)}}
    (output / "interpretability.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def summarize(output):
    archive = np.load(output / "trajectories.npz")
    metadata = json.loads((output / "metadata.json").read_text())
    n = metadata["arguments"]["seeds"]
    resamples = np.random.default_rng(175).integers(n, size=(20000, n))
    summary = {"paired_bootstrap_seed": 175, "paired_bootstrap_replicates": 20000, "n_seeds": n, "teacher_true_kl_mean": float(np.mean(metadata["teacher_true_kl"])), "methods": {}}
    for method in metadata["methods"]:
        final = archive[method][:, -1]
        difference = final[:, 0] - archive["direct_mle"][:, -1, 0]
        summary["methods"][method] = {"initial_true_kl_mean": float(archive[method][:, 0, 0].mean()), "final_true_kl_mean": float(final[:, 0].mean()), "final_true_kl_sd": float(final[:, 0].std(ddof=1)), "final_true_kl_range": [float(final[:, 0].min()), float(final[:, 0].max())], "kl_under_point_one": int((final[:, 0] < 0.1).sum()), "all_four_modes_above_point_zero_five": int((final[:, 6] > 0.05).sum()), "paired_difference_from_direct_mle": float(difference.mean()), "paired_difference_bootstrap_95_ci": np.quantile(difference[resamples].mean(axis=1), [0.025, 0.975]).tolist()}
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    load_dotenv(Path.home() / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "practice_density_sigmoid")
    parser.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parents[1] / "tex" / "archive" / "figures")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=9600)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--real-samples", type=int, default=512)
    parser.add_argument("--init-scale", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    args = parser.parse_args()
    assert args.steps % args.eval_every == 0 and not args.output.exists()
    args.output.mkdir(parents=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    states, pairs, theta, features, _, modes = density.make_problem(6)
    theta[:6] = 0.0
    log_true = features @ theta
    log_true -= density.logsumexp(log_true)
    methods = ["unfiltered", "score", "free_energy", "importance", "direct_mle", "joint_free_energy", "joint_importance"]
    metadata = {"uk_start": datetime.now(ZoneInfo("Europe/London")).isoformat(), "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "shared_code_sha256": hashlib.sha256(Path(density.__file__).read_bytes()).hexdigest(), "checks": audit(states, features, log_true), "methods": methods, "metrics": density.METRICS, "teacher_true_kl": [], "stopping": {method: [] for method in methods}, "true_entropy": float(-np.exp(log_true) @ log_true)}
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "run_source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "shared_source.py").write_bytes(Path(density.__file__).read_bytes())
    archive = {"states": states, "true_theta": theta, "log_true": log_true, "steps": np.arange(0, args.steps + 1, args.eval_every), **{method: [] for method in methods}}
    for seed_index in tqdm(range(args.seeds), desc="Sigmoid density seeds"):
        seed = args.seed_start + seed_index
        rng = np.random.default_rng(seed)
        data = states[rng.choice(len(states), size=args.real_samples, p=np.exp(log_true))]
        teacher, history = density.fit_pseudolikelihood(data, pairs, 0.001)
        log_teacher = features @ teacher
        log_teacher -= density.logsumexp(log_teacher)
        metadata["teacher_true_kl"].append(float(np.exp(log_true) @ (log_true - log_teacher)))
        archive[f"teacher_{seed_index}"], archive[f"recognizer_{seed_index}"], archive[f"real_data_{seed_index}"] = teacher, history, data
        init_rng = np.random.default_rng(seed + 100000)
        initial = (init_rng.normal(0, args.init_scale, (6, 2)), np.zeros(6), np.zeros(2))
        warmup = [[0, *evaluate(states, *initial, log_true, log_teacher, modes), float(-log_density(data, *initial).mean())]]
        for step in range(1, args.warmup_steps + 1):
            derivatives = gradient(data, None, *initial, np.full(len(data), 1.0 / len(data)), False)
            for parameter, derivative in zip(initial, derivatives):
                parameter += args.learning_rate * derivative
            warmup.append([step, *evaluate(states, *initial, log_true, log_teacher, modes), float(-log_density(data, *initial).mean())])
        archive[f"warmup_{seed_index}"] = np.array(warmup)
        for method_index, method in enumerate(tqdm(methods, desc=f"Seed {seed}", leave=False)):
            result = train(np.random.default_rng(seed + 200000 + method_index * 1000), states, pairs, teacher, log_true, log_teacher, modes, initial, data, method, args)
            archive[method].append(result[0])
            archive[f"{method}_weight_{seed_index}"], archive[f"{method}_bias_{seed_index}"], archive[f"{method}_prior_{seed_index}"] = result[1:4]
            metadata["stopping"][method].append({"step": result[4], "reason": result[5]})
    for method in methods:
        archive[method] = np.stack(archive[method])
    np.savez_compressed(args.output / "trajectories.npz", **archive)
    metadata["summary"] = {method: {metric: {"initial_mean": float(archive[method][:, 0, index].mean()), "final_mean": float(archive[method][:, -1, index].mean()), "final_sd": float(archive[method][:, -1, index].std(ddof=1))} for index, metric in enumerate(density.METRICS)} for method in methods}
    metadata["uk_finish"] = datetime.now(ZoneInfo("Europe/London")).isoformat()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summarize(args.output)
    plot(args.output, args.figure_dir)
    interpretability_plots(args.output, args.figure_dir)
    print(json.dumps({"output": str(args.output), "teacher_kl": np.mean(metadata["teacher_true_kl"]), "checks": metadata["checks"], "final_true_kl": {method: metadata["summary"][method]["true_forward_kl"] for method in methods}}, indent=2))


if __name__ == "__main__":
    main()
