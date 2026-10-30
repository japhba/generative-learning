"""CPU experiment: compile learned local energies into a shallow stochastic generator.

The real distribution is accessed only in make_teacher_data and evaluate. During
practice every observed spin vector is sampled from the current generator; the
fixed recognizer contributes an unnormalized local energy score. Exact state
enumeration is used only for evaluation and gradient audits.
"""

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


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def logsumexp(x, axis=-1):
    maximum = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True)), axis=axis)


def softmax(x, axis=-1):
    return np.exp(x - np.expand_dims(logsumexp(x, axis), axis))


def atom_features(y, pairs):
    return np.concatenate((y, np.stack([y[:, i] * y[:, j] for i, j in pairs], axis=1)), axis=1)


def make_problem(dimension):
    assert dimension == 6
    states = np.array(list(itertools.product((-1.0, 1.0), repeat=dimension)))
    pairs = list(itertools.combinations(range(dimension), 2))
    theta = np.zeros(dimension + len(pairs))
    theta[:3], theta[3:6] = 0.04, -0.025
    for p, (i, j) in enumerate(pairs):
        theta[dimension + p] = 0.8 if i // 3 == j // 3 else 0.0
    features = atom_features(states, pairs)
    log_true = features @ theta
    log_true -= logsumexp(log_true)
    modes = (states[:, :3].sum(axis=1) > 0).astype(int) * 2 + (states[:, 3:].sum(axis=1) > 0).astype(int)
    return states, pairs, theta, features, log_true, modes


def fit_pseudolikelihood(data, pairs, penalty):
    """Newton fit of a symmetric Ising model using only local conditionals."""
    n, dimension = data.shape
    design = np.zeros((n, dimension, dimension + len(pairs)))
    for i in range(dimension):
        design[:, i, i] = 1.0
    for p, (i, j) in enumerate(pairs):
        design[:, i, dimension + p] = data[:, j]
        design[:, j, dimension + p] = data[:, i]
    design = (2.0 * data[:, :, None] * design).reshape(n * dimension, -1)
    theta = np.zeros(design.shape[1])
    history = []
    for iteration in range(50):
        margin = design @ theta
        wrong = sigmoid(-margin)
        loss = np.logaddexp(0.0, -margin).mean() + penalty * np.square(theta).sum() / 2
        grad = -(design.T @ wrong) / len(design) + penalty * theta
        history.append([iteration, float(loss), float(np.linalg.norm(grad)), *theta])
        if np.linalg.norm(grad) < 1e-10:
            break
        hessian = (design.T @ (design * (wrong * (1.0 - wrong))[:, None])) / len(design) + penalty * np.eye(len(theta))
        direction = np.linalg.solve(hessian, grad)
        step = 1.0
        for _ in range(40):
            candidate = theta - step * direction
            candidate_loss = np.logaddexp(0.0, -(design @ candidate)).mean() + penalty * np.square(candidate).sum() / 2
            if candidate_loss <= loss - 1e-4 * step * (grad @ direction):
                break
            step *= 0.5
        assert candidate_loss <= loss
        theta = candidate
    assert history[-1][2] < 1e-8, history[-1]
    return theta, np.array(history)


def generator(y, weight, bias):
    """Return log q, posterior hidden activations, and local output probabilities."""
    bits = (y + 1.0) / 2.0
    probability = sigmoid(weight)
    log_alpha = bias - logsumexp(bias)
    log_joint = bits @ weight.T - np.logaddexp(0.0, weight).sum(axis=1)[None, :] + log_alpha[None, :]
    log_q = logsumexp(log_joint, axis=1)
    responsibility = np.exp(log_joint - log_q[:, None])
    return log_q, responsibility, probability


def score_average(y, weight, bias, coefficient):
    _, responsibility, probability = generator(y, weight, bias)
    hidden = coefficient[:, None] * responsibility
    weight_gradient = hidden.T @ ((y + 1.0) / 2.0) - hidden.sum(axis=0)[:, None] * probability
    bias_gradient = hidden.sum(axis=0) - coefficient.sum() * softmax(bias)
    return weight_gradient, bias_gradient


def sample_generator(rng, weight, bias, batch):
    hidden = rng.choice(len(bias), size=batch, p=softmax(bias))
    return 2.0 * (rng.random((batch, weight.shape[1])) < sigmoid(weight)[hidden]) - 1.0, hidden


def joint_score_average(y, hidden, weight, bias, coefficient):
    """A scalar reward times the local residual at the actually active component."""
    gradient_w = np.zeros_like(weight)
    np.add.at(gradient_w, hidden, coefficient[:, None] * ((y + 1.0) / 2.0 - sigmoid(weight)[hidden]))
    gradient_b = np.bincount(hidden, weights=coefficient, minlength=len(bias)) - coefficient.sum() * softmax(bias)
    return gradient_w, gradient_b


def evaluate(states, weight, bias, log_true, log_teacher, modes):
    log_q, _, _ = generator(states, weight, bias)
    q, true, teacher = np.exp(log_q), np.exp(log_true), np.exp(log_teacher)
    mode_q = np.bincount(modes, weights=q, minlength=4)
    mode_true = np.bincount(modes, weights=true, minlength=4)
    return np.array([
        true @ (log_true - log_q), teacher @ (log_teacher - log_q),
        q @ (log_q - log_teacher), -(q @ log_q),
        0.5 * np.abs(q - true).sum(), 0.5 * np.abs(mode_q - mode_true).sum(),
        mode_q.min(), np.max(np.abs(weight - weight.mean(axis=0))), *mode_q,
    ])


METRICS = ["true_forward_kl", "teacher_forward_kl", "teacher_reverse_kl", "entropy", "true_total_variation", "mode_total_variation", "minimum_mode_probability", "component_asymmetry", "mode_0", "mode_1", "mode_2", "mode_3"]


def audit_gradients(states, pairs, features):
    rng = np.random.default_rng(99471)
    weight, bias = rng.normal(size=(4, 6)), rng.normal(size=4)
    theta = rng.normal(size=features.shape[1]) * 0.2
    energy = features @ theta
    log_q, _, _ = generator(states, weight, bias)
    q = np.exp(log_q)
    teacher = softmax(energy)
    coefficient = {
        "free_energy": q * (energy - log_q),
        "importance": teacher,
        "score": softmax(log_q + energy),
        "unfiltered": q,
    }
    objective = {
        "free_energy": lambda logp: np.exp(logp) @ (energy - logp),
        "importance": lambda logp: teacher @ logp,
        "score": lambda logp: logsumexp(logp + energy),
        "unfiltered": lambda logp: np.exp(logp).sum(),
    }
    errors = {}
    epsilon = 1e-5
    for method in coefficient:
        grad_w, grad_b = score_average(states, weight, bias, coefficient[method])
        analytical = np.r_[grad_w.ravel(), grad_b]
        parameter = np.r_[weight.ravel(), bias]
        numerical = np.zeros_like(parameter)
        for index in range(len(parameter)):
            plus, minus = parameter.copy(), parameter.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_logq = generator(states, plus[:24].reshape(4, 6), plus[24:])[0]
            minus_logq = generator(states, minus[:24].reshape(4, 6), minus[24:])[0]
            numerical[index] = (objective[method](plus_logq) - objective[method](minus_logq)) / (2 * epsilon)
        errors[method] = float(np.max(np.abs(analytical - numerical)))
        assert errors[method] < 1e-8, errors
    gradient_w, gradient_b = score_average(states, weight, bias, q)
    assert np.max(np.abs(gradient_w)) < 1e-14
    assert np.max(np.abs(gradient_b)) < 1e-14
    # The joint sampled-hidden score averages to the marginal responsibility score.
    bits = (states + 1.0) / 2.0
    logq, responsibility, probability = generator(states, weight, bias)
    reward = energy - logq
    joint_w = np.einsum("n,n,nk,nkd->kd", q, reward, responsibility, bits[:, None, :] - probability[None, :, :])
    marginal_w = score_average(states, weight, bias, q * reward)[0]
    errors["joint_marginal_identity"] = float(np.max(np.abs(joint_w - marginal_w)))
    assert errors["joint_marginal_identity"] < 1e-14
    expanded_y = np.repeat(states, 4, axis=0)
    expanded_hidden = np.tile(np.arange(4), len(states))
    joint_coefficient = (q[:, None] * responsibility * reward[:, None]).ravel()
    checked_w, checked_b = joint_score_average(expanded_y, expanded_hidden, weight, bias, joint_coefficient)
    expected_w, expected_b = score_average(states, weight, bias, q * reward)
    errors["implemented_joint_score"] = float(max(np.max(np.abs(checked_w - expected_w)), np.max(np.abs(checked_b - expected_b))))
    assert errors["implemented_joint_score"] < 1e-14
    return errors


def train(rng, states, pairs, theta_teacher, log_true, log_teacher, modes, initial_weight, initial_bias, method, steps, batch, learning_rate, eval_every, real_data):
    weight, bias = initial_weight.copy(), initial_bias.copy()
    records = [evaluate(states, weight, bias, log_true, log_teacher, modes)]
    diagnostics = []
    monitored_loss = []
    stop_reason = "maximum budget"
    objective_method = method.removeprefix("joint_")
    for step in range(1, steps + 1):
        if method == "direct_mle":
            y = real_data
        else:
            y, hidden = sample_generator(rng, weight, bias, batch)
        energy = atom_features(y, pairs) @ theta_teacher
        log_q = generator(y, weight, bias)[0]
        if objective_method == "free_energy":
            reward = energy - log_q
            # Leave-one-out baseline: unbiased, unlike including the sample itself.
            coefficient = (reward - reward.mean()) / (batch - 1)
        elif objective_method == "importance":
            coefficient = softmax(energy - log_q)
        elif objective_method == "score":
            coefficient = softmax(energy)
        elif method in ["unfiltered", "direct_mle"]:
            coefficient = np.full(len(y), 1.0 / len(y))
        else:
            raise ValueError(method)
        gradient_w, gradient_b = joint_score_average(y, hidden, weight, bias, coefficient) if method.startswith("joint_") else score_average(y, weight, bias, coefficient)
        weight += learning_rate * gradient_w
        bias += learning_rate * gradient_b
        bias -= bias.mean()
        assert np.isfinite(weight).all() and np.isfinite(bias).all()
        if step % eval_every == 0:
            records.append(evaluate(states, weight, bias, log_true, log_teacher, modes))
            importance = softmax(energy - log_q)
            diagnostics.append([step, float(1.0 / np.square(importance).sum()), float(np.linalg.norm(gradient_w)), float(energy.mean())])
            if method == "direct_mle":
                loss = -generator(real_data, weight, bias)[0].mean()
            elif objective_method == "free_energy":
                loss = records[-1][2]
            elif objective_method in ["importance", "unfiltered"]:
                loss = records[-1][1] - np.exp(log_teacher) @ log_teacher
            elif objective_method == "score":
                state_energy = atom_features(states, pairs) @ theta_teacher
                loss = state_energy.max() - logsumexp(generator(states, weight, bias)[0] + state_energy)
            monitored_loss.append(float(loss))
            # Halt only after BOTH optimized loss and teacher-forward-KL flatten.
            # The true distribution never participates in this stopping decision.
            if step >= 800 and len(monitored_loss) >= 16:
                slopes = []
                for start, end in [(-16, -8), (-8, None)]:
                    log_steps = np.log(np.array(diagnostics)[start:end, 0])
                    slopes.extend([np.polyfit(log_steps, np.log(np.array(monitored_loss)[start:end] + 1e-12), 1)[0], np.polyfit(log_steps, np.log(np.array(records[1:])[start:end, 1] + 1e-12), 1)[0]])
                if max(abs(slope) for slope in slopes) < 0.02:
                    stop_reason = "stationary loss and teacher forward KL"
                    break
    assert steps % eval_every == 0
    stop_step = step
    for step in range(stop_step + eval_every, steps + 1, eval_every):
        records.append(records[-1].copy())
        diagnostics.append([step, *diagnostics[-1][1:]])
    return np.array(records), np.array(diagnostics), weight, bias, stop_step, stop_reason


def make_plots(output, figure_dir):
    arrays = np.load(output / "trajectories.npz")
    metadata = json.loads((output / "metadata.json").read_text())
    x = arrays["steps"]
    labels = {"unfiltered": "Unfiltered self-imitation", "score": "Positive energy weighting", "free_energy": "Free energy (reverse KL)", "importance": "Importance correction (forward KL)", "direct_mle": "Direct MLE on the same real data", "joint_free_energy": "Free energy, sampled hidden", "joint_importance": "Importance correction, sampled hidden"}
    main = [name for name in metadata["configurations"] if name.startswith("main_") and "joint" not in name]
    fig, ax = plt.subplots()
    for name in main:
        values = arrays[name][:, :, 0]
        label = labels[name.removeprefix("main_")]
        line, = ax.plot(x, values.mean(axis=0), label=label)
        low, high = np.quantile(values, [0.1, 0.9], axis=0)
        ax.fill_between(x, low, high, alpha=0.2, color=line.get_color())
    ax.axhline(np.mean(metadata["teacher_true_kl"]), color="black", linestyle="--", label="Frozen learned density")
    ax.set(xlabel="Practice updates", ylabel="KL(true distribution || generator)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-learning.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for name in main:
        values = arrays[name][:, :, 3]
        line, = ax.plot(x, values.mean(axis=0), label=labels[name.removeprefix("main_")])
        low, high = np.quantile(values, [0.1, 0.9], axis=0)
        ax.fill_between(x, low, high, alpha=0.2, color=line.get_color())
    ax.axhline(metadata["true_entropy"], color="black", linestyle="--", label="True entropy")
    ax.set(xlabel="Practice updates", ylabel="Generator entropy (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-entropy.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    names = ["main_importance", "main_free_energy", "single_importance", "single_free_energy", "symmetric_importance", "symmetric_free_energy"]
    for name in names:
        ax.plot(x, arrays[name][:, :, 0].mean(axis=0), label=name.replace("_", " "))
    ax.axhline(np.mean(metadata["teacher_true_kl"]), color="black", linestyle="--", label="Frozen learned density")
    ax.set(xlabel="Practice updates", ylabel="KL(true distribution || generator)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-controls.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    positions = np.arange(4)
    ax.plot(positions, metadata["true_modes"], marker="o", linestyle="--", color="black", label="True")
    for name in main:
        values = arrays[name][:, -1, 8:12]
        ax.plot(positions, values.mean(axis=0), marker="o", label=labels[name.removeprefix("main_")])
    ax.set(xlabel="Signs of the two three-spin blocks", ylabel="Final mode probability", xticks=positions, xticklabels=["--", "-+", "+-", "++"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-modes.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for seed in range(metadata["arguments"]["seeds"]):
        history = arrays[f"recognizer_{seed}"]
        ax.plot(history[:, 0], history[:, 1])
    ax.set(xlabel="Recognizer Newton updates", ylabel="Penalized pseudolikelihood loss")
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-recognizer-learning.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for method in ["free_energy", "importance", "joint_free_energy", "joint_importance"]:
        values = arrays[f"main_{method}"][:, :, 0]
        line, = ax.plot(x, values.mean(axis=0), label=labels[method])
        low, high = np.quantile(values, [0.1, 0.9], axis=0)
        ax.fill_between(x, low, high, alpha=0.2, color=line.get_color())
    ax.axhline(np.mean(metadata["teacher_true_kl"]), color="black", linestyle="--", label="Frozen learned density")
    ax.set(xlabel="Practice updates", ylabel="KL(true distribution || generator)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "practice-density-local-rule.png")
    plt.close(fig)


def main():
    load_dotenv(Path.home() / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "practice_density")
    parser.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parents[1] / "tex" / "archive" / "figures")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=8310)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--real-samples", type=int, default=512)
    parser.add_argument("--penalty", type=float, default=0.001)
    parser.add_argument("--init-scale", type=float, default=0.8)
    parser.add_argument("--components", type=int, default=4)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    states, pairs, true_theta, features, log_true, modes = make_problem(6)
    configurations = {f"main_{method}": (args.components, False, method) for method in ["unfiltered", "score", "free_energy", "importance", "direct_mle", "joint_free_energy", "joint_importance"]}
    configurations |= {f"{kind}_{method}": (1 if kind == "single" else 4, kind == "symmetric", method) for kind in ["single", "symmetric"] for method in ["free_energy", "importance"]}
    metadata = {
        "uk_start": datetime.now(ZoneInfo("Europe/London")).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configurations": configurations, "metrics": METRICS,
        "gradient_audit": audit_gradients(states, pairs, features),
        "true_entropy": float(-(np.exp(log_true) @ log_true)),
        "true_modes": np.bincount(modes, weights=np.exp(log_true), minlength=4).tolist(),
        "teacher_true_kl": [], "recognizer_parameter_error": [],
        "stopping": {name: [] for name in configurations},
        "practice_information": "Only current-generator samples and the fixed learned local atom score; exact enumerated probabilities are used for evaluation and gradient audits only.",
    }
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    archive = {"states": states, "true_theta": true_theta, "log_true": log_true, "steps": np.arange(0, args.steps + 1, args.eval_every)}
    for name in configurations:
        archive[name], archive[name + "_diagnostics"] = [], []
    for seed_index in tqdm(range(args.seeds), desc="Density experiment seeds"):
        seed = args.seed_start + seed_index
        data_rng = np.random.default_rng(seed)
        data = states[data_rng.choice(len(states), size=args.real_samples, p=np.exp(log_true))]
        theta, history = fit_pseudolikelihood(data, pairs, args.penalty)
        log_teacher = features @ theta
        log_teacher -= logsumexp(log_teacher)
        metadata["teacher_true_kl"].append(float(np.exp(log_true) @ (log_true - log_teacher)))
        metadata["recognizer_parameter_error"].append(float(np.linalg.norm(theta - true_theta)))
        archive[f"recognizer_{seed_index}"] = history
        archive[f"teacher_{seed_index}"] = theta
        archive[f"real_data_{seed_index}"] = data
        for config_index, (name, (components, symmetric, method)) in enumerate(tqdm(configurations.items(), desc=f"Seed {seed}", leave=False)):
            init_rng = np.random.default_rng(seed + 100000)
            initial_weight = np.zeros((components, 6)) if symmetric else init_rng.normal(0, args.init_scale, size=(components, 6))
            initial_bias = np.zeros(components)
            result = train(np.random.default_rng(seed + 200000 + 1000 * config_index), states, pairs, theta, log_true, log_teacher, modes, initial_weight, initial_bias, method, args.steps, args.batch, args.learning_rate, args.eval_every, data)
            archive[name].append(result[0])
            archive[name + "_diagnostics"].append(result[1])
            archive[f"{name}_weight_{seed_index}"] = result[2]
            archive[f"{name}_bias_{seed_index}"] = result[3]
            metadata["stopping"][name].append({"step": result[4], "reason": result[5]})
    for name in configurations:
        archive[name] = np.stack(archive[name])
        archive[name + "_diagnostics"] = np.stack(archive[name + "_diagnostics"])
    np.savez_compressed(args.output / "trajectories.npz", **archive)
    metadata["uk_finish"] = datetime.now(ZoneInfo("Europe/London")).isoformat()
    summary = {}
    for name in configurations:
        summary[name] = {metric: {"initial_mean": float(archive[name][:, 0, index].mean()), "final_mean": float(archive[name][:, -1, index].mean()), "final_sd": float(archive[name][:, -1, index].std(ddof=1)), "final_min": float(archive[name][:, -1, index].min()), "final_max": float(archive[name][:, -1, index].max())} for index, metric in enumerate(METRICS)}
    metadata["summary"] = summary
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_plots(args.output, args.figure_dir)
    print(json.dumps({"output": str(args.output), "teacher_true_kl_mean": float(np.mean(metadata["teacher_true_kl"])), "gradient_audit": metadata["gradient_audit"], "final_true_kl": {name: summary[name]["true_forward_kl"] for name in configurations}}, indent=2))


if __name__ == "__main__":
    main()
