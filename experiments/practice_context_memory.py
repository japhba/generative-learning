"""Run practice entirely through a growing context; audit against weight SGD.

The learner retains fixed base weights and stores latent keys plus scored output
residuals. A separate explicit-weight learner is only an equivalence audit.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

import practice_density as density
import practice_density_sigmoid as sbn


def main():
    load_dotenv(Path.home() / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="importance")
    parser.add_argument("--source", type=Path, default=Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "practice_density_sigmoid")
    parser.add_argument("--method", choices=["importance", "free_energy"], default="importance")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=193000)
    args = parser.parse_args()
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_context" / args.name
    assert not output.exists()
    output.mkdir(parents=True)
    source = np.load(args.source / "trajectories.npz")
    source_meta = json.loads((args.source / "metadata.json").read_text())
    states, pairs, _, features, _, modes = density.make_problem(6)
    teacher, log_true = source["teacher_0"], source["log_true"]
    log_teacher = features @ teacher
    log_teacher -= density.logsumexp(log_teacher)
    initial_rng = np.random.default_rng(source_meta["arguments"]["seed_start"] + 100000)
    base_w, base_b, base_a = initial_rng.normal(0, 0.5, (6, 2)), np.zeros(6), np.zeros(2)
    base_augmented = np.column_stack((base_w, base_b))
    queries = np.column_stack((sbn.HIDDEN, np.ones(4)))
    weight, bias, prior = base_w.copy(), base_b.copy(), base_a.copy()
    keys, output_values, prior_values = np.empty((args.steps * args.batch, 3)), np.empty((args.steps * args.batch, 6)), np.empty((args.steps * args.batch, 2))
    rng = np.random.default_rng(args.seed)
    curves, max_error, first_error = [], 0.0, 0.0
    for step in tqdm(range(args.steps + 1), desc="Context-only practice"):
        count = step * args.batch
        # These are actual memory queries; no compressed fast-weight matrix.
        logits = queries @ base_augmented.T + (queries @ keys[:count].T) @ output_values[:count]
        context_prior = base_a + prior_values[:count].sum(axis=0)
        log_prior = ((sbn.HIDDEN + 1) / 2) @ context_prior - np.logaddexp(0, context_prior).sum()
        expected_logits, expected_log_prior = sbn.components(weight, bias, prior)
        error = max(np.max(np.abs(logits - expected_logits)), np.max(np.abs(log_prior - expected_log_prior)))
        max_error = max(max_error, float(error))
        assert error < 1e-8, (step, error)
        if step % 20 == 0:
            metrics = density.evaluate(states, logits, log_prior, log_true, log_teacher, modes)
            explicit = sbn.evaluate(states, weight, bias, prior, log_true, log_teacher, modes)
            curves.append({"step": step, "context_entries": count, "true_kl": float(metrics[0]), "explicit_true_kl": float(explicit[0]), "teacher_kl": float(metrics[1]), "entropy": float(metrics[3]), "max_logit_error": float(error)})
            assert np.isfinite(metrics).all()
        if step == args.steps:
            break
        h_uniforms, y_uniforms = rng.random((args.batch, 2)), rng.random((args.batch, 6))
        h_bits = h_uniforms < density.sigmoid(context_prior)
        hidden = 2.0 * h_bits - 1
        hidden_ids = 2 * h_bits[:, 0].astype(int) + h_bits[:, 1]
        probability = density.sigmoid(logits[hidden_ids])
        y = 2.0 * (y_uniforms < probability) - 1
        logq = density.generator(y, logits, log_prior)[0]
        energy = density.atom_features(y, pairs) @ teacher
        if args.method == "importance":
            coefficient = density.softmax(energy - logq)
        else:
            reward = energy - logq
            coefficient = (reward - reward.mean()) / (args.batch - 1)
        keys[count:count + args.batch] = np.column_stack((hidden, np.ones(args.batch)))
        output_values[count:count + args.batch] = args.learning_rate * coefficient[:, None] * ((y + 1) / 2 - probability)
        prior_values[count:count + args.batch] = args.learning_rate * coefficient[:, None] * (h_bits - density.sigmoid(context_prior))
        # Separately execute the same random rollout and update in weight space.
        explicit_hidden = 2.0 * (h_uniforms < density.sigmoid(prior)) - 1
        explicit_y = 2.0 * (y_uniforms < density.sigmoid(explicit_hidden @ weight.T + bias)) - 1
        assert np.array_equal(hidden, explicit_hidden) and np.array_equal(y, explicit_y)
        explicit_logq = sbn.log_density(explicit_y, weight, bias, prior)
        if args.method == "importance":
            explicit_coefficient = density.softmax(energy - explicit_logq)
        else:
            explicit_reward = energy - explicit_logq
            explicit_coefficient = (explicit_reward - explicit_reward.mean()) / (args.batch - 1)
        dw, db, da = sbn.gradient(explicit_y, explicit_hidden, weight, bias, prior, explicit_coefficient, True)
        weight += args.learning_rate * dw
        bias += args.learning_rate * db
        prior += args.learning_rate * da
    metadata = {"uk_finish": datetime.now(ZoneInfo("Europe/London")).isoformat(), "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "maximum_logit_discrepancy": max_error, "curves": curves, "frozen_base_weights": base_w.tolist(), "frozen_base_bias": base_b.tolist(), "frozen_base_prior": base_a.tolist(), "learned_teacher": teacher.tolist(), "explicit_final_weights": weight.tolist(), "explicit_final_bias": bias.tolist(), "explicit_final_prior": prior.tolist()}
    (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "source.py").write_text(Path(__file__).read_text())
    np.savez_compressed(output / "context.npz", keys=keys, output_values=output_values, prior_values=prior_values)
    figure_dir = Path(__file__).resolve().parents[1] / "tex" / "archive" / "figures"
    fig, ax = plt.subplots()
    ax.plot([r["context_entries"] for r in curves], [r["true_kl"] for r in curves], label="Fixed weights + growing context")
    ax.plot([r["context_entries"] for r in curves], [r["explicit_true_kl"] for r in curves], linestyle="--", label="Explicit local weight updates")
    ax.axhline(np.exp(log_true) @ (log_true - log_teacher), linestyle=":", color="gray", label="Frozen learned density")
    ax.set(xlabel="Stored practice rollouts", ylabel="KL(true distribution || generator)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"practice-context-{args.method}-learning.png")
    plt.close(fig)
    print(json.dumps({"output": str(output), "maximum_logit_discrepancy": max_error, "first": curves[0], "final": curves[-1]}, indent=2))


if __name__ == "__main__":
    main()
