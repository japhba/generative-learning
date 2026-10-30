"""Validate the Zipf coverage toy with dense linear weights and actual row SGD."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from zipf_task import ALPHA, BUDGETS, LABELS, P, VOCABULARY

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("weighted_mse", "normalized_overlap", "weighted_argmax_error", "bayes_unseen_error")
ETAS = (1.0, 0.25)
POLICIES = ("iid", "ranked")


def measure(weights, truth, labels, counts, eta, probabilities):
    """Read metrics from actual dense weights, then check independent count identities."""
    chance_error = 1 - 1 / truth.shape[1]
    residual = weights - truth
    mse = float(np.einsum("rk,rk,r->", residual, residual, probabilities))
    overlap = float(np.einsum("rk,rk,r->", weights, truth, probabilities) / chance_error)
    predictions = np.argmax(weights, axis=1)  # Adding the same uniform baseline leaves argmax unchanged.
    classification = float(probabilities @ (predictions != labels))
    unseen = counts == 0
    bayes = float(chance_error * probabilities[unseen].sum())
    attenuation = (1 - eta) ** counts
    np.testing.assert_allclose(weights, (1 - attenuation[:, None]) * truth, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(mse, chance_error * (probabilities @ attenuation**2), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(overlap, probabilities @ (1 - attenuation), rtol=1e-12, atol=1e-14)
    assert np.array_equal(predictions[~unseen], labels[~unseen])
    np.testing.assert_allclose(classification, probabilities @ (unseen & (labels != 0)), rtol=1e-12, atol=1e-14)
    if eta == 1: np.testing.assert_allclose(mse, bayes, rtol=1e-12, atol=1e-14)
    return (mse, overlap, classification, bayes)


def exact_expectations(policy, eta, budgets, probabilities, classes):
    chance_error = 1 - 1 / classes
    answer = []
    for budget in budgets:
        if policy == "iid":
            bayes = chance_error * np.sum(probabilities * (1 - probabilities) ** budget)
            mse = chance_error * np.sum(probabilities * (1 - probabilities * (2 * eta - eta**2)) ** budget)
            overlap = np.sum(probabilities * (1 - (1 - eta * probabilities) ** budget))
        else:
            coverage = probabilities[:budget].sum()
            bayes = chance_error * probabilities[budget:].sum()
            mse = chance_error * (1 - (2 * eta - eta**2) * coverage)
            overlap = eta * coverage
        answer.append((mse, overlap, bayes, bayes))
    return np.asarray(answer)


def simulate(worlds, first_seed):
    classes, budgets = len(LABELS), np.asarray(BUDGETS)
    conditions = [(policy, eta) for policy in POLICIES for eta in ETAS]
    values = np.empty((len(conditions), worlds, len(budgets), len(METRICS)))
    tiny_weights = []
    checkpoint_indices = {int(budget): index for index, budget in enumerate(budgets)}
    for world in tqdm(range(worlds), desc="Dense-weight worlds"):
        rng = np.random.default_rng(first_seed + world)
        labels = rng.integers(classes, size=VOCABULARY)
        truth = np.eye(classes)[labels] - 1 / classes
        iid_ranks = rng.choice(VOCABULARY, size=int(budgets[-1]), p=P)
        for condition, (policy, eta) in enumerate(conditions):
            ranks = iid_ranks if policy == "iid" else np.arange(budgets[-1])
            weights, counts = np.zeros_like(truth), np.zeros(VOCABULARY, dtype=np.int64)
            for presentation, rank in enumerate(ranks, start=1):
                weights[rank] += eta * (truth[rank] - weights[rank])
                counts[rank] += 1
                if presentation in checkpoint_indices:
                    values[condition, world, checkpoint_indices[presentation]] = measure(weights, truth, labels, counts, eta, P)
                if world == 0 and policy == "ranked" and presentation == 4:
                    tiny_weights.append({"eta": eta, "budget": presentation, "ranks": [1, 2, 3],
                                         "true_labels": [LABELS[index] for index in labels[:3]],
                                         "true_weights": truth[:3].tolist(), "learned_weights": weights[:3].tolist()})
    for policy_index in range(len(POLICIES)):
        np.testing.assert_array_equal(values[2 * policy_index, :, :, 2:], values[2 * policy_index + 1, :, :, 2:])
    result = []
    for index, (policy, eta) in enumerate(conditions):
        expected = exact_expectations(policy, eta, budgets, P, classes)
        average, standard_error = values[index].mean(axis=0), values[index].std(axis=0, ddof=1) / np.sqrt(worlds)
        if policy == "ranked": np.testing.assert_allclose(average[:, [0, 1, 3]], expected[:, [0, 1, 3]], rtol=1e-12, atol=1e-14)
        result.append({"policy": policy, "eta": eta,
                       "world_metrics": {name: values[index, :, :, column].tolist() for column, name in enumerate(METRICS)},
                       "mean": {name: average[:, column].tolist() for column, name in enumerate(METRICS)},
                       "standard_error": {name: standard_error[:, column].tolist() for column, name in enumerate(METRICS)},
                       "exact_expectation": {name: expected[:, column].tolist() for column, name in enumerate(METRICS)},
                       "max_absolute_mean_minus_expectation": {name: float(np.max(np.abs(average[:, column] - expected[:, column]))) for column, name in enumerate(METRICS)}})
    window = budgets >= 8
    exponents = {row["policy"]: -float(np.polyfit(np.log(budgets[window]), np.log(np.asarray(row["exact_expectation"]["weighted_mse"])[window]), 1)[0])
                 for row in result if row["eta"] == 1}
    return result, tiny_weights, exponents


def plot(result, output, worlds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    for row in result:
        policy, eta = row["policy"], row["eta"]
        color, style = ("C0" if policy == "iid" else "C1"), ("-" if eta == 1 else "--")
        label = f"{'IID' if policy == 'iid' else 'Ranked'}, η={eta:g}"
        axis.plot(BUDGETS, row["exact_expectation"]["weighted_mse"], color=color, linestyle=style, label=label)
        axis.errorbar(BUDGETS, row["mean"]["weighted_mse"], yerr=row["standard_error"]["weighted_mse"],
                      color=color, marker="o" if eta == 1 else "s", linestyle="none")
    axis.set(xscale="log", yscale="log", xlabel="Label presentations", ylabel="Test-weighted squared weight error",
             title=f"Actual row SGD: mean of {worlds} dictionaries\nLines: exact expectations; bars: ±1 standard error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "weighted_mse.png")
    plt.close(figure)

    figure, axis = plt.subplots()
    for row in result:
        if row["eta"] != 1: continue
        color = "C0" if row["policy"] == "iid" else "C1"
        label = f"{'IID' if row['policy'] == 'iid' else 'Ranked'} (η=1 and 0.25)"
        axis.plot(BUDGETS, row["exact_expectation"]["weighted_argmax_error"], color=color, label=label)
        axis.errorbar(BUDGETS, row["mean"]["weighted_argmax_error"], yerr=row["standard_error"]["weighted_argmax_error"],
                      color=color, marker="o", linestyle="none")
    axis.set(xscale="log", yscale="log", xlabel="Label presentations", ylabel="Test-weighted argmax classification error",
             title=f"Same classification at both learning rates\nLines: exact expectations; points: {worlds}-world means ±1 SE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "classification_error.png")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "weight_coverage_toy")
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--first-seed", type=int, default=8800)
    args = parser.parse_args()
    assert args.worlds >= 2
    load_dotenv(Path.home() / ".env")
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "matplotlib")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    conditions, tiny_weights, exponents = simulate(args.worlds, args.first_seed)
    result = {"config": {"vocabulary": VOCABULARY, "classes": len(LABELS), "zipf_exponent": ALPHA, "budgets": BUDGETS,
                         "etas": list(ETAS), "worlds": args.worlds, "first_seed": args.first_seed,
                         "seeds": list(range(args.first_seed, args.first_seed + args.worlds)),
                         "weights": "Dense float64 single-layer matrices updated by actual supervised row SGD; no hidden layers.",
                         "pairing": "Within each world, the true dictionary and IID presentation sequence are identical at both learning rates; ranked queries use the same dictionary.",
                         "argmax": "Deterministic ties predict label A; reported realized weighted error is distinct from posterior Bayes unseen error.",
                         "error_bars": "Standard error across independent dictionary/exposure worlds; no hypothesis testing or exponent fitting to noisy simulations.",
                         "source_sha256": {name: hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest() for name in ("simulate_weight_coverage.py", "zipf_task.py")}},
              "conditions": conditions, "illustrative_weights": tiny_weights, "exact_eta1_exponents_n8_to128": exponents,
              "numerical_validation": "All dense matrices, MSE, overlap, realized argmax error and eta1 posterior identities checked against independent count formulas at all checkpoints; eta-invariant classification checked exactly.",
              "simulation_seconds": time.monotonic() - start}
    (output / "simulation.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    plot(conditions, output, args.worlds)
    print(json.dumps({"simulation_seconds": result["simulation_seconds"], "exact_eta1_exponents": exponents,
                      "endpoints": [{"policy": row["policy"], "eta": row["eta"], "mean": {name: values[-1] for name, values in row["mean"].items()},
                                     "exact": {name: values[-1] for name, values in row["exact_expectation"].items()}} for row in conditions]}, indent=2))
    print(output / "weighted_mse.png")
    print(output / "classification_error.png")


if __name__ == "__main__": main()
