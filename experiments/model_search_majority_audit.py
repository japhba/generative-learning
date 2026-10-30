"""One-shot bypasses and independent checks for the majority-circuit search."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit
from tqdm.auto import tqdm

from model_search_majority import CATEGORIES, D, FLOOR, INVERSE, POWERS, X, Z, expected_counts, law, local_step


SIZES = np.bincount(INVERSE)
SINGLE = np.flatnonzero(SIZES == 1)
SINGLE_CAUSES = np.array([np.flatnonzero(INVERSE == i)[0] for i in SINGLE])
SINGLE_DESIGN = np.column_stack((np.ones(len(SINGLE)), Z[SINGLE_CAUSES] - 0.5))
WITNESS = []
for bit in range(D):
    flipped = INVERSE[np.arange(len(Z)) ^ (1 << bit)]
    graph = coo_matrix((np.ones(len(Z)), (INVERSE, flipped)), shape=(len(CATEGORIES), len(CATEGORIES))).tocsr()
    n_components, component = connected_components(graph, directed=False)
    sums = np.bincount(INVERSE, weights=Z[:, bit])
    determined = (sums == 0) | (sums == SIZES)
    good = np.bincount(component, weights=(~determined).astype(int), minlength=n_components) == 0
    WITNESS.append((good[component], sums == SIZES))


def log_inverse(counts, alpha, ridge, weighted):
    design = SINGLE_DESIGN
    target = np.log(counts[SINGLE] + alpha)
    weights = counts[SINGLE] + alpha if weighted else np.ones(len(SINGLE))
    penalty = np.diag([0] + [ridge] * D)
    coefficient = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty, design.T @ (weights * target))
    return np.clip(expit(coefficient[1:]), FLOOR, 1 - FLOOR)


def witnesses(counts, alpha):
    return np.clip(np.array([(counts[mask & value].sum() + alpha) / (counts[mask].sum() + 2 * alpha) for mask, value in WITNESS]), FLOOR, 1 - FLOOR)


MENU = [("log", a, r, w) for a in [0.01, 0.1, 0.5, 1, 5] for r in [0, 1, 10, 100] for w in [False, True]] + [("witness", a, 0, False) for a in [0.1, 0.5, 1, 5, 10]]


def estimate(counts, option):
    kind, alpha, ridge, weighted = option
    return log_inverse(counts, alpha, ridge, weighted) if kind == "log" else witnesses(counts, alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    records = []
    source = list(sorted(args.directory.glob("world-*-practice.json")))
    for file in tqdm(source, desc="one-shot controls"):
        result = json.loads(file.read_text())
        counts = np.array(result["data"])
        teacher = np.array(result["teacher"])
        _, truth = law(teacher)
        rng = np.random.default_rng(result["seed"] + 200000)
        order = rng.permutation(np.repeat(np.arange(len(counts)), counts))
        cv = []
        for option in MENU:
            score = 0.0
            for fold in np.array_split(order, 4):
                held = np.bincount(fold, minlength=len(counts))
                _, prediction = law(estimate(counts - held, option))
                score -= held @ np.log(prediction)
            cv.append(score)
        selected = int(np.argmin(cv))
        candidate_results = []
        for option in MENU:
            p = estimate(counts, option)
            _, prediction = law(p)
            candidate_results.append({"option": option, "kl": float(truth @ np.log(truth / prediction)), "nll": float(-(counts / counts.sum()) @ np.log(prediction))})
        record = {"seed": result["seed"], "selected": candidate_results[selected], "oracle_menu_kl": min(row["kl"] for row in candidate_results), "all": candidate_results, "practice_kl": result["history"][-1]["kl"], "exact_likelihood_gain": -result["history"][-1]["nll"] - result["initial_likelihood"], "bound_certificate": result["history"][-1]["certificate"], "initial_kl": result["history"][0]["kl"], "stop_step": result["history"][-1]["step"]}
        records.append(record)
    rng = np.random.default_rng(20106999)
    bootstrap = rng.integers(0, len(records), size=(20000, len(records)))
    contrast = np.array([row["practice_kl"] - row["selected"]["kl"] for row in records])
    summary = {"single_preimages": int(len(SINGLE)), "log_design_rank": int(np.linalg.matrix_rank(SINGLE_DESIGN)), "witness_probability_uniform": [float(SIZES[mask].sum() / len(Z)) for mask, _ in WITNESS], "mean_cv_shortcut_kl": float(np.mean([row["selected"]["kl"] for row in records])), "mean_oracle_menu_kl": float(np.mean([row["oracle_menu_kl"] for row in records])), "practice_minus_cv": float(contrast.mean()), "practice_minus_cv_ci": np.quantile(contrast[bootstrap].mean(axis=1), [0.025, 0.975]).tolist(), "minimum_actual_likelihood_gain": min(row["exact_likelihood_gain"] for row in records), "minimum_bound_certificate": min(row["bound_certificate"] for row in records), "positive_bound_certificates": sum(row["bound_certificate"] > 0 for row in records), "records": records}
    original = X.astype(int)
    independent = Z.astype(int)
    for _ in range(2): independent = np.array([[int(row[(i - 1) % D] + row[i] + row[(i + 1) % D] >= 2) for i in range(D)] for row in independent])
    assert np.array_equal(original, independent)
    assert np.array_equal((independent @ POWERS).astype(int), CATEGORIES[INVERSE])
    gradients = []
    for _ in range(20):
        p = 0.1 + 0.8 * rng.random(D)
        frequencies = rng.dirichlet(np.ones(len(CATEGORIES)))
        m, mass, _, _ = expected_counts(p, frequencies, np.ones(len(Z), dtype=bool))
        s = (p - FLOOR) / (1 - 2 * FLOOR)
        analytic = (1 - 2 * FLOOR) * s * (1 - s) * (m / p - (mass - m) / (1 - p))
        u = np.log(s / (1 - s))
        numerical = []
        for j in range(D):
            shift = np.zeros(D)
            shift[j] = 1e-5
            plus = law(FLOOR + (1 - 2 * FLOOR) * expit(u + shift))[1]
            minus = law(FLOOR + (1 - 2 * FLOOR) * expit(u - shift))[1]
            numerical.append(frequencies @ (np.log(plus) - np.log(minus)) / 2e-5)
        gradients.append(np.max(np.abs(analytic - numerical)))
        local_step(p, m, mass)
    summary["maximum_gradient_error"] = float(max(gradients))
    (args.directory / "shortcut-audit.json").write_text(json.dumps(summary, indent=2))
    printable = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__": main()
