"""Shared task, acquisition streams, and evaluation for the matched Zipf experiment."""

import json
from pathlib import Path

import numpy as np

from zipf_task import BUDGETS, LABELS, P, VOCABULARY, label, make_world

ROOT = Path(__file__).resolve().parents[1]
PILOT_SEEDS = (51000, 51001, 51002, 51003)
MAIN_SEEDS = tuple(range(4000, 4032))
CURRICULA = ("iid", "ranked", "historical_qwen")


def prompt(key):
    assert 1 <= key <= VOCABULARY
    return f"Dictionary key: {key}\nLabel:"


def oracle(seed, key):
    return LABELS.index(label({"seed": seed}, key))


def historical_queries(model, seed, run):
    path = ROOT / f"results/{run}/world_results/{model.replace('/', '__')}__{seed}.json"
    rows = json.loads(path.read_text())
    return next(row["query_keys"] for row in rows if row["condition"] == "active_generate" and row["budget"] == 128)


def acquisition(seed, curriculum):
    if curriculum == "iid": return make_world(seed)["passive_keys"]
    if curriculum == "ranked": return list(range(1, 129))
    assert curriculum == "historical_qwen"
    return historical_queries("qwen/qwen3-30b-a3b-instruct-2507", seed, "zipf_replication")


def replay_indices(seed, repeats):
    rng = np.random.default_rng(seed + 720000)
    return np.array([[[n, *rng.integers(0, n + 1, 31).tolist()] for _ in range(repeats)] for n in range(128)], dtype=np.int64)


def evaluate_probabilities(seed, queried, keys, probabilities, include_test):
    keys = np.asarray(keys, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    predictions = probabilities.argmax(axis=1)
    labels = np.array([oracle(seed, int(key)) for key in keys])
    indices = {int(key): i for i, key in enumerate(keys)}
    seen = sorted(set(queried))
    seen_rows = np.array([indices[key] for key in seen], dtype=int)
    coverage = float(P[np.array(seen, dtype=int) - 1].sum())
    seen_mistakes = predictions[seen_rows] != labels[seen_rows]
    seen_error_mass = float(P[np.array(seen, dtype=int) - 1] @ seen_mistakes)
    result = {"observed_unique": len(seen), "known_mass": coverage, "coverage_error": .75 * (1 - coverage),
              "seen_accuracy": float(1 - seen_mistakes.mean()) if seen else None,
              "seen_error_mass": seen_error_mass, "marginalized_error": .75 * (1 - coverage) + seen_error_mass}
    if include_test:
        world = make_world(seed)
        test_rows = np.array([indices[key] for key in world["test_keys"]])
        test_p = probabilities[test_rows]
        targets = labels[test_rows]
        weights = np.array(world["test_weights"])
        result.update({"error": float(weights @ (predictions[test_rows] != targets)),
                       "probability_mse": float(weights @ np.sum((test_p - np.eye(4)[targets]) ** 2, axis=1)),
                       "functional_overlap": float(weights @ (test_p[np.arange(len(targets)), targets] - .25) / .75)})
    return result


def verify_original_worlds():
    saved = json.loads((ROOT / "results/zipf_main/worlds.json").read_text())
    assert saved == [make_world(seed) for seed in MAIN_SEEDS]
    for seed in MAIN_SEEDS:
        assert historical_queries("google/gemini-2.5-flash-lite", seed, "zipf_main") == acquisition(seed, "ranked")
        assert historical_queries("mistralai/mistral-small-3.2-24b-instruct", seed, "zipf_replication") == acquisition(seed, "ranked")

