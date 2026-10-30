"""Zipf-distributed random associations with exact coverage learning curves."""

from hashlib import blake2b
from operator import index

import numpy as np

VOCABULARY = 65_536
ALPHA = 1.5
LABELS = "ABCD"
BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128]
P = np.arange(1, VOCABULARY + 1, dtype=np.float64) ** -ALPHA
P /= P.sum()
P.setflags(write=False)
TEST_BINS = ((9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 1024), (1025, 4096), (4097, VOCABULARY))


def label(world: dict, key: int) -> str:
    """Return a reproducible random label; the episode seed stays private."""
    key = index(key)
    if not 1 <= key <= VOCABULARY: raise ValueError(f"Key must lie between 1 and {VOCABULARY}: {key}")
    digest = blake2b(f"{world['seed']}:{key}".encode(), digest_size=1).digest()
    return LABELS[digest[0] % len(LABELS)]


def make_world(seed: int) -> dict:
    """Sample an IID curriculum and a weighted, stratified 40-item test set.

    Ranks 1--8 are evaluated exactly. Every remaining stratum contributes four
    draws with replacement from its conditional Zipf distribution. A draw's
    weight is its stratum's mass divided by four, giving an unbiased estimate
    of full-distribution error for any fixed prediction function.
    """
    rng = np.random.default_rng(seed)
    passive_keys = (rng.choice(VOCABULARY, size=max(BUDGETS), p=P) + 1).tolist()
    test_keys, test_weights = list(range(1, 9)), P[:8].tolist()
    for lower, upper in TEST_BINS:
        probabilities = P[lower - 1:upper]
        mass = float(probabilities.sum())
        keys = rng.choice(upper - lower + 1, size=4, replace=True, p=probabilities / mass) + lower
        test_keys.extend(keys.tolist())
        test_weights.extend([mass / 4] * 4)
    world = {"seed": seed, "passive_keys": passive_keys, "test_keys": test_keys, "test_weights": test_weights}
    world["test_labels"] = [label(world, key) for key in test_keys]
    return world


def coverage_error(keys: list[int]) -> float:
    """Bayes error after observing these keys: seen labels known, others uniform."""
    unique = np.unique(np.asarray([index(key) for key in keys], dtype=np.int64))
    if np.any((unique < 1) | (unique > VOCABULARY)): raise ValueError(f"Keys must lie between 1 and {VOCABULARY}.")
    return float((1 - 1 / len(LABELS)) * (1 - P[unique - 1].sum()))


def ideal_curves(budgets: list[int]) -> dict:
    """Exact finite-vocabulary error averaged over labels and IID curricula."""
    budgets = [index(budget) for budget in budgets]
    if any(not 0 <= budget <= VOCABULARY for budget in budgets): raise ValueError(f"Budgets must lie between 0 and {VOCABULARY}.")
    chance_error = 1 - 1 / len(LABELS)
    return {
        "budgets": budgets,
        "iid_errors": [float(chance_error * np.sum(P * (1 - P) ** budget)) for budget in budgets],
        "ranked_errors": [float(chance_error * P[budget:].sum()) for budget in budgets],
    }
