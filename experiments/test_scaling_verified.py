"""Check the rejection rule and the heterogeneous accepted-data mechanism."""

import itertools
import unittest

import numpy as np


class VerifiedSamplingTest(unittest.TestCase):
    def test_selection_never_trains_a_failed_group(self):
        import torch
        from train_scaling_verified import choose
        pairs = torch.tensor(list(itertools.product(range(4), repeat=2)))
        for target in range(4):
            chosen, accepted, _, _ = choose(pairs, torch.full((len(pairs),), target))
            torch.testing.assert_close(accepted, (pairs == target).any(-1))
            self.assertTrue(bool((chosen[accepted] == target).all()))

    def test_accepted_mean_removes_homogeneous_yield_factor(self):
        # Exact enumeration of two questions, with two candidates each.
        p = .71
        for source_p in (.23, .71):
            q = 1 - (1 - source_p) ** 2
            total = 0.
            for a, b in itertools.product((0, 1), repeat=2):
                probability = q ** (a + b) * (1 - q) ** (2 - a - b)
                if a + b: total += probability * (1 - p)
            self.assertAlmostEqual(total, (1 - (1 - q) ** 2) * (1 - p))

    def test_accepted_distribution_changes_under_refresh(self):
        rng = np.random.default_rng(9123)
        initial = np.array([.2, .8]); current = np.array([.8, .95])
        for source_p in (initial, current):
            classes = rng.integers(2, size=300000)
            draws = rng.random((len(classes), 2)) < source_p[classes, None]
            accepted = draws.any(-1)
            empirical = np.bincount(classes[accepted], minlength=2) / accepted.sum()
            q = 1 - (1 - source_p) ** 2
            np.testing.assert_allclose(empirical, q / q.sum(), atol=.004)


if __name__ == "__main__": unittest.main()
