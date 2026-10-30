"""Offline checks of Zipf sampling, weighted evaluation, and Bayes references."""

import unittest

import numpy as np

from experiments.zipf_task import BUDGETS, LABELS, P, TEST_BINS, VOCABULARY, coverage_error, ideal_curves, label, make_world


class ZipfTaskTests(unittest.TestCase):
    def test_world_is_reproducible_and_labels_are_private_functions(self):
        world = make_world(42)
        self.assertEqual(world, make_world(42))
        self.assertEqual(len(world["passive_keys"]), 128)
        self.assertTrue(all(1 <= key <= VOCABULARY for key in world["passive_keys"]))
        self.assertEqual(world["test_labels"], [label(world, key) for key in world["test_keys"]])
        self.assertTrue(all(value in LABELS for value in world["test_labels"]))
        other = make_world(43)
        self.assertNotEqual([label(world, key) for key in range(1, 17)], [label(other, key) for key in range(1, 17)])
        self.assertEqual(label(world, np.int64(17)), label(world, 17))

    def test_stratified_evaluation_preserves_probability_mass(self):
        world = make_world(17)
        keys, weights = world["test_keys"], world["test_weights"]
        self.assertEqual(len(keys), 40)
        self.assertEqual(len(weights), 40)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertEqual(keys[:8], list(range(1, 9)))
        np.testing.assert_allclose(weights[:8], P[:8])
        for i, (lower, upper) in enumerate(TEST_BINS):
            start = 8 + 4 * i
            self.assertTrue(all(lower <= key <= upper for key in keys[start:start + 4]))
            self.assertAlmostEqual(sum(weights[start:start + 4]), float(P[lower - 1:upper].sum()))

    def test_coverage_depends_only_on_unique_observed_keys(self):
        self.assertEqual(coverage_error([]), 0.75)
        self.assertEqual(coverage_error([1, 2, 1, 2]), coverage_error([2, 1]))
        self.assertAlmostEqual(coverage_error([1, 2]), 0.75 * (1 - P[0] - P[1]))
        self.assertAlmostEqual(coverage_error(list(range(1, VOCABULARY + 1))), 0.0)

    def test_exact_reference_endpoints_and_one_sample_identity(self):
        curves = ideal_curves([0, 1, *BUDGETS[1:], VOCABULARY])
        self.assertAlmostEqual(curves["iid_errors"][0], 0.75)
        self.assertAlmostEqual(curves["ranked_errors"][0], 0.75)
        self.assertAlmostEqual(curves["iid_errors"][1], 0.75 * (1 - np.sum(P ** 2)))
        self.assertEqual(curves["ranked_errors"][-1], 0.0)
        for budget, ranked in zip(curves["budgets"], curves["ranked_errors"], strict=True):
            self.assertAlmostEqual(ranked, coverage_error(list(range(1, budget + 1))))
        self.assertTrue(np.all(np.diff(curves["iid_errors"]) < 0))
        self.assertTrue(np.all(np.diff(curves["ranked_errors"]) < 0))

    def test_invalid_keys_and_budgets_fail(self):
        world = make_world(1)
        for key in (0, VOCABULARY + 1):
            with self.assertRaises(ValueError): label(world, key)
            with self.assertRaises(ValueError): coverage_error([key])
        with self.assertRaises(TypeError): label(world, 1.5)
        with self.assertRaises(TypeError): coverage_error([1.5])
        with self.assertRaises(ValueError): ideal_curves([-1])


if __name__ == "__main__": unittest.main()
