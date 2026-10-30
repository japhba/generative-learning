"""Independent checks of the selected score and Gaussian overlap equations."""

from itertools import product
import unittest

import numpy as np

from bestn_theory import drift, multiplier, overlap_velocity, sigmoid


class BestNTests(unittest.TestCase):
    def test_full_categorical_score_identity(self):
        probabilities = np.array([.07, .13, .21, .26, .33])
        reward = np.array([0, 1, 0, 1, 0])
        p = probabilities @ reward
        expected_reward_gradient = probabilities * (reward - p)
        for n in (1, 2, 3, 4):
            score = np.zeros(5)
            for group in product(range(5), repeat=n):
                group = np.array(group)
                tied = group[reward[group] == reward[group].max()]
                selected = np.bincount(tied, minlength=5) / len(tied)
                score += np.prod(probabilities[group]) * (selected - probabilities)
            np.testing.assert_allclose(score, multiplier(p, n) * expected_reward_gradient, atol=3e-15)

    def test_overlap_equation(self):
        w = np.array([.3, .7, -.2])
        norm = np.linalg.norm(w)
        gradient = (np.eye(3)[0] - w[0] * w / norm**2) / norm
        for n in (1, 2, 4, 8):
            np.testing.assert_allclose(gradient @ drift(w, w, 1.5, n, "current"), overlap_velocity(w, 1.5, n), atol=2e-15)

    def test_initial_current_and_frozen_agree(self):
        w = np.array([-.2, .8])
        for n in (1, 2, 4):
            np.testing.assert_allclose(drift(w, w, 1.5, n, "current"), drift(w, w, 1.5, n, "frozen"), atol=2e-15)

    def test_population_drift_against_finite_sampling(self):
        rng = np.random.default_rng(81732)
        z = rng.normal(size=(300000, 3)) + np.array([1.5, 0, 0])
        w, initial = np.array([.3, .7, -.2]), np.array([-.1, .4, .2])
        for source in ("current", "frozen"):
            proposal = w if source == "current" else initial
            correctness = (rng.random((len(z), 4)) < sigmoid(2 * (z @ proposal))[:, None]).any(axis=1)
            scores = z * (2 * correctness - 1 - np.tanh(z @ w))[:, None]
            error = np.abs(scores.mean(axis=0) - drift(w, initial, 1.5, 4, source))
            assert np.all(error < 5 * scores.std(axis=0) / np.sqrt(len(z)))

    def test_compute_crossover(self):
        threshold = (3 - np.sqrt(5)) / 2
        self.assertAlmostEqual(multiplier(threshold, 4) / 4, .5)
        self.assertGreater(multiplier(.2, 4) / 4, .5)
        self.assertLess(multiplier(.7, 4) / 4, .5)


if __name__ == "__main__": unittest.main()
