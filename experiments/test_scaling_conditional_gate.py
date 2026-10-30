import unittest

import numpy as np

from scaling_conditional_gate import forecast


class ConditionalGateTest(unittest.TestCase):
    def test_best_two_score_is_success_gradient(self):
        rng = np.random.default_rng(73)
        for size in (2, 3, 101):
            logits = rng.normal(size=size)
            policy = np.exp(logits) / np.exp(logits).sum()
            reward = np.arange(size) == 0
            chosen = policy * (1 + reward - policy[0])
            np.testing.assert_allclose(chosen.sum(), 1)
            score = np.eye(size) - policy[None, :]
            expected = chosen @ score
            np.testing.assert_allclose(expected, policy[0] * (reward - policy), atol=1e-16)

    def test_forecast_uses_initial_overlap(self):
        result = forecast([.3, .7], [.2, .8], 1e-6, 2)
        for source in ("current", "frozen"):
            curve = result["curves"][source]
            self.assertAlmostEqual(curve[1]["accuracy"] - curve[0]["accuracy"], 1e-6, places=11)
        self.assertGreater(result["curves"]["current"][2]["accuracy"], result["curves"]["frozen"][2]["accuracy"])


if __name__ == "__main__": unittest.main()
