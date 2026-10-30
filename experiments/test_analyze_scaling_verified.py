import unittest

import numpy as np

from analyze_scaling_verified import bootstrap, crossing


class AnalysisTest(unittest.TestCase):
    def test_crossing_and_censoring(self):
        x = np.array([10., 100., 1000.]); y = 1 / x
        self.assertAlmostEqual(crossing(x, y, .02)["budget"], 50.)
        self.assertEqual(crossing(x, y, .2)["status"], "before_first_positive_checkpoint")
        self.assertEqual(crossing(x, y, .0001)["status"], "not_reached")

    def test_paired_exponents_and_speedup(self):
        steps = np.array([128, 256, 512, 1024, 2048, 4096])
        curves = np.stack([4 / steps, 8 / steps])
        errors = np.broadcast_to(curves[:, None, :, None], (2, 3, len(steps), 12))
        results = bootstrap(steps, errors, np.zeros_like(errors), np.repeat([0, 1, 2], 4), repetitions=10)
        for point in results["pointwise"]: self.assertAlmostEqual(point["frozen_over_current"], 2.)
        for window in results["windows"]:
            self.assertAlmostEqual(window["current_beta"], 1.); self.assertAlmostEqual(window["frozen_beta"], 1.)
        target = next(r for r in results["crossings"] if r["error"] == .01)
        self.assertAlmostEqual(target["candidate_speedup"], 2.)
        matched = next(r for r in results["cross_budget"] if r["current_updates"] == 512 and r["frozen_updates"] == 1024)
        self.assertAlmostEqual(matched["frozen_minus_current_error"], 0.)

    def test_unequal_strata_retain_equal_class_weight(self):
        steps = np.array([128, 256, 512, 1024, 2048, 4096]); labels = np.array([0, 1, 1, 2, 2, 2])
        individual = np.array([.3, .1, .1, .02, .02, .02])
        errors = np.broadcast_to(individual, (2, 3, len(steps), len(labels)))
        results = bootstrap(steps, errors, np.zeros_like(errors), labels, repetitions=10)
        for row in results["pointwise"]: self.assertAlmostEqual(row["current_error"], (.3 + .1 + .02) / 3)


if __name__ == "__main__": unittest.main()
