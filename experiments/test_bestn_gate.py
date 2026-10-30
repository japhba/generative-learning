"""Independent joint-outcome enumeration and immutable forecast reproduction."""

from itertools import product
import json
from pathlib import Path
import unittest

import numpy as np

from bestn_gate import population_curve, selected_gate_probability


class GateTests(unittest.TestCase):
    def test_selection_against_joint_enumeration(self):
        gate, a, b = .61, .43, .08
        probabilities = np.array([(1 - gate) * (1 - b), (1 - gate) * b, gate * (1 - a), gate * a])
        modes, rewards = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
        for n in (1, 2, 3, 4):
            expected = 0.
            for group in product(range(4), repeat=n):
                group = np.array(group)
                winners = group[rewards[group] == rewards[group].max()]
                expected += np.prod(probabilities[group]) * modes[winners].mean()
            self.assertAlmostEqual(expected, selected_gate_probability(gate, a, b, n), places=13)

    def test_reproduce_frozen_population_forecast(self):
        root = Path(__file__).resolve().parents[1]
        forecast = json.loads((root / "results/bestn_b2k/gate_prediction/prospective.json").read_text())
        for source in ("current", "frozen"):
            reproduced = population_curve(forecast["parameters"], source, 32)
            saved = [r for r in forecast["rows"] if r["source"] == source]
            for left, right in zip(reproduced, saved, strict=True):
                for metric in ("accuracy", "mode_probability", "weight_projection"):
                    self.assertAlmostEqual(left[metric], right[metric], places=13)


if __name__ == "__main__": unittest.main()
