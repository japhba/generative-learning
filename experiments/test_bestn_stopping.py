import unittest

from train_bestn import stationary


class StoppingTests(unittest.TestCase):
    def history(self, plateau):
        evaluations, updates = [], []
        for step in (4, 8, 12, 16):
            accuracy = .3 if plateau == "error" else 1 - .8 * (step / 4)**(-.2)
            loss = 1. if plateau == "loss" else (step / 4)**(-.2)
            evaluations.append({"step": step, "candidates": step * 32, "accuracy": accuracy})
            updates.extend({"step": i, "loss": loss} for i in range(step - 3, step + 1))
        return {"step": 16, "evaluations": evaluations, "updates": updates}

    def test_either_plateau_stops(self):
        for field in ("loss", "error"): self.assertTrue(stationary(self.history(field))[0])

    def test_improving_curves_continue(self):
        self.assertFalse(stationary(self.history("neither"))[0])

    def test_minimum_observation_window(self):
        history = self.history("error")
        history["step"] = 12
        self.assertFalse(stationary(history)[0])


if __name__ == "__main__": unittest.main()
