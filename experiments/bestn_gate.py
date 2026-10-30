"""One conditional Boltzmann gate; reproduces the frozen population forecast."""

import numpy as np

from bestn_theory import multiplier, sigmoid


def logit(probability): return np.log(probability) - np.log1p(-probability)


def selected_gate_probability(probability, good_accuracy, other_accuracy, n):
    success = other_accuracy + (good_accuracy - other_accuracy) * probability
    return probability + multiplier(success, n) * (good_accuracy - other_accuracy) * probability * (1 - probability)


def population_curve(parameters, source, steps):
    assert source in ("current", "frozen")
    train, evaluation = parameters["train"], parameters["evaluation"]
    z, records = 0., []
    fixed_target = selected_gate_probability(train["lambda0"], train["a"], train["b"], 2)
    for step in range(steps + 1):
        gate_train = sigmoid(logit(train["lambda0"]) + z)
        gate_eval = sigmoid(logit(evaluation["lambda0"]) + z)
        records.append({"source": source, "step": step, "candidates": 32 * step, "mode_probability": float(gate_eval),
                        "accuracy": float(evaluation["b"] + (evaluation["a"] - evaluation["b"]) * gate_eval),
                        "weight_projection": float(z / parameters["logit_gradient_norm"])})
        target = selected_gate_probability(gate_train, train["a"], train["b"], 2) if source == "current" else fixed_target
        z += parameters["effective_step_coefficient"] * (target - gate_train)
    return records
