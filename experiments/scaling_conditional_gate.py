"""A normalized conditional Boltzmann forecast, fixed before LLM training."""

import numpy as np


def probabilities(logits): return 1 / (1 + np.exp(-logits))


def forecast(train_probabilities, eval_probabilities, initial_gain, steps):
    train, evaluation = [np.asarray(values, dtype=float) for values in (train_probabilities, eval_probabilities)]
    assert np.all((train > 0) & (train < 1)) and np.all((evaluation > 0) & (evaluation < 1))
    train_bias, eval_bias = [np.log(values) - np.log1p(-values) for values in (train, evaluation)]
    train_sensitivity, eval_sensitivity = [float(np.mean(values * (1 - values))) for values in (train, evaluation)]
    rate = initial_gain / (train_sensitivity * eval_sensitivity)
    selected_initial = float(np.mean(2 * train - train**2))
    curves = {}
    for source in ("current", "frozen"):
        shift, rows = 0., []
        for step in range(steps + 1):
            p = probabilities(train_bias + shift)
            rows.append({"step": step, "accuracy": float(np.mean(probabilities(eval_bias + shift))), "logit_shift": shift})
            force = float(np.mean(p * (1 - p))) if source == "current" else selected_initial - float(np.mean(p))
            shift += rate * force
        curves[source] = rows
    return {"rate": rate, "train_sensitivity": train_sensitivity, "eval_sensitivity": eval_sensitivity,
            "initial_gain": initial_gain, "frozen_selected_training_success": selected_initial, "curves": curves}
