"""Exact small-state checks of nonlinear and Gaussian spectral practice laws."""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

import practice_density as density
import practice_density_sigmoid as sigmoid_model


def finite_difference(function, parameter):
    derivative = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        plus, minus = parameter.copy(), parameter.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        derivative[index] = (function(plus) - function(minus)) / 2e-6
    return derivative


def sigmoid_audit(rng):
    states, pairs, theta, features, _, _ = density.make_problem(6)
    couplings = np.array([0.8, 0.35])
    theta[:6] = 0
    interaction = np.zeros((6, 6))
    for index, (i, j) in enumerate(pairs):
        theta[6 + index] = couplings[i // 3] if i // 3 == j // 3 else 0
        interaction[i, j] = interaction[j, i] = theta[6 + index]
    energy = features @ theta
    log_target = energy - density.logsumexp(energy)
    target = np.exp(log_target)
    raw_second_moment = (states.T * target) @ states
    basis = np.zeros((6, 2))
    basis[:3, 0], basis[3:, 1] = 1 / np.sqrt(3), 1 / np.sqrt(3)
    correlation = (np.exp(4 * couplings) - 1) / (np.exp(4 * couplings) + 3)
    modal_checks = []
    for gains in [np.array([0.1, 0.2]), np.array([0.7, 1.3]), np.array([2., 4.]), np.array([5.5, 3.])]:
        weight = basis * gains
        mean = np.tanh(gains / (2 * np.sqrt(3)))
        logq = sigmoid_model.log_density(states, weight, np.zeros(6), np.zeros(2))
        derivatives = {
            "forward": np.sqrt(3) * mean * (correlation - mean ** 2) / (1 + 3 * mean ** 2),
            "reverse": np.sqrt(3) / 4 * mean * (1 - mean ** 2) * (4 * couplings - np.log((1 + 3 * mean ** 2) / (1 - mean ** 2))),
        }
        coefficients = {"forward": target, "reverse": np.exp(logq) * (energy - logq)}
        for name in ("forward", "reverse"):
            gradient, bias_gradient, prior_gradient = sigmoid_model.gradient(states, None, weight, np.zeros(6), np.zeros(2), coefficients[name], False)
            objective = lambda candidate: target @ sigmoid_model.log_density(states, basis * candidate, np.zeros(6), np.zeros(2))
            if name == "reverse":
                objective = lambda candidate: np.exp(sigmoid_model.log_density(states, basis * candidate, np.zeros(6), np.zeros(2))) @ (energy - sigmoid_model.log_density(states, basis * candidate, np.zeros(6), np.zeros(2)))
            numerical = finite_difference(objective, gains)
            error = float(np.max(np.abs(gradient - basis * derivatives[name])))
            finite_error = float(np.max(np.abs(numerical - derivatives[name])))
            assert error < 1e-12 and finite_error < 1e-8
            assert np.max(np.abs(bias_gradient)) < 1e-12 and np.max(np.abs(prior_gradient)) < 1e-12
            modal_checks.append({"objective": name, "gains": gains.tolist(), "gain_velocity": derivatives[name].tolist(), "full_gradient_error": error, "finite_difference_error": finite_error})
    base = rng.normal(size=(6, 2))
    early_checks = []
    for scale in (0.01, 0.005, 0.0025):
        weight = scale * base
        logq = sigmoid_model.log_density(states, weight, np.zeros(6), np.zeros(2))
        forward = sigmoid_model.gradient(states, None, weight, np.zeros(6), np.zeros(2), target, False)[0]
        reverse = sigmoid_model.gradient(states, None, weight, np.zeros(6), np.zeros(2), np.exp(logq) * (energy - logq), False)[0]
        early_checks.append({"scale": scale, "forward_remainder_norm": float(np.linalg.norm(forward - (raw_second_moment - np.eye(6)) @ weight / 4)), "reverse_remainder_norm": float(np.linalg.norm(reverse - interaction @ weight / 4))})
    for objective in ("forward", "reverse"):
        errors = np.array([entry[objective + "_remainder_norm"] for entry in early_checks])
        assert np.all((errors[:-1] / errors[1:] > 7.8) & (errors[:-1] / errors[1:] < 8.2))
    ideal_gains = 2 * np.sqrt(3) * np.arctanh(np.sqrt(correlation))
    capacity_error = float(np.max(np.abs(sigmoid_model.log_density(states, basis * ideal_gains, np.zeros(6), np.zeros(2)) - log_target)))
    assert capacity_error < 1e-12
    equal_theta = theta.copy()
    for index, (i, j) in enumerate(pairs):
        equal_theta[6 + index] = 0.8 if i // 3 == j // 3 else 0
    equal_log_target = features @ equal_theta
    equal_log_target -= density.logsumexp(equal_log_target)
    equal_weight = basis * ideal_gains[0]
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = equal_weight @ rotation
    rotated_logq = sigmoid_model.log_density(states, rotated, np.zeros(6), np.zeros(2))
    rotation_kl = float(np.exp(equal_log_target) @ (equal_log_target - rotated_logq))
    gram_error = float(np.max(np.abs(equal_weight @ equal_weight.T - rotated @ rotated.T)))
    assert gram_error < 1e-12 and rotation_kl > 0.1
    return {"scope": "Population marginal-score gradients; binary hidden prior uniform and visible biases zero; exact modal equations on the supplied block-aligned invariant manifold; C is the raw second moment E[yy^T].", "couplings": couplings.tolist(), "target_correlations": correlation.tolist(), "ideal_singular_gains": ideal_gains.tolist(), "modal_checks": modal_checks, "early_matrix_checks": early_checks, "early_matrix_base": base.tolist(), "exact_capacity_error": capacity_error, "svd_rotation_counterexample": {"angle_radians": angle, "same_gram_error": gram_error, "true_to_rotated_generator_kl": rotation_kl, "interpretation": "The Gram matrix and singular values cannot distinguish these decoders, but the independent binary latent distribution can."}}


def matrix_product(layers):
    product = layers[0]
    for layer in layers[1:]:
        product = layer @ product
    return product


def gaussian_audit(rng):
    checks = []
    gains = np.array([0.35, 1.1])
    for noise_variance, dimension in [(0.2, 6), (0., 2)]:
        basis = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
        eigenvalues = np.array([0.5, 1.2, 2.3, 3.1, 4., 5.])[:dimension]
        precision = (basis * eigenvalues) @ basis.T
        def loss(layers):
            product = matrix_product(layers)
            covariance = product @ product.T + noise_variance * np.eye(dimension)
            sign, logdet = np.linalg.slogdet(covariance)
            assert sign > 0
            return float((np.trace(precision @ covariance) - logdet) / 2)
        for depth in (1, 2, 3):
            layer_gain = gains ** (1 / depth)
            layers = [np.diag(layer_gain) for _ in range(depth - 1)] + [basis[:, :2] * layer_gain]
            product = matrix_product(layers)
            covariance = product @ product.T + noise_variance * np.eye(dimension)
            end_gradient = (np.linalg.inv(covariance) - precision) @ product
            gradients, velocity = [], np.zeros_like(product)
            maximum_finite_error = 0.
            for index, layer in enumerate(layers):
                before = np.eye(2) if index == 0 else matrix_product(layers[:index])
                after = np.eye(dimension) if index == depth - 1 else matrix_product(layers[index + 1:])
                gradient = after.T @ end_gradient @ before.T
                numerical = finite_difference(lambda candidate: -loss(layers[:index] + [candidate] + layers[index + 1:]), layer)
                maximum_finite_error = max(maximum_finite_error, float(np.max(np.abs(gradient - numerical))))
                gradients.append(gradient)
                velocity += after @ gradient @ before
            predicted = depth * gains ** (3 - 2 / depth) * (1 / (gains ** 2 + noise_variance) - eigenvalues[:2])
            velocity_error = float(np.max(np.abs(velocity - basis[:, :2] * predicted)))
            balance_error = 0.
            for index in range(depth - 1):
                first = gradients[index] @ layers[index].T + layers[index] @ gradients[index].T
                second = gradients[index + 1].T @ layers[index + 1] + layers[index + 1].T @ gradients[index + 1]
                balance_error = max(balance_error, float(np.max(np.abs(first - second))))
            assert maximum_finite_error < 1e-8 and velocity_error < 1e-12 and balance_error < 1e-12
            entry = {"depth": depth, "noise_variance": noise_variance, "dimension": dimension, "gains": gains.tolist(), "gain_velocity": predicted.tolist(), "factor_gradient_finite_difference_error": maximum_finite_error, "end_map_velocity_error": velocity_error, "balanced_invariance_derivative_error": balance_error}
            if noise_variance == 0:
                alternate = depth * gains ** (1 - 2 / depth) * (1 - eigenvalues[:2] * gains ** 2)
                entry["zero_noise_reduction_error"] = float(np.max(np.abs(predicted - alternate)))
                assert entry["zero_noise_reduction_error"] < 1e-12
            checks.append(entry)
    return {"scope": "Reverse KL, fixed learned Gaussian precision, deterministic balanced aligned depth factors, output noise only; zero-noise check uses a square full-rank generator.", "checks": checks}


def main():
    load_dotenv(Path.home() / ".env")
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_spectral_theory_audit"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8271)
    result = {"sigmoid": sigmoid_audit(rng), "gaussian": gaussian_audit(rng), "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("practice_density.py", "practice_density_sigmoid.py", "practice_spectral_theory_audit.py")}}
    (output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
