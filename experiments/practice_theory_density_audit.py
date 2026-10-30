"""Independent exact joint-state audit of the two-hidden-spin density toy."""

import hashlib
import itertools
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from scipy.special import expit, logsumexp

import practice_density_sigmoid as model


def distribution(parameter, y, hidden):
    weight, bias, prior = parameter[:12].reshape(6, 2), parameter[12:18], parameter[18:]
    logits = hidden @ weight.T + bias
    hidden_logp = ((hidden + 1) / 2) @ prior - np.logaddexp(0, prior).sum()
    joint_logp = ((y + 1) / 2) @ logits.T - np.logaddexp(0, logits).sum(axis=1)[None] + hidden_logp[None]
    return joint_logp, logsumexp(joint_logp, axis=1)


def finite_difference(function, parameter):
    result = np.zeros_like(parameter)
    for index in range(len(parameter)):
        plus, minus = parameter.copy(), parameter.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        result[index] = (function(plus) - function(minus)) / 2e-6
    return result


def categorical_resampling(target, proposal, batch):
    categories = len(target)
    ratio = target / proposal
    draws = np.array(list(itertools.product(range(categories), repeat=batch)))
    draw_probability = proposal[draws].prod(axis=1)
    weights = ratio[draws] / ratio[draws].sum(axis=1, keepdims=True)
    selected = np.sum(np.eye(categories)[draws] * weights[:, :, None], axis=1)
    marginal = draw_probability @ selected
    others = np.array(list(itertools.product(range(categories), repeat=batch - 1)), dtype=int).reshape(-1, batch - 1) if batch > 1 else np.empty((1, 0), dtype=int)
    other_probability = proposal[others].prod(axis=1)
    kernel = np.zeros((categories, categories))
    for retained in range(categories):
        conditional = np.column_stack([np.full(len(others), retained), others])
        conditional_weights = ratio[conditional] / ratio[conditional].sum(axis=1, keepdims=True)
        selected = np.sum(np.eye(categories)[conditional] * conditional_weights[:, :, None], axis=1)
        kernel[retained] = other_probability @ selected
    return marginal, kernel


def categorical_snis_audit():
    rng = np.random.default_rng(128461)
    kl = lambda first, second: float(first @ np.log(first / second))
    maximum = {name: 0.0 for name in ["resampling_kernel_identity_error", "target_stationarity_error", "detailed_balance_error", "natural_flow_covariance_error", "logit_lyapunov_identity_error", "one_candidate_null_error", "target_fixed_point_error", "stationary_sgd_variance_error"]}
    minimum = {name: float("inf") for name in ["forward_kl_decrease", "reverse_kl_decrease", "damped_forward_kl_decrease", "natural_flow_descent", "logit_lyapunov_descent", "logit_step_descent_bound_slack"]}
    for batch in [2, 3, 4]:
        for _ in range(30):
            target, proposal = rng.dirichlet(np.full(3, 0.7), size=2)
            selected, kernel = categorical_resampling(target, proposal, batch)
            ratio, gradient = target / proposal, selected - proposal
            multiplier = selected / proposal
            theta_error = np.log(proposal / target); theta_error -= theta_error.mean()
            ratio_covariance = proposal @ ((ratio - proposal @ ratio) * (multiplier - proposal @ multiplier))
            log_covariance = proposal @ ((np.log(ratio) - proposal @ np.log(ratio)) * (multiplier - proposal @ multiplier))
            maximum["resampling_kernel_identity_error"] = max(maximum["resampling_kernel_identity_error"], float(np.max(np.abs(proposal @ kernel - selected))))
            maximum["target_stationarity_error"] = max(maximum["target_stationarity_error"], float(np.max(np.abs(target @ kernel - target))))
            balance = target[:, None] * kernel
            maximum["detailed_balance_error"] = max(maximum["detailed_balance_error"], float(np.max(np.abs(balance - balance.T))))
            maximum["natural_flow_covariance_error"] = max(maximum["natural_flow_covariance_error"], abs(float(ratio @ gradient) - ratio_covariance))
            maximum["logit_lyapunov_identity_error"] = max(maximum["logit_lyapunov_identity_error"], abs(float(-theta_error @ gradient) - log_covariance), abs(log_covariance - (kl(proposal, target) - kl(selected, target) + kl(selected, proposal))))
            maximum["one_candidate_null_error"] = max(maximum["one_candidate_null_error"], float(np.max(np.abs(categorical_resampling(target, proposal, 1)[0] - proposal))))
            maximum["target_fixed_point_error"] = max(maximum["target_fixed_point_error"], float(np.max(np.abs(categorical_resampling(target, target, batch)[0] - target))))
            target_draws = np.array(list(itertools.product(range(len(target)), repeat=batch)))
            target_probability = target[target_draws].prod(axis=1)
            target_score = np.eye(len(target))[target_draws].mean(axis=1) - target
            target_variance = target_probability @ np.square(target_score).sum(axis=1)
            maximum["stationary_sgd_variance_error"] = max(maximum["stationary_sgd_variance_error"], abs(float(target_variance - (1 - target @ target) / batch)))
            minimum["forward_kl_decrease"] = min(minimum["forward_kl_decrease"], kl(target, proposal) - kl(target, selected))
            minimum["reverse_kl_decrease"] = min(minimum["reverse_kl_decrease"], kl(proposal, target) - kl(selected, target))
            damped = 0.6 * proposal + 0.4 * selected
            minimum["damped_forward_kl_decrease"] = min(minimum["damped_forward_kl_decrease"], kl(target, proposal) - kl(target, damped))
            minimum["natural_flow_descent"] = min(minimum["natural_flow_descent"], ratio_covariance)
            minimum["logit_lyapunov_descent"] = min(minimum["logit_lyapunov_descent"], log_covariance)
            eta = 1.5
            decrease = (theta_error @ theta_error - (theta_error + eta * gradient) @ (theta_error + eta * gradient)) / 2
            guaranteed = eta * (1 - eta / 2) * (gradient @ gradient)
            minimum["logit_step_descent_bound_slack"] = min(minimum["logit_step_descent_bound_slack"], float(decrease - guaranteed))
    assert max(maximum.values()) < 1e-10, maximum
    assert min(minimum.values()) > 0, minimum
    target, proposal = np.array([0.01, 0.09, 0.9]), np.array([0.25, 0.74, 0.01])
    selected, kernel = categorical_resampling(target, proposal, 2)
    gradient = selected - proposal
    derivative = float((proposal - target) @ gradient)
    theta = np.log(proposal)
    objective = lambda t: kl(target, np.exp(t - logsumexp(t)))
    numerical = (objective(theta + 1e-6 * gradient) - objective(theta - 1e-6 * gradient)) / 2e-6
    assert derivative > 0 and abs(derivative - numerical) < 1e-8
    counterexample = {"target": target.tolist(), "proposal": proposal.tolist(), "expected_resampled": selected.tolist(), "kernel": kernel.tolist(), "logit_forward_kl_derivative": derivative, "derivative_finite_difference_error": abs(derivative - numerical), "probability_flow_forward_kl_derivative": float(-(target / proposal) @ gradient), "forward_kl_before": kl(target, proposal), "forward_kl_after_logit_step_0.01": objective(theta + 0.01 * gradient), "forward_kl_after_probability_step_1": kl(target, selected)}
    return {"maximum_identity_errors": maximum, "minimum_strict_descent_quantities": minimum, "counterexample": counterexample, "cases": "90 independent positive three-category examples, B=2,3,4; exact finite batch enumeration", "scope": "Deterministic expected updates in an unrestricted categorical model, not realized noisy updates or the restricted neural generator."}


def main():
    load_dotenv(Path.home() / ".env")
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_density_audit"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(706883)
    y = np.array(list(itertools.product((-1., 1.), repeat=6)))
    hidden = np.array(list(itertools.product((-1., 1.), repeat=2)))
    parameter = rng.normal(0, 0.4, 20)
    weight, bias, prior = parameter[:12].reshape(6, 2), parameter[12:18], parameter[18:]
    pairs = list(itertools.combinations(range(6), 2))
    phi = np.column_stack([y, *[y[:, i] * y[:, j] for i, j in pairs]])
    energy = phi @ rng.normal(0, 0.3, phi.shape[1])
    joint_logp, logq = distribution(parameter, y, hidden)
    joint, q, target = np.exp(joint_logp), np.exp(logq), np.exp(energy - logsumexp(energy))
    joint_y, joint_h = np.repeat(y, 4, axis=0), np.tile(hidden, (64, 1))
    joint_q, joint_reward = joint.ravel(), np.repeat(energy - logq, 4)
    residual = (joint_y + 1) / 2 - expit(joint_h @ weight.T + bias)
    score = np.column_stack([(residual[:, :, None] * joint_h[:, None]).reshape(256, 12), residual, (joint_h + 1) / 2 - expit(prior)])
    result = {"normalization_error": float(abs(q.sum() - 1)), "mean_joint_score_error": float(np.max(np.abs(joint_q @ score))), "marginal_density_error": float(np.max(np.abs(model.log_density(y, weight, bias, prior) - logq)))}
    free_gradient = (joint_q * joint_reward) @ score
    forward_gradient = (joint_q * np.repeat(target / q, 4)) @ score
    objectives = {"free_energy": lambda p: np.exp(distribution(p, y, hidden)[1]) @ (energy - distribution(p, y, hidden)[1]), "forward_likelihood": lambda p: target @ distribution(p, y, hidden)[1]}
    expected = {"free_energy": free_gradient, "forward_likelihood": forward_gradient}
    for name, objective in objectives.items():
        result[name + "_finite_difference_error"] = float(np.max(np.abs(expected[name] - finite_difference(objective, parameter))))
    coefficients = {"free_energy": q * (energy - logq), "forward_likelihood": target}
    for name, coefficient in coefficients.items():
        for local in (False, True):
            values = coefficient if not local else (coefficient[:, None] * joint / q[:, None]).ravel()
            gradient = model.gradient(y if not local else joint_y, None if not local else joint_h, weight, bias, prior, values, local)
            result[name + ("_joint" if local else "_marginal") + "_implementation_error"] = float(np.max(np.abs(np.concatenate([value.ravel() for value in gradient]) - expected[name])))
    pair_probability = joint_q[:, None] * joint_q[None]
    pair_difference = (joint_reward[:, None] - joint_reward[None]) / 2
    loo = np.einsum("ij,ij,ijk->k", pair_probability, pair_difference, score[:, None] - score[None])
    result["exact_two_candidate_loo_error"] = float(np.max(np.abs(loo - free_gradient)))
    ratio = np.exp(joint_reward)
    normalized_two_gradient = np.einsum("ij,ij,ik->k", pair_probability, ratio[:, None] / (ratio[:, None] + ratio[None]), score) * 2
    result["two_candidate_importance_bias"] = float(np.max(np.abs(normalized_two_gradient - forward_gradient)))
    result["one_candidate_importance_mean_gradient"] = float(np.max(np.abs(joint_q @ score)))
    result["source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("practice_density.py", "practice_density_sigmoid.py", "practice_theory_density_audit.py")}
    for name, value in result.items():
        if name.endswith("error"):
            assert value < 1e-8, (name, value)
    (output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    categorical = categorical_snis_audit()
    categorical["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "categorical_snis_checks.json").write_text(json.dumps(categorical, indent=2) + "\n")
    (output / "categorical_snis_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
    print(json.dumps(categorical, indent=2))


if __name__ == "__main__": main()
