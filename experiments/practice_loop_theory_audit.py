"""Independent algebra, locality, and memory-access checks for the loop toy."""

import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from practice_loop_theory import CONFIG, EDGES, FEATURES, OUT, STATES, TRUE_C, TRUE_H, TRUE_LOGP, TRUE_P, covariance, distribution, empirical_nll, screening


def finite_difference(function, vector, epsilon=1e-5):
    eye = np.eye(len(vector))
    return np.array([(function(vector + epsilon * direction) - function(vector - epsilon * direction)) / (2 * epsilon) for direction in eye])


def masked_loss_gradient(states, prob, interaction):
    field = states @ interaction.T
    loss = np.sum(prob * (np.logaddexp(field, -field) - states * field).mean(axis=1))
    response = np.tanh(field)
    gradient = ((states.T @ (prob[:, None] * response)) + (response.T @ (prob[:, None] * states)) - 2 * (states.T @ (prob[:, None] * states))) / states.shape[1]
    np.fill_diagonal(gradient, 0)
    return loss, gradient


def main():
    rng = np.random.default_rng(500713)
    frequency = rng.dirichlet(np.ones(8))[None]
    cdata = covariance(frequency)[0]
    h = np.array([0.9, -0.4, 0.3])
    analytic = (covariance(distribution(h[None])[1])[0] - cdata) @ h
    numeric = finite_difference(lambda v: empirical_nll(v[None], frequency)[0], h)
    screening_analytic = screening(h[None], frequency)[1][0]
    screening_numeric = finite_difference(lambda v: screening(v[None], frequency)[0][0], h)
    assert np.max(np.abs(analytic - numeric)) < 1e-9
    assert np.max(np.abs(screening_analytic - screening_numeric)) < 1e-9

    curve_data = np.load(OUT / "curves_and_data.npz")
    empirical = curve_data["empirical_frequencies"]
    c_emp = curve_data["empirical_covariances"]
    symmetrized = (empirical + empirical[:, ::-1]) / 2
    recovered_law = (1 + np.einsum("sk,tk->st", empirical @ FEATURES, FEATURES)) / 8
    recovered_b = np.log(symmetrized) @ FEATURES / 8
    assert np.max(np.abs(recovered_law - symmetrized)) < 1e-15
    assert np.max(np.abs(recovered_b - curve_data["mle_b"])) < 1e-7
    assert np.max(np.abs(screening(curve_data["mle_h"], empirical)[1])) < 1e-7

    _, u = np.linalg.eigh(TRUE_C)
    ae = u[:, -1]
    ae *= np.sign(ae @ TRUE_H)
    projector = np.eye(3) - np.outer(ae, ae)
    positive_angle = projector @ TRUE_C @ ae
    negative_angle = -projector @ covariance(distribution(ae[None])[1])[0] @ ae
    tangent_toward_truth = TRUE_H / np.linalg.norm(TRUE_H) - ae * (ae @ TRUE_H) / np.linalg.norm(TRUE_H)
    angular_speed_toward_truth = negative_angle @ tangent_toward_truth / np.linalg.norm(tangent_toward_truth)
    assert np.linalg.norm(positive_angle) < 1e-14
    assert angular_speed_toward_truth > 0.04

    gauge = np.array([1, -1, -1])
    witness = np.array([0.5, 0.3, 0.2]) * gauge
    logq = FEATURES @ witness
    logq -= logsumexp(logq)
    gradient = np.exp(logq) @ FEATURES - TRUE_P[0] @ FEATURES
    witness_kl = TRUE_P[0] @ (TRUE_LOGP[0] - logq)
    ae_density_lower_bound = witness_kl - np.max(np.abs(gradient)) - gradient @ witness
    add_one_risk_bound = 3 / (CONFIG["examples"] + 1)
    assert ae_density_lower_bound > 0.05
    assert add_one_risk_bound < ae_density_lower_bound

    hbase = rng.normal(size=(3, 2))
    visible = rng.choice([-1., 1.], size=(17, 3))
    hidden = rng.normal(size=(17, 2))
    coefficients = rng.normal(size=17)
    hnew = hbase + np.einsum("n,ni,nj->ij", coefficients, visible, hidden)
    query_visible, query_hidden = rng.normal(size=3), rng.normal(size=2)
    encoder_memory = hbase.T @ query_visible + np.einsum("n,nj,n->j", coefficients, hidden, visible @ query_visible)
    decoder_memory = hbase @ query_hidden + np.einsum("n,ni,n->i", coefficients, visible, hidden @ query_hidden)
    attention_error = max(np.max(np.abs(encoder_memory - hnew.T @ query_visible)), np.max(np.abs(decoder_memory - hnew @ query_hidden)))
    assert attention_error < 1e-13

    five = np.array(list(itertools.product([-1., 1.], repeat=5)))
    parity = np.prod(five[:, :4], axis=1)
    epsilon, coupling = 0.5, 0.7
    plus, minus = (1 + epsilon * parity) / 32, (1 - epsilon * parity) / 32
    star = np.zeros((5, 5))
    star[4, :4] = star[:4, 4] = coupling
    plus_gradient = masked_loss_gradient(five, plus, star)[1]
    minus_gradient = masked_loss_gradient(five, minus, star)[1]
    explicit_difference = epsilon * (np.tanh(4 * coupling) - 2 * np.tanh(2 * coupling)) / 20
    covariance_difference = np.max(np.abs(five.T @ ((plus - minus)[:, None] * five)))
    assert covariance_difference < 1e-14
    assert abs((plus_gradient - minus_gradient)[4, 0] - explicit_difference) < 1e-14
    assert abs(explicit_difference) > 0.01
    five_h = np.array([0.9, 0.6, -0.4, 0.3, 0.8])
    rank_one = np.outer(five_h, five_h)
    np.fill_diagonal(rank_one, 0)
    rank_one_difference = (masked_loss_gradient(five, plus, rank_one)[1] - masked_loss_gradient(five, minus, rank_one)[1]) @ five_h
    assert np.linalg.norm(rank_one_difference) > 1e-4

    four = np.array(list(itertools.product([-1., 1.], repeat=4)))
    four_parity = np.prod(four, axis=1)
    four_h = np.array([0.9, 0.6, -0.4, 0.3])
    four_interaction = np.outer(four_h, four_h)
    np.fill_diagonal(four_interaction, 0)
    four_difference = masked_loss_gradient(four, (1 + epsilon * four_parity) / 16, four_interaction)[1] - masked_loss_gradient(four, (1 - epsilon * four_parity) / 16, four_interaction)[1]
    assert np.max(np.abs(four_difference)) < 1e-14

    empirical_improvement = curve_data["metrics_gibbs"][0, :, 4] - curve_data["metrics_gibbs"][-1, :, 4]
    initial, final = curve_data["weights_gibbs"][0], curve_data["weights_gibbs"][-1]
    norm_product_difference = np.sqrt((np.sum(initial ** 2, axis=1) + np.sum(final ** 2, axis=1)) ** 2 - 4 * np.sum(initial * final, axis=1) ** 2)
    op_error = np.linalg.norm(c_emp - TRUE_C, ord=2, axis=(1, 2))
    actual_covariance_certificate = empirical_improvement - 0.5 * norm_product_difference * op_error
    # This diagnostic uses the known simulated population only in evaluation.
    checks = {"python_executable": sys.executable, "nll_gradient_error": float(np.max(np.abs(analytic - numeric))), "screening_gradient_error": float(np.max(np.abs(screening_analytic - screening_numeric))), "three_spin_empirical_law_error": float(np.max(np.abs(recovered_law - symmetrized))), "three_spin_closed_form_mle_error": float(np.max(np.abs(recovered_b - curve_data["mle_b"]))), "positive_angular_velocity": positive_angle.tolist(), "negative_angular_velocity": negative_angle.tolist(), "negative_angular_norm": float(np.linalg.norm(negative_angle)), "angular_speed_toward_truth": float(angular_speed_toward_truth), "explicit_ae_density_lower_bound": float(ae_density_lower_bound), "ae_lower_bound_witness": witness.tolist(), "add_one_expected_risk_bound": float(add_one_risk_bound), "attention_identity_error": float(attention_error), "five_spin_same_covariance_error": float(covariance_difference), "five_spin_masked_gradient_difference": float((plus_gradient - minus_gradient)[4, 0]), "five_spin_closed_formula_difference": float(explicit_difference), "five_spin_rank_one_masked_gradient_difference": rank_one_difference.tolist(), "four_spin_masked_gradient_difference": float(np.max(np.abs(four_difference))), "actual_population_covariance_certificate_count_positive": int(np.sum(actual_covariance_certificate > 0)), "actual_population_covariance_certificates": actual_covariance_certificate.tolist(), "scope": "Population covariance certificates use simulated truth for evaluation only. The add-one estimator is unrestricted three-edge Ising; its rank-one representability is not unconditional. Five-spin laws are exact finite empirical count distributions, not asserted pairwise-Ising population laws."}
    (OUT / "theory_checks.json").write_text(json.dumps(checks, indent=2))
    shutil.copy2(__file__, OUT / "audit_source.py")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
