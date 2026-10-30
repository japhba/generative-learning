"""Independent finite-state gradient audits of the recognition-gated toys."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from practice_atoms import atom_features, features, score_atoms, sigmoid, signs, true_trace


def finite_difference(function, weights):
    derivative = np.empty_like(weights)
    for index in np.ndindex(weights.shape):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        derivative[index] = (function(plus) - function(minus)) / 2e-6
    return derivative


def atom_audit(rng):
    x, outputs = signs(3), signs(3)
    h = features(x, "monomial", 3)
    v = np.array([0.11, -0.21, 0.07, 1.4])
    w, w0 = rng.normal(0, 0.3, (3, h.shape[1])), rng.normal(0, 0.3, (3, h.shape[1]))
    atoms = score_atoms(x[:, None], np.broadcast_to(outputs[None], (len(x), len(outputs), 3)), v)
    global_reward, local_reward = atoms.prod(axis=-1), atoms.mean(axis=-1)
    adjacent_reward = (atoms + np.concatenate([atoms[..., 1:], np.zeros(atoms.shape[:-1] + (1,))], axis=-1)) / 3
    def probabilities(weights):
        p = sigmoid(h @ weights.T)
        q = np.prod(np.where(outputs[None] == 1, p[:, None], 1 - p[:, None]), axis=-1)
        return p, q
    p, q = probabilities(w); _, q0 = probabilities(w0)
    residual = (outputs[None] + 1) / 2 - p[:, None]
    gradient_global = np.einsum("xy,xy,xyl,xh->lh", q, global_reward, residual, h) / len(x)
    gradient_local = np.einsum("xy,xyl,xyl,xh->lh", q, adjacent_reward, residual, h) / len(x)
    gradient_frozen = np.einsum("xy,xy,xyl,xh->lh", q0, global_reward, residual, h) / len(x)
    numeric_global = finite_difference(lambda candidate: np.mean(np.sum(probabilities(candidate)[1] * global_reward, axis=-1)), w)
    numeric_local = finite_difference(lambda candidate: np.mean(np.sum(probabilities(candidate)[1] * local_reward, axis=-1)), w)
    numeric_frozen = finite_difference(lambda candidate: np.mean(np.sum(q0 * global_reward * np.log(probabilities(candidate)[1]), axis=-1)), w)
    loo_values = np.zeros((len(x), 3))
    for i in range(len(outputs)):
        for j in range(len(outputs)):
            candidate_pair_value = ((global_reward[:, i] - global_reward[:, j])[:, None] * (residual[:, i] - residual[:, j])) / 2
            loo_values += (q[:, i] * q[:, j])[:, None] * candidate_pair_value
    loo_gradient = loo_values.T @ h / len(x)
    unfiltered = np.einsum("xy,xyl,xh->lh", q, residual, h) / len(x)
    result = {"global_gradient_error": float(np.max(np.abs(gradient_global - numeric_global))), "local_adjacent_gradient_error": float(np.max(np.abs(gradient_local - numeric_local))), "frozen_cross_entropy_gradient_error": float(np.max(np.abs(gradient_frozen - numeric_frozen))), "global_two_candidate_loo_error": float(np.max(np.abs(loo_gradient - gradient_global))), "unfiltered_population_gradient": float(np.max(np.abs(unfiltered)))}
    assert max(result.values()) < 1e-8, result
    truth = true_trace(x, 3)
    correct = np.all(outputs[None] == truth[:, None], axis=-1)
    success = np.sum(q * correct, axis=-1)
    acceptance = np.sum(q * global_reward, axis=-1)
    tilted = q * global_reward / acceptance[:, None]
    selected_success = np.sum(tilted * correct, axis=-1)
    covariance = np.sum(q * global_reward * correct, axis=-1) - acceptance * success
    result["conditional_improvement_identity_error"] = float(np.max(np.abs(selected_success - success - covariance / acceptance)))
    result["global_verifier_modal_accuracy"] = float(np.mean(np.all(outputs[global_reward.argmax(axis=-1)] == truth, axis=-1)))
    result["fraction_inputs_positive_true_improvement"] = float(np.mean(selected_success > success))
    result["true_success_before_by_input"] = success.tolist()
    result["true_success_after_exact_tilt_by_input"] = selected_success.tolist()
    pairs = signs(2)
    local_correct_probability = np.array([0.8, 0.99, 0.8, 0.55])
    table_logits = pairs.prod(axis=-1) * np.log(local_correct_probability / (1 - local_correct_probability))
    wrong_global_v = np.linalg.solve(atom_features(*pairs.T), table_logits)
    bad_x = np.ones((1, 3))
    bad_scores = score_atoms(bad_x[:, None], outputs[None], wrong_global_v).prod(axis=-1)[0]
    true_index = int(np.flatnonzero(np.all(outputs == 1, axis=-1))[0])
    q_t = np.ones(len(outputs)) / len(outputs); trace = [float(q_t[true_index])]
    for _ in range(8):
        q_t = q_t * bad_scores; q_t /= q_t.sum(); trace.append(float(q_t[true_index]))
    result["atom_perfect_global_wrong_counterexample"] = {"pairs": pairs.tolist(), "local_correct_probabilities": local_correct_probability.tolist(), "v": wrong_global_v.tolist(), "x": bad_x.tolist(), "true_trace_score": float(bad_scores[true_index]), "mode_trace": outputs[bad_scores.argmax()].tolist(), "mode_trace_score": float(bad_scores.max()), "true_success_under_repeated_exact_tilts": trace}
    return result


def hidden_audit(rng):
    x, hidden_states, output_signs = signs(2), (signs(3) + 1) / 2, signs(2)
    xb = np.column_stack([x, np.ones(len(x))]); hb = np.column_stack([hidden_states, np.ones(len(hidden_states))])
    outputs = (output_signs + 1) / 2
    v = np.array([0.11, -0.21, 0.07, 1.4])
    rewards = score_atoms(x[:, None], np.broadcast_to(output_signs[None], (len(x), len(outputs), 2)), v).prod(axis=-1)
    b, w = rng.normal(0, 0.3, (3, xb.shape[1])), rng.normal(0, 0.3, (2, hb.shape[1]))
    def joint(candidate_b, candidate_w):
        ph = sigmoid(xb @ candidate_b.T)
        qh = np.prod(np.where(hidden_states[None] == 1, ph[:, None], 1 - ph[:, None]), axis=-1)
        py = sigmoid(hb @ candidate_w.T)
        qy = np.prod(np.where(outputs[None] == 1, py[:, None], 1 - py[:, None]), axis=-1)
        return ph, py, qh[:, :, None] * qy[None]
    ph, py, q = joint(b, w)
    output_residual = outputs[None] - py[:, None]
    hidden_residual = hidden_states[None] - ph[:, None]
    gb = np.einsum("xhy,xy,xhk,xd->kd", q, rewards, hidden_residual, xb) / len(x)
    gw = np.einsum("xhy,xy,hyl,hk->lk", q, rewards, output_residual, hb) / len(x)
    objective = lambda cb, cw: float(np.einsum("xhy,xy->", joint(cb, cw)[2], rewards) / len(x))
    nb = finite_difference(lambda candidate: objective(candidate, w), b)
    nw = finite_difference(lambda candidate: objective(b, candidate), w)
    count = 50000
    sampled_h = (rng.random((len(x), count, 3)) < ph[:, None]).astype(float)
    sampled_hb = np.concatenate([sampled_h, np.ones(sampled_h.shape[:-1] + (1,))], axis=-1)
    sample_py = sigmoid(sampled_hb @ w.T)
    sampled_y = (rng.random(sample_py.shape) < sample_py).astype(float)
    sample_reward = score_atoms(x[:, None], 2 * sampled_y - 1, v).prod(axis=-1)
    advantage = (count * sample_reward - sample_reward.sum(axis=1, keepdims=True)) / (count - 1)
    values_b = advantage[..., None] * (sampled_h - ph[:, None])
    values_w = advantage[..., None] * (sampled_y - sample_py)
    sampled_gb = values_b.mean(axis=1).T @ xb / len(x)
    sampled_gw = np.einsum("xml,xmh->lh", values_w, sampled_hb) / (len(x) * count)
    b_step, w_step = b + sampled_gb, w + sampled_gw
    explicit_hidden_logits = xb @ b_step.T
    attention_hidden_logits = xb @ b.T + (xb @ xb.T) @ values_b.mean(axis=1) / len(x)
    fresh_uniforms = rng.random((len(x), 3))
    fresh_h = (fresh_uniforms < sigmoid(explicit_hidden_logits)).astype(float)
    attention_h = (fresh_uniforms < sigmoid(attention_hidden_logits)).astype(float)
    assert np.array_equal(fresh_h, attention_h)
    fresh_hb = np.column_stack([fresh_h, np.ones(len(x))])
    explicit_output_logits = fresh_hb @ w_step.T
    attention_output_logits = fresh_hb @ w.T + np.einsum("qh,xmh,xml->ql", fresh_hb, sampled_hb, values_w) / (len(x) * count)
    result = {"hidden_population_gradient_error": float(np.max(np.abs(gb - nb))), "output_population_gradient_error": float(np.max(np.abs(gw - nw))), "hidden_mc_loo_error_200000_rollouts": float(np.max(np.abs(sampled_gb - gb))), "output_mc_loo_error_200000_rollouts": float(np.max(np.abs(sampled_gw - gw))), "hidden_attention_error": float(np.max(np.abs(explicit_hidden_logits - attention_hidden_logits))), "output_attention_error": float(np.max(np.abs(explicit_output_logits - attention_output_logits)))}
    assert result["hidden_population_gradient_error"] < 1e-8 and result["output_population_gradient_error"] < 1e-8
    assert result["hidden_mc_loo_error_200000_rollouts"] < 0.002 and result["output_mc_loo_error_200000_rollouts"] < 0.002
    assert result["hidden_attention_error"] < 1e-10 and result["output_attention_error"] < 1e-10
    return result


def main():
    load_dotenv(Path.home() / ".env")
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_gradient_audit"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(194537)
    result = {"atom": atom_audit(rng), "stochastic_hidden": hidden_audit(rng), "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("practice_atoms.py", "practice_stochastic_hidden.py", "practice_gradient_audit.py")}}
    (output / "checks.json").write_text(json.dumps(result, indent=2))
    (output / "source.py").write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
