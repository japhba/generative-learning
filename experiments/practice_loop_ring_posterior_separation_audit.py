"""Constants and inequalities for finite-posterior ring separation; no training."""

from datetime import datetime
from itertools import permutations
import json
from math import ceil, exp, log
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

from practice_loop_ring import atomic, exact, sigmoid


def main():
    load_dotenv(Path.home() / '.env')
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/posterior_separation_audit'
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(__file__, out / 'source.py')
    shutil.copy2(root / 'experiments/practice_loop_ring.py', out / 'learner_source.py')
    n, horizon, beta, delta, rho = 7, 3, 6., 1e-8, .01
    states = np.arange(n)
    truth = np.stack([(states + 1) % n, (states - 1) % n], axis=1)
    queries = np.array([(s, g) for s in states for g in states if s != g])
    good, bad = [2 * np.pi * k * states / n for k in [1, 2]]
    Jgood, Jbad = [float(exact(theta[None], truth[None], queries[None], horizon, beta)[0]) for theta in [good, bad]]
    top = float(np.sin(4 * np.pi / n))
    gap = float(top - np.sin(2 * np.pi / n))
    slope = float(sigmoid(-beta * top))
    residual = delta / ((1 - delta) * slope * gap)
    route_error = horizon * beta / np.sqrt(2) * np.sqrt(residual)
    good_lower = float(sigmoid(beta * np.sin(np.pi / n)) ** horizon)
    bad_upper = float(1 / 3 + sigmoid(beta * np.sin(8 * np.pi / 7)) / 3 + 2 * sigmoid(beta * np.sin(12 * np.pi / 7)))
    assert Jgood >= good_lower and Jbad <= bad_upper
    models = []
    for tail in permutations(range(1, n)):
        order = np.array((0,) + tail)
        model = np.empty((n, 2), int)
        model[order, 0] = np.roll(order, -1)
        model[order, 1] = np.roll(order, 1)
        models.append(model)
    models = np.array(models)
    differences = (models != truth).sum(axis=(1, 2))
    minimum = int(differences[differences > 0].min())
    assert len(models) == 720 and minimum == 6
    finite_data = []
    for epsilon in [.35, .55]:
        correct = 1 - epsilon + epsilon / n
        wrong = epsilon / n
        llr = log(correct / wrong)
        count = ceil(max(8 * log((len(models) - 1) / rho) / (minimum * (1 - epsilon) ** 2), 2 * log((len(models) - 1) / delta) / (minimum * (1 - epsilon) * llr)))
        failure = (len(models) - 1) * exp(-count * minimum * (1 - epsilon) ** 2 / 8)
        posterior_error = (len(models) - 1) * exp(-.5 * count * minimum * (1 - epsilon) * llr)
        assert failure <= rho and posterior_error <= delta
        finite_data.append({'corruption': epsilon, 'correct_observation_probability': correct, 'other_observation_probability': wrong, 'log_likelihood_ratio_magnitude': llr, 'observations_per_context': count, 'total_observations': 2 * n * count, 'failure_probability_upper': failure, 'posterior_error_upper_on_good_event': posterior_error})
    rng = np.random.default_rng(107031)
    perturbations = np.concatenate([bad + scale * rng.normal(size=(128, n)) for scale in [.0001, .01, .1, 1.]])
    loss = atomic(perturbations, np.broadcast_to(truth, (len(perturbations), n, 2)), beta)[0]
    optimum = float(np.logaddexp(0, -beta * top))
    power = abs(np.exp(1j * perturbations) @ np.exp(-1j * bad) / n) ** 2
    r = np.maximum(0, 1 - power)
    assert np.all(loss - optimum >= beta * slope * gap * r - 1e-12)
    route = exact(perturbations, np.broadcast_to(truth, (len(perturbations), n, 2)), np.broadcast_to(queries, (len(perturbations), len(queries), 2)), horizon, beta)
    assert np.all(abs(route - Jbad) <= horizon * beta / np.sqrt(2) * np.sqrt(r) + 1e-12)
    values, counts = np.unique(differences, return_counts=True)
    lower_gain, reverse_mass = 2., .1
    opposite_results = []
    for winding, theta in [(1, good), (2, bad)]:
        p = sigmoid(lower_gain * np.sin(2 * np.pi * winding * np.arange(1, 4) / n))
        def formula(probabilities):
            p1, p2, p3 = probabilities
            return float(p1 * (1 + p2 * (2 - p1 + p3)) / 3)
        forward_success, reverse_success = formula(p), formula(1 - p)
        forward_dp = float(exact(theta[None], truth[None], queries[None], horizon, lower_gain)[0])
        reverse_dp = float(exact(theta[None], truth[None, :, ::-1], queries[None], horizon, lower_gain)[0])
        assert abs(forward_success - forward_dp) < 1e-12 and abs(reverse_success - reverse_dp) < 1e-12
        opposite_results.append({'winding': winding, 'forward_success': forward_success, 'reverse_success': reverse_success, 'posterior_success': (1 - reverse_mass) * forward_success + reverse_mass * reverse_success})
    opposite_loss = (1 - reverse_mass) * atomic(perturbations, np.broadcast_to(truth, (len(perturbations), n, 2)), lower_gain)[0] + reverse_mass * atomic(perturbations, np.broadcast_to(truth[:, ::-1], (len(perturbations), n, 2)), lower_gain)[0]
    opposite_minimum = float(np.logaddexp(0, -lower_gain * top) + reverse_mass * lower_gain * top)
    assert np.all(opposite_loss >= opposite_minimum - 1e-12)
    opposite_noise = 7 / 15
    probability_correct, probability_other = 1 - opposite_noise + opposite_noise / n, opposite_noise / n
    posterior_forward = probability_correct / (probability_correct + probability_other)
    assert abs(posterior_forward - .9) < 1e-12
    orientation_control = {'prior': 'Uniform on the known undirected cycle with either action orientation.', 'gain': lower_gain, 'posterior_weights': [.9, .1], 'decreasing_objective_threshold': float(sigmoid(-lower_gain * top)), 'atomic_global_minimum': opposite_minimum, 'corruption': opposite_noise, 'correct_observation_probability': probability_correct, 'other_observation_probability': probability_other, 'one_observation_informative_probability': probability_correct + probability_other, 'posterior_after_forward_consistent_observation': posterior_forward, 'policies': opposite_results, 'posterior_success_gap': opposite_results[0]['posterior_success'] - opposite_results[1]['posterior_success']}
    result = {'timestamp_uk': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'states': n, 'horizon': horizon, 'gain': beta, 'query_count': len(queries), 'query_law': 'Uniform all ordered distinct pairs, independent of unknown cycle.', 'posterior_error_target': delta, 'confidence_failure_target': rho, 'good_success': Jgood, 'bad_success': Jbad, 'true_success_gap': Jgood - Jbad, 'top_eigenvalue': top, 'spectral_gap': gap, 'slope_factor': slope, 'residual_power_upper': residual, 'route_deviation_upper': float(route_error), 'posterior_gap_lower_using_dp': float((1 - delta) * (Jgood - Jbad - route_error) - delta), 'good_success_elementary_lower': good_lower, 'bad_success_elementary_upper': bad_upper, 'posterior_gap_elementary_lower': float((1 - delta) * (good_lower - bad_upper - route_error) - delta), 'minimum_differing_contexts': minimum, 'differing_context_histogram': {str(v): int(c) for v, c in zip(values, counts)}, 'finite_data_bounds': finite_data, 'checked_perturbations': len(perturbations), 'scope': 'Analytic sufficient conditions plus exact finite-state audits; no claim that the one-observation-per-context experiment satisfies the concentration premise, and no practice-convergence theorem.'}
    result['two_orientation_control'] = orientation_control
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'audit_arrays.npz', models=models, truth=truth, queries=queries, perturbations=perturbations, atomic_loss=loss, residual_power=r, success=route)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
