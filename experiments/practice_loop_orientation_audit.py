"""Independent check of the conditional two-orientation practice demonstration."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import trange

import practice_loop_ring_audit as audit


def posterior_success(theta, worlds, queries):
    worlds_success = np.stack([audit.enumerate_routes(theta, np.broadcast_to(world, (len(theta), 7, 2)), np.broadcast_to(queries, (len(theta), 42, 2)), 3, 2.) for world in worlds], axis=1)
    return worlds_success @ np.array([.9, .1]), worlds_success


def main():
    load_dotenv(Path.home() / '.env')
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/orientation'
    out = base / 'audit'; out.mkdir(parents=True, exist_ok=True)
    ring = audit.load_source(base / 'practice_loop_ring.py', 'orientation_saved_ring')
    bayes = audit.load_source(base / 'practice_loop_bayesian.py', 'orientation_saved_bayes')
    bayes.ring = ring
    data = np.load(base / 'trajectories.npz')
    report = json.loads((base / 'summary.json').read_text())
    worlds, posterior, transitions, queries, initial = [data[key] for key in ['worlds', 'posterior', 'transitions', 'queries', 'initial']]
    np.testing.assert_array_equal(queries, np.array([(s, g) for s in range(7) for g in range(7) if s != g]))
    np.testing.assert_array_equal(posterior, np.broadcast_to([.9, .1], (32, 2)))
    np.testing.assert_array_equal(initial, np.broadcast_to(4 * np.pi * np.arange(7) / 7, (32, 7)))
    cumulative = posterior.cumsum(axis=1); cumulative[:, -1] = 1.
    phases = {arm: initial.copy() for arm in ['practice', 'atomic']}
    rng = np.random.default_rng(1097123)
    for _ in range(100):
        phases['practice'] += .1 * bayes.sampled(phases['practice'], worlds, cumulative, queries, rng, 2., 3, 64)
        phases['atomic'] += bayes.atomic(phases['atomic'], transitions, 2.)[1]
    replay = {arm: audit.error(theta, data[f'angles_{arm}'][1]) for arm, theta in phases.items()}
    assert max(replay.values()) == 0
    final_checks = {}
    for arm in phases:
        expected, per_world = posterior_success(data[f'angles_{arm}'][-1], worlds, queries)
        final_checks[arm] = max(audit.error(expected, data[f'curves_{arm}'][-1, :, 0]), audit.error(per_world, data[f'curves_{arm}'][-1, :, 2:]))
        assert final_checks[arm] < 1e-14
    gains = data['curves_practice'][-1, :, 0] - data['curves_atomic'][-1, :, 0]
    indices = np.random.default_rng(1097124).integers(32, size=(10000, 32))
    interval = np.quantile(gains[indices].mean(axis=1), [.025, .975])
    assert audit.error(interval, report['paired_posterior_gain']['bootstrap95']) == 0
    winding = np.array([2 * np.pi * np.arange(7) / 7, 4 * np.pi * np.arange(7) / 7])
    winding_values, winding_world_values = posterior_success(winding, worlds, queries)
    analytic = []
    for k in [1, 2]:
        p = 1 / (1 + np.exp(-2 * np.sin(2 * np.pi * k * np.arange(1, 4) / 7)))
        formula = lambda x: x[0] / 3 * (1 + x[1] * (2 - x[0] + x[2]))
        analytic.append(.9 * formula(p) + .1 * formula(1 - p))
    assert audit.error(analytic, winding_values) < 1e-14
    corruption = 7 / 15
    a, b = 1 - corruption + corruption / 7, corruption / 7
    acquired_posterior = a / (a + b)
    assert abs(acquired_posterior - .9) < 1e-14
    s = np.sin(4 * np.pi / 7)
    optimum_loss = .9 * np.logaddexp(0, -2 * s) + .1 * np.logaddexp(0, 2 * s)
    assert abs(optimum_loss - report['initial'][1]) < 1e-14
    # Independently differentiate exact path sums, without the directional helper.
    point = initial[:1]
    value = posterior_success(point, worlds, queries)[0][0]
    step = 5e-4
    basis = np.eye(7)
    def second(direction): return (posterior_success(point + step * direction, worlds, queries)[0][0] + posterior_success(point - step * direction, worlds, queries)[0][0] - 2 * value) / step ** 2
    diagonal = np.array([second(direction) for direction in basis])
    hessian = np.diag(diagonal)
    for i in range(7):
        for j in range(i + 1, 7): hessian[i, j] = hessian[j, i] = (second(basis[i] + basis[j]) - diagonal[i] - diagonal[j]) / 2
    hessian_error = audit.error(hessian, data['hessian'])
    assert hessian_error < 1e-7
    phase_rng = np.random.default_rng(441901)
    test = np.stack([initial[0], initial[0] + phase_rng.normal(0, .2, 7), data['angles_practice'][-1, 0], data['angles_practice'][-1, 1]])
    fd_step = 1e-5
    exact_gradient = np.column_stack([(posterior_success(test + fd_step * direction, worlds, queries)[0] - posterior_success(test - fd_step * direction, worlds, queries)[0]) / (2 * fd_step) for direction in basis])
    sample_rng = np.random.default_rng(441902)
    samples = np.array([bayes.sampled(test, worlds, cumulative[:4], queries, sample_rng, 2., 3, 64) for _ in trange(4096, desc='Conditional posterior gradient')])
    mean, sem = samples.mean(axis=0), samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    z = (mean - exact_gradient) / sem
    assert np.max(abs(z)) < 5
    np.savez_compressed(out / 'checks.npz', finite_difference_hessian=hessian, exact_gradient=exact_gradient, gradient_samples=samples, test_phases=test)
    result = {'uk_time': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'first100_update_replay_errors': replay, 'final_independent_eight_string_enumeration_errors': final_checks, 'bootstrap95_reproduced': interval.tolist(), 'winding_one_then_two_success': winding_values.tolist(), 'winding_one_then_two_success_by_world': winding_world_values.tolist(), 'closed_form_success_max_error': audit.error(analytic, winding_values), 'one_observation_posterior': acquired_posterior, 'global_atomic_optimum_loss': float(optimum_loss), 'strictly_decreasing_convex_edge_loss_slope_at_top_eigenvalue': float(2 * (.1 - 1 / (1 + np.exp(2 * s)))), 'independent_second_difference_hessian_error': hessian_error, 'sampled_gradient_max_standardized_error': float(np.max(abs(z))), 'sampled_gradient_batches': 4096, 'trajectories_per_batch': 64, 'minority_world_initial_success': report['initial'][3], 'minority_world_final_success': report['final']['practice'][3], 'scope': 'One specified conditional acquisition event,32optimizer realizations; intervals describe optimization randomness only. Known undirected graph and known corruption channel are supplied. Posterior(.9,.1)is hardcoded as the analytically derived conditional posterior, not estimated from32fresh datasets. No observed true-world identity or reward is used. Updates use audited fixed-posterior world sampling; stop criteria use posterior success and posterior atomic loss only. The theorem certifies the starting atomic global optimum and existence of a better policy, while the32runs supply empirical practice evidence, not a convergence theorem.'}
    sources = [Path(__file__), Path(audit.__file__), base / 'practice_loop_orientation.py', base / 'practice_loop_bayesian.py', base / 'practice_loop_ring.py']
    result['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
