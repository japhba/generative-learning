"""Independent Bayesian recognizer, route-correlation, and policy-gradient checks."""

from datetime import datetime
from itertools import product
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


class TracedWorldLookup:
    def __init__(self, worlds): self.worlds, self.reads = worlds, []

    def __getitem__(self, key):
        self.reads.append(tuple(np.asarray(value).copy() for value in key))
        return self.worlds[key]


def direct_posterior(observations, worlds, corruption):
    states = worlds.shape[1]
    match = observations[:, None] == worlds[None]
    likelihood = np.where(match, 1 - corruption + corruption / states, corruption / states).prod(axis=(2, 3))
    return likelihood / likelihood.sum(axis=1, keepdims=True), likelihood


def exhaustive_bayes_risk(theta, posterior, worlds, queries, horizon, beta):
    configs, models = posterior.shape
    phase = np.repeat(theta, models, axis=0)
    transitions = np.tile(worlds, (configs, 1, 1))
    task = np.broadcast_to(queries, (configs * models, len(queries), 2))
    success = audit.enumerate_routes(phase, transitions, task, horizon, beta)
    return np.sum(success.reshape(configs, models) * posterior, axis=1)


def calibration_identity(bayes, corruption):
    """Exhaust every dataset and true world in an auxiliary three-state check."""
    _, worlds = bayes.cycle_prior(3)
    observations = np.array(list(product(range(3), repeat=6))).reshape(-1, 3, 2)
    posterior, likelihood = direct_posterior(observations, worlds, corruption)
    fitted, _ = bayes.infer(np.eye(3, dtype=int)[observations], worlds, corruption)
    assert audit.error(posterior, fitted) < 2e-14
    queries = np.array([(s, g) for s in range(3) for g in range(3) if s != g])
    candidates = np.array([2 * np.pi * np.arange(3) / 3, -2 * np.pi * np.arange(3) / 3])
    success = np.stack([audit.enumerate_routes(np.broadcast_to(theta, (2, 3)), worlds, np.broadcast_to(queries, (2, 6, 2)), 1, 6.) for theta in candidates])
    conditional_risk = posterior @ success.T
    chosen = conditional_risk.argmax(axis=1)
    joint = likelihood / len(worlds)
    data_probability = joint.sum(axis=1)
    assert abs(data_probability.sum() - 1) < 2e-14
    prior_expected_gain = np.sum(joint * (success[chosen] - .5))
    posterior_expected_gain = np.sum(data_probability * (conditional_risk[np.arange(len(chosen)), chosen] - .5))
    discrepancy = abs(prior_expected_gain - posterior_expected_gain)
    assert discrepancy < 2e-14
    return {'states': 3, 'possible_complete_datasets': len(observations), 'possible_oriented_worlds': len(worlds), 'scope': 'Exact auxiliary calibration check, not another training experiment. Select between two fixed oriented policies using each dataset posterior; compare to a uniform-action baseline.', 'adaptive_policy_prior_expected_gain': float(prior_expected_gain), 'adaptive_policy_posterior_expected_gain': float(posterior_expected_gain), 'identity_error': float(discrepancy), 'probability_selected_policy_worse_in_realized_world': float(np.sum(joint * (success[chosen] < .5)))}


def main():
    load_dotenv(Path.home() / '.env')
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_bayesian'
    out = base / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    ring = audit.load_source(base / 'shared_source.py', 'bayes_saved_ring')
    bayes = audit.load_source(base / 'run_source.py', 'bayes_saved_source')
    bayes.ring = ring
    summary = json.loads((base / 'summary.json').read_text())
    tests = {}
    config_phase, config_posterior, config_labels = [], [], []
    test_rng = np.random.default_rng(447901)
    for corruption in [.35, .55]:
        name = f'noise{int(round(corruption * 100))}'
        data = np.load(base / f'{name}.npz')
        worlds, queries = data['worlds'], data['queries']
        assert worlds.shape == (720, 7, 2)
        assert len(set(map(tuple, worlds.reshape(720, -1)))) == 720
        for world in worlds:
            np.testing.assert_array_equal(world[world[:, 0], 1], np.arange(7))
            path, current = [], 0
            for _ in range(7):
                path.append(current)
                current = world[current, 0]
            assert len(set(path)) == 7 and current == 0
        np.testing.assert_array_equal(queries, np.array([(s, g) for s in range(7) for g in range(7) if s != g]))
        independent, likelihood = direct_posterior(data['observation'], worlds, corruption)
        posterior_error = audit.error(independent, data['posterior'])
        assert posterior_error < 1e-12
        np.testing.assert_array_equal(data['counts'], np.eye(7, dtype=int)[data['observation']])
        transition = np.zeros((32, 7, 2, 7))
        for world, weight in zip(worlds, independent.T):
            transition[:, np.arange(7)[:, None], np.arange(2)[None], world] += weight[:, None, None]
        transition_error = audit.error(transition, data['transition_posterior'])
        assert transition_error < 1e-12
        # Exact reproduction of finite exposure and the independent initial actor.
        rng = np.random.default_rng(1007000 + round(1000 * corruption))
        truth_index = rng.integers(720, size=32)
        observation = worlds[truth_index].copy()
        noisy = rng.random(observation.shape) < corruption
        observation[noisy] = rng.integers(7, size=noisy.sum())
        initial = rng.uniform(-np.pi, np.pi, (32, 7))
        np.testing.assert_array_equal(truth_index, data['truth_index'])
        np.testing.assert_array_equal(observation, data['observation'])
        np.testing.assert_array_equal(initial, data['pretrain_angles'][0])
        theta = initial.copy()
        for _ in range(200): theta += bayes.atomic(theta, data['transition_posterior'], 6.)[1]
        assert audit.error(theta, data['pretrain_angles'][1]) == 0
        cumulative = data['posterior'].cumsum(axis=1)
        cumulative[:, -1] = 1
        phases = {arm: data['pretrain_angles'][-1].copy() for arm in ['practice', 'atomic']}
        for _ in range(200):
            phases['practice'] += .1 * bayes.sampled(phases['practice'], worlds, cumulative, queries, rng, 6., 3, 64)
            phases['atomic'] += bayes.atomic(phases['atomic'], data['transition_posterior'], 6.)[1]
        replay = {arm: audit.error(theta, data[f'angles_{arm}'][1]) for arm, theta in phases.items()}
        assert max(replay.values()) == 0
        gain = data['curves_practice'][-1, :, 0] - data['curves_atomic'][-1, :, 0]
        boot_rng = np.random.default_rng(1009000 + round(1000 * corruption))
        interval = np.quantile(gain[boot_rng.integers(32, size=(10000, 32))].mean(axis=1), [.025, .975])
        assert audit.error(interval, summary['results'][name]['paired_true_success_gain']['bootstrap95']) == 0
        # A fixed cycle always reverses a forward step with its backward action.
        # Redrawing world independently per step would violate this certainty.
        independent_return = np.einsum('sij,sji->si', transition[:, :, 0], transition[:, :, 1])
        tests[name] = {'direct_channel_product_posterior_max_error': posterior_error, 'independent_transition_marginal_max_error': transition_error, 'exposure_and_actor_start_regenerate_exactly': True, 'pretraining_first200_update_max_error': 0., 'offline_first200_update_max_errors': replay, 'bootstrap95_reproduced': interval.tolist(), 'actual_worlds_worse': int(np.sum(gain < 0)), 'forward_then_backward_return_probability_fixed_world': 1., 'forward_then_backward_mean_return_if_world_redrawn_each_step': float(independent_return.mean()), 'forward_then_backward_min_return_if_world_redrawn_each_step': float(independent_return.min()), 'exact_calibration_identity': calibration_identity(bayes, corruption)}
        config_phase.extend([test_rng.uniform(-np.pi, np.pi, 7), data['pretrain_angles'][-1, 1], data['angles_practice'][-1, 2]])
        config_posterior.extend(data['posterior'][:3])
        config_labels.extend([name + '_random_world0', name + '_pretrained_world1', name + '_final_world2'])
    phase, posterior = np.array(config_phase), np.array(config_posterior)
    cumulative = posterior.cumsum(axis=1)
    cumulative[:, -1] = 1
    exact_risk = exhaustive_bayes_risk(phase, posterior, worlds, queries, 3, 6.)
    implementation_risk = bayes.internal_success(phase, worlds, posterior, queries, 3, 6.)
    assert audit.error(exact_risk, implementation_risk) < 1e-12
    epsilon = 1e-4
    gradients = []
    for step in [epsilon, epsilon / 2]:
        gradient = []
        for coordinate in trange(7, desc=f'Posterior-risk derivative {step}'):
            offset = np.eye(7)[coordinate] * step
            gradient.append((exhaustive_bayes_risk(phase + offset, posterior, worlds, queries, 3, 6.) - exhaustive_bayes_risk(phase - offset, posterior, worlds, queries, 3, 6.)) / (2 * step))
        gradients.append(np.array(gradient).T)
    assert audit.error(*gradients) < 1e-7
    traced = TracedWorldLookup(worlds)
    bayes.sampled(phase, traced, cumulative, queries, np.random.default_rng(447902), 6., 3, 64)
    assert len(traced.reads) == 3
    for read in traced.reads:
        np.testing.assert_array_equal(read[0], traced.reads[0][0])
    sample_rng = np.random.default_rng(447903)
    samples = np.array([bayes.sampled(phase, worlds, cumulative, queries, sample_rng, 6., 3, 64) for _ in trange(4096, desc='Posterior-world score expectation')])
    mean, sem = samples.mean(axis=0), samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    z = (mean - gradients[-1]) / sem
    assert np.max(abs(z)) < 5
    np.savez_compressed(out / 'gradient_samples.npz', samples=samples, phase=phase, posterior=posterior, exact_risk=exact_risk, exact_gradient=gradients[-1], coarse_gradient=gradients[0], sampled_world_indices=traced.reads[0][0])
    result = {'uk_time': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'cases': tests, 'gradient_check': {'configurations': config_labels, 'exact_posterior_risk': exact_risk.tolist(), 'enumeration_vs_dynamic_program_max_error': audit.error(exact_risk, implementation_risk), 'finite_difference_step_halving_max_change': audit.error(*gradients), 'monte_carlo_batches': 4096, 'trajectories_per_batch': 64, 'max_standardized_error': float(np.max(abs(z))), 'standardized_error_by_coordinate': z.tolist(), 'world_index_unchanged_at_all_route_steps': True}, 'scope': 'Correct conditional Bayes identity requires the stated correctly specified world prior and observation channel. For any actor selected from the data and independent internal randomness, its posterior risk equals conditional expected true-world risk. It does not guarantee improvement in each realized world; this experiment retains actual negative-world outcomes. The main exact atomic-optimum theorem does not apply here.', 'leakage_audit': 'The optimizer sees only frozen posterior/world hypothesis list, posterior transition marginals, actor phases, all42topology-independent tasks, and internal RNG. The true world index and true-world outcomes enter saved diagnostics only. The known cycle hypothesis class and known corruption rate are substantive prior assumptions. Posterior-world sampling occurs once before each complete trajectory; return baselines use independent same-query trajectories.'}
    sources = [Path(__file__), Path(audit.__file__), base / 'run_source.py', base / 'shared_source.py']
    result['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
