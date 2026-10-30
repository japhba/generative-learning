"""Independent replay, exhaustive evaluation, and attention-read audits; no training study."""

from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(a, b): return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def enumerate_routes(theta, transition, queries, horizon, beta):
    """Sum probabilities of all action strings, independently of dynamic programming."""
    seeds, tasks = queries.shape[:2]
    strings = np.arange(2 ** horizon)
    state = np.broadcast_to(queries[:, :, 0, None], (seeds, tasks, len(strings))).copy()
    goal = queries[:, :, 1, None]
    sid = np.arange(seeds)[:, None, None]
    mass = np.ones(state.shape)
    for step in range(horizon):
        alive = state != goal
        logits = beta * np.sin(theta[sid, goal] - theta[sid, state])
        probability = 1 / (1 + np.exp(-logits))
        forward = ((strings >> step) & 1).astype(bool)
        mass *= np.where(forward, probability, 1 - probability)
        state = np.where(alive, transition[sid, state, 1 - forward.astype(int)], state)
    assert error(mass.sum(axis=2), 1.) < 2e-14
    return (mass * (state == goal)).sum(axis=2).mean(axis=1)


def transfer_audit(root):
    directory = root / 'transfer'
    ring = load_source(directory / 'shared_source.py', 'transfer_saved_ring')
    data = np.load(directory / 'trajectories.npz')
    original = np.load(root / 'confirmation/trajectories.npz')
    controls = np.load(root / 'controls/trajectories.npz')
    report = json.loads((directory / 'summary.json').read_text())
    config = report['config']
    for key in ['truth', 'learned', 'counts', 'ordering']:
        np.testing.assert_array_equal(data[key], original[key])
    np.testing.assert_array_equal(data['all_queries'], original['queries'])
    np.testing.assert_array_equal(data['learned'], data['counts'].argmax(axis=3))
    rng = np.random.default_rng(config['goal_seed'])
    held = np.stack([rng.choice(12, 3, replace=False) for _ in range(32)])
    np.testing.assert_array_equal(held, data['held_out_goals'])
    for seed, queries in enumerate(data['all_queries']):
        mask = np.isin(queries[:, 1], held[seed])
        np.testing.assert_array_equal(data['test_queries'][seed], queries[mask])
        np.testing.assert_array_equal(data['train_queries'][seed], queries[~mask])
        assert not set(data['train_queries'][seed, :, 1]) & set(held[seed])
        assert set(held[seed]) <= set(data['train_queries'][seed, :, 0])
        expected = np.array([(s, g) for s in range(12) for g in range(12) if s != g and g not in data['learned'][seed, s]])
        np.testing.assert_array_equal(queries, expected)
    aliases = np.rint(original['initial'] / (np.pi / 2)).astype(int) % 4
    starts = {'natural': controls['pretrain_angles'][-1], 'alias': np.array([0., np.pi / 2, -np.pi, -np.pi / 2])[aliases]}
    replay, evaluation, effects, role_changes = {}, {}, {}, {}
    bootstrap_rng = np.random.default_rng(905715)
    for si, start in enumerate(tqdm(['natural', 'alias'], desc='Transfer replay')):
        for ai, arm in enumerate(['partial_goals', 'all_goals', 'atomic_only']):
            phase = starts[start].copy()
            np.testing.assert_array_equal(phase, data[f'angles_{start}_{arm}'][0])
            rng = np.random.default_rng(config['training_seed'] + 100 * si + ai)
            # These updates use no true transition, held-out query, or evaluation outcome.
            query = data['train_queries' if arm == 'partial_goals' else 'all_queries']
            for _ in range(100):
                gradient = ring.atomic(phase, data['learned'], config['beta'])[1] if arm == 'atomic_only' else ring.sampled(phase, data['learned'], query, config['horizon'], config['beta'], rng, config['batch'], 'practice')
                phase += (config['atomic_rate'] if arm == 'atomic_only' else config['practice_rate']) * gradient
            replay[f'{start}_{arm}'] = error(phase, data[f'angles_{start}_{arm}'][1])
            assert replay[f'{start}_{arm}'] == 0
            final = data[f'angles_{start}_{arm}'][-1]
            success = enumerate_routes(final, data['truth'], data['test_queries'], config['horizon'], config['beta'])
            evaluation[f'{start}_{arm}'] = error(success, data[f'curves_{start}_{arm}'][-1, :, 0])
            assert evaluation[f'{start}_{arm}'] < 2e-14
        difference = data[f'curves_{start}_partial_goals'][-1, :, 0] - data[f'curves_{start}_atomic_only'][-1, :, 0]
        indices = bootstrap_rng.integers(32, size=(10000, 32))
        effect = {'mean': float(difference.mean()), 'bootstrap95': np.quantile(difference[indices].mean(axis=1), [.025, .975]).tolist()}
        assert effect == report['summary'][start]['paired_held_goal_practice_minus_atomic']
        effects[start] = effect
        displacement = data[f'angles_{start}_partial_goals'][-1] - starts[start]
        held_displacement = displacement[np.arange(32)[:, None], held]
        role_changes[start] = {'changed_held_goal_parameters': int(np.count_nonzero(np.abs(held_displacement) > 1e-12)), 'total_held_goal_parameters': held.size, 'mean_absolute_displacement': float(np.mean(abs(held_displacement)))}
    return {'replay_first100_update_max_errors': replay, 'independent_all64_action_strings_evaluation_max_errors': evaluation, 'paired_effects_recomputed': effects, 'held_goal_shared_parameter_changes': role_changes, 'all_held_goal_identities_are_practice_start_states': True, 'goal_split_reproduces_from_declared_rng': True, 'initial_states_match_prepractice_archives': True, 'all_atomic_modes_correct': bool(np.all(data['learned'] == data['truth'])), 'scope': 'Goals withheld only in the goal role. Their state identities, atomic transitions, and shared phase parameters remain available during practice. Same32worlds, no unseen-state or unseen-world claim. Original queries are determined by learned transitions, not a separately supplied true adjacency.'}


def sizes_audit(root, learned_queries):
    directory = root / ('sizes_learned_queries' if learned_queries else 'sizes')
    ring = load_source(directory / 'shared_source.py', 'sizes_saved_ring')
    summary = json.loads((directory / 'summary.json').read_text())
    results = {}
    for d in tqdm(summary['sizes'], desc='Size replay'):
        data = np.load(directory / f'states{d}.npz')
        count = np.zeros_like(data['counts'])
        truth = np.zeros_like(data['truth'])
        for seed in range(16):
            rng = np.random.default_rng(991000 + 100 * d + seed)
            order = rng.permutation(d)
            truth[seed, order, 0] = np.roll(order, -1)
            truth[seed, order, 1] = np.roll(order, 1)
            for state in range(d):
                for action in range(2):
                    observed = np.full(5, truth[seed, state, action])
                    noisy = rng.random(5) < .05
                    observed[noisy] = rng.integers(d, size=noisy.sum())
                    np.add.at(count[seed, state, action], observed, 1)
        np.testing.assert_array_equal(truth, data['truth'])
        np.testing.assert_array_equal(count, data['counts'])
        np.testing.assert_array_equal(count.argmax(axis=3), data['learned'])
        task_transition = data['learned'] if learned_queries else truth
        queries = [np.array([(s, g) for s in range(d) for g in range(d) if s != g and g not in model[s]]) for model in task_transition]
        size = np.lcm.reduce([len(query) for query in queries])
        expected_queries = np.array([np.tile(query, (size // len(query), 1)) for query in queries])
        np.testing.assert_array_equal(expected_queries, data['queries'])
        rng = np.random.default_rng(993000 + d)
        initial = rng.uniform(-np.pi, np.pi, (16, d))
        np.testing.assert_array_equal(initial, data['pretrain_angles'][0])
        phase = initial.copy()
        for _ in range(100): phase += ring.atomic(phase, data['learned'], 6.)[1]
        assert error(phase, data['pretrain_angles'][1]) == 0
        phases = {arm: data['pretrain_angles'][-1].copy() for arm in ['practice', 'atomic', 'shuffled']}
        for _ in range(100):
            for arm, phase in phases.items():
                phase += ring.atomic(phase, data['learned'], 6.)[1] if arm == 'atomic' else .1 * ring.sampled(phase, data['learned'], data['queries'], d // 2, 6., rng, 64, 'practice' if arm == 'practice' else 'shuffled')
        replay = {arm: error(phase, data[f'angles_{arm}'][1]) for arm, phase in phases.items()}
        assert max(replay.values()) == 0
        evaluation = {}
        for arm in phases:
            value = enumerate_routes(data[f'angles_{arm}'][-1], data['truth'], data['queries'], d // 2, 6.)
            evaluation[arm] = error(value, data[f'curves_{arm}'][-1, :, 0])
            assert evaluation[arm] < 2e-14
        difference = data['curves_practice'][-1, :, 0] - data['curves_atomic'][-1, :, 0]
        bootstrap_rng = np.random.default_rng(994000 + d)
        interval = np.quantile(difference[bootstrap_rng.integers(16, size=(10000, 16))].mean(axis=1), [.025, .975])
        assert error(interval, summary['results'][str(d)]['practice_minus_atomic']['bootstrap95']) == 0
        incorrect = np.any(data['truth'] != data['learned'], axis=(1, 2))
        original_difference = None
        if learned_queries and d != 10:
            previous = np.load(root / 'sizes' / f'states{d}.npz')
            original_difference = max(error(data[key], previous[key]) for key in data.files)
            assert original_difference == 0
        results[str(d)] = {'first100_update_replay_errors': replay, 'independent_enumeration_max_errors': evaluation, 'mean_practice_minus_atomic': float(difference.mean()), 'bootstrap95': interval.tolist(), 'unique_queries_per_world': [len(query) for query in queries], 'repeated_query_array_length': int(size), 'original_true_query_run_max_difference_when_tables_correct': original_difference, 'incorrect_checker_world_indices': np.flatnonzero(incorrect).tolist(), 'checker_accuracy_in_incorrect_worlds': np.mean(data['truth'] == data['learned'], axis=(1, 2))[incorrect].tolist(), 'incorrect_world_final_true_internal_practice': data['curves_practice'][-1, incorrect, :2].tolist()}
    scope = 'Corrected task catalogue depends only on the finite learned table. Each unique allowed pair is repeated equally within a world; common LCM padding preserves its uniform task law. No true topology enters training queries. All other sizes reproduce the earlier arrays exactly.' if learned_queries else 'Original query set is exhaustively filtered using true one-step adjacency. Its exclusions therefore reveal undirected neighbor sets, even though the optimizer only receives sampled start-goal identities and the learned table. For the one inaccurate10-stateworld this task distribution contains true adjacency information missing from its fitted table. This retained original run is not strictly atom-only information.'
    return {'sizes': results, 'query_side_information': scope}


def softmax_audit(directory):
    data = np.load(directory / 'trajectories.npz')
    keys, values = data['memory_state_keys'], data['memory_values']
    states = 12
    alpha = np.log(states + 1.)
    tests = []
    for step in [0, 20, 200, 1000, 3000]:
        n = step + 1
        length = states * n
        key, value = keys[:length], values[:, :length]
        np.testing.assert_array_equal(np.bincount(key), np.full(states, n))
        # Actual finite-logit, normalized attention, reading every occupied record.
        logits = alpha * (np.arange(states)[:, None] == key[None, :])
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        output = np.einsum('im,sm->si', weights, value, optimize=False)
        raw_difference = output[:, None, :] - output[:, :, None]
        scale = n * (np.exp(alpha) + states - 1) / (np.exp(alpha) - 1)
        corrected = scale * raw_difference
        linear_output = np.einsum('im,sm->si', np.arange(states)[:, None] == key[None, :], value, optimize=False)
        expected = linear_output[:, None, :] - linear_output[:, :, None]
        discrepancy = error(corrected, expected)
        assert discrepancy < 1e-10
        tests.append({'step': step, 'records': length, 'copies_per_state': n, 'readout_scale': float(scale), 'corrected_all_state_pair_max_error': discrepancy, 'uncorrected_all_state_pair_max_error': error(raw_difference, expected)})
    return {'matching_logit': float(alpha), 'nonmatching_logit': 0., 'results': tests, 'identity': 'With n records per state and d states, A_i=((exp(alpha)-1)*theta_i+sum_j theta_j)/(n*(exp(alpha)+d-1)). Two heads for goal and current state followed by subtraction and scale n*(exp(alpha)+d-1)/(exp(alpha)-1) give theta_goal-theta_state exactly. This audits only the read circuit, not a full vanilla-transformer learning loop.', 'extra_gate': 'For d=12 and alpha=log13, multiply the difference of the two normalized heads by2n. This length-dependent multiplication is extra to fixed linear attention output projections.', 'parameter_records_at_end': int(len(keys)), 'generated_action_events_over_run': 3000 * 32 * 6, 'context_count_excludes': ['rollout scratch tokens', 'eligibilities awaiting terminal rewards', 'pair identifiers and counter/controller state', 'frozen24entry learned transition table'], 'source_implementation': 'Saved full simulation uses signed unnormalized linear attention; this targeted probe implements actual normalized finite-logit reads on its saved memories.'}


def gain_audit(root, out):
    directory = root / 'trainable_gain'
    ring = load_source(directory / 'shared_source.py', 'gain_saved_ring')
    gain = load_source(directory / 'run_source.py', 'gain_saved_source')
    gain.ring = ring
    original = np.load(root / 'confirmation/trajectories.npz')
    data = np.load(directory / 'trajectories.npz')
    report = json.loads((directory / 'summary.json').read_text())
    rng = np.random.default_rng(906815)
    phase = rng.uniform(-np.pi, np.pi, (32, 12))
    raw = np.full(32, np.log(np.expm1(6.)))
    np.testing.assert_array_equal(phase, data['pretrain_angles'][0])
    np.testing.assert_array_equal(raw, data['pretrain_gain_raw'][0])
    for _ in range(100):
        _, gp, gr = gain.atomic(phase, raw, original['learned'])
        phase += gp
        raw += gr
    np.testing.assert_array_equal(phase, data['pretrain_angles'][1])
    np.testing.assert_array_equal(raw, data['pretrain_gain_raw'][1])
    weights = {arm: (data['pretrain_angles'][-1].copy(), data['pretrain_gain_raw'][-1].copy()) for arm in ['practice', 'atomic']}
    for _ in range(100):
        for arm, (phase, raw) in weights.items():
            if arm == 'atomic':
                _, gp, gr = gain.atomic(phase, raw, original['learned'])
                phase += gp
                raw += gr
            else:
                gp, gr = gain.sampled(phase, raw, original['learned'], original['queries'], rng)
                phase += .1 * gp
                raw += .1 * gr
    replay = {}
    evaluation = {}
    for arm, (phase, raw) in weights.items():
        replay[arm] = max(error(phase, data[f'angles_{arm}'][1]), error(raw, data[f'gain_raw_{arm}'][1]))
        assert replay[arm] == 0
        success = enumerate_routes(data[f'angles_{arm}'][-1], original['truth'], original['queries'], 6, gain.scale(data[f'gain_raw_{arm}'][-1])[:, None, None])
        evaluation[arm] = error(success, data[f'curves_{arm}'][-1, :, 0])
        assert evaluation[arm] < 2e-14
    difference = data['curves_practice'][-1, :, 0] - data['curves_atomic'][-1, :, 0]
    bootstrap = difference[np.random.default_rng(906816).integers(32, size=(10000, 32))].mean(axis=1)
    effect = {'mean': float(difference.mean()), 'bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist()}
    assert effect == report['paired_practice_minus_atomic']
    # Four random configurations span small to large gains; two pretrained and
    # two final configurations test the actual operating regime. No updates occur.
    test_rng = np.random.default_rng(408911)
    test_phase = np.concatenate([test_rng.uniform(-np.pi, np.pi, (4, 12)), data['pretrain_angles'][-1, :2], data['angles_practice'][-1, :2]])
    test_raw = np.concatenate([np.array([-2., 0., 2., 6.]), data['pretrain_gain_raw'][-1, :2], data['gain_raw_practice'][-1, :2]])
    indices = np.array([0, 1, 2, 3, 0, 1, 0, 1])
    transition, queries = original['learned'][indices], original['queries'][indices]
    epsilon = 1e-4
    exact = (enumerate_routes(test_phase, transition, queries, 6, gain.scale(test_raw + epsilon)[:, None, None]) - enumerate_routes(test_phase, transition, queries, 6, gain.scale(test_raw - epsilon)[:, None, None])) / (2 * epsilon)
    half_step = (enumerate_routes(test_phase, transition, queries, 6, gain.scale(test_raw + epsilon / 2)[:, None, None]) - enumerate_routes(test_phase, transition, queries, 6, gain.scale(test_raw - epsilon / 2)[:, None, None])) / epsilon
    assert error(exact, half_step) < 1e-8
    score_rng = np.random.default_rng(408912)
    samples = np.array([gain.sampled(test_phase, test_raw, transition, queries, score_rng, batch=64, horizon=6)[1] for _ in tqdm(range(4096), desc='Gain-score expectation')])
    mean = samples.mean(axis=0)
    sem = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    standardized = (mean - exact) / sem
    assert np.max(abs(standardized)) < 5
    np.savez_compressed(out / 'gain_score_samples.npz', gradient_samples=samples, phase=test_phase, raw_gain=test_raw, transition=transition, queries=queries, exact_gradient_fd=exact, exact_gradient_halfstep_fd=half_step)
    return {'first100_update_replay_errors': replay, 'atomic_pretraining_first100_bitwise_identical': True, 'independent_enumeration_final_success_errors': evaluation, 'bootstrap_reproduced': effect, 'monte_carlo': {'batch_count': len(samples), 'trajectories_per_batch': 64, 'configuration_seed': 408911, 'sampling_seed': 408912, 'gains': gain.scale(test_raw).tolist(), 'exact_expected_gain_gradient_finite_difference': exact.tolist(), 'finite_difference_step_halving_max_change': error(exact, half_step), 'sampled_gradient_means': mean.tolist(), 'sampled_mean_standard_errors': sem.tolist(), 'standardized_errors': standardized.tolist(), 'max_absolute_standardized_error': float(np.max(abs(standardized)))}, 'source_leakage_audit': 'Both atomic and sampled updates receive only learned transitions, allowed queries, phases/raw gains, and RNG. True transitions enter evaluation only; stopping uses internal risk and atomic loss. Fixed pretraining5000updates is not selected using true outcomes. Query catalogue inherits the primary learned-only task law.'}


def main():
    load_dotenv(Path.home() / '.env')
    fast = Path(os.environ['FAST_CACHE_DIR'])
    root = fast / 'generative_practice_loop_ring'
    out = fast / 'generative_practice_loop_ring_audit'
    out.mkdir(parents=True, exist_ok=True)
    results = {'uk_time': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'transfer': transfer_audit(root), 'original_sizes_true_queries': sizes_audit(root, False), 'corrected_sizes_learned_queries': sizes_audit(root, True), 'softmax_read': softmax_audit(fast / 'generative_practice_loop_ring_context'), 'trainable_gain': gain_audit(root, out)}
    sources = [Path(__file__), root / 'transfer/source.py', root / 'transfer/shared_source.py', root / 'sizes/run_source.py', root / 'sizes/shared_source.py', root / 'sizes_learned_queries/run_source.py', root / 'sizes_learned_queries/shared_source.py', root / 'trainable_gain/run_source.py', root / 'trainable_gain/shared_source.py', fast / 'generative_practice_loop_ring_context/run_source.py']
    results['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py')
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
