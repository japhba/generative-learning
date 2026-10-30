"""Full growing-context implementation of the cyclic composer's local practice rule.

One-hot keys are stored losslessly as state indices. Every policy read compares
its query with every occupied key and sums every value; no phase cache is used.
Only within-batch increments are compressed into one record per state.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange

import practice_loop_ring as ring


class ExplicitLearner:
    def __init__(self, initial): self.phase = initial.copy()

    def delta(self, state, goal):
        index = np.arange(len(state))[:, None]
        return self.phase[index, goal] - self.phase[index, state]

    def update(self, change): self.phase += change


class ContextLearner:
    def __init__(self, initial, updates):
        seeds, states = initial.shape
        self.state_keys = np.tile(np.arange(states), updates + 1)
        self.values = np.zeros((seeds, len(self.state_keys)))
        self.values[:, :states] = initial
        self.length = states
        self.states = states

    def delta(self, state, goal):
        # This is the linear-attention query e_goal - e_state, with keys e_i.
        key = self.state_keys[None, None, :self.length]
        score = (goal[:, :, None] == key).astype(float)
        score -= state[:, :, None] == key
        return np.einsum('sbm,sm->sb', score, self.values[:, :self.length], optimize=False)

    def read(self, state):
        score = state[:, :, None] == self.state_keys[None, None, :self.length]
        return np.einsum('sbm,sm->sb', score, self.values[:, :self.length], optimize=False)

    def update(self, change):
        following = self.length + self.states
        self.values[:, self.length:following] = change
        self.length = following


def rollout(learner, transition, queries, uniforms, beta):
    seeds, batch = queries.shape[:2]
    state, goal = queries[:, :, 0].copy(), queries[:, :, 1]
    index = np.arange(seeds)[:, None]
    alive = state != goal
    records, action_history, state_history, probability_history, delta_history = [], [], [state.copy()], [], []
    for uniform in uniforms:
        difference = learner.delta(state, goal)
        probability = ring.sigmoid(beta * np.sin(difference))
        action = uniform < probability
        local = (action - probability) * beta * np.cos(difference) * alive
        records.append((state.copy(), local))
        state = np.where(alive, transition[index, state, 1 - action.astype(int)], state)
        alive &= state != goal
        action_history.append(action); state_history.append(state.copy())
        probability_history.append(probability); delta_history.append(difference)
    reward = (state == goal).astype(float)
    paired = reward.reshape(seeds, batch // 2, 2)
    coefficient = (paired - paired[:, :, ::-1]).reshape(seeds, batch) / batch
    gradient = np.zeros((seeds, transition.shape[1]))
    for current, local in records:
        value = coefficient * local
        np.add.at(gradient, (index, goal), value)
        np.add.at(gradient, (index, current), -value)
    return gradient, np.array(action_history), np.array(state_history), np.array(probability_history), np.array(delta_history), reward


def evaluate(phase, learned, truth, queries, horizon, beta):
    return np.stack([ring.exact(phase, truth, queries, horizon, beta), ring.exact(phase, learned, queries, horizon, beta), ring.atomic(phase, learned, beta)[0]], axis=1)


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/confirmation')
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring_context')
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--random-seed', type=int, default=416901)
    args = parser.parse_args()
    assert args.batch % 2 == 0
    args.output.mkdir(parents=True, exist_ok=True)
    source_data = np.load(args.input / 'trajectories.npz')
    source_metadata = json.loads((args.input / 'summary.json').read_text())
    seeds = args.seeds
    learned, truth, queries = [source_data[name][:seeds].copy() for name in ['learned', 'truth', 'queries']]
    states = learned.shape[1]
    parameters = source_metadata['parameters']
    beta, horizon, rate = parameters['beta'], parameters['horizon'], parameters['rate']
    # Preserve only the four aliased phases, not the unwrapped cycle index.
    discrete_phase = np.rint(source_data['initial'][:seeds] / (np.pi / 2)).astype(int) % 4
    initial = np.array([0., np.pi / 2, -np.pi, -np.pi / 2])[discrete_phase]
    initial_invariance_error = float(np.max(np.abs(ring.exact(initial, learned, queries, horizon, beta) - ring.exact(source_data['initial'][:seeds], learned, queries, horizon, beta))))
    assert initial_invariance_error < 1e-12
    assert np.all(np.unique(initial) == np.array([-np.pi, -np.pi / 2, 0., np.pi / 2]))
    explicit, context = ExplicitLearner(initial), ContextLearner(initial, args.steps)
    rng = np.random.default_rng(args.random_seed)
    all_states = np.broadcast_to(np.arange(states), (seeds, states))
    metric = evaluate(initial, learned, truth, queries, horizon, beta)
    times = [0]; curves = [metric]; context_curves = [metric.copy()]
    explicit_phases = [initial.copy()]; context_phases = [context.read(all_states)]
    discrepancies, rewards = [], []
    metadata = {'start_uk': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'parameters': {'states': states, 'horizon': horizon, 'beta': beta, 'rate': rate, 'data_seed_ids': list(range(parameters['seed_start'], parameters['seed_start'] + seeds))}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'shared_source_sha256': hashlib.sha256(Path(ring.__file__).read_bytes()).hexdigest(), 'input_sha256': hashlib.sha256((args.input / 'trajectories.npz').read_bytes()).hexdigest(), 'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}, 'metrics': ['true_future_success', 'learned_future_success', 'atomic_inverse_nll'], 'error_columns': ['phase_difference', 'action_probability', 'gradient', 'effective_phase', 'action_mismatches', 'state_mismatches', 'reward_mismatches'], 'initial_wrapping_policy_error': initial_invariance_error, 'protocol': 'Fixed-horizon implementation audit, not statistical confirmation. Full growing memory reads for every sampled action; one-hot keys stored exactly as state indices. Each batch compresses only its own summed phase increments into12new signed value tokens. No cross-batch memory compression, cached phase parameter, teacher transition in learner, or explicit-branch read by context. Exact evaluation uses a temporary diagnostic read and never feeds back into updates.'}
    (args.output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    started = perf_counter()
    for step in trange(1, args.steps + 1, desc='Full ring context cycles'):
        chosen = rng.integers(queries.shape[1], size=(seeds, args.batch // 2))
        query = np.repeat(queries[np.arange(seeds)[:, None], chosen], 2, axis=1)
        uniforms = rng.random((horizon, seeds, args.batch))
        reference = rollout(explicit, learned, query, uniforms, beta)
        memory = rollout(context, learned, query, uniforms, beta)
        explicit.update(rate * reference[0]); context.update(rate * memory[0])
        diagnostic_phase = context.read(all_states)
        error = [np.max(np.abs(reference[4] - memory[4])), np.max(np.abs(reference[3] - memory[3])), np.max(np.abs(reference[0] - memory[0])), np.max(np.abs(explicit.phase - diagnostic_phase)), np.count_nonzero(reference[1] != memory[1]), np.count_nonzero(reference[2] != memory[2]), np.count_nonzero(reference[5] != memory[5])]
        assert max(error) < 1e-7, (step, error)
        assert np.isfinite(diagnostic_phase).all()
        discrepancies.append(error); rewards.append(memory[5].mean(axis=1))
        if step % args.eval_every == 0:
            times.append(step)
            curves.append(evaluate(explicit.phase, learned, truth, queries, horizon, beta))
            context_curves.append(evaluate(diagnostic_phase, learned, truth, queries, horizon, beta))
            explicit_phases.append(explicit.phase.copy()); context_phases.append(diagnostic_phase.copy())
    curves, context_curves, discrepancies = map(np.array, [curves, context_curves, discrepancies])
    metadata.update({'finish_uk': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'duration_seconds': perf_counter() - started, 'memory_records': context.length, 'maximum_discrepancies': dict(zip(metadata['error_columns'], discrepancies.max(axis=0).tolist())), 'maximum_metric_discrepancy': float(np.max(np.abs(curves - context_curves))), 'initial_mean_metrics': curves[0].mean(axis=0).tolist(), 'final_mean_metrics': curves[-1].mean(axis=0).tolist(), 'final_metrics_by_seed': curves[-1].tolist()})
    np.savez_compressed(args.output / 'trajectories.npz', times=times, explicit_metrics=curves, context_metrics=context_curves, errors=discrepancies, rewards=rewards, explicit_phases=explicit_phases, context_phases=context_phases, memory_state_keys=context.state_keys[:context.length], memory_values=context.values[:, :context.length], initial=initial, learned=learned, truth=truth, queries=queries, counts=source_data['counts'][:seeds])
    (args.output / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'shared_source.py').write_bytes(Path(ring.__file__).read_bytes())
    fig, axes = plt.subplots(3, 1, layout='constrained')
    for axis, index, ylabel in [(axes[0], 0, 'Future success'), (axes[1], 2, 'Atomic NLL')]:
        axis.plot(times, curves[:, :, index].mean(axis=1), label='Explicit phases')
        axis.plot(times, context_curves[:, :, index].mean(axis=1), linestyle='--', label='Growing context')
        axis.set(ylabel=ylabel)
    axes[0].legend()
    for index in [0, 1, 3]: axes[2].plot(np.arange(1, args.steps + 1), discrepancies[:, index], label=metadata['error_columns'][index].replace('_', ' '))
    axes[2].set(xlabel='Offline update', ylabel='Absolute discrepancy'); axes[2].legend()
    figure = args.output / 'practice-loop-ring-context.png'
    fig.savefig(figure); plt.close(fig)
    (Path(__file__).resolve().parents[1] / 'tex/figures' / figure.name).write_bytes(figure.read_bytes())
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__': main()
