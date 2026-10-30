"""Finite learned dynamics, latent routing, and offline goal-conditioned practice.

Only the observation constructor and evaluation use the true environment.
All practice arms use the same frozen transition table estimated from finite data.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


METHODS = ['practice', 'practice_readout', 'reconstruction', 'unfiltered', 'shuffled', 'frozen', 'hindsight', 'hindsight_erased', 'planning_gradient']
LABELS = {'practice': 'Goal-checked practice', 'practice_readout': 'Practice, hidden weights frozen', 'reconstruction': 'Original-data inverse MLE', 'unfiltered': 'Unfiltered self-imitation', 'shuffled': 'Shuffled checks', 'frozen': 'Frozen accepted proposals', 'hindsight': 'Hindsight synthetic reconstruction', 'hindsight_erased': 'Hindsight with loop erasure', 'planning_gradient': 'Exact planning gradient'}


def sigmoid(x): return np.exp(-np.logaddexp(0, -x))


def softmax(x):
    value = np.exp(x - x.max(axis=-1, keepdims=True))
    return value / value.sum(axis=-1, keepdims=True)


def init_weights(rng, states, hidden):
    return [rng.normal(0, 0.5, (hidden, states)), rng.normal(0, 0.5, (hidden, states)), np.zeros(hidden), rng.normal(0, 0.5, (2, hidden)), np.zeros(2)]


def copy_weights(weights): return [x.copy() for x in weights]


def zeros_like(weights): return [np.zeros_like(x) for x in weights]


def add_step(weights, gradient, rate, tied=False):
    if tied:
        weights[1] += rate * (gradient[1] - gradient[0])
        weights[0] = -weights[1].copy()
    for index, (weight, change) in enumerate(zip(weights, gradient)):
        if tied and index < 2: continue
        weight += rate * change
        assert np.isfinite(weight).all()


def hidden_states(weights): return np.array(list(itertools.product((0.0, 1.0), repeat=len(weights[2]))))


def marginal(weights, queries):
    us, ug, bias, readout, output_bias = weights
    hidden = hidden_states(weights)
    logits = us[:, queries[:, 0]].T + ug[:, queries[:, 1]].T + bias
    probability = sigmoid(logits)
    log_hidden = logits @ hidden.T - np.logaddexp(0, logits).sum(axis=1, keepdims=True)
    ph = np.exp(log_hidden)
    pa = softmax(hidden @ readout.T + output_bias)
    return ph @ pa, probability, ph, pa, hidden


def probability_gradient(weights, queries, coefficient):
    _, probability, ph, pa, hidden = marginal(weights, queries)
    conditional_value = coefficient @ pa.T
    weighted = ph * conditional_value
    hidden_delta = weighted @ hidden - weighted.sum(axis=1, keepdims=True) * probability
    gradient = zeros_like(weights)
    np.add.at(gradient[0].T, queries[:, 0], hidden_delta)
    np.add.at(gradient[1].T, queries[:, 1], hidden_delta)
    gradient[2] = hidden_delta.sum(axis=0)
    action_delta = (ph.T @ coefficient) * pa - np.sum(ph * conditional_value, axis=0)[:, None] * pa
    gradient[3] = action_delta.T @ hidden
    gradient[4] = action_delta.sum(axis=0)
    return gradient


def mle_gradient(weights, queries, actions):
    policy = marginal(weights, queries)[0]
    coefficient = np.eye(2)[actions] / policy / len(queries)
    return probability_gradient(weights, queries, coefficient)


def make_world(rng, states, observations, noise, split_type):
    labels = rng.permutation(states)
    actions = rng.permutation(2)
    truth = np.empty((states, 2), dtype=int)
    for position, label in enumerate(labels):
        truth[label, actions[0]] = labels[max(0, position - 1)]
        truth[label, actions[1]] = labels[min(states - 1, position + 1)]
    counts = np.zeros((states, 2, states), dtype=int)
    for state, action in itertools.product(range(states), range(2)):
        targets = np.full(observations, truth[state, action])
        corrupt = rng.random(observations) < noise
        targets[corrupt] = rng.integers(states, size=corrupt.sum())
        np.add.at(counts[state, action], targets, 1)
    learned = counts.argmax(axis=2)
    atomic_queries = np.array([(s, learned[s, a]) for s, a in itertools.product(range(states), range(2))])
    atomic_actions = np.tile(np.arange(2), states)
    distant = np.array([(s, g) for s, g in itertools.product(range(states), repeat=2) if s != g and g not in learned[s]])
    rng.shuffle(distant)
    if split_type == 'goal':
        heldout = rng.choice(states, size=2, replace=False)
        test_mask = np.isin(distant[:, 1], heldout)
        train, test = distant[~test_mask], distant[test_mask]
    elif split_type == 'all':
        train, test = distant.copy(), distant.copy()
    else:
        split = len(distant) // 2
        train, test = distant[:split], distant[split:]
    return truth, learned, counts, labels, atomic_queries, atomic_actions, train, test


def all_queries(states): return np.array(list(itertools.product(range(states), repeat=2)))


def policy_values(weights, transition, horizon):
    states = len(transition)
    policy = marginal(weights, all_queries(states))[0].reshape(states, states, 2)
    values = [np.eye(states)]
    goal = np.arange(states)[None, :]
    for _ in range(horizon):
        new = sum(policy[:, :, a] * values[-1][transition[:, a, None], goal] for a in range(2))
        np.fill_diagonal(new, 1)
        values.append(new)
    return policy, values


def evaluate(weights, truth, learned, train, test, atomic_queries, atomic_actions, horizon):
    _, values = policy_values(weights, truth, horizon)
    _, internal = policy_values(weights, learned, horizon)
    atomic = marginal(weights, atomic_queries)[0][np.arange(len(atomic_actions)), atomic_actions]
    return np.array([values[-1][test[:, 0], test[:, 1]].mean(), values[-1][train[:, 0], train[:, 1]].mean(), internal[-1][train[:, 0], train[:, 1]].mean(), -np.log(atomic).mean(), atomic.mean()])


def planning_gradient(weights, transition, queries, horizon):
    states = len(transition)
    policy, values = policy_values(weights, transition, horizon)
    occupancy = np.zeros((states, states))
    np.add.at(occupancy, (queries[:, 0], queries[:, 1]), 1 / len(queries))
    coefficient = np.zeros_like(policy)
    goals = np.arange(states)
    for step in range(horizon):
        np.fill_diagonal(occupancy, 0)
        following = np.zeros_like(occupancy)
        for action in range(2):
            coefficient[:, :, action] += occupancy * values[horizon - step - 1][transition[:, action, None], goals[None, :]]
            np.add.at(following, transition[:, action], occupancy * policy[:, :, action])
        occupancy = following
    return probability_gradient(weights, all_queries(states), coefficient.reshape(-1, 2))


def rollout(weights, transition, queries, horizon, rng):
    us, ug, bias, readout, output_bias = weights
    state, goal = queries.T.copy()
    alive = state != goal
    records = []
    for _ in range(horizon):
        probability = sigmoid(us[:, state].T + ug[:, goal].T + bias)
        hidden = (rng.random(probability.shape) < probability).astype(float)
        action_probability = softmax(hidden @ readout.T + output_bias)
        action = (rng.random(len(state)) > action_probability[:, 0]).astype(int)
        records.append((state.copy(), hidden, action, alive.copy()))
        state = np.where(alive, transition[state, action], state)
        alive &= state != goal
    return (state == goal).astype(float), records, state


def eligibility_gradient(weights, queries, records, coefficient):
    us, ug, bias, readout, output_bias = weights
    goal = queries[:, 1]
    gradient = zeros_like(weights)
    for state, hidden, action, alive in records:
        probability = sigmoid(us[:, state].T + ug[:, goal].T + bias)
        action_probability = softmax(hidden @ readout.T + output_bias)
        weight = coefficient * alive
        hidden_delta = (hidden - probability) * weight[:, None]
        action_delta = (np.eye(2)[action] - action_probability) * weight[:, None]
        np.add.at(gradient[0].T, state, hidden_delta)
        np.add.at(gradient[1].T, goal, hidden_delta)
        gradient[2] += hidden_delta.sum(axis=0)
        gradient[3] += action_delta.T @ hidden
        gradient[4] += action_delta.sum(axis=0)
    return gradient


def hindsight_gradient(weights, records, endpoint, allowed_goals, erase):
    allowed = np.isin(endpoint, allowed_goals)
    if not erase:
        states = np.concatenate([row[0][row[3] & allowed] for row in records])
        actions = np.concatenate([row[2][row[3] & allowed] for row in records])
        goals = np.concatenate([endpoint[row[3] & allowed] for row in records])
    else:
        states, actions, goals = [], [], []
        for n in np.flatnonzero(allowed):
            visited = [row[0][n] for row in records if row[3][n]] + [endpoint[n]]
            proposed = [row[2][n] for row in records if row[3][n]]
            path, chosen = [visited[0]], []
            for action, following in zip(proposed, visited[1:]):
                if following in path:
                    index = path.index(following)
                    path, chosen = path[:index + 1], chosen[:index]
                else:
                    path.append(following); chosen.append(action)
            states.extend(path[:-1]); actions.extend(chosen); goals.extend([endpoint[n]] * len(chosen))
        states, actions, goals = np.array(states, dtype=int), np.array(actions, dtype=int), np.array(goals, dtype=int)
    assert len(states) > 0
    return mle_gradient(weights, np.column_stack([states, goals]), actions)


def checker_diagnostics(truth, learned, counts):
    frequency = counts / counts.sum(axis=2, keepdims=True)
    confidence = frequency.max(axis=2).ravel()
    correct = (learned == truth).ravel()
    ece = 0.0
    for lower, upper in zip(np.arange(0, 1, .2), np.arange(.2, 1.01, .2)):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any(): ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return {'mode_accuracy': float(correct.mean()), 'mean_mode_frequency': float(confidence.mean()), 'mode_ece_five_bins': float(ece), 'brier_to_true_transition': float(np.sum((frequency - np.eye(len(truth))[truth]) ** 2, axis=2).mean())}


def bounded_search(learned, truth, queries, budget):
    outcomes, checks = [], []
    for start, goal in queries:
        frontier, visited, used, solution = [(int(start), [])], {int(start)}, 0, []
        while frontier and used < budget and not solution:
            state, path = frontier.pop(0)
            for action in range(2):
                if used == budget: break
                following = int(learned[state, action]); used += 1
                proposed = [*path, action]
                if following == goal:
                    solution = proposed
                    break
                if following not in visited:
                    visited.add(following); frontier.append((following, proposed))
        state = start
        for action in solution: state = truth[state, action]
        outcomes.append(float(state == goal)); checks.append(used)
    return {'true_success': float(np.mean(outcomes)), 'mean_transition_queries': float(np.mean(checks)), 'success_by_query': outcomes}


def audit_gradient(weights, transition, queries, horizon):
    analytic = planning_gradient(weights, transition, queries, horizon)
    maximum = 0.0
    for group, parameter in enumerate(weights):
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old + 1e-5
            plus = policy_values(weights, transition, horizon)[1][-1][queries[:, 0], queries[:, 1]].mean()
            parameter[index] = old - 1e-5
            minus = policy_values(weights, transition, horizon)[1][-1][queries[:, 0], queries[:, 1]].mean()
            parameter[index] = old
            maximum = max(maximum, abs((plus - minus) / 2e-5 - analytic[group][index]))
    assert maximum < 1e-7
    return maximum


def plot(output, figure_dir):
    metadata = json.loads((output / 'metadata.json').read_text())
    archive = np.load(output / 'curves.npz')
    success_label = 'Success on practiced query population' if metadata['arguments']['split'] == 'all' else 'Success on held-out goal pairs'
    for metric, suffix, ylabel in [(0, 'learning', success_label), (3, 'reconstruction', 'Original atomic inverse NLL')]:
        fig, ax = plt.subplots()
        for method in ['practice', 'practice_readout', 'reconstruction', 'hindsight_erased', 'planning_gradient']:
            values = archive[method][:, :, metric]
            line, = ax.plot(archive['steps'], values.mean(axis=0), label=LABELS[method])
            low, high = np.quantile(values, [.1, .9], axis=0)
            ax.fill_between(archive['steps'], low, high, alpha=.15, color=line.get_color())
        ax.set(xlabel='Offline updates', ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f'practice-loop-candidate-{suffix}.png')
        plt.close(fig)
    fig, ax = plt.subplots()
    initial = archive['pretraining'][:, :, 1]
    line, = ax.plot(archive['pretrain_steps'], initial.mean(axis=0))
    low, high = np.quantile(initial, [.1, .9], axis=0)
    ax.fill_between(archive['pretrain_steps'], low, high, alpha=.2, color=line.get_color())
    ax.set(xlabel='Original-data updates', ylabel='Atomic inverse NLL')
    fig.tight_layout(); fig.savefig(figure_dir / 'practice-loop-candidate-pretraining.png'); plt.close(fig)


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--seed-start', type=int, default=12000)
    parser.add_argument('--states', type=int, default=12)
    parser.add_argument('--hidden', type=int, default=4)
    parser.add_argument('--observations', type=int, default=5)
    parser.add_argument('--observation-noise', type=float, default=0.05)
    parser.add_argument('--split', choices=['goal', 'pair', 'all'], default='goal')
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--pretrain-steps', type=int, default=4000)
    parser.add_argument('--reference-steps', type=int, default=0)
    parser.add_argument('--tied', action='store_true')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--rate', type=float, default=.3)
    parser.add_argument('--pretrain-rate', type=float, default=1.)
    parser.add_argument('--eval-every', type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.figure_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'metrics': ['test_success', 'train_success', 'internal_train_success', 'atomic_inverse_nll', 'atomic_action_probability'], 'methods': METHODS, 'checker_accuracy': [], 'checker_diagnostics': [], 'bounded_search': [], 'full_search': [], 'gradient_audit': None, 'stopping': {method: [] for method in METHODS}}
    (args.output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    reference_steps = max(args.pretrain_steps, args.reference_steps)
    archive = {'steps': np.arange(0, args.steps + 1, args.eval_every), 'pretrain_steps': np.arange(0, reference_steps + 1, args.eval_every), 'pretraining': [], **{method: [] for method in METHODS}}
    for index in tqdm(range(args.seeds), desc='Latent routing seeds'):
        seed = args.seed_start + index
        truth, learned, counts, ordering, atomic_queries, atomic_actions, train, test = make_world(np.random.default_rng(seed), args.states, args.observations, args.observation_noise, args.split)
        initial = init_weights(np.random.default_rng(seed + 10000), args.states, args.hidden)
        if args.tied: initial[0] = -initial[1].copy()
        metadata['checker_accuracy'].append(float(np.mean(truth == learned)))
        metadata['checker_diagnostics'].append(checker_diagnostics(truth, learned, counts))
        metadata['bounded_search'].append(bounded_search(learned, truth, test, args.horizon))
        metadata['full_search'].append(bounded_search(learned, truth, test, 2 * args.states))
        if args.split != 'all': assert not set(map(tuple, train)) & set(map(tuple, test))
        assert not set(map(tuple, atomic_queries)) & set(map(tuple, test))
        if args.split == 'goal': assert not set(train[:, 1]) & set(test[:, 1])
        for name, value in [('truth', truth), ('learned', learned), ('counts', counts), ('ordering', ordering), ('atomic_queries', atomic_queries), ('atomic_actions', atomic_actions), ('train', train), ('test', test)]: archive[f'{name}_{index}'] = value
        if index == 0: metadata['gradient_audit'] = audit_gradient(initial, learned, train[:3], 3)
        pretraining = []
        for step in range(reference_steps + 1):
            if step % args.eval_every == 0:
                score = evaluate(initial, truth, learned, train, test, atomic_queries, atomic_actions, args.horizon)
                pretraining.append([score[0], score[3]])
            if step == args.pretrain_steps: warmup = copy_weights(initial)
            if step < reference_steps: add_step(initial, mle_gradient(initial, atomic_queries, atomic_actions), args.pretrain_rate, args.tied)
        archive['pretraining'].append(pretraining)
        initial = warmup
        for group, value in enumerate(initial): archive[f'initial_{index}_{group}'] = value
        for method in tqdm(METHODS, desc=f'Seed {seed}', leave=False):
            rng = np.random.default_rng(seed + 20000)
            weights = copy_weights(initial)
            curve = [evaluate(weights, truth, learned, train, test, atomic_queries, atomic_actions, args.horizon)]
            weight_history = [np.concatenate([value.ravel() for value in weights])]
            for step in range(1, args.steps + 1):
                if method == 'reconstruction': gradient = mle_gradient(weights, atomic_queries, atomic_actions)
                elif method == 'planning_gradient': gradient = planning_gradient(weights, learned, train, args.horizon)
                else:
                    queries = np.repeat(train[rng.integers(len(train), size=args.batch)], 2, axis=0)
                    reward, records, endpoint = rollout(initial if method == 'frozen' else weights, learned, queries, args.horizon, rng)
                    if method in ['hindsight', 'hindsight_erased']: gradient = hindsight_gradient(weights, records, endpoint, np.unique(train[:, 1]), method == 'hindsight_erased')
                    else:
                        if method == 'shuffled': reward = rng.permutation(reward)
                        if method in ['practice', 'practice_readout', 'shuffled']:
                            paired = reward.reshape(-1, 2)
                            coefficient = (paired - paired[:, ::-1]).ravel() / len(reward)
                        elif method == 'unfiltered': coefficient = np.full(len(reward), 1 / len(reward))
                        else: coefficient = reward / len(reward)
                        gradient = eligibility_gradient(weights, queries, records, coefficient)
                if method == 'practice_readout':
                    for group in range(3): gradient[group].fill(0)
                add_step(weights, gradient, args.rate, args.tied)
                if step % args.eval_every == 0:
                    curve.append(evaluate(weights, truth, learned, train, test, atomic_queries, atomic_actions, args.horizon))
                    weight_history.append(np.concatenate([value.ravel() for value in weights]))
                    if step >= 1000 and len(curve) >= 17:
                        recent = np.array(curve[-16:])
                        times = np.arange(step - 15 * args.eval_every, step + 1, args.eval_every)
                        optimized = recent[:, 3] if method == 'reconstruction' else 1 - recent[:, 2]
                        slopes = [np.polyfit(np.log(times[start:end]), np.log(optimized[start:end] + 1e-12), 1)[0] for start, end in [(0, 8), (8, 16)]]
                        if max(abs(np.array(slopes))) < .02: break
            metadata['stopping'][method].append(step)
            while len(curve) < len(archive['steps']): curve.append(curve[-1].copy()); weight_history.append(weight_history[-1].copy())
            archive[method].append(curve)
            archive[f'{method}_weight_history_{index}'] = weight_history
            for group, value in enumerate(weights): archive[f'{method}_{index}_{group}'] = value
        np.savez_compressed(args.output / 'curves.npz', **archive)
    metadata['summary'] = {method: {'initial_mean': np.mean(archive[method], axis=0)[0].tolist(), 'final_mean': np.mean(archive[method], axis=0)[-1].tolist(), 'final_by_seed': np.asarray(archive[method])[:, -1].tolist()} for method in METHODS}
    metadata['uk_finish'] = datetime.now(ZoneInfo('Europe/London')).isoformat()
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    plot(args.output, args.figure_dir)
    print(json.dumps({method: values['final_mean'] for method, values in metadata['summary'].items()}, indent=2))


if __name__ == '__main__':
    main()
