"""A cyclic composer initialized at the exact constrained atomic reconstruction optimum."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange


def sigmoid(x): return np.exp(-np.logaddexp(0, -x))


def exact(theta, transitions, queries, horizon, beta, gradient=False):
    seeds, states = theta.shape
    delta = theta[:, None, :] - theta[:, :, None]
    probability = sigmoid(beta * np.sin(delta))
    values = [np.broadcast_to(np.eye(states), (seeds, states, states)).copy()]
    sid = np.arange(seeds)[:, None, None]
    goals = np.arange(states)[None, None, :]
    for _ in range(horizon):
        value = probability * values[-1][sid, transitions[:, :, 0, None], goals] + (1 - probability) * values[-1][sid, transitions[:, :, 1, None], goals]
        value[:, np.arange(states), np.arange(states)] = 1
        values.append(value)
    success = values[-1][np.arange(seeds)[:, None], queries[:, :, 0], queries[:, :, 1]].mean(axis=1)
    if not gradient: return success
    occupancy = np.zeros((seeds, states, states))
    np.add.at(occupancy, (np.arange(seeds)[:, None], queries[:, :, 0], queries[:, :, 1]), 1 / queries.shape[1])
    derivative = np.zeros_like(probability)
    for step in range(horizon):
        occupancy[:, np.arange(states), np.arange(states)] = 0
        q0 = values[horizon - step - 1][sid, transitions[:, :, 0, None], goals]
        q1 = values[horizon - step - 1][sid, transitions[:, :, 1, None], goals]
        derivative += occupancy * (q0 - q1) * probability * (1 - probability) * beta * np.cos(delta)
        following = np.zeros_like(occupancy)
        np.add.at(following, (sid, transitions[:, :, 0, None], goals), occupancy * probability)
        np.add.at(following, (sid, transitions[:, :, 1, None], goals), occupancy * (1 - probability))
        occupancy = following
    return success, derivative.sum(axis=1) - derivative.sum(axis=2)


def atomic(theta, transitions, beta):
    seeds, states = theta.shape
    source = np.broadcast_to(np.arange(states)[None, :, None], transitions.shape)
    sid = np.arange(seeds)[:, None, None]
    delta = theta[sid, transitions] - theta[sid, source]
    labels = np.broadcast_to(np.array([1., 0.]), transitions.shape)
    logits = beta * np.sin(delta)
    loss = (np.logaddexp(0, logits) - labels * logits).mean(axis=(1, 2))
    local = (labels - sigmoid(logits)) * beta * np.cos(delta) / (2 * states)
    gradient = np.zeros_like(theta)
    np.add.at(gradient, (sid, transitions), local)
    np.add.at(gradient, (sid, source), -local)
    return loss, gradient


def sampled(theta, transitions, queries, horizon, beta, rng, batch, mode):
    seeds, states = theta.shape
    chosen = rng.integers(queries.shape[1], size=(seeds, batch // 2))
    pair = queries[np.arange(seeds)[:, None], chosen]
    pair = np.repeat(pair, 2, axis=1)
    state, goal = pair[:, :, 0].copy(), pair[:, :, 1]
    sid = np.arange(seeds)[:, None]
    alive = state != goal
    records = []
    for _ in range(horizon):
        delta = theta[sid, goal] - theta[sid, state]
        probability = sigmoid(beta * np.sin(delta))
        action = rng.random(state.shape) < probability
        local = (action - probability) * beta * np.cos(delta) * alive
        records.append((state.copy(), local))
        state = np.where(alive, transitions[sid, state, 1 - action.astype(int)], state)
        alive &= state != goal
    reward = (state == goal).astype(float)
    if mode == 'unfiltered': coefficient = np.ones_like(reward) / batch
    else:
        if mode == 'shuffled': reward = rng.permuted(reward, axis=1)
        paired = reward.reshape(seeds, batch // 2, 2)
        coefficient = (paired - paired[:, :, ::-1]).reshape(seeds, batch) / batch
    gradient = np.zeros_like(theta)
    for state, local in records:
        value = coefficient * local
        np.add.at(gradient, (sid, goal), value)
        np.add.at(gradient, (sid, state), -value)
    return gradient


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/pilot')
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--seed-start', type=int, default=86100)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    d, horizon, beta, rate, batch, observations, corruption = 12, 6, 6., .1, 64, 5, .05
    methods = ['practice', 'atomic_reconstruction', 'exact_gradient', 'unfiltered', 'shuffled']
    rng = np.random.default_rng(args.seed_start + 10000)
    truths, learned, counts, orders, queries = [], [], [], [], []
    for seed in range(args.seeds):
        data_rng = np.random.default_rng(args.seed_start + seed)
        labels = data_rng.permutation(d)
        truth = np.empty((d, 2), dtype=int)
        truth[labels, 0], truth[labels, 1] = np.roll(labels, -1), np.roll(labels, 1)
        count = np.zeros((d, 2, d), dtype=int)
        for s in range(d):
            for a in range(2):
                observed = np.full(observations, truth[s, a])
                noisy = data_rng.random(observations) < corruption
                observed[noisy] = data_rng.integers(d, size=noisy.sum())
                np.add.at(count[s, a], observed, 1)
        model = count.argmax(axis=2)
        order = [0]
        for _ in range(d - 1): order.append(int(model[order[-1], 0]))
        assert len(set(order)) == d and model[order[-1], 0] == 0, 'Learned forward cycle is broken; do not silently discard this world.'
        assert np.all(model[np.array(order), 1] == np.roll(order, 1)), 'Learned backward cycle is inconsistent.'
        q = np.array([(s, g) for s in range(d) for g in range(d) if s != g and g not in model[s]])
        truths.append(truth); learned.append(model); counts.append(count); orders.append(order); queries.append(q)
    truths, learned, counts, orders, queries = map(np.array, (truths, learned, counts, orders, queries))
    initial = np.zeros((args.seeds, d))
    initial[np.arange(args.seeds)[:, None], orders] = np.arange(d)[None] * np.pi / 2
    target = np.zeros_like(initial)
    target[np.arange(args.seeds)[:, None], orders] = np.arange(d)[None] * 2 * np.pi / d
    theta = {name: initial.copy() for name in methods}
    checkpoints = np.arange(0, args.steps + 1, 20)
    curves = {name: np.zeros((len(checkpoints), args.seeds, 3)) for name in methods}
    histories = {name: np.zeros((len(checkpoints), args.seeds, d)) for name in methods}
    _, initial_gradient = exact(initial, learned, queries, horizon, beta, True)
    analytic = exact(initial + .1 * rng.normal(size=initial.shape), learned, queries, horizon, beta, True)
    test_theta = initial + .1 * rng.normal(size=initial.shape)
    exact_gradient = exact(test_theta, learned, queries, horizon, beta, True)[1]
    checks = []
    for i in range(d):
        offset = np.zeros_like(initial); offset[:, i] = 1e-5
        numeric = (exact(test_theta + offset, learned, queries, horizon, beta) - exact(test_theta - offset, learned, queries, horizon, beta)) / 2e-5
        checks.append(np.max(np.abs(numeric - exact_gradient[:, i])))
    assert max(checks) < 1e-7
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'parameters': vars(args) | {'output': str(args.output), 'states': d, 'horizon': horizon, 'beta': beta, 'rate': rate, 'batch': batch, 'observations_per_atomic_context': observations, 'observation_corruption': corruption}, 'methods': methods, 'metrics': ['true_success', 'internal_success', 'atomic_inverse_nll'], 'scope': 'Exact unit-circle constrained atomic optimum (winding three), all distant goals practiced, future-rollout success computed exactly. Entire learned table constructed from finite noisy observations. No true order or outcomes used by training. At exact winding, deterministic gradient vanishes by symmetry; stochastic updates may break symmetry. Fixed 3000-step mechanism diagnostic, retain null result.', 'gradient_audit': float(max(checks)), 'initial_exact_gradient_norm': np.linalg.norm(initial_gradient, axis=1).tolist(), 'checker_accuracy': np.mean(learned == truths, axis=(1, 2)).tolist(), 'ideal_winding_one_success': exact(target, truths, queries, horizon, beta).tolist(), 'atomic_global_optimum': float(np.logaddexp(0, -beta))}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(__file__, args.output / 'run_source.py')
    for step in trange(args.steps + 1, desc='Cyclic latent practice'):
        if step % 20 == 0:
            for name, angles in theta.items():
                curves[name][step // 20] = np.stack([exact(angles, truths, queries, horizon, beta), exact(angles, learned, queries, horizon, beta), atomic(angles, learned, beta)[0]], axis=1)
                histories[name][step // 20] = angles
        if step == args.steps: break
        for name, angles in theta.items():
            if name == 'atomic_reconstruction': gradient = atomic(angles, learned, beta)[1]
            elif name == 'exact_gradient': gradient = exact(angles, learned, queries, horizon, beta, True)[1]
            else: gradient = sampled(angles, learned, queries, horizon, beta, rng, batch, name)
            angles += rate * gradient
            assert np.all(np.isfinite(angles))
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'initial': curves['practice'][0].mean(axis=0).tolist(), 'final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'final_by_seed': {name: value[-1].tolist() for name, value in curves.items()}}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(args.output / 'trajectories.npz', steps=checkpoints, initial=initial, target=target, truth=truths, learned=learned, counts=counts, ordering=orders, queries=queries, **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in histories.items()})
    fig, axes = plt.subplots(2, 1)
    for name, value in curves.items():
        line, = axes[0].plot(checkpoints, value[:, :, 0].mean(axis=1), label=name.replace('_', ' '))
        axes[1].plot(checkpoints, value[:, :, 2].mean(axis=1), color=line.get_color())
    axes[0].axhline(np.mean(protocol['ideal_winding_one_success']), color='black', linestyle=':', label='Known winding-one representation')
    axes[0].set(ylabel='Fresh-rollout success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates', ylabel='Atomic reconstruction NLL')
    fig.tight_layout(); fig.savefig(args.output / 'practice-loop-ring.png'); plt.close(fig)
    print(json.dumps({'initial': result['initial'], 'final': result['final'], 'max_gradient_error': max(checks)}, indent=2))


if __name__ == '__main__': main()
