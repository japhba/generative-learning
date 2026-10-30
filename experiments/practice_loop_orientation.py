"""Conditional two-orientation posterior practice from the exact atomic optimum."""

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

import practice_loop_bayesian as bayes
import practice_loop_ring as ring
from practice_loop_ring_theory_audit import directional


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/orientation'
    out.mkdir(parents=True, exist_ok=True)
    seeds, n, horizon, beta, batch, rate, steps, cadence = 32, 7, 3, 2., 64, .1, 6000, 100
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'optimization_seeds': seeds, 'random_seed': 1097123, 'states': n, 'horizon': horizon, 'gain': beta, 'batch': batch, 'practice_rate': rate, 'atomic_rate': 1., 'maximum_updates': steps, 'evaluation_cadence': cadence, 'posterior': [.9, .1], 'query_law': 'All42ordered distinct pairs.', 'initialization': 'Exact unique posterior atomic global optimum, winding2, with fixedunitnorm andgain2.', 'acquisition': 'Uniform prior on two orientations of a known undirected7cycle; one forward-consistent observation under uniform-replacement corruption7/15 gives posterior(.9,.1). This is one conditional dataset and32independent optimization realizations, not32independent acquired datasets.', 'methods': ['practice', 'atomic'], 'stopping': 'After2000updates, stop an arm after3consecutive checks when both mean internal failure and atomic loss have absolute log-log slope<.01 in each of the lasttwo10-evaluation windows. No true-world or heldout metric affects stopping.', 'scope': 'Secondary conditional demonstration, no hyperparameter search. Each simulated route draws one posteriorworld and keepsitfixed; paired baselines use independent worlds. Posterior risk is normative, not monotonic risk in each orientation.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for path in [Path(__file__), Path(bayes.__file__), Path(ring.__file__), root / 'experiments/practice_loop_ring_theory_audit.py']:
        shutil.copy2(path, out / path.name)
    states = np.arange(n)
    forward = np.stack([(states + 1) % n, (states - 1) % n], axis=1)
    worlds = np.stack([forward, forward[:, ::-1]])
    queries = np.array([(s, g) for s in states for g in states if s != g])
    posterior = np.broadcast_to([.9, .1], (seeds, 2)).copy()
    cumulative = posterior.cumsum(axis=1); cumulative[:, -1] = 1
    transitions = np.einsum('sm,miag->siag', posterior, np.eye(n)[worlds])
    initial = np.broadcast_to(4 * np.pi * states / n, (seeds, n)).copy()
    basis = np.eye(n)
    def second(vector): return sum(weight * directional(initial[0], vector, world, queries, horizon, beta)[2] for weight, world in zip(posterior[0], worlds))
    diagonal = np.array([second(v) for v in basis])
    hessian = np.diag(diagonal)
    for i in range(n):
        for j in range(i + 1, n): hessian[i, j] = hessian[j, i] = (second(basis[i] + basis[j]) - diagonal[i] - diagonal[j]) / 2
    def gradient(theta): return sum(weight * ring.exact(theta[None], world[None], queries[None], horizon, beta, True)[1][0] for weight, world in zip(posterior[0], worlds))
    numerical = np.column_stack([(gradient(initial[0] + 1e-5 * v) - gradient(initial[0] - 1e-5 * v)) / 2e-5 for v in basis])
    hessian_error = float(abs(hessian - numerical).max()); assert hessian_error < 1e-8
    fourier = [float(np.real(np.exp(-2j * np.pi * k * states / n) @ hessian @ np.exp(2j * np.pi * k * states / n) / n)) for k in range(n)]
    def metrics(theta):
        task = np.broadcast_to(queries, (seeds, len(queries), 2))
        success = np.stack([ring.exact(theta, np.broadcast_to(world, (seeds, n, 2)), task, horizon, beta) for world in worlds], axis=1)
        return np.column_stack([(success * posterior).sum(axis=1), bayes.atomic(theta, transitions, beta)[0], success])
    theta = {name: initial.copy() for name in protocol['methods']}
    curves, angles = {name: [] for name in theta}, {name: [] for name in theta}
    active, stationary, stopped = {name: True for name in theta}, {name: 0 for name in theta}, {}
    checkpoints = np.arange(0, steps + 1, cadence)
    rng = np.random.default_rng(protocol['random_seed'])
    for step in trange(steps + 1, desc='Two-orientation practice'):
        if step % cadence == 0:
            for name, phase in theta.items():
                curves[name].append(metrics(phase)); angles[name].append(phase.copy())
                if step >= 2000 and active[name]:
                    data = np.array(curves[name])
                    risk = np.stack([1 - data[:, :, 0].mean(axis=1), data[:, :, 1].mean(axis=1)], axis=1)
                    slopes = np.array([[np.polyfit(np.log(checkpoints[:len(data)][part]), np.log(risk[part, column]), 1)[0] for column in range(2)] for part in [slice(-20, -10), slice(-10, None)]])
                    stationary[name] = stationary[name] + 1 if np.all(abs(slopes) < .01) else 0
                    if stationary[name] == 3:
                        active[name] = False; stopped[name] = {'step': step, 'reason': 'stationary_internal_objectives', 'slopes': slopes.tolist()}
        if step == steps: break
        for name, phase in theta.items():
            if not active[name]: continue
            change = bayes.atomic(phase, transitions, beta)[1] if name == 'atomic' else rate * bayes.sampled(phase, worlds, cumulative, queries, rng, beta, horizon, batch)
            phase += change
            assert np.isfinite(phase).all()
    for name in theta:
        if active[name]: stopped[name] = {'step': steps, 'reason': 'budget'}
    curves, angles = {k: np.array(v) for k, v in curves.items()}, {k: np.array(v) for k, v in angles.items()}
    gains = curves['practice'][-1, :, 0] - curves['atomic'][-1, :, 0]
    bootstrap_rng = np.random.default_rng(1097124)
    bootstrap = gains[bootstrap_rng.integers(seeds, size=(10000, seeds))].mean(axis=1)
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'metric_columns': ['posterior_success', 'posterior_atomic_nll', 'forward_world_success', 'reverse_world_success'], 'initial': curves['practice'][0].mean(axis=0).tolist(), 'final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'practice_success_by_seed': curves['practice'][-1, :, 0].tolist(), 'paired_posterior_gain': {'mean': float(gains.mean()), 'bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist(), 'improved_seeds': int((gains > 0).sum())}, 'hessian_fourier_eigenvalues': fourier, 'hessian_finite_difference_error': hessian_error, 'initial_goal_gradient_norm': float(np.linalg.norm(gradient(initial[0]))), 'stopping': stopped}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'trajectories.npz', worlds=worlds, posterior=posterior, transitions=transitions, queries=queries, initial=initial, steps=checkpoints, hessian=hessian, **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in angles.items()})
    (out / 'rng_final.json').write_text(json.dumps(rng.bit_generator.state, indent=2) + '\n')
    fig, axes = plt.subplots(2, 1)
    for name, data in curves.items():
        line, = axes[0].plot(checkpoints, data[:, :, 0].mean(axis=1), label=name)
        axes[1].plot(checkpoints, data[:, :, 1].mean(axis=1), color=line.get_color())
        if name == 'practice':
            axes[0].plot(checkpoints, data[:, :, 0], color=line.get_color(), alpha=.15)
            axes[1].plot(checkpoints, data[:, :, 1], color=line.get_color(), alpha=.15)
    axes[0].set(ylabel='Posterior route success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates', ylabel='Posterior atomic loss')
    fig.tight_layout()
    figure = root / 'tex/archive/figures/practice-loop-orientation.png'
    fig.savefig(figure); fig.savefig(out / figure.name); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
