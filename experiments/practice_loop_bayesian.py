"""Offline cyclic practice with an exact, uncertain frozen Bayesian recognizer."""

from datetime import datetime
from itertools import permutations
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange

import practice_loop_ring as ring


def cycle_prior(states):
    ordering = np.array([(0,) + order for order in permutations(range(1, states))])
    worlds = np.zeros((len(ordering), states, 2), int)
    worlds[np.arange(len(worlds))[:, None], ordering, 0] = np.roll(ordering, -1, axis=1)
    worlds[np.arange(len(worlds))[:, None], ordering, 1] = np.roll(ordering, 1, axis=1)
    return ordering, worlds


def infer(counts, worlds, corruption):
    states = worlds.shape[1]
    matches = np.einsum('siag,miag->sm', counts, np.eye(states)[worlds])
    logits = matches * np.log((1 - corruption + corruption / states) / (corruption / states))
    log_probability = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    posterior = np.exp(log_probability)
    transitions = np.einsum('sm,miag->siag', posterior, np.eye(states)[worlds])
    return posterior, transitions


def atomic(theta, transitions, beta):
    states = theta.shape[1]
    delta = theta[:, None, :] - theta[:, :, None]
    logits = beta * np.sin(delta)
    probability = ring.sigmoid(logits)
    loss = (transitions[:, :, 0] * np.logaddexp(0, -logits) + transitions[:, :, 1] * np.logaddexp(0, logits)).sum(axis=(1, 2)) / (2 * states)
    local = (transitions[:, :, 0] * (1 - probability) - transitions[:, :, 1] * probability) * beta * np.cos(delta) / (2 * states)
    gradient = local.sum(axis=1) - local.sum(axis=2)
    return loss, gradient


def sampled(theta, worlds, cumulative, queries, rng, beta, horizon, batch):
    seeds = len(theta); sid = np.arange(seeds)[:, None]
    chosen = rng.integers(len(queries), size=(seeds, batch // 2))
    pairs = np.repeat(queries[chosen], 2, axis=1)
    state, goal = pairs[:, :, 0].copy(), pairs[:, :, 1]
    uniforms = rng.random((seeds, batch))
    model = np.array([np.searchsorted(cdf, uniform) for cdf, uniform in zip(cumulative, uniforms)])
    alive = state != goal
    records = []
    for _ in range(horizon):
        delta = theta[sid, goal] - theta[sid, state]
        probability = ring.sigmoid(beta * np.sin(delta))
        action = rng.random(state.shape) < probability
        records.append((state.copy(), (action - probability) * beta * np.cos(delta) * alive))
        state = np.where(alive, worlds[model, state, 1 - action.astype(int)], state)
        alive &= state != goal
    reward = (state == goal).reshape(seeds, batch // 2, 2).astype(float)
    coefficient = (reward - reward[:, :, ::-1]).reshape(seeds, batch) / batch
    gradient = np.zeros_like(theta)
    for state, eligibility in records:
        np.add.at(gradient, (sid, goal), coefficient * eligibility)
        np.add.at(gradient, (sid, state), -coefficient * eligibility)
    return gradient


def internal_success(theta, worlds, posterior, queries, horizon, beta):
    seeds, states = theta.shape; models = len(worlds)
    phase = np.repeat(theta, models, axis=0)
    transitions = np.tile(worlds, (seeds, 1, 1))
    task = np.broadcast_to(queries, (seeds * models, len(queries), 2))
    success = ring.exact(phase, transitions, task, horizon, beta).reshape(seeds, models)
    return (success * posterior).sum(axis=1)


def metrics(theta, worlds, posterior, transitions, truth, queries, horizon, beta):
    task = np.broadcast_to(queries, (len(theta), len(queries), 2))
    return np.stack([ring.exact(theta, truth, task, horizon, beta), internal_success(theta, worlds, posterior, queries, horizon, beta), atomic(theta, transitions, beta)[0]], axis=1)


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_bayesian'; out.mkdir(parents=True, exist_ok=True)
    states, seeds, beta, horizon, batch, rate = 7, 32, 6., 3, 64, .1
    pretrain_steps, steps, cadence = 5000, 6000, 200
    ordering, worlds = cycle_prior(states)
    queries = np.array([(s, g) for s in range(states) for g in range(states) if s != g])
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'states': states, 'world_prior': 'Uniform over720oriented cycles withstate0anchored; true worlds are independent draws from this prior.', 'corruption_rates': [.35, .55], 'observations_per_context': 1, 'observations_total': 14, 'seeds_per_noise_level': seeds, 'seed_formula': '1007000+round(1000*corruption)', 'pretraining_updates': pretrain_steps, 'maximum_offline_updates': steps, 'cadence': cadence, 'beta': beta, 'horizon': horizon, 'batch': batch, 'practice_rate': rate, 'atomic_rate': 1., 'metrics': ['true_world_success', 'posterior_expected_success', 'posterior_expected_atomic_nll'], 'recognizer': 'Exact posterior using known cycle prior and uniform observation-corruption channel, frozen after14noisyatomicobservations. This is a richer and more expensive recognizer than the main modal table.', 'scope': 'Each practice trajectory samples one posterior world and keeps it fixed for every step, preserving epistemic correlations. Partners have independent world and action noise but the same query. All42distinctstate-goalpairs are supplied independently of true or inferred topology. No true world index, transition, or success reaches the optimizer. Atomic pretraining and its continuation minimize posterior-expected one-step inverse CE. This auxiliary test addresses uncertain recognition and conditional Bayes risk, not the main exact atomic-optimum theorem. Complete worlds drawn from the prior make the posterior-risk identity correctly specified. No universal frequentist true-world improvement is claimed. Stop on three stationary checks of both internal failure risk and atomic loss using two ten-evaluation log-log windows after4000updates.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py'); shutil.copy2(Path(ring.__file__), out / 'shared_source.py')
    summaries = {}
    fig, axes = plt.subplots(2, 1)
    for corruption in tqdm(protocol['corruption_rates'], desc='Frozen uncertain recognition'):
        case = f'noise{int(round(100 * corruption))}'
        rng = np.random.default_rng(1007000 + round(1000 * corruption))
        truth_index = rng.integers(len(worlds), size=seeds); truth = worlds[truth_index]
        observation = truth.copy(); noisy = rng.random(truth.shape) < corruption
        observation[noisy] = rng.integers(states, size=noisy.sum())
        counts = np.eye(states, dtype=int)[observation]
        posterior, transitions = infer(counts, worlds, corruption)
        cumulative = posterior.cumsum(axis=1); cumulative[:, -1] = 1.
        theta = rng.uniform(-np.pi, np.pi, (seeds, states))
        probe = theta[:3]; delta = np.eye(states) * 1e-5
        numeric = np.column_stack([-(atomic(probe + offset, transitions[:3], beta)[0] - atomic(probe - offset, transitions[:3], beta)[0]) / 2e-5 for offset in delta])
        gradient_error = np.max(np.abs(numeric - atomic(probe, transitions[:3], beta)[1])); assert gradient_error < 1e-8
        degenerate = np.eye(states)[truth]
        reference_loss, reference_gradient = ring.atomic(theta, truth, beta)
        new_loss, new_gradient = atomic(theta, degenerate, beta)
        special_case_error = max(np.max(np.abs(reference_loss - new_loss)), np.max(np.abs(reference_gradient - new_gradient))); assert special_case_error < 1e-12
        pretraining, pretrain_angles = [], []
        for step in trange(pretrain_steps + 1, desc=f'{case}: atomic learning', leave=False):
            if step % cadence == 0:
                pretraining.append(metrics(theta, worlds, posterior, transitions, truth, queries, horizon, beta)); pretrain_angles.append(theta.copy())
            if step < pretrain_steps: theta += atomic(theta, transitions, beta)[1]
        phases = {name: theta.copy() for name in ['practice', 'atomic']}
        curves = {name: [] for name in phases}; history = {name: [] for name in phases}
        active = {name: True for name in phases}; stationary = {name: 0 for name in phases}; stopping = {}
        checkpoints = np.arange(0, steps + 1, cadence)
        for step in trange(steps + 1, desc=f'{case}: posterior-world practice', leave=False):
            if step % cadence == 0:
                for name, phase in phases.items():
                    curves[name].append(metrics(phase, worlds, posterior, transitions, truth, queries, horizon, beta)); history[name].append(phase.copy())
                    if step >= 4000 and active[name]:
                        past = np.array(curves[name]); risk = np.stack([1 - past[:, :, 1].mean(axis=1), past[:, :, 2].mean(axis=1)], axis=1)
                        slopes = np.array([[np.polyfit(np.log(checkpoints[:len(past)][part]), np.log(risk[part, column]), 1)[0] for column in range(2)] for part in [slice(-20, -10), slice(-10, None)]])
                        stationary[name] = stationary[name] + 1 if np.all(np.abs(slopes) < .01) else 0
                        if stationary[name] == 3:
                            active[name] = False; stopping[name] = {'step': step, 'reason': 'internal_objectives_stationary', 'slopes': slopes.tolist()}
            if step == steps: break
            for name, phase in phases.items():
                if not active[name]: continue
                phase += atomic(phase, transitions, beta)[1] if name == 'atomic' else rate * sampled(phase, worlds, cumulative, queries, rng, beta, horizon, batch)
                assert np.isfinite(phase).all()
        for name in phases:
            if active[name]: stopping[name] = {'step': steps, 'reason': 'budget'}
        curves, pretraining = {name: np.array(value) for name, value in curves.items()}, np.array(pretraining)
        gain = curves['practice'][-1, :, 0] - curves['atomic'][-1, :, 0]
        bootstrap_rng = np.random.default_rng(1009000 + round(1000 * corruption))
        bootstrap = gain[bootstrap_rng.integers(seeds, size=(10000, seeds))].mean(axis=1)
        summaries[case] = {'corruption': corruption, 'recognizer_modal_transition_accuracy': float(np.mean(transitions.argmax(axis=3) == truth)), 'recognizer_correct_MAP_world_fraction': float(np.mean(posterior.argmax(axis=1) == truth_index)), 'posterior_mass_on_true_world_mean': float(posterior[np.arange(seeds), truth_index].mean()), 'posterior_entropy_mean': float(-(posterior * np.log(posterior)).sum(axis=1).mean()), 'gradient_check_error': float(gradient_error), 'degenerate_posterior_equivalence_error': float(special_case_error), 'pretrained': pretraining[-1].mean(axis=0).tolist(), 'final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'paired_true_success_gain': {'mean': float(gain.mean()), 'bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist(), 'by_seed': gain.tolist()}, 'passive_best_common_true_success': float(np.concatenate([pretraining[:, :, 0], curves['atomic'][1:, :, 0]]).mean(axis=1).max()), 'worlds_worse_under_practice': int(np.sum(gain < 0)), 'stopping': stopping}
        np.savez_compressed(out / f'{case}.npz', worlds=worlds, ordering=ordering, truth_index=truth_index, observation=observation, counts=counts, posterior=posterior, transition_posterior=transitions, queries=queries, steps=checkpoints, pretrain_steps=np.arange(0, pretrain_steps + 1, cadence), pretraining=pretraining, pretrain_angles=pretrain_angles, **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in history.items()})
        line, = axes[0].plot(checkpoints, curves['practice'][:, :, 0].mean(axis=1), label=f'Noise {corruption:.0%}: practice')
        axes[0].plot(checkpoints, curves['atomic'][:, :, 0].mean(axis=1), color=line.get_color(), linestyle=':')
        axes[1].plot(checkpoints, curves['practice'][:, :, 1].mean(axis=1), color=line.get_color())
        axes[1].plot(checkpoints, curves['atomic'][:, :, 1].mean(axis=1), color=line.get_color(), linestyle=':')
        (out / 'summary.json').write_text(json.dumps(protocol | {'results': summaries}, indent=2) + '\n')
    axes[0].set(ylabel='True-world future success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates after finite atomic exposure', ylabel='Posterior expected success', title='Dotted: continue original atomic objective')
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-bayesian.png'); plt.close(fig)
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'results': summaries}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
