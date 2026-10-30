"""Check whether cyclic practice gains survive a trainable positive logit scale."""

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

import practice_loop_ring as ring


def scale(raw): return np.logaddexp(0, raw)


def atomic(theta, raw, transitions):
    beta = scale(raw)[:, None, None]
    loss, phase_gradient = ring.atomic(theta, transitions, beta)
    delta = theta[np.arange(len(theta))[:, None, None], transitions] - theta[:, :, None]
    gain_gradient = ((np.array([1., 0.]) - ring.sigmoid(beta * np.sin(delta))) * np.sin(delta)).mean(axis=(1, 2)) * ring.sigmoid(raw)
    return loss, phase_gradient, gain_gradient


def sampled(theta, raw, transitions, queries, rng, batch=64, horizon=6):
    seeds, states = theta.shape
    sid = np.arange(seeds)[:, None]
    chosen = rng.integers(queries.shape[1], size=(seeds, batch // 2))
    pair = np.repeat(queries[sid, chosen], 2, axis=1)
    state, goal = pair[:, :, 0].copy(), pair[:, :, 1]
    alive = state != goal
    records = []
    for _ in range(horizon):
        delta = theta[sid, goal] - theta[sid, state]
        probability = ring.sigmoid(scale(raw)[:, None] * np.sin(delta))
        action = rng.random(state.shape) < probability
        residual = (action - probability) * alive
        phase_eligibility = residual * scale(raw)[:, None] * np.cos(delta)
        gain_eligibility = residual * ring.sigmoid(raw)[:, None] * np.sin(delta)
        records.append((state.copy(), phase_eligibility, gain_eligibility))
        state = np.where(alive, transitions[sid, state, 1 - action.astype(int)], state)
        alive &= state != goal
    paired = (state == goal).reshape(seeds, batch // 2, 2).astype(float)
    coefficient = (paired - paired[:, :, ::-1]).reshape(seeds, batch) / batch
    phase_gradient, gain_gradient = np.zeros_like(theta), np.zeros_like(raw)
    for state, phase, gain in records:
        np.add.at(phase_gradient, (sid, goal), coefficient * phase)
        np.add.at(phase_gradient, (sid, state), -coefficient * phase)
        gain_gradient += (coefficient * gain).sum(axis=1)
    return phase_gradient, gain_gradient


def metrics(theta, raw, truth, learned, queries):
    beta = scale(raw)
    return np.stack([ring.exact(theta, truth, queries, 6, beta[:, None, None]), ring.exact(theta, learned, queries, 6, beta[:, None, None]), atomic(theta, raw, learned)[0], beta], axis=1)


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring'
    out = base / 'trainable_gain'; out.mkdir(parents=True, exist_ok=True)
    original = np.load(base / 'confirmation/trajectories.npz')
    truth, learned, queries = original['truth'], original['learned'], original['queries']
    seeds, states = truth.shape[:2]
    pretrain_steps, steps, cadence, rate = 5000, 6000, 100, .1
    rng = np.random.default_rng(906815)
    phase = rng.uniform(-np.pi, np.pi, (seeds, states))
    raw = np.full(seeds, np.log(np.expm1(6.)))
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'seed': 906815, 'worlds': 'Same32finite worlds as primary confirmation; fresh natural generator initialization and sampling.', 'pretrain_steps': pretrain_steps, 'maximum_offline_steps': steps, 'cadence': cadence, 'atomic_rate': 1., 'practice_rate': rate, 'batch': 64, 'horizon': 6, 'gain_parameterization': 'beta=softplus(raw); train phases and raw with the same step size within each arm; no ceiling, gain regularizer, or target-risk tuning.', 'metrics': ['true_success', 'internal_success', 'atomic_nll', 'beta'], 'scope': 'Secondary control removing fixed-logit-scale assumption. The analytic fixed-beta global-minimum theorem does not directly apply. Both initial atomic pretraining and offline learning update gain. Stopping uses internal failure risk and atomic loss, two ten-evaluation log-log windows and three consecutive abs(slopes)<.01 checks after2000updates.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py'); shutil.copy2(Path(ring.__file__), out / 'shared_source.py')
    epsilon = 1e-5
    numeric = -(atomic(phase, raw + epsilon, learned)[0] - atomic(phase, raw - epsilon, learned)[0]) / (2 * epsilon)
    analytic = atomic(phase, raw, learned)[2]
    gradient_error = np.max(np.abs(numeric - analytic)); assert gradient_error < 1e-8
    fixed_gradient = ring.sampled(phase, learned, queries, 6, scale(raw)[:, None], np.random.default_rng(42), 64, 'practice')
    new_gradient = sampled(phase, raw, learned, queries, np.random.default_rng(42))[0]
    phase_rule_error = np.max(np.abs(fixed_gradient - new_gradient)); assert phase_rule_error < 1e-12
    pretraining, pretrain_phase, pretrain_raw = [], [], []
    for step in trange(pretrain_steps + 1, desc='Atomic learning with trainable gain'):
        if step % cadence == 0:
            pretraining.append(metrics(phase, raw, truth, learned, queries)); pretrain_phase.append(phase.copy()); pretrain_raw.append(raw.copy())
        if step < pretrain_steps:
            _, gp, gr = atomic(phase, raw, learned); phase += gp; raw += gr
    weights = {name: (phase.copy(), raw.copy()) for name in ['practice', 'atomic']}
    curves = {name: [] for name in weights}; phases = {name: [] for name in weights}; raws = {name: [] for name in weights}
    active = {name: True for name in weights}; stationary = {name: 0 for name in weights}; stopping = {}
    checkpoints = np.arange(0, steps + 1, cadence)
    for step in trange(steps + 1, desc='Practice or reconstruction with trainable gain'):
        if step % cadence == 0:
            for name, (theta, gain) in weights.items():
                curves[name].append(metrics(theta, gain, truth, learned, queries)); phases[name].append(theta.copy()); raws[name].append(gain.copy())
                if step >= 2000 and active[name]:
                    history = np.array(curves[name]); risk = np.stack([1 - history[:, :, 1].mean(axis=1), history[:, :, 2].mean(axis=1)], axis=1)
                    slopes = np.array([[np.polyfit(np.log(checkpoints[:len(history)][segment]), np.log(risk[segment, metric]), 1)[0] for metric in range(2)] for segment in [slice(-20, -10), slice(-10, None)]])
                    stationary[name] = stationary[name] + 1 if np.all(np.abs(slopes) < .01) else 0
                    if stationary[name] == 3:
                        active[name] = False; stopping[name] = {'step': step, 'reason': 'internal_objectives_stationary', 'log_log_slopes': slopes.tolist()}
        if step == steps: break
        for name, (theta, gain) in weights.items():
            if not active[name]: continue
            if name == 'atomic':
                _, gp, gr = atomic(theta, gain, learned); theta += gp; gain += gr
            else:
                gp, gr = sampled(theta, gain, learned, queries, rng); theta += rate * gp; gain += rate * gr
            assert np.isfinite(theta).all() and np.isfinite(gain).all()
    for name in weights:
        if active[name]: stopping[name] = {'step': steps, 'reason': 'budget'}
    curves, pretraining = {name: np.array(value) for name, value in curves.items()}, np.array(pretraining)
    difference = curves['practice'][-1, :, 0] - curves['atomic'][-1, :, 0]
    bootstrap = difference[np.random.default_rng(906816).integers(seeds, size=(10000, seeds))].mean(axis=1)
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'gradient_check_error': float(gradient_error), 'phase_rule_equivalence_error': float(phase_rule_error), 'pretrained': pretraining[-1].mean(axis=0).tolist(), 'final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'paired_practice_minus_atomic': {'mean': float(difference.mean()), 'bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist()}, 'passive_best_common_success': float(np.concatenate([pretraining[:, :, 0], curves['atomic'][1:, :, 0]]).mean(axis=1).max()), 'stopping': stopping}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'trajectories.npz', steps=checkpoints, pretrain_steps=np.arange(0, pretrain_steps + 1, cadence), pretraining=pretraining, pretrain_angles=pretrain_phase, pretrain_gain_raw=pretrain_raw, **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in phases.items()}, **{f'gain_raw_{name}': value for name, value in raws.items()})
    fig, axes = plt.subplots(2, 1)
    for name, values in curves.items():
        line, = axes[0].plot(checkpoints, values[:, :, 0].mean(axis=1), label=name)
        axes[1].plot(checkpoints, values[:, :, 3].mean(axis=1), color=line.get_color(), label=name)
    axes[0].set(ylabel='Fresh-route success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates after atomic pretraining', ylabel='Learned logit scale'); axes[1].legend()
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-gain.png'); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
