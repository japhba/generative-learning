"""Natural atomic pretraining and noise controls for the cyclic latent toy."""

import json
import os
from pathlib import Path
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange

import practice_loop_ring as ring


def metrics(angles, truth, learned, queries, beta, horizon):
    return np.stack([ring.exact(angles, truth, queries, horizon, beta), ring.exact(angles, learned, queries, horizon, beta), ring.atomic(angles, learned, beta)[0]], axis=1)


def main():
    load_dotenv(Path.home() / '.env')
    fast = Path(os.environ['FAST_CACHE_DIR'])
    out = fast / 'generative_practice_loop_ring/controls'
    out.mkdir(parents=True, exist_ok=True)
    archive = np.load(fast / 'generative_practice_loop_ring/confirmation/trajectories.npz')
    initial = archive['initial'].copy()
    truth, learned, queries = archive['truth'], archive['learned'], archive['queries']
    seeds, dimension = initial.shape
    rng = np.random.default_rng(170036)
    beta, horizon, rate, batch, pretrain_steps, steps = 6., 6, .1, 64, 5000, 3000
    natural = rng.uniform(-np.pi, np.pi, size=initial.shape)
    modes = ['natural_practice', 'natural_atomic', 'natural_exact', 'alias_perturbed_exact', 'alias_matched_noise', 'alias_noise_plus_exact']
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'seeds': seeds, 'beta': beta, 'horizon': horizon, 'practice_rate': rate, 'atomic_pretraining_rate': 1., 'atomic_pretraining_steps': pretrain_steps, 'steps': steps, 'batch': batch, 'seed': 170036, 'modes': modes, 'specification': 'Secondary controls on same32finiteworlds; decisions fixed before running. Natural generator angles are independent uniform phases, followed by5000atomic-only fullbatchupdates. Alias exact-gradient control has isotropic .01-radian centered jitter. Matched-noise controls obtain update norm from their own sampled reward gradient, replace direction by centered isotropic noise of that norm, optionally add exact model-gradient. This grants norm information, so it is a conservative directional diagnostic. All evaluation integrates fresh future-rollout randomness. No teacher used in training.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py')
    shutil.copy2(Path(ring.__file__), out / 'shared_source.py')
    pretraining, pretrain_angles = [], []
    for step in trange(pretrain_steps + 1, desc='Natural atomic pretraining'):
        if step % 50 == 0:
            pretraining.append(metrics(natural, truth, learned, queries, beta, horizon)); pretrain_angles.append(natural.copy())
        if step < pretrain_steps: natural += ring.atomic(natural, learned, beta)[1]
    jitter = rng.normal(0, .01, initial.shape)
    jitter -= jitter.mean(axis=1, keepdims=True)
    angles = {name: natural.copy() if name.startswith('natural') else initial.copy() for name in modes}
    angles['alias_perturbed_exact'] += jitter
    checkpoints = np.arange(0, steps + 1, 20)
    curves = {name: np.zeros((len(checkpoints), seeds, 3)) for name in modes}
    histories = {name: np.zeros((len(checkpoints), seeds, dimension)) for name in modes}
    norms = {name: [] for name in modes if 'noise' in name}
    for step in trange(steps + 1, desc='Natural and symmetry controls'):
        if step % 20 == 0:
            for name, theta in angles.items():
                curves[name][step // 20] = metrics(theta, truth, learned, queries, beta, horizon)
                histories[name][step // 20] = theta
        if step == steps: break
        for name, theta in angles.items():
            if name == 'natural_atomic': theta += ring.atomic(theta, learned, beta)[1]; continue
            if name in ['natural_exact', 'alias_perturbed_exact']: gradient = ring.exact(theta, learned, queries, horizon, beta, True)[1]
            else:
                gradient = ring.sampled(theta, learned, queries, horizon, beta, rng, batch, 'practice')
                if 'noise' in name:
                    scale = np.linalg.norm(gradient, axis=1, keepdims=True)
                    noise = rng.normal(size=theta.shape); noise -= noise.mean(axis=1, keepdims=True)
                    gradient = noise * scale / np.linalg.norm(noise, axis=1, keepdims=True)
                    norms[name].append(scale[:, 0])
                    if name == 'alias_noise_plus_exact': gradient += ring.exact(theta, learned, queries, horizon, beta, True)[1]
            theta += rate * gradient
            assert np.isfinite(theta).all()
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'natural_start': pretraining[0].mean(axis=0).tolist(), 'natural_after_atomic_pretraining': pretraining[-1].mean(axis=0).tolist(), 'final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'final_by_seed': {name: value[-1].tolist() for name, value in curves.items()}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'trajectories.npz', steps=checkpoints, pretrain_steps=np.arange(0, pretrain_steps + 1, 50), pretraining=pretraining, pretrain_angles=pretrain_angles, **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in histories.items()}, **{f'gradient_norms_{name}': value for name, value in norms.items()})
    fig, axes = plt.subplots(2, 1)
    for name in ['natural_practice', 'natural_atomic', 'natural_exact']:
        value = curves[name]
        line, = axes[0].plot(checkpoints, value[:, :, 0].mean(axis=1), label=name.replace('_', ' '))
        axes[1].plot(checkpoints, value[:, :, 2].mean(axis=1), color=line.get_color())
    axes[0].set(ylabel='Fresh-rollout success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates after atomic pretraining', ylabel='Atomic reconstruction NLL')
    fig.tight_layout(); fig.savefig(out / 'practice-loop-ring-natural.png'); plt.close(fig)
    print(json.dumps({key: result[key] for key in ['natural_start', 'natural_after_atomic_pretraining', 'final']}, indent=2))


if __name__ == '__main__': main()
