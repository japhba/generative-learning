"""Raw noisy-example replay and original-transition slow-feature learning controls."""

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


def raw_objective(theta, counts, beta):
    phase = theta[:, None, :] - theta[:, :, None]
    logits = beta * np.sin(phase)
    p = ring.sigmoid(logits)
    positive, negative = counts[:, :, 0], counts[:, :, 1]
    total = counts.sum(axis=(1, 2, 3))
    loss = ((positive * np.logaddexp(0, -logits) + negative * np.logaddexp(0, logits)).sum(axis=(1, 2)) / total)
    derivative = (positive - (positive + negative) * p) * beta * np.cos(phase) / total[:, None, None]
    return loss, derivative.sum(axis=1) - derivative.sum(axis=2)


def whiten(unit):
    unit = unit - unit.mean(axis=2, keepdims=True)
    gram = unit @ unit.transpose(0, 2, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    assert np.all(eigenvalues > 1e-12)
    transform = (eigenvectors * np.sqrt(6 / eigenvalues)[:, None]) @ eigenvectors.transpose(0, 2, 1)
    return transform @ unit


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring'
    out = base / 'data_bypasses'; out.mkdir(parents=True, exist_ok=True)
    data = np.load(base / 'confirmation/trajectories.npz')
    natural = np.load(base / 'controls/trajectories.npz')['pretrain_angles'][-1]
    rng = np.random.default_rng(170061)
    rate, steps = .2, 10000
    theta = {'alias_raw_replay': data['initial'].copy(), 'natural_raw_replay': natural.copy()}
    saved = np.arange(0, steps + 1, 100)
    curves = {name: np.zeros((len(saved), len(natural), 2)) for name in theta}
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'raw_steps': steps, 'raw_rate': rate, 'spectral_steps': 200, 'spectral_rate': .1, 'seed': 170061, 'scope': 'Raw replay fits every original noisy transition triple weighted by saved counts. Slow features use only learned one-step transitions, temporal attraction and exact2x2row whitening/centering. Convert2Dfeatures to unit phases for same generator; select chirality using only atomic inverse CE. This is an explicit alternative objective/circuit, not generated practice. Initial spectral jitter .01 is declared before run.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py'); shutil.copy2(Path(ring.__file__), out / 'shared_source.py')
    for step in trange(steps + 1, desc='Raw original-example reconstruction'):
        for name, phase in theta.items():
            loss, gradient = raw_objective(phase, data['counts'], 6.)
            if step % 100 == 0: curves[name][step // 100] = np.stack([ring.exact(phase, data['truth'], data['queries'], 6, 6.), loss], axis=1)
            if step < steps: phase += rate * gradient
    seeds, dimension = natural.shape
    adjacency = np.zeros((seeds, dimension, dimension))
    np.add.at(adjacency, (np.arange(seeds)[:, None, None], np.arange(dimension)[None, :, None], data['learned']), .5)
    assert np.max(np.abs(adjacency - adjacency.transpose(0, 2, 1))) == 0
    unit = np.stack([np.cos(data['initial']), np.sin(data['initial'])], axis=1)
    unit = whiten(unit + rng.normal(0, .01, unit.shape))
    spectral, spectral_units = [], []
    for step in trange(201, desc='Original-pair slow feature control'):
        phases = np.arctan2(unit[:, 1], unit[:, 0])
        atom_forward = ring.atomic(phases, data['learned'], 6.)[0]
        atom_reverse = ring.atomic(-phases, data['learned'], 6.)[0]
        phases *= np.where(atom_forward <= atom_reverse, 1, -1)[:, None]
        smoothness = np.sum(unit * (unit - unit @ adjacency), axis=(1, 2))
        spectral.append(np.stack([ring.exact(phases, data['truth'], data['queries'], 6, 6.), smoothness], axis=1))
        spectral_units.append(unit.copy())
        if step < 200: unit = whiten(unit + .1 * (unit @ adjacency))
    spectral = np.array(spectral)
    summary = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'raw_final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'raw_oracle_best_common_success': {name: float(value[:, :, 0].mean(axis=1).max()) for name, value in curves.items()}, 'spectral_initial_mean': spectral[0].mean(axis=0).tolist(), 'spectral_final_mean': spectral[-1].mean(axis=0).tolist(), 'spectral_final_success_by_seed': spectral[-1, :, 0].tolist(), 'interpretation': 'The spectral original-pair objective is a successful bypass if its output succeeds; it uses additional global whitening and a different target, so it does not explain continued inverse-action CE descent.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(out / 'trajectories.npz', raw_steps=saved, **curves, spectral_steps=np.arange(201), spectral_curves=spectral, spectral_units=spectral_units, adjacency=adjacency)
    fig, axes = plt.subplots(2, 1)
    for name, value in curves.items(): axes[0].plot(saved, value[:, :, 0].mean(axis=1), label=name.replace('_', ' '))
    axes[0].set(xlabel='Original-example updates', ylabel='Fresh-route success'); axes[0].legend()
    axes[1].plot(np.arange(201), spectral[:, :, 0].mean(axis=1), label='Original-transition slow features')
    axes[1].set(xlabel='Spectral updates', ylabel='Fresh-route success'); axes[1].legend()
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-data-bypasses.png'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
