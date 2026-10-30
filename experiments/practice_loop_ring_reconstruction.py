"""Strong original-data-only state-identity reconstruction controls for the ring."""

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


def identity(theta, decoder, bias):
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    logits = decoder @ unit + bias[:, :, None]
    log_probability = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    error = np.eye(theta.shape[1])[None] - np.exp(log_probability)
    loss = -np.diagonal(log_probability, axis1=1, axis2=2).mean(axis=1)
    weight_gradient = error @ unit.transpose(0, 2, 1) / theta.shape[1]
    bias_gradient = error.mean(axis=2)
    unit_gradient = decoder.transpose(0, 2, 1) @ error / theta.shape[1]
    phase_gradient = np.sum(np.stack([-np.sin(theta), np.cos(theta)], axis=1) * unit_gradient, axis=1)
    return loss, phase_gradient, weight_gradient, bias_gradient


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring'
    out = base / 'state_reconstruction'
    out.mkdir(parents=True, exist_ok=True)
    original = np.load(base / 'confirmation/trajectories.npz')
    natural = np.load(base / 'controls/trajectories.npz')['pretrain_angles'][-1]
    initial = original['initial']
    seeds, states = initial.shape
    rng = np.random.default_rng(170045)
    decoder0 = rng.normal(0, .1, (seeds, states, 2))
    bias0 = np.zeros((seeds, states))
    steps, rate, beta, horizon = 10000, .2, 6., 6
    conditions = {f'{start}_gamma{gamma:g}': (phase.copy(), decoder0.copy(), bias0.copy(), gamma) for start, phase in [('alias', initial), ('natural', natural)] for gamma in [.1, 1., 10.]}
    saved = np.arange(0, steps + 1, 100)
    curves = {name: np.zeros((len(saved), seeds, 3)) for name in conditions}
    histories = {name: np.zeros((len(saved), seeds, states)) for name in conditions}
    configuration = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'seed': 170045, 'steps': steps, 'rate': rate, 'gammas': [.1, 1., 10.], 'scope': 'Original-data only, no generated trajectories. Auxiliary decoder predicts each observed state identity from the same2Dphase embedding. Optimize sum of atomic inverse CE and gamma*state identity CE. Each gamma/start chosen before the run, no hyperparameter choice based on success. Best true-success checkpoint envelope is oracle diagnostic, not an admissible stopping rule.'}
    (out / 'protocol.json').write_text(json.dumps(configuration, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py'); shutil.copy2(Path(ring.__file__), out / 'shared_source.py')
    probe = natural[:2] + .1
    analytic = identity(probe, decoder0[:2], bias0[:2])[1]
    numeric = np.column_stack([-(identity(probe + np.eye(states)[i] * 1e-5, decoder0[:2], bias0[:2])[0] - identity(probe - np.eye(states)[i] * 1e-5, decoder0[:2], bias0[:2])[0]) / 2e-5 for i in range(states)])
    gradient_error = np.max(np.abs(analytic - numeric)); assert gradient_error < 1e-8
    for step in trange(steps + 1, desc='Original-only identity reconstruction'):
        for name, (phase, decoder, bias, gamma) in conditions.items():
            atom_loss, atom_gradient = ring.atomic(phase, original['learned'], beta)
            id_loss, phase_gradient, decoder_gradient, bias_gradient = identity(phase, decoder, bias)
            if step % 100 == 0:
                curves[name][step // 100] = np.stack([ring.exact(phase, original['truth'], original['queries'], horizon, beta), atom_loss, id_loss], axis=1)
                histories[name][step // 100] = phase
            if step == steps: continue
            phase += rate * (atom_gradient + gamma * phase_gradient)
            decoder += rate * gamma * decoder_gradient
            bias += rate * gamma * bias_gradient
            assert np.all(np.isfinite(phase)) and np.all(np.isfinite(decoder))
    summary = configuration | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'gradient_check_max_error': float(gradient_error), 'final': {name: values[-1].mean(axis=0).tolist() for name, values in curves.items()}, 'oracle_best_common_success': {name: float(values[:, :, 0].mean(axis=1).max()) for name, values in curves.items()}, 'oracle_per_seed_best_mean': {name: float(values[:, :, 0].max(axis=0).mean()) for name, values in curves.items()}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(out / 'trajectories.npz', steps=saved, **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in histories.items()}, **{f'decoder_{name}': value[1] for name, value in conditions.items()}, **{f'bias_{name}': value[2] for name, value in conditions.items()})
    fig, ax = plt.subplots()
    for name, values in curves.items(): ax.plot(saved, values[:, :, 0].mean(axis=1), label=name.replace('_', ' '))
    ax.set(xlabel='Original-data updates', ylabel='Fresh-route success'); ax.legend()
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-reconstruction.png'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
