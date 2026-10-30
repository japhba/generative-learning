"""Remove the practice drift while preserving each sampled pair's update magnitude."""

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


def scrambled(theta, worlds, cumulative, queries, rng, beta, horizon, batch):
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
    sign = 2 * rng.integers(2, size=(seeds, batch // 2, 1)) - 1
    coefficient = ((reward - reward[:, :, ::-1]) * sign).reshape(seeds, batch) / batch
    gradient = np.zeros_like(theta)
    for state, eligibility in records:
        np.add.at(gradient, (sid, goal), coefficient * eligibility)
        np.add.at(gradient, (sid, state), -coefficient * eligibility)
    return gradient


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/orientation'
    out = base / 'noise_control'; out.mkdir(parents=True, exist_ok=True)
    data = np.load(base / 'trajectories.npz')
    worlds, posterior, transitions, queries, initial = [data[key] for key in ['worlds', 'posterior', 'transitions', 'queries', 'initial']]
    steps, cadence, rate, batch = 4300, 100, .1, 64
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'source': str(base / 'protocol.json'), 'optimizer_runs': 32, 'conditional_datasets': 1, 'steps': steps, 'cadence': cadence, 'rate': rate, 'batch': batch, 'rng_seeds': {'practice': 1097223, 'sign_scrambled': 1097224}, 'method': 'Multiply every complete paired score update by an independent fair sign. Conditional on its query and trajectories, the pair retains its outer product and norm but its mean becomes zero. Raw second moments match at a fixed policy; covariances differ by the removed mean outer product away from a stationary point.', 'scope': 'Secondary drift-removal control using the original configuration and practice stopping horizon. Fresh optimization randomness, same conditional posterior. The uniform random signs do not use true worlds, gradients, Hessians or learned Fourier geometry. This is not a new finite-data confirmation.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for source in [Path(__file__), Path(bayes.__file__), Path(ring.__file__)]: shutil.copy2(source, out / source.name)
    cumulative = posterior.cumsum(axis=1); cumulative[:, -1] = 1
    theta = {name: initial.copy() for name in protocol['rng_seeds']}
    random = {name: np.random.default_rng(seed) for name, seed in protocol['rng_seeds'].items()}
    curves, angles = {name: [] for name in theta}, {name: [] for name in theta}
    for step in trange(steps + 1, desc='Reward-alignment control'):
        if step % cadence == 0:
            for name, phase in theta.items():
                values = np.column_stack([bayes.internal_success(phase, worlds, posterior, queries, 3, 2.), bayes.atomic(phase, transitions, 2.)[0]])
                curves[name].append(values); angles[name].append(phase.copy())
        if step == steps: break
        for name, phase in theta.items():
            sampler = bayes.sampled if name == 'practice' else scrambled
            phase += rate * sampler(phase, worlds, cumulative, queries, random[name], 2., 3, batch)
            assert np.isfinite(phase).all()
    curves, angles = {name: np.array(value) for name, value in curves.items()}, {name: np.array(value) for name, value in angles.items()}
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'metric_columns': ['posterior_success', 'posterior_atomic_nll'], 'initial': curves['practice'][0].mean(axis=0).tolist(), 'final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'final_success_by_seed': {name: value[-1, :, 0].tolist() for name, value in curves.items()}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'trajectories.npz', steps=np.arange(0, steps + 1, cadence), **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in angles.items()})
    fig, axes = plt.subplots(2, 1)
    for name, values in curves.items():
        line, = axes[0].plot(np.arange(0, steps + 1, cadence), values[:, :, 0].mean(axis=1), label=name.replace('_', ' '))
        sem = values[:, :, 0].std(axis=1, ddof=1) / np.sqrt(len(initial))
        axes[0].fill_between(np.arange(0, steps + 1, cadence), values[:, :, 0].mean(axis=1) - sem, values[:, :, 0].mean(axis=1) + sem, color=line.get_color(), alpha=.2)
        axes[1].plot(np.arange(0, steps + 1, cadence), values[:, :, 1].mean(axis=1), color=line.get_color())
    axes[0].set(ylabel='Posterior route success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates', ylabel='Posterior atomic loss')
    fig.tight_layout()
    figure = root / 'tex/figures/practice-loop-orientation-noise.png'
    fig.savefig(figure); fig.savefig(out / figure.name); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
