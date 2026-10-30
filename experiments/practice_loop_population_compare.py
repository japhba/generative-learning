"""Compare matched-world hidden practice and original-only scalar reconstruction."""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    load_dotenv(Path.home() / '.env')
    cache = Path(os.environ['FAST_CACHE_DIR'])
    hidden_root = cache / 'generative_practice_loop_candidate/all_goal_confirmation'
    shallow_root = cache / 'generative_practice_loop_shallow'
    output = hidden_root / 'population_comparison'
    output.mkdir(parents=True, exist_ok=True)
    hidden = np.load(hidden_root / 'curves.npz')
    shallow = np.load(shallow_root / 'curves_and_worlds.npz')
    metadata = json.loads((hidden_root / 'metadata.json').read_text())
    seeds = metadata['arguments']['seeds']
    assert seeds == 16
    assert metadata['arguments']['split'] == 'all'
    for index in range(seeds):
        seed = metadata['arguments']['seed_start'] + index
        for hidden_key, shallow_key in [('truth', 'truth'), ('learned', 'learned'), ('counts', 'counts'), ('ordering', 'ordering'), ('atomic_queries', 'queries'), ('atomic_actions', 'actions'), ('test', 'test')]:
            assert np.array_equal(hidden[f'{hidden_key}_{index}'], shallow[f'{shallow_key}_{seed}']), (seed, hidden_key)
        assert np.array_equal(hidden[f'train_{index}'], hidden[f'test_{index}'])
    bootstrap = np.random.default_rng(34052).integers(seeds, size=(20000, seeds))
    practice = hidden['practice'][:, -1, 0]
    results = {'matched_worlds_and_populations': seeds, 'arguments': metadata['arguments'], 'comparisons': {}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'hidden_archive_sha256': hashlib.sha256((hidden_root / 'curves.npz').read_bytes()).hexdigest(), 'shallow_archive_sha256': hashlib.sha256((shallow_root / 'curves_and_worlds.npz').read_bytes()).hexdigest(), 'note': 'Scalar and hidden actors use the same finite observations and exact evaluation population, but different parameterizations and initial functions. Original-only scalar baseline is a structural bypass, not a compute-matched intervention.'}
    for checkpoint in [500, 10000]:
        baseline = shallow['curves'][:, np.flatnonzero(shallow['steps'] == checkpoint)[0], 0]
        difference = practice - baseline
        results['comparisons'][f'practice_minus_scalar_{checkpoint}'] = {'practice_mean': float(practice.mean()), 'scalar_mean': float(baseline.mean()), 'difference': float(difference.mean()), 'ci95': np.quantile(difference[bootstrap].mean(axis=1), [.025, .975]).tolist(), 'by_seed': difference.tolist()}
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    fig, axes = plt.subplots(2, 1, layout='constrained')
    warmup = metadata['arguments']['pretrain_steps']
    for ax, metric, label in [(axes[0], 0, 'Exact population success'), (axes[1], 3, 'Atomic inverse NLL')]:
        shallow_metric = metric
        ax.plot(shallow['steps'], shallow['curves'][:, :, shallow_metric].mean(axis=0), color='black', label='Scalar: original atomic data only')
        reference_metric = 0 if metric == 0 else 1
        ax.plot(hidden['pretrain_steps'], hidden['pretraining'][:, :, reference_metric].mean(axis=0), label='Hidden actor: original atomic data only')
        for method, name in [('practice', 'Hidden actor: checked practice'), ('hindsight', 'Hidden actor: hindsight')]:
            ax.plot(warmup + hidden['steps'], hidden[method][:, :, metric].mean(axis=0), label=name)
        ax.axvline(warmup, color='black', linestyle=':')
        ax.set(xlabel='Updates from initialization', ylabel=label)
    axes[0].legend()
    figure = output / 'practice-loop-population-bypass.png'
    fig.savefig(figure); plt.close(fig)
    (Path(__file__).resolve().parents[1] / 'tex/archive/figures' / figure.name).write_bytes(figure.read_bytes())
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
