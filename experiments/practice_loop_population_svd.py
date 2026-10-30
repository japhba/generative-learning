"""Post-training causal SVD interventions on the tied line actor's embeddings."""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import practice_loop_candidate as candidate


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_candidate/all_goal_confirmation'
    output = root / 'causal_svd'
    output.mkdir(parents=True, exist_ok=True)
    archive = np.load(root / 'curves.npz')
    metadata = json.loads((root / 'metadata.json').read_text())
    methods = ['initial', *metadata['methods']]
    conditions = ['full', 'leading_only', 'leading_removed', 'mean_only']
    seeds = metadata['arguments']['seeds']
    results = np.zeros((len(methods), len(conditions), seeds, 5))
    spectra = np.zeros((len(methods), seeds, metadata['arguments']['hidden']))
    for m, method in enumerate(methods):
        for seed in range(seeds):
            weights = [archive[f'{method}_{seed}_{group}'].copy() for group in range(5)]
            assert np.max(np.abs(weights[0] + weights[1])) < 1e-13
            mean = weights[1].mean(axis=1, keepdims=True)
            centered = weights[1] - mean
            left, singular, right = np.linalg.svd(centered, full_matrices=False)
            leading = np.outer(left[:, 0], right[0]) * singular[0]
            spectra[m, seed] = singular
            replacements = [centered, leading, centered - leading, np.zeros_like(centered)]
            for c, replacement in enumerate(replacements):
                intervention = candidate.copy_weights(weights)
                intervention[1] = mean + replacement
                intervention[0] = -intervention[1].copy()
                for group in range(2, 5): assert np.array_equal(intervention[group], weights[group])
                results[m, c, seed] = candidate.evaluate(intervention, archive[f'truth_{seed}'], archive[f'learned_{seed}'], archive[f'train_{seed}'], archive[f'test_{seed}'], archive[f'atomic_queries_{seed}'], archive[f'atomic_actions_{seed}'], metadata['arguments']['horizon'])
            reference = archive['practice'][:, 0] if method == 'initial' else archive[method][:, -1]
            assert np.max(np.abs(results[m, 0, seed] - reference[seed])) < 1e-13
    resamples = np.random.default_rng(23997).integers(seeds, size=(20000, seeds))
    summary = {'methods': methods, 'conditions': conditions, 'metric_columns': metadata['metrics'], 'seed_ids': list(range(metadata['arguments']['seed_start'], metadata['arguments']['seed_start'] + seeds)), 'protocol': 'Post hoc causal geometry diagnostic, not a training algorithm. Center each row of G across state identities, keep or remove its leading SVD component, restore row mean, enforce U=-G. Hidden bias and output readout/bias stay exactly fixed. No outcome-selected component and no refitting. Exact future success uses the identical all-goal population.', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'shared_source_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(), 'input_sha256': hashlib.sha256((root / 'curves.npz').read_bytes()).hexdigest(), 'by_method': {}}
    for m, method in enumerate(methods):
        record = {'mean_metrics': {condition: results[m, c].mean(axis=0).tolist() for c, condition in enumerate(conditions)}, 'mean_singular_values': spectra[m].mean(axis=0).tolist(), 'success_by_seed': {condition: results[m, c, :, 0].tolist() for c, condition in enumerate(conditions)}, 'differences_from_full': {}}
        for c, condition in enumerate(conditions[1:], 1):
            difference = results[m, c, :, 0] - results[m, 0, :, 0]
            record['differences_from_full'][condition] = {'mean': float(difference.mean()), 'ci95': np.quantile(difference[resamples].mean(axis=1), [.025, .975]).tolist()}
        summary['by_method'][method] = record
    (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(output / 'interventions.npz', metrics=results, spectra=spectra)
    (output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    (output / 'shared_source.py').write_bytes(Path(candidate.__file__).read_bytes())
    values = results[:, :, :, 0].mean(axis=2)
    names = ['Initial', 'Checked practice', 'Hidden frozen', 'Atomic replay', 'Unfiltered', 'Shuffled', 'Frozen accepted', 'Hindsight', 'Loop-erased hindsight', 'Exact planning']
    fig, ax = plt.subplots(layout='constrained')
    im = ax.imshow(values, vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=np.arange(len(conditions)), xticklabels=['Full', 'Leading', 'Remainder', 'Mean only'], yticks=np.arange(len(methods)), yticklabels=names)
    for index in np.ndindex(values.shape): ax.text(index[1], index[0], f'{values[index]:.3f}', ha='center', va='center', color='black' if values[index] > .5 else 'white')
    fig.colorbar(im, ax=ax, label='Exact population success')
    figure = output / 'practice-loop-population-causal-svd.png'
    fig.savefig(figure); plt.close(fig)
    (Path(__file__).resolve().parents[1] / 'tex/archive/figures' / figure.name).write_bytes(figure.read_bytes())
    print(json.dumps({method: record['mean_metrics'] for method, record in summary['by_method'].items()}, indent=2))


if __name__ == '__main__': main()
