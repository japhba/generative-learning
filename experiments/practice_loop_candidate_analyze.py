"""Audit held-out goals, reconstruction envelopes, and latent geometry of routing runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import practice_loop_candidate as candidate


PRIMARY = ['practice', 'practice_readout', 'reconstruction', 'hindsight', 'planning_gradient']
SHORT = {'practice': 'Goal-checked practice', 'practice_readout': 'Hidden weights frozen', 'reconstruction': 'Original-data replay', 'hindsight': 'Hindsight reconstruction', 'planning_gradient': 'Exact planning gradient'}


def spectrum(weights, ordering):
    contrast = (weights[0] - weights[1]) / 2
    contrast -= contrast.mean(axis=1, keepdims=True)
    coordinate = np.empty(len(ordering))
    coordinate[ordering] = np.arange(len(ordering))
    coordinate -= coordinate.mean()
    coordinate /= np.linalg.norm(coordinate)
    _, singular, right = np.linalg.svd(contrast, full_matrices=False)
    return np.array([abs(right[0] @ coordinate), np.linalg.norm(contrast @ coordinate) ** 2 / np.linalg.norm(contrast) ** 2, np.linalg.norm(right @ coordinate), *singular])


def unpack(vector, shapes):
    ends = np.cumsum([np.prod(shape) for shape in shapes])
    return [part.reshape(shape) for part, shape in zip(np.split(vector, ends[:-1]), shapes)]


def summarize(archive, metadata):
    methods = metadata['methods']
    seeds = metadata['arguments']['seeds']
    rng = np.random.default_rng(779)
    bootstrap = rng.integers(seeds, size=(20000, seeds))
    initial = archive['practice'][:, 0, 0]
    reference = archive['pretraining'][:, :, 0]
    reference_mean = reference.mean(axis=0)
    best_index = int(np.argmax(reference_mean))
    summary = {'original_data_reference': {'initial_success': float(reference[:, 0].mean()), 'per_seed_oracle_best_mean': float(reference.max(axis=1).mean()), 'per_seed_oracle_best': reference.max(axis=1).tolist(), 'best_common_checkpoint': int(archive['pretrain_steps'][best_index]), 'best_common_checkpoint_mean': float(reference_mean[best_index]), 'last_checkpoint_mean': float(reference[:, -1].mean())}, 'methods': {}, 'paired_differences': {}, 'checker_diagnostics': metadata['checker_diagnostics'], 'bounded_search_success': float(np.mean([row['true_success'] for row in metadata['bounded_search']])), 'full_search_success': float(np.mean([row['true_success'] for row in metadata['full_search']])), 'bounded_search_mean_queries': float(np.mean([row['mean_transition_queries'] for row in metadata['bounded_search']])), 'geometry_metrics': ['leading_coordinate_cosine', 'coordinate_energy_fraction', 'rowspace_coordinate_cosine', 'singular_values_descending'], 'heldout_goals': []}
    summary['evaluation_population'] = metadata['arguments']['split']
    summary['goals_are_withheld_from_practice'] = metadata['arguments']['split'] == 'goal'
    geometries = {}
    for index in range(seeds):
        train, test = archive[f'train_{index}'], archive[f'test_{index}']
        if metadata['arguments']['split'] == 'goal': assert not set(train[:, 1]) & set(test[:, 1])
        else: assert np.array_equal(train, test)
        summary['heldout_goals'].append({'seed': metadata['arguments']['seed_start'] + index, 'goal_ids': np.unique(test[:, 1]).tolist(), 'test_pairs': len(test), 'train_pairs': len(train)})
    for method in methods:
        curves = archive[method]
        values = curves[:, -1, 0]
        shapes = [archive[f'initial_0_{group}'].shape for group in range(5)]
        geometry = np.stack([[spectrum(unpack(vector, shapes), archive[f'ordering_{index}']) for vector in archive[f'{method}_weight_history_{index}']] for index in range(seeds)])
        geometries[method] = geometry
        summary['methods'][method] = {'initial_success': float(initial.mean()), 'final_success': float(values.mean()), 'final_success_by_seed': values.tolist(), 'final_mean_metrics': curves[:, -1].mean(axis=0).tolist(), 'initial_geometry': geometry[:, 0].mean(axis=0).tolist(), 'final_geometry': geometry[:, -1].mean(axis=0).tolist(), 'final_geometry_by_seed': geometry[:, -1].tolist()}
    practice = archive['practice'][:, -1, 0]
    comparisons = [('initial', initial), ('oracle_reconstruction_checkpoint', reference.max(axis=1)), ('best_common_reconstruction_checkpoint', reference[:, best_index])]
    comparisons += [(name, archive[name][:, -1, 0]) for name in methods if name != 'practice']
    for name, comparator in comparisons:
        difference = practice - comparator
        ci = np.quantile(difference[bootstrap].mean(axis=1), [.025, .975])
        summary['paired_differences'][f'practice_minus_{name}'] = {'mean': float(difference.mean()), 'ci95': ci.tolist(), 'by_seed': difference.tolist()}
    return summary, geometries


def save(fig, output, destination, filename):
    fig.savefig(output / filename)
    (destination / filename).write_bytes((output / filename).read_bytes())
    plt.close(fig)


def plot(archive, metadata, summary, geometry, output, destination, prefix):
    population = 'Independent success on practiced query population' if metadata['arguments']['split'] == 'all' else 'Success on withheld goal identities'
    fig, ax = plt.subplots()
    for method in PRIMARY:
        values = archive[method][:, :, 0]
        line, = ax.plot(archive['steps'], values.mean(axis=0), label=SHORT[method])
        lower, upper = np.quantile(values, [.1, .9], axis=0)
        ax.fill_between(archive['steps'], lower, upper, color=line.get_color(), alpha=.15)
    ax.axhline(summary['bounded_search_success'], color='black', linestyle='--', label='BFS: 12 transition queries')
    ax.set(xlabel='Offline updates', ylabel=population)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save(fig, output, destination, f'{prefix}-learning.png')
    fig, ax = plt.subplots()
    reference = archive['pretraining'].mean(axis=0)
    ax.plot(archive['pretrain_steps'], reference[:, 0], label='Original-data reconstruction')
    warmup = metadata['arguments']['pretrain_steps']
    for method in ['practice', 'hindsight']:
        ax.plot(warmup + archive['steps'], archive[method][:, :, 0].mean(axis=0), label=SHORT[method])
    ax.axvline(warmup, color='black', linestyle=':', label='Shared practice initialization')
    ax.set(xlabel='Updates from common random initialization', ylabel=population)
    ax.legend(); fig.tight_layout()
    save(fig, output, destination, f'{prefix}-reconstruction-envelope.png')
    fig, axes = plt.subplots(2, 1, layout='constrained')
    for method in ['practice', 'practice_readout', 'reconstruction', 'hindsight']:
        axes[0].plot(archive['steps'], archive[method][:, :, 3].mean(axis=0), label=SHORT[method])
        axes[1].plot(archive['steps'], geometry[method][:, :, 1].mean(axis=0))
    axes[0].set(ylabel='Atomic inverse NLL'); axes[0].legend()
    axes[1].set(xlabel='Offline updates', ylabel='Coordinate-aligned weight energy')
    save(fig, output, destination, f'{prefix}-latent-geometry.png')
    fig, axes = plt.subplots(2, 1, layout='constrained')
    initial_singular = geometry['practice'][:, 0, 3:].mean(axis=0)
    modes = np.arange(1, len(initial_singular) + 1)
    axes[0].plot(modes, initial_singular, marker='o', color='black', linestyle=':', label='Before practice')
    for method in ['practice', 'practice_readout', 'reconstruction', 'hindsight']:
        axes[0].plot(modes, geometry[method][:, -1, 3:].mean(axis=0), marker='o', label=SHORT[method])
        axes[1].plot(archive['steps'], geometry[method][:, :, 0].mean(axis=0), label=SHORT[method])
    axes[0].set(xlabel='Mode', ylabel='Singular value', xticks=modes); axes[0].legend()
    axes[1].set(xlabel='Offline update', ylabel='Coordinate alignment')
    save(fig, output, destination, f'{prefix}-spectrum.png')
    index = 0
    ordering = archive[f'ordering_{index}']
    right_action = int(np.flatnonzero(archive[f'truth_{index}'][ordering[0]] == ordering[1])[0])
    heldout_positions = [] if metadata['arguments']['split'] == 'all' else [int(np.flatnonzero(ordering == goal)[0]) for goal in np.unique(archive[f'test_{index}'][:, 1])]
    fig, axes = plt.subplots(1, 3, layout='constrained')
    for ax, method, title in zip(axes, ['initial', 'practice', 'hindsight'], ['Before', 'Practice', 'Hindsight']):
        weights = [archive[f'{method}_{index}_{group}'] for group in range(5)]
        policy = candidate.marginal(weights, candidate.all_queries(len(ordering)))[0].reshape(len(ordering), len(ordering), 2)
        view = policy[:, :, right_action][np.ix_(ordering, ordering)]
        im = ax.imshow(view, origin='lower', vmin=0, vmax=1)
        for goal in heldout_positions: ax.add_patch(Rectangle((goal - .5, -.5), 1, len(ordering), fill=False, color='red'))
        ax.set(title=title, xlabel='Goal position')
    axes[0].set(ylabel='Current position')
    fig.colorbar(im, ax=axes, label='Probability of moving right')
    save(fig, output, destination, f'{prefix}-policy.png')
    fig, ax = plt.subplots()
    for method in ['practice', 'hindsight']:
        line, = ax.plot(archive['steps'], archive[method][:, :, 2].mean(axis=0), label=f'{SHORT[method]}: learned model')
        ax.plot(archive['steps'], archive[method][:, :, 1].mean(axis=0), color=line.get_color(), linestyle='--', label=f'{SHORT[method]}: true world')
    ax.set(xlabel='Offline updates', ylabel='Success on practice goal identities')
    ax.legend(); fig.tight_layout()
    save(fig, output, destination, f'{prefix}-model-error.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path, required=True)
    parser.add_argument('--prefix', default='practice-loop-candidate')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.figure_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.input / 'metadata.json').read_text())
    archive = np.load(args.input / 'curves.npz')
    for key in archive.files: assert np.isfinite(archive[key]).all(), key
    assert hashlib.sha256((args.input / 'run_source.py').read_bytes()).hexdigest() == metadata['code_sha256']
    summary, geometry = summarize(archive, metadata)
    summary['data_source'] = str(args.input)
    summary['data_sha256'] = hashlib.sha256((args.input / 'curves.npz').read_bytes()).hexdigest()
    summary['analysis_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['shared_code_sha256'] = hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()
    summary['note'] = 'Paired seed bootstrap intervals; reconstruction checkpoint oracle uses held-out outcomes and is only a diagnostic upper envelope. Plot bands are seed 10th–90th percentiles.'
    (args.output / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'shared_source.py').write_bytes(Path(candidate.__file__).read_bytes())
    np.savez_compressed(args.output / 'geometry.npz', **geometry)
    plot(archive, metadata, summary, geometry, args.output, args.figure_dir, args.prefix)
    print(json.dumps({'method_success': {name: row['final_success'] for name, row in summary['methods'].items()}, 'original_data_reference': summary['original_data_reference'], 'paired_differences': summary['paired_differences']}, indent=2))


if __name__ == '__main__':
    main()
