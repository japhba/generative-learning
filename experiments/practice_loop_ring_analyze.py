"""Effect intervals, passive checkpoint envelope, and Fourier interventions for cyclic practice."""

import json
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

import practice_loop_ring as ring


def interval(values, rng):
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'paired_bootstrap_95': np.quantile(samples, [.025, .975]).tolist(), 'by_seed': values.tolist()}


def geometry(angles, ordering):
    ordered = np.take_along_axis(angles, ordering, axis=-1)
    unit = np.stack([np.cos(ordered), np.sin(ordered)], axis=-2)
    d = angles.shape[-1]
    phi = 2 * np.pi * np.arange(d) / d
    projectors = {k: (np.outer(np.cos(k * phi), np.cos(k * phi)) + np.outer(np.sin(k * phi), np.sin(k * phi))) * 2 / d for k in [1, 3]}
    coefficients = np.linalg.svd(unit, full_matrices=False)
    basis = coefficients[2]
    overlap = {k: np.einsum('...id,de,...ie->...', basis, p, basis) / 2 for k, p in projectors.items()}
    return unit, overlap, coefficients[1], projectors


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/analysis'
    out.mkdir(parents=True, exist_ok=True)
    base = out.parent
    main = np.load(base / 'confirmation/trajectories.npz')
    control = np.load(base / 'controls/trajectories.npz')
    ordering = main['ordering']
    rng = np.random.default_rng(630418)
    final = main['curves_practice'][-1, :, 0]
    result = {'exact_optimum_gain': interval(final - main['curves_atomic_reconstruction'][-1, :, 0], rng), 'practice_vs_shuffled': interval(final - main['curves_shuffled'][-1, :, 0], rng), 'natural_practice_vs_continue_atoms': interval(control['curves_natural_practice'][-1, :, 0] - control['curves_natural_atomic'][-1, :, 0], rng), 'confirmation_above_point9': int(np.sum(final > .9)), 'confirmation_seed_count': len(final)}
    passive = np.concatenate([control['pretraining'][:, :, 0], control['curves_natural_atomic'][1:, :, 0]], axis=0)
    common = passive.mean(axis=1).argmax()
    result['natural_passive_envelope'] = {'best_common_true_success': float(passive[common].mean()), 'mean_per_seed_oracle_best': float(passive.max(axis=0).mean()), 'practice_minus_common': interval(control['curves_natural_practice'][-1, :, 0] - passive[common], rng), 'practice_minus_per_seed_oracle': interval(control['curves_natural_practice'][-1, :, 0] - passive.max(axis=0), rng), 'scope': 'Oracle checkpoint diagnostics selected using evaluation truth, not admissible training or stopping.'}
    angles = main['angles_practice']
    unit, overlap, singular, projectors = geometry(angles, ordering[None])
    result['geometry'] = {'initial_mode1_overlap': float(overlap[1][0].mean()), 'final_mode1_overlap': float(overlap[1][-1].mean()), 'initial_mode3_overlap': float(overlap[3][0].mean()), 'final_mode3_overlap': float(overlap[3][-1].mean()), 'initial_singular_values': singular[0].mean(axis=0).tolist(), 'final_singular_values': singular[-1].mean(axis=0).tolist()}
    interventions = {}
    for name, matrix in [('keep_mode1', projectors[1]), ('remove_mode1', np.eye(12) - projectors[1]), ('keep_mode3', projectors[3])]:
        changed = unit[-1] @ matrix
        norms = np.linalg.norm(changed, axis=1, keepdims=True)
        assert np.all(norms > 1e-12)
        changed /= norms
        transformed_ordered = np.arctan2(changed[:, 1], changed[:, 0])
        transformed = np.empty_like(transformed_ordered)
        transformed[np.arange(len(ordering))[:, None], ordering] = transformed_ordered
        success = ring.exact(transformed, main['truth'], main['queries'], 6, 6.)
        interventions[name] = interval(success, rng)
    result['spectral_interventions'] = interventions
    result['orientation_interventions'] = {name: interval(ring.exact(phase, main['truth'], main['queries'], 6, 6.), rng) for name, phase in [('reflect_final', -angles[-1]), ('reflect_ideal_winding_one', -main['target'])]}
    result['intervention_scope'] = 'Posthoc causal evaluation only: project trained2Dembedding onto graph Fourier subspace, renormalize each state to unit length; true/learned recovered ordering used solely for interpretation. Projection changes amplitudes and individual directions, so this is not a pure hidden-unit lesion. Mode1 versusmode3 is identified by subspace, global phase is gauge.'
    reconstruction = np.load(base / 'state_reconstruction/trajectories.npz')
    diagnostics = {}
    for name in [key.removeprefix('decoder_') for key in reconstruction.files if key.startswith('decoder_')]:
        phase = reconstruction[f'angles_{name}'][-1]
        unit_embedding = np.stack([np.cos(phase), np.sin(phase)], axis=1)
        decoder, bias = reconstruction[f'decoder_{name}'], reconstruction[f'bias_{name}']
        prediction = (decoder @ unit_embedding + bias[:, :, None]).argmax(axis=1)
        separation = np.abs(np.angle(np.exp(1j * (phase[:, :, None] - phase[:, None, :]))))
        separation[:, np.arange(12), np.arange(12)] = np.inf
        diagnostics[name] = {'identity_accuracy': float(np.mean(prediction == np.arange(12)[None])), 'identity_accuracy_by_seed': np.mean(prediction == np.arange(12)[None], axis=1).tolist(), 'decoder_frobenius_norm_mean': float(np.linalg.norm(decoder, axis=(1, 2)).mean()), 'minimum_phase_separation_mean': float(separation.min(axis=(1, 2)).mean())}
    (base / 'state_reconstruction/identity_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    result['identity_reconstruction_diagnostics'] = diagnostics
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'geometry.npz', mode1_overlap=overlap[1], mode3_overlap=overlap[3], singular_values=singular, ordered_unit_embeddings=unit)
    shutil.copy2(__file__, out / 'run_source.py')
    fig, axes = plt.subplots(2, 1)
    labels = {'practice': 'Internally judged practice', 'atomic_reconstruction': 'Continue optimal atomic reconstruction', 'unfiltered': 'Unjudged activity', 'shuffled': 'Shuffled judgements'}
    for name, label in labels.items():
        curve = main[f'curves_{name}']
        line, = axes[0].plot(main['steps'], curve[:, :, 0].mean(axis=1), label=label)
        lower, upper = np.quantile(curve[:, :, 0], [.1, .9], axis=1)
        axes[0].fill_between(main['steps'], lower, upper, color=line.get_color(), alpha=.15)
        axes[1].plot(main['steps'], curve[:, :, 2].mean(axis=1), color=line.get_color())
    axes[0].set(ylabel='Fresh-route success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates', ylabel='Atomic reconstruction NLL', yscale='log')
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-learning.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 1)
    for k in [1, 3]:
        values = overlap[k]
        line, = axes[0].plot(main['steps'], values.mean(axis=1), label=f'Winding {k} subspace')
        low, high = np.quantile(values, [.1, .9], axis=1)
        axes[0].fill_between(main['steps'], low, high, color=line.get_color(), alpha=.15)
    axes[0].set(ylabel='Embedding-subspace overlap'); axes[0].legend()
    for j in range(2): axes[1].plot(main['steps'], singular[:, :, j].mean(axis=1), label=f'Singular value {j + 1}')
    axes[1].set(xlabel='Offline updates', ylabel='Singular value'); axes[1].legend()
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-modes.png'); plt.close(fig)
    fig, ax = plt.subplots()
    for index, label in [(0, 'Initial aliased embedding'), (-1, 'After practice')]:
        coordinate = unit[index, 0]
        ax.plot(coordinate[0], coordinate[1], marker='o', label=label)
        for position in range(12): ax.annotate(str(position), coordinate[:, position])
    ax.set(xlabel='Embedding coordinate 1', ylabel='Embedding coordinate 2', aspect='equal'); ax.legend()
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-example.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 1)
    for name, label in [('natural_practice', 'Judged practice'), ('natural_atomic', 'Continue original-atom reconstruction'), ('natural_exact', 'Exact model-based policy gradient')]:
        curve = control[f'curves_{name}']
        line, = axes[0].plot(control['steps'], curve[:, :, 0].mean(axis=1), label=label)
        lower, upper = np.quantile(curve[:, :, 0], [.1, .9], axis=1)
        axes[0].fill_between(control['steps'], lower, upper, color=line.get_color(), alpha=.15)
        axes[1].plot(control['steps'], curve[:, :, 2].mean(axis=1), color=line.get_color())
    axes[0].axhline(result['natural_passive_envelope']['best_common_true_success'], color='black', linestyle=':', label='Best passive checkpoint (oracle)')
    axes[0].set(ylabel='Fresh-route success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates after atomic pretraining', ylabel='Atomic reconstruction NLL', yscale='log')
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-natural.png'); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
