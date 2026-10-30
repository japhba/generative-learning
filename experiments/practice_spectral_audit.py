"""Audit saved sigmoid-generator weights; no fitting or training is performed."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ATOMS = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=float) / np.sqrt(3)
METHODS = ['unfiltered', 'score', 'free_energy', 'importance', 'direct_mle', 'joint_free_energy', 'joint_importance']
LABELS = ['Self', 'Score', 'FE', 'IW', 'MLE', 'LFE', 'LIW']
HIDDEN = np.array(list(itertools.product((-1.0, 1.0), repeat=2)))


def matched_atoms(columns):
    normalized = columns / np.linalg.norm(columns, axis=0)
    similarities = ATOMS.T @ normalized
    permutation = max(itertools.permutations(range(2)), key=lambda order: sum(abs(similarities[j, order[j]]) for j in range(2)))
    cosines = np.clip(np.array([abs(similarities[j, permutation[j]]) for j in range(2)]), 0, 1)
    signs = np.array([np.sign(similarities[j, permutation[j]]) for j in range(2)])
    return {'cosines': cosines.tolist(), 'mean_cosine': float(cosines.mean()), 'minimum_cosine': float(cosines.min()), 'angles_degrees': np.degrees(np.arccos(cosines)).tolist(), 'permutation': list(permutation), 'signs': signs.tolist()}


def spectral_metrics(weight):
    left, singular, _ = np.linalg.svd(weight, full_matrices=False)
    assert singular[-1] > np.finfo(float).eps * singular[0]
    cosines = np.clip(np.linalg.svd(ATOMS.T @ left, compute_uv=False), 0, 1)
    angles = np.degrees(np.arccos(cosines))
    return {'weight': weight.tolist(), 'singular_values': singular.tolist(), 'principal_angles_degrees': angles.tolist(), 'maximum_principal_angle_degrees': float(angles.max()), 'subspace_overlap': float(np.mean(cosines ** 2)), 'atom_alignment': matched_atoms(weight), 'left_singular_vector_alignment': matched_atoms(left)}


def initial_kl(states, log_true, weight):
    logits = HIDDEN @ weight.T
    log_component = ((states + 1) / 2) @ logits.T - np.logaddexp(0, logits).sum(axis=1)
    logq = np.logaddexp.reduce(log_component - np.log(4), axis=1)
    return float(np.exp(log_true) @ (log_true - logq))


def summarize(records):
    summary = {}
    for name, extract in [
        ('singular_values', lambda row: row['singular_values']),
        ('principal_angles_degrees', lambda row: row['principal_angles_degrees']),
        ('maximum_principal_angle_degrees', lambda row: row['maximum_principal_angle_degrees']),
        ('subspace_overlap', lambda row: row['subspace_overlap']),
        ('atom_mean_cosine', lambda row: row['atom_alignment']['mean_cosine']),
        ('atom_minimum_cosine', lambda row: row['atom_alignment']['minimum_cosine']),
        ('svd_atom_mean_cosine', lambda row: row['left_singular_vector_alignment']['mean_cosine']),
        ('true_forward_kl', lambda row: row['true_forward_kl']),
    ]:
        values = np.array([extract(row) for row in records])
        summary[name] = {key: np.asarray(value).tolist() for key, value in [('mean', values.mean(axis=0)), ('median', np.median(values, axis=0)), ('minimum', values.min(axis=0)), ('maximum', values.max(axis=0)), ('q10', np.quantile(values, 0.1, axis=0)), ('q90', np.quantile(values, 0.9, axis=0))]}
    return summary


def audit_cohort(path, warmup):
    metadata = json.loads((path / 'metadata.json').read_text())
    archive = np.load(path / 'trajectories.npz')
    expected_theta = np.array([0.0] * 6 + [0.8 if i // 3 == j // 3 else 0.0 for i, j in itertools.combinations(range(6), 2)])
    assert np.array_equal(archive['true_theta'], expected_theta)
    args = metadata['arguments']
    source = (path / 'run_source.py').read_text()
    assert 'init_rng = np.random.default_rng(seed + 100000)' in source
    assert 'init_rng.normal(0, args.init_scale, (6, 2))' in source
    assert hashlib.sha256(source.encode()).hexdigest() == metadata['code_sha256']
    result = {'source': str(path), 'data_sha256': hashlib.sha256((path / 'trajectories.npz').read_bytes()).hexdigest(), 'source_sha256': metadata['code_sha256'], 'seeds': list(range(args['seed_start'], args['seed_start'] + args['seeds'])), 'warmup_steps': warmup, 'practice_initial_weights_saved': False, 'records': {'random_initial': [], **{method: [] for method in METHODS}}}
    for index, seed in enumerate(result['seeds']):
        weight = np.random.default_rng(seed + 100000).normal(0, args['init_scale'], (6, 2))
        kl = initial_kl(archive['states'], archive['log_true'], weight)
        reference_kl = archive[f'warmup_{index}'][0, 1] if warmup else archive['unfiltered'][index, 0, 0]
        assert abs(kl - reference_kl) < 1e-12
        result['records']['random_initial'].append({'seed': seed, **spectral_metrics(weight), 'true_forward_kl': kl, 'initial_kl_reconstruction_error': float(abs(kl - reference_kl))})
        for method in METHODS:
            result['records'][method].append({'seed': seed, **spectral_metrics(archive[f'{method}_weight_{index}']), 'true_forward_kl': float(archive[method][index, -1, 0]), 'generator_entropy': float(archive[method][index, -1, 3]), 'bias_norm': float(np.linalg.norm(archive[f'{method}_bias_{index}'])), 'prior_logit_norm': float(np.linalg.norm(archive[f'{method}_prior_{index}']))})
    result['summary'] = {method: summarize(rows) for method, rows in result['records'].items()}
    result['initial_weight_status'] = 'Random initial weights reconstructed exactly from archived RNG specification; initial KL checked. Warm-start checkpoint weights are absent and are not reconstructed.' if warmup else 'Practice initial weights reconstructed exactly from archived RNG specification; initial KL checked.'
    return result


def plot(result, output, figure):
    fig, axes = plt.subplots(2, 2, layout='constrained')
    for ax, (cohort, title) in zip(axes[0], [('random', 'Random start: spectrum'), ('warm100', 'Warm start: spectrum')]):
        methods = ['random_initial', *METHODS] if cohort == 'random' else METHODS
        labels = ['Init', *LABELS] if cohort == 'random' else LABELS
        summary = result['cohorts'][cohort]['summary']
        for index, label in enumerate(['First', 'Second']):
            means = np.array([summary[method]['singular_values']['mean'][index] for method in methods])
            low = np.array([summary[method]['singular_values']['q10'][index] for method in methods])
            high = np.array([summary[method]['singular_values']['q90'][index] for method in methods])
            line, = ax.plot(range(len(methods)), means, marker='o', label=label)
            ax.fill_between(range(len(methods)), low, high, alpha=0.2, color=line.get_color())
        ax.axhline(result['teacher']['singular_values'][0], color='black', linestyle='--', label='Teacher')
        ax.set(title=title, ylabel='Singular value', xticks=range(len(methods)), xticklabels=labels)
        ax.tick_params(axis='x', labelrotation=60)
    axes[0, 0].legend()
    for ax, metric, title, ylabel in [
        (axes[1, 0], 'maximum_principal_angle_degrees', 'Column-space recovery', 'Largest angle (degrees)'),
        (axes[1, 1], 'atom_mean_cosine', 'Individual atom recovery', 'Matched column cosine'),
    ]:
        for cohort, label in [('random', 'Random start'), ('warm100', 'Warm start')]:
            summary = result['cohorts'][cohort]['summary']
            means = [summary[method][metric]['mean'] for method in METHODS]
            low = [summary[method][metric]['q10'] for method in METHODS]
            high = [summary[method][metric]['q90'] for method in METHODS]
            line, = ax.plot(range(len(METHODS)), means, marker='o', label=label)
            ax.fill_between(range(len(METHODS)), low, high, alpha=0.2, color=line.get_color())
        ax.set(title=title, ylabel=ylabel, xticks=range(len(METHODS)), xticklabels=LABELS)
        ax.tick_params(axis='x', labelrotation=60)
    axes[1, 0].legend()
    fig.savefig(output / 'practice-spectral-audit.png')
    figure.write_bytes((output / 'practice-spectral-audit.png').read_bytes())
    plt.close(fig)


def main():
    load_dotenv(Path.home() / '.env')
    fast = Path(os.environ['FAST_CACHE_DIR'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-dir', type=Path, default=fast / 'generative-learning/practice_density_sigmoid')
    parser.add_argument('--warm-dir', type=Path, default=fast / 'generative-learning/practice_density_sigmoid_warm100')
    parser.add_argument('--output', type=Path, default=fast / 'generative_practice_spectral_audit')
    parser.add_argument('--figure', type=Path, default=Path(__file__).resolve().parents[1] / 'tex/archive/figures/practice-spectral-audit.png')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    coupling = 0.8
    atom_coefficient = 2 * np.arctanh(np.sqrt((np.exp(4 * coupling) - 1) / (np.exp(4 * coupling) + 3)))
    teacher = spectral_metrics(ATOMS * np.sqrt(3) * atom_coefficient)
    teacher['atom_coefficient'] = float(atom_coefficient)
    cohorts = {name: audit_cohort(path, warmup) for name, path, warmup in tqdm([('random', args.random_dir, 0), ('warm100', args.warm_dir, 100)], desc='Saved-weight cohorts')}
    archive = np.load(args.random_dir / 'trajectories.npz')
    teacher_weight = np.array(teacher['weight'])
    rotation = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    rotated = spectral_metrics(teacher_weight @ rotation)
    rotated['true_forward_kl'] = initial_kl(archive['states'], archive['log_true'], teacher_weight @ rotation)
    assert np.allclose(rotated['singular_values'], teacher['singular_values'], atol=1e-12, rtol=0)
    assert rotated['maximum_principal_angle_degrees'] < 1e-5
    assert abs(initial_kl(archive['states'], archive['log_true'], teacher_weight)) < 1e-12
    sign_permuted = teacher_weight[:, [1, 0]] * np.array([-1, 1])
    assert abs(initial_kl(archive['states'], archive['log_true'], sign_permuted)) < 1e-12
    result = {'uk_time': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'teacher': teacher, 'teacher_rotated_45_degrees': rotated, 'cohorts': cohorts, 'plot_labels': dict(zip(LABELS, METHODS)), 'notes': ['No training or fitting is performed.', 'Bands are seed-wise 10th–90th percentiles, not confidence intervals.', 'Principal angles compare span(W) to the two normalized three-spin block indicators.', 'Atom alignment permits only hidden-column sign flips and permutations, not arbitrary rotations.', 'Equal teacher singular values make individual SVD directions nonidentified; independent binary latent axes do not have continuous rotational symmetry.', 'The teacher rotation is a post-hoc algebraic counterexample: identical spectrum and subspace, different generated density. Sign/permutation invariance is checked numerically.', 'Warm-start checkpoint weights were not saved. Its random_initial records refer to before real-data warmup, not the start of practice.']}
    (args.output / 'spectral_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    plot(result, args.output, args.figure)
    print(args.output / 'spectral_audit.json')
    print(args.figure)


if __name__ == '__main__':
    main()
