"""Posterior-risk, latent-geometry, and uncertainty diagnostics from saved Bayesian runs."""

import json
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

from practice_loop_ring_analyze import geometry


def interval(values, rng):
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'bootstrap95': np.quantile(means, [.025, .975]).tolist(), 'by_seed': values.tolist()}


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_bayesian'
    out = base / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1009990)
    summaries = {}
    fig, axes = plt.subplots(2, 1)
    for case, noise in [('noise35', 35), ('noise55', 55)]:
        data = np.load(base / f'{case}.npz')
        practice, atomic = data['curves_practice'], data['curves_atomic']
        _, overlap, singular, _ = geometry(data['angles_practice'], data['ordering'][data['truth_index']][None])
        summaries[case] = {'posterior_success_gain': interval(practice[-1, :, 1] - atomic[-1, :, 1], rng), 'realized_true_minus_posterior_after_practice': interval(practice[-1, :, 0] - practice[-1, :, 1], rng), 'initial_first_fourier_subspace_overlap': float(overlap[1][0].mean()), 'final_first_fourier_subspace_overlap': float(overlap[1][-1].mean()), 'overlap_change': interval(overlap[1][-1] - overlap[1][0], rng), 'initial_singular_values': singular[0].mean(axis=0).tolist(), 'final_singular_values': singular[-1].mean(axis=0).tolist(), 'scope': 'Truth-based Fourier ordering is evaluation only. Subspace overlap omits chirality. The finite-world true-minus-posterior difference is a sampling diagnostic, not a fitted calibration correction.'}
        np.savez_compressed(out / f'{case}_geometry.npz', steps=data['steps'], overlap_mode1=overlap[1], overlap_mode3=overlap[3], singular_values=singular)
        for axis, metric in zip(axes, [0, 1]):
            values = practice[:, :, metric]
            line, = axis.plot(data['steps'], values.mean(axis=1), label=f'{noise}% noise: practice')
            sem = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
            axis.fill_between(data['steps'], values.mean(axis=1) - sem, values.mean(axis=1) + sem, color=line.get_color(), alpha=.15)
            axis.plot(data['steps'], atomic[:, :, metric].mean(axis=1), color=line.get_color(), linestyle=':')
    axes[0].set(ylabel='True-world future success'); axes[0].legend()
    axes[1].set(xlabel='Offline updates after finite atomic exposure', ylabel='Posterior expected success', title='Dotted: continue original atomic objective')
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-bayesian.png'); plt.close(fig)
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n'); shutil.copy2(__file__, out / 'run_source.py')
    print(json.dumps({name: {key: value for key, value in result.items() if key not in ['posterior_success_gain', 'realized_true_minus_posterior_after_practice', 'overlap_change']} for name, result in summaries.items()}, indent=2))


if __name__ == '__main__': main()
