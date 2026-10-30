"""Exact block-masked reconstruction controls for hierarchical majority data."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit
from tqdm.auto import tqdm

from model_search_hierarchical_majority import CATEGORIES, D, parameter_law, parameter_vector


def mask_groups(size):
    masks = np.array([sum(1 << bit for bit in chosen) for chosen in itertools.combinations(range(D), size)])
    keys = ((CATEGORIES[None, :] & ~masks[:, None]) + (1 << D) * np.arange(len(masks))[:, None])
    _, groups = np.unique(keys, return_inverse=True)
    return groups.reshape(keys.shape)


def objective(theta, empirical, groups, empirical_groups):
    p, score = parameter_law(theta)
    group_p = np.bincount(groups.ravel(), weights=np.broadcast_to(p, groups.shape).ravel())
    value = empirical @ np.log(p) - empirical_groups @ np.log(group_p) / len(groups)
    coefficient = empirical - p * (empirical_groups / group_p)[groups].mean(axis=0)
    return -value, -(coefficient @ score)


def fit(theta0, empirical, truth, groups):
    empirical_groups = np.bincount(groups.ravel(), weights=np.broadcast_to(empirical, groups.shape).ravel())
    arguments = (empirical, groups, empirical_groups)
    analytic = objective(theta0, *arguments)[1]
    shifts = np.eye(len(theta0)) * 1e-5
    numerical = np.array([(objective(theta0 + shift, *arguments)[0] - objective(theta0 - shift, *arguments)[0]) / 2e-5 for shift in shifts])
    error = float(np.max(np.abs(analytic - numerical)))
    assert error < 1e-7
    history = []

    def record(theta):
        p, _ = parameter_law(theta)
        loss, gradient = objective(theta, *arguments)
        history.append({'iteration': len(history), 'kl': float(truth @ np.log(truth / p)), 'masked_nll': float(loss), 'nll': float(-empirical @ np.log(p)), 'gradient_norm': float(np.linalg.norm(gradient))})

    record(theta0)
    result = minimize(objective, theta0, args=arguments, jac=True, method='L-BFGS-B', bounds=[(logit(.05), logit(.95))] * len(theta0), callback=record, options={'maxiter': 3000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8})
    assert result.success, result.message
    return {'theta': result.x.tolist(), 'history': history, 'gradient_error': error, 'optimizer_message': result.message}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--masks', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--world-limit', type=int)
    parser.add_argument('--restarts', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(args.directory.glob('world-*-initial.json'), key=lambda f: int(f.name.split('-')[1]))
    if args.world_limit is not None:
        files = files[:args.world_limit]
    summaries = []
    all_groups = {size: mask_groups(size) for size in args.masks}
    for file in tqdm(files, desc='Hierarchy masked worlds'):
        initial = json.loads(file.read_text())
        empirical = np.array(initial['data']) / sum(initial['data'])
        truth = np.array(initial['truth'])
        theta0 = parameter_vector(initial['pi'], np.array(initial['p']))
        record = {'seed': initial['seed'], 'source': str(file), 'fits': {}}
        for size, groups in all_groups.items():
            attempts = [fit(theta0, empirical, truth, groups)]
            rng = np.random.default_rng(initial['seed'] + 450000)
            base = theta0[1:].reshape(2, D).mean(axis=0)
            for restart in range(args.restarts):
                contrast = rng.normal(0, 1.5, D)
                candidate = np.concatenate(([0.], np.clip(base[None, :] + np.array([-1., 1.])[:, None] * contrast, logit(.05), logit(.95)).ravel()))
                attempts.append(fit(candidate, empirical, truth, groups))
            selected = int(np.argmin([attempt['history'][-1]['masked_nll'] for attempt in attempts]))
            record['fits'][str(size)] = attempts[selected] | {'selected_restart': selected, 'restart_final_metrics': [attempt['history'][-1] for attempt in attempts]}
        (args.output / file.name.replace('initial', 'masked')).write_text(json.dumps(record, indent=2))
        summaries.append(record)
    summary = {'source': str(args.directory), 'worlds': len(summaries), 'scores': {str(size): {key: float(np.mean([r['fits'][str(size)]['history'][-1][key] for r in summaries])) for key in ('kl', 'nll', 'masked_nll', 'iteration', 'gradient_norm')} for size in args.masks}, 'maximum_gradient_error': max(r['fits'][str(size)]['gradient_error'] for r in summaries for size in args.masks)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    for size in args.masks:
        records = [r['fits'][str(size)]['history'] for r in summaries]
        grid = np.arange(max(len(r) for r in records))
        for ax, key in zip(axes, ('kl', 'masked_nll')):
            curves = np.array([np.interp(grid, [v['iteration'] for v in r], [v[key] for v in r]) for r in records])
            ax.plot(grid, curves.mean(axis=0), label=f'{size}-bit masks')
            ax.set(xlabel='L-BFGS-B iterations', ylabel=key.replace('_', ' '))
    axes[0].legend()
    fig.savefig(args.output / 'learning.png')
    print(json.dumps(summary, indent=2))
    print(args.output / 'learning.png')


if __name__ == '__main__':
    main()
