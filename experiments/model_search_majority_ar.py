"""Autoregressive inverse control for the known two-layer majority renderer."""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from tqdm.auto import tqdm

from model_search_majority import CATEGORIES, D, FEATURES, FIBERS, FLOOR, INVERSE, POWERS, STRIDE, VISIBLE_D, Z, expected_counts, law, local_step


NF = VISIBLE_D + 1
MASK = np.column_stack((np.ones((D, NF)), np.tril(np.ones((D, D)), -1)))
ALL_FEATURES = np.column_stack((FEATURES[INVERSE], 2 * Z - 1))


def propose(weights, targets, rng):
    features = FEATURES[targets]
    z = np.zeros((len(targets), D))
    for bit in range(D):
        logits = features @ weights[bit, :NF] + (2 * z[:, :bit] - 1) @ weights[bit, NF:NF + bit]
        z[:, bit] = rng.random(len(targets)) < .02 + .96 * expit(logits)
    return (z @ POWERS).astype(int)


def train(seed, steps, mode, cover_all):
    rng = np.random.default_rng(seed)
    teacher = FLOOR + (1 - 2 * FLOOR) * expit(rng.normal(0, 1.5, D))
    true_latent, truth = law(teacher)
    data = rng.multinomial(128, truth)
    frequency = data / data.sum()
    initial_means, _, _, _ = expected_counts(np.full(D, .5), frequency, np.ones(len(Z), dtype=bool))
    p = np.clip(initial_means, FLOOR + 1e-6, 1 - FLOOR - 1e-6)
    latent, predicted = law(p)
    initial_likelihood = frequency @ np.log(predicted)
    included = np.zeros(len(Z), dtype=bool)
    included[rng.choice(len(Z), size=1024, p=latent)] = True
    included &= data[INVERSE] > 0
    if cover_all:
        for target in np.flatnonzero(data):
            included[rng.choice(FIBERS[target])] = True
    initial_cache = np.flatnonzero(included).tolist()
    weights = np.zeros((D, NF + D))
    weights[:, 0] = logit(p)
    rng = np.random.default_rng(seed + 1000000)
    history, previous_bound = [], -np.inf
    stopping = False
    for step in range(steps + 1):
        means, mass, bound, predicted = expected_counts(p, frequency, included)
        if cover_all:
            assert abs(mass - 1) < 1e-12
        assert bound >= previous_bound - 1e-10
        previous_bound = bound
        if step % 20 == 0 or step == steps:
            recognition = .02 + .96 * expit(ALL_FEATURES @ weights.T)
            log_prob = (Z * np.log(recognition) + (1 - Z) * np.log1p(-recognition)).sum(axis=1)
            latent, _ = law(p)
            internal_ce = -latent @ log_prob
            acceptance = np.bincount(INVERSE, weights=np.exp(log_prob), minlength=len(CATEGORIES))
            record = {'step': step, 'kl': float(truth @ np.log(truth / predicted)), 'nll': float(-frequency @ np.log(predicted)), 'bound': float(bound), 'certificate': float(bound - initial_likelihood), 'internal_cause_ce': float(internal_ce), 'cause_ce': float(-true_latent @ log_prob), 'valid_proposal': float(truth @ acceptance), 'coverage': float(mass), 'cached_causes': int(included.sum()), 'renderer_calls': int(1024 + 384 * step), 'reconstruction': 0.}
            history.append(record)
            if step >= 400 and len(history) >= 9:
                recent = history[-9:]
                abscissa = np.log([row['step'] for row in recent])
                slopes = [np.polyfit(abscissa, np.log([-row['bound'] for row in recent]), 1)[0], np.polyfit(abscissa, np.log([row['internal_cause_ce'] for row in recent]), 1)[0]]
                stopping = bool(max(abs(slope) for slope in slopes) < .002)
        if step == steps or stopping:
            break
        p = local_step(p, means, mass)
        if mode in ('ar_practice', 'factorized_practice'):
            latent, _ = law(p)
            dreamed = rng.choice(len(Z), size=256, p=latent)
            features = ALL_FEATURES[dreamed]
            residual = Z[dreamed] - expit(features @ weights.T)
            gradient = residual.T @ features / len(dreamed)
            gradient *= MASK
            if mode == 'factorized_practice':
                gradient[:, NF:] = 0.
            weights += .3 * gradient
            nproposals = 128
        else:
            nproposals = 384
        targets = rng.choice(np.flatnonzero(data), size=nproposals)
        proposed = propose(weights, targets, rng)
        accepted = INVERSE[proposed] == targets
        included[proposed[accepted]] = True
    return {'seed': seed, 'mode': mode, 'teacher': teacher.tolist(), 'data': data.tolist(), 'initial': initial_means.tolist(), 'final': p.tolist(), 'recognition_weights': weights.tolist(), 'initial_cache': initial_cache, 'final_cache': np.flatnonzero(included).tolist(), 'history': history, 'stopped': stopping}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20106200)
    parser.add_argument('--worlds', type=int, default=12)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--observed-stride', type=int, default=1)
    parser.add_argument('--cover-all', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR']) / 'model_search_majority_ar')
    args = parser.parse_args()
    assert args.observed_stride == STRIDE
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ('ar_practice', 'factorized_practice', 'frozen_recognition')
    protocol = {'args': vars(args) | {'output': str(args.output)}, 'active_inverse_weights': int(MASK.sum()), 'practice_budget': '256 rendered dreams + 128 checked proposals versus 384 checked frozen proposals', 'stopping': 'log-log slopes of negative internal bound and recognition CE under current generator; no teacher quantity enters decisions', 'recognition': 'AR logistic conditionals on x and preceding z bits; local residual coactivity updates'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = []
    for offset in tqdm(range(args.worlds), desc='AR inverse worlds'):
        for mode in modes:
            result = train(args.seed + offset, args.steps, mode, args.cover_all)
            (args.output / f'world-{offset}-{mode}.json').write_text(json.dumps(result, indent=2))
            results.append(result)
    summary = {mode: {key: float(np.mean([r['history'][-1][key] for r in results if r['mode'] == mode])) for key in ('kl', 'nll', 'certificate', 'cause_ce', 'valid_proposal', 'cached_causes', 'step')} for mode in modes}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    grid = np.arange(0, args.steps + 1, 20)
    for mode in modes:
        records = [r for r in results if r['mode'] == mode]
        for ax, key in zip(axes.ravel(), ('kl', 'certificate', 'valid_proposal', 'cached_causes')):
            values = np.asarray([np.interp(grid, [h['step'] for h in r['history']], [h[key] for h in r['history']]) for r in records])
            mean, error = values.mean(axis=0), values.std(axis=0) / np.sqrt(len(records))
            ax.plot(grid, mean, label=mode.replace('_', ' '))
            ax.fill_between(grid, mean - error, mean + error, alpha=.2)
            ax.set(xlabel='Offline steps', ylabel=key.replace('_', ' '))
    axes[0, 0].legend()
    fig.savefig(args.output / 'learning.png')
    print(json.dumps(summary, indent=2))
    print(args.output / 'learning.png')


if __name__ == '__main__':
    main()
