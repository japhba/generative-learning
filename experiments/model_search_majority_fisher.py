"""One-shot Fisher and observed-Newton shortcuts for the majority renderer."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from tqdm.auto import tqdm

from model_search_majority import CATEGORIES, D, FLOOR, INVERSE, Z, law


def features(p, second=False):
    latent, observed = law(p)
    first = np.column_stack([np.bincount(INVERSE, weights=latent * Z[:, j], minlength=len(CATEGORIES)) / observed for j in range(D)])
    if not second:
        return observed, first
    products = np.empty((len(CATEGORIES), D, D))
    for i in range(D):
        for j in range(D):
            products[:, i, j] = np.bincount(INVERSE, weights=latent * Z[:, i] * Z[:, j], minlength=len(CATEGORIES)) / observed
    return observed, first, products


UNIFORM_LAW, UNIFORM_FEATURES = features(np.full(D, .5))
RIDGES = [0, .01, .03, .1, .3, 1, 3]
MULTIPLIERS = [.25, .5, .75, 1, 1.5, 2, 3]
MENU = [(family, ridge, scale) for family in ['uniform_fisher', 'warm_fisher', 'warm_newton'] for ridge in RIDGES for scale in MULTIPLIERS]


def candidates(counts):
    freq = counts / counts.sum()
    initial = np.clip(freq @ UNIFORM_FEATURES, FLOOR, 1 - FLOOR)
    answers = []
    for family in ['uniform_fisher', 'warm_fisher', 'warm_newton']:
        p = np.full(D, .5) if family == 'uniform_fisher' else initial
        if family == 'uniform_fisher':
            px, first = UNIFORM_LAW, UNIFORM_FEATURES
        elif family == 'warm_fisher':
            px, first = features(p)
        else:
            px, first, products = features(p, second=True)
        score = first - p
        g = freq @ score
        if family == 'warm_newton':
            matrix = np.diag(p * (1 - p)) - np.einsum('x,xij->ij', freq, products - first[:, :, None] * first[:, None, :])
        else:
            matrix = score.T @ (px[:, None] * score)
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, 0)
        magnitude = values.mean()
        for ridge in RIDGES:
            inverse = np.zeros_like(values)
            active = values + ridge * magnitude > 1e-12
            inverse[active] = 1 / (values[active] + ridge * magnitude)
            direction = vectors @ (inverse * (vectors.T @ g))
            for scale in MULTIPLIERS:
                answers.append(np.clip(expit(logit(p) + scale * direction), FLOOR, 1 - FLOOR))
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--observed-stride', type=int, default=1)
    args = parser.parse_args()
    records = []
    for file in tqdm(sorted(args.directory.glob('world-*-practice.json')), desc='Fisher shortcuts'):
        source = json.loads(file.read_text())
        counts = np.array(source['data'])
        rng = np.random.default_rng(source['seed'] + 300000)
        order = rng.permutation(np.repeat(np.arange(len(counts)), counts))
        scores = np.zeros(len(MENU))
        for heldout in np.array_split(order, 4):
            held = np.bincount(heldout, minlength=len(counts))
            estimates = candidates(counts - held)
            for j, p in enumerate(estimates):
                scores[j] += held @ np.log(law(p)[1])
        estimates = candidates(counts)
        truth = law(np.array(source['teacher']))[1]
        risks = [float(truth @ np.log(truth / law(p)[1])) for p in estimates]
        best = int(np.argmax(scores))
        by_family = {}
        for family in ['uniform_fisher', 'warm_fisher', 'warm_newton']:
            ids = [j for j, item in enumerate(MENU) if item[0] == family]
            chosen = ids[np.argmax(scores[ids])]
            by_family[family] = dict(kl=risks[chosen], option=MENU[chosen], oracle_kl=min(risks[j] for j in ids))
        records.append(dict(seed=source['seed'], practice_kl=source['history'][-1]['kl'], selected_kl=risks[best], selected_option=MENU[best], oracle_kl=min(risks), families=by_family, all_risks=risks, all_cv=scores.tolist()))
    differences = np.array([r['practice_kl'] - r['selected_kl'] for r in records])
    rng = np.random.default_rng(73383)
    boots = differences[rng.integers(0, len(records), (20000, len(records)))].mean(axis=1)
    result = dict(practice_kl=float(np.mean([r['practice_kl'] for r in records])), selected_kl=float(np.mean([r['selected_kl'] for r in records])), oracle_kl=float(np.mean([r['oracle_kl'] for r in records])), difference=float(differences.mean()), confidence_interval=np.quantile(boots, [.025, .975]).tolist(), family_means={family: float(np.mean([r['families'][family]['kl'] for r in records])) for family in ['uniform_fisher', 'warm_fisher', 'warm_newton']}, records=records)
    (args.directory / 'fisher-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
