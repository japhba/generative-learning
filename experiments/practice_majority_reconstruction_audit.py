"""One-shot exact-reconstruction controls for the majority Helmholtz model."""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from model_search_majority import D, FIBERS, FLOOR, INVERSE, VISIBLE, Z, expected_counts, law


def choices():
    scores = dict(minimum_ones=Z.sum(axis=1), maximum_ones=-Z.sum(axis=1), nearest_output=np.sum(np.abs(Z - VISIBLE[INVERSE]), axis=1), lexical=np.arange(len(Z)))
    rng = np.random.default_rng(20106400)
    for repeat in range(16):
        scores[f'random_rank_{repeat}'] = rng.random(len(Z))
    return {name: np.array([fiber[np.argmin(score[fiber])] for fiber in FIBERS]) for name, score in scores.items()}


def estimators(counts, canonical):
    frequency = counts / counts.sum()
    initial = expected_counts(np.full(D, 0.5), frequency, np.ones(len(Z), bool))[0]
    initial = np.clip(initial, FLOOR, 1 - FLOOR)
    logprior = Z @ np.log(initial) + (1 - Z) @ np.log1p(-initial)
    initial_map = np.array([fiber[np.argmax(logprior[fiber])] for fiber in FIBERS])
    decoded = {name: frequency @ Z[indices] for name, indices in (canonical | {'initial_map': initial_map}).items()}
    decoded['uniform_inverse'] = initial
    _, post_mean = law(initial)
    fixed_latent, _ = law(initial)
    posterior = fixed_latent * frequency[INVERSE] / post_mean[INVERSE]
    decoded['one_posterior'] = posterior @ Z
    probabilities = {}
    for name, estimate in decoded.items():
        for shrinkage in [0, 0.05, 0.1, 0.25, 0.5]:
            p = np.clip((1 - shrinkage) * estimate + shrinkage * 0.5, FLOOR, 1 - FLOOR)
            probabilities[f'{name}_shrink_{shrinkage}'] = law(p)[1]
    return probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    canonical = choices()
    records = []
    for path in tqdm(sorted(args.source.glob('world-*-practice.json')), desc='Exact reconstruction shortcuts'):
        data = json.loads(path.read_text())
        counts = np.array(data['data'])
        truth = law(np.array(data['teacher']))[1]
        rng = np.random.default_rng(data['seed'] + 50000)
        observations = np.repeat(np.arange(len(counts)), counts)
        rng.shuffle(observations)
        folds = np.array_split(observations, 4)
        candidates = estimators(counts, canonical)
        scores = {name: 0.0 for name in candidates}
        for validation in folds:
            validation_counts = np.bincount(validation, minlength=len(counts))
            training = counts - validation_counts
            for name, probability in estimators(training, canonical).items():
                scores[name] -= validation_counts @ np.log(probability)
        selected = min(scores, key=scores.__getitem__)
        risks = {name: float(truth @ np.log(truth / probability)) for name, probability in candidates.items()}
        record = dict(seed=data['seed'], selected=selected, cv_kl=risks[selected], oracle_menu_kl=min(risks.values()), practice_kl=data['history'][-1]['kl'], all_kl=risks, validation_nll=scores)
        records.append(record)
    summary = dict(source=str(args.source), menu_size=len(candidates), worlds=records)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({name: float(np.mean([r[name] for r in records])) for name in ['cv_kl', 'oracle_menu_kl', 'practice_kl']}, indent=2))
    print([r['selected'] for r in records])
    print(args.out / 'summary.json')


if __name__ == '__main__':
    main()
