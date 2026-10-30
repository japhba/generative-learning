"""One-shot spectral mixture fits, selected solely by original-data validation."""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from model_search_majority import D, FLOOR, INVERSE, VISIBLE, Z
from model_search_hierarchical_majority import counts, law


def candidate_models(data):
    frequency = data / data.sum()
    _, uniform, uniform_x = law(.5, np.full((2, D), .5))
    posterior_weights = frequency[INVERSE] * uniform / uniform_x[INVERSE]
    mean_uniform = posterior_weights @ Z
    p0 = np.clip(mean_uniform, FLOOR, 1 - FLOOR)
    _, initial, initial_x = law(.5, np.stack([p0, p0]))
    initial_weights = frequency[INVERSE] * initial / initial_x[INVERSE]
    moment_sources = {'uniform_inverse': (posterior_weights @ Z, Z.T @ (posterior_weights[:, None] * Z)), 'initial_inverse': (initial_weights @ Z, Z.T @ (initial_weights[:, None] * Z)), 'observed_bits': (frequency @ VISIBLE, VISIBLE.T @ (frequency[:, None] * VISIBLE))}
    candidates = {}
    for source, (mean, second) in moment_sources.items():
        covariance = second - np.outer(mean, mean)
        np.fill_diagonal(covariance, 0.)
        values, vectors = np.linalg.eigh(covariance)
        eigenvalue, direction = values[-1], vectors[:, -1]
        for pi in [.2, .35, .5, .65, .8]:
            contrast = np.sqrt(max(eigenvalue, 0.) * D / ((D - 1) * pi * (1 - pi))) * direction
            for amplitude in [.25, .5, 1., 1.5, 2., 3.]:
                p = np.clip(np.stack([mean - pi * amplitude * contrast, mean + (1 - pi) * amplitude * contrast]), FLOOR, 1 - FLOOR)
                name = f'{source}_pi{pi}_amplitude{amplitude}'
                candidates[name] = (pi, p)
                means, mass, _, _ = counts(pi, p, frequency, data[INVERSE] > 0)
                post_pi = float(np.clip(mass[1], FLOOR, 1 - FLOOR))
                post_p = np.clip(means / mass[:, None], FLOOR, 1 - FLOOR)
                candidates[name + '_one_posterior'] = (post_pi, post_p)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for path in tqdm(sorted(args.source.glob('world-*-initial.json')), desc='Spectral mixture shortcuts'):
        initial = json.loads(path.read_text())
        data = np.array(initial['data'])
        truth = np.array(initial['truth'])
        rng = np.random.default_rng(initial['seed'] + 70000)
        observations = np.repeat(np.arange(len(data)), data)
        rng.shuffle(observations)
        folds = np.array_split(observations, 4)
        candidates = candidate_models(data)
        scores = {name: 0. for name in candidates}
        for validation in folds:
            validation_counts = np.bincount(validation, minlength=len(data))
            for name, (pi, p) in candidate_models(data - validation_counts).items():
                scores[name] -= validation_counts @ np.log(law(pi, p)[2])
        selected = min(scores, key=scores.__getitem__)
        risks = {name: float(truth @ np.log(truth / law(pi, p)[2])) for name, (pi, p) in candidates.items()}
        practice_path = path.with_name(path.name.replace('-initial.json', '-practice.json'))
        practice = json.loads(practice_path.read_text())
        pi, p = candidates[selected]
        records.append(dict(seed=initial['seed'], selected=selected, cv_kl=risks[selected], oracle_menu_kl=min(risks.values()), practice_kl=practice['history'][-1]['kl'], pi=pi, p=p.tolist(), all_kl=risks, validation_nll=scores))
        (args.out / 'summary.json').write_text(json.dumps(dict(source=str(args.source), menu_size=len(candidates), worlds=records), indent=2))
    print(json.dumps({name: float(np.mean([r[name] for r in records])) for name in ['cv_kl', 'oracle_menu_kl', 'practice_kl']}, indent=2))
    print([r['selected'] for r in records])
    print(args.out / 'summary.json')


if __name__ == '__main__':
    main()
