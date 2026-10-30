"""One-shot moment and ten-partition audit of the two-motif RBM teacher."""
from pathlib import Path
import itertools
import json
import os

from dotenv import load_dotenv
import numpy as np

from practice_binary_rbm_search import D, K, X, quantities, likelihood, metrics, unpack, pack


def main():
    load_dotenv(Path.home() / '.env')
    directory = Path(os.environ['FAST_CACHE_DIR']) / 'generative_binary_rbm_search/n1024_s16_e1000_p9000'
    source = json.loads((directory / 'results.json').read_text())
    truth = np.array(source['teacher'])
    teacher_w = unpack(truth)[0]
    teacher = quantities(truth)[0]
    rows = []
    for seed in range(len(source['curves'])):
        rng = np.random.default_rng(seed)
        counts = np.bincount(rng.choice(len(X), size=source['args']['examples'], p=teacher), minlength=len(X))
        empirical = counts / source['args']['examples']
        correlations = X.T @ (empirical[:, None] * X)
        means = empirical @ X
        candidates = []
        for others in itertools.combinations(range(1, D), 2):
            first = (0, *others)
            second = tuple(j for j in range(D) if j not in first)
            w, b = np.zeros((D, K)), np.zeros(K)
            valid = True
            for k, group in enumerate((first, second)):
                i, j, l = group
                if correlations[i, j] * correlations[i, l] * correlations[j, l] <= 0:
                    valid = False
                    break
                squared = correlations[i, j] * correlations[i, l] / correlations[j, l]
                m = np.array([np.sqrt(squared), correlations[i, j] / np.sqrt(squared), correlations[i, l] / np.sqrt(squared)])
                hidden_mean = np.mean(means[list(group)] / m)
                if np.max(np.abs(m)) >= 1 or abs(hidden_mean) >= 1:
                    valid = False
                    break
                w[list(group), k] = np.arctanh(m)
                b[k] = np.arctanh(hidden_mean)
            if valid:
                t = pack(w, np.zeros(D), b)
                candidates.append((likelihood(t, empirical)[0], t, first, second))
        best = max(candidates, key=lambda c: c[0])
        rows.append({'seed': seed, 'admissible_partitions': len(candidates), 'chosen_partition': best[2:], 'metrics': metrics(best[1], empirical, teacher, teacher_w).tolist(), 'weights': best[1].tolist()})
    payload = {'source': str(directory / 'results.json'), 'results': rows}
    (directory / 'moments_results.json').write_text(json.dumps(payload, indent=2))
    print('Mean metrics:', np.mean([row['metrics'] for row in rows], axis=0))
    print('Correct partitions:', sum(row['chosen_partition'] == ((0, 1, 2), (3, 4, 5)) for row in rows))
    print(directory / 'moments_results.json')


if __name__ == '__main__':
    main()
