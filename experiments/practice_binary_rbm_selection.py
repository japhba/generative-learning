"""Accepted-fantasy RBM updates with a finite-step likelihood certificate.

The internal mismatch g is exact, so this tests the selection pathway rather
than learning a noisy mismatch estimator. The generated-state batch is sampled.
"""
from pathlib import Path
import json
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from practice_binary_rbm_search import D, K, X, Z, quantities, likelihood, metrics, unpack


TX = np.repeat(X, len(Z), axis=0)
TZ = np.tile(Z, (len(X), 1))
T = np.concatenate(((TX[:, :, None] * TZ[:, None, :]).reshape(len(TX), D * K), TX, TZ), axis=1)
P = T.shape[1]


def selected_direction(t, empirical, rng, dreams, hard):
    ell, g = likelihood(t, empirical)
    p, q, *_ = quantities(t)
    counts = rng.multinomial(dreams, (p[:, None] * q).ravel())
    # Sampling this joint is distributionally identical to generating x,
    # discarding its original cause, and recognizing a fresh h ~ p(h | x).
    mu = counts @ T / dreams
    centered = T - mu
    rewards = centered @ g
    if hard:
        weighted = counts * np.maximum(rewards, 0.)
        assert weighted.sum() > 0
        d = weighted @ centered / weighted.sum()
        c = g @ d
        assert c > 0
        alpha = c / (P * (d @ d))
        delta = alpha * d
        certificate = c * c / (2 * P * (d @ d))
        fraction = counts[rewards > 0].sum() / dreams
    else:
        # Positive attention weights 1 + reward/(8P), centered on the same batch.
        assert np.max(np.abs(rewards)) <= 4 * P + 1e-12
        d = ((counts * rewards) @ centered / dreams) / P
        delta = d / P
        certificate = (g @ d) / P - .5 * (d @ d) / P
        fraction = 1.
    next_ell = likelihood(t + delta, empirical)[0]
    assert next_ell >= ell + certificate - 1e-12
    return t + delta, certificate, fraction


def plot_comparison(source, records, moments):
    times = np.arange(0, source['args']['practice_steps'] + 1, 10)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    for ax, index in zip(axes.ravel(), (0, 1, 2, 4)):
        for name, curves in [('likelihood', source['curves']), ('reconstruction', source['curves']), ('positive soft gate', records), ('positive hard gate', records)]:
            values = np.asarray([row[name] for row in curves])[:, :, index]
            mean = values.mean(axis=0)
            error = values.std(axis=0) / np.sqrt(len(records))
            ax.plot(times, mean, label=name)
            ax.fill_between(times, mean - error, mean + error, alpha=.2)
        ax.axhline(np.mean([row['metrics'][index] for row in moments['results']]), color='black', linestyle='--', label='one-shot moments')
        ax.set_xlabel('Offline updates')
        ax.set_ylabel(source['metric_order'][index].replace('_', ' '))
    axes[0, 0].legend()
    figure = Path(__file__).resolve().parents[1] / 'tex/archive/figures/practice-binary-rbm-selection.png'
    fig.savefig(figure)
    return figure


def main():
    load_dotenv(Path.home() / '.env')
    directory = Path(os.environ['FAST_CACHE_DIR']) / 'generative_binary_rbm_search/n1024_s16_e1000_p9000'
    source = json.loads((directory / 'results.json').read_text())
    truth = np.array(source['teacher'])
    teacher_w = unpack(truth)[0]
    teacher = quantities(truth)[0]
    records, final_weights, min_certificates, selected_fractions = [], [], [], []
    for seed, saved in enumerate(tqdm(source['final_weights'], desc='Accepted-fantasy datasets')):
        rng = np.random.default_rng(seed)
        counts = np.bincount(rng.choice(len(X), size=source['args']['examples'], p=teacher), minlength=len(X))
        empirical = counts / source['args']['examples']
        arms = {name: np.array(saved['self']) for name in ('positive soft gate', 'positive hard gate')}
        histories = {name: [metrics(t, empirical, teacher, teacher_w).tolist()] for name, t in arms.items()}
        minimum, fraction = np.inf, []
        for j in range(source['args']['practice_steps']):
            for name in arms:
                arms[name], certificate, retained = selected_direction(arms[name], empirical, rng, source['args']['dreams'], name == 'positive hard gate')
                minimum = min(minimum, certificate)
                if name == 'positive hard gate':
                    fraction.append(retained)
            if (j + 1) % 10 == 0:
                for name, t in arms.items():
                    histories[name].append(metrics(t, empirical, teacher, teacher_w).tolist())
        records.append(histories)
        final_weights.append({name: t.tolist() for name, t in arms.items()})
        min_certificates.append(minimum)
        selected_fractions.append(float(np.mean(fraction)))
    payload = {'curves': records, 'final_weights': final_weights, 'minimum_certified_gain': min_certificates, 'mean_retained_fraction': selected_fractions, 'source': str(directory / 'results.json')}
    (directory / 'selection_results.json').write_text(json.dumps(payload, indent=2))
    moments = json.loads((directory / 'moments_results.json').read_text())
    figure = plot_comparison(source, records, moments)
    print(json.dumps({name: np.mean([r[name][-1] for r in records], axis=0).tolist() for name in arms}, indent=2))
    print('Mean retained fraction:', np.mean(selected_fractions))
    print(directory / 'selection_results.json')
    print(figure)


if __name__ == '__main__':
    main()
