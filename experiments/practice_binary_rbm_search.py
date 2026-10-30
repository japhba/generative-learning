"""Exact finite binary RBM diagnostic: likelihood versus reconstruction learning."""
from pathlib import Path
import argparse
import itertools
import json
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


D, K = 6, 2
X = np.array(list(itertools.product((-1., 1.), repeat=D)))
Z = np.array(list(itertools.product((-1., 1.), repeat=K)))
NW = D * K


def unpack(t):
    return t[:NW].reshape(D, K), t[NW:NW + D], t[NW + D:]


def pack(w, a, b):
    return np.concatenate((w.ravel(), a, b))


def softmax(logp, axis):
    p = np.exp(logp - np.max(logp, axis=axis, keepdims=True))
    return p / p.sum(axis=axis, keepdims=True)


def quantities(t):
    w, a, b = unpack(t)
    posterior_fields = X @ w + b
    means = np.tanh(posterior_fields)
    q = softmax(posterior_fields @ Z.T, 1)
    logv = X @ a + np.logaddexp(posterior_fields, -posterior_fields).sum(axis=1)
    p = softmax(logv, 0)
    decoder_fields = Z @ w.T + a
    decoder_means = np.tanh(decoder_fields)
    ce = np.logaddexp(decoder_fields, -decoder_fields).sum(axis=1)[None, :] - X @ decoder_fields.T
    return p, q, means, decoder_means, ce


def likelihood(t, empirical):
    p, q, means, decoder_means, ce = quantities(t)
    delta = empirical - p
    grad = pack(X.T @ (delta[:, None] * means), delta @ X, delta @ means)
    return empirical @ np.log(p), grad


def sampled_likelihood_gradient(t, empirical, rng, dreams):
    p, q, means, decoder_means, ce = quantities(t)
    negative_frequency = rng.multinomial(dreams, p) / dreams
    delta = empirical - negative_frequency
    return pack(X.T @ (delta[:, None] * means), delta @ X, delta @ means)


def reconstruction(t, empirical):
    p, q, means, decoder_means, ce = quantities(t)
    joint = empirical[:, None] * q
    encoder_error = Z[None, :, :] - means[:, None, :]
    decoder_error = decoder_means[None, :, :] - X[:, None, :]
    gw_encoder = np.einsum('xz,xzj,xi->ij', joint * ce, encoder_error, X)
    gw_decoder = np.einsum('xz,xzi,zj->ij', joint, decoder_error, Z)
    grad = pack(gw_encoder + gw_decoder, np.einsum('xz,xzi->i', joint, decoder_error), np.einsum('xz,xzj->j', joint * ce, encoder_error))
    return np.sum(joint * ce), grad


def step(t, empirical, objective, maximize, rate=.3):
    value, g = objective(t, empirical)
    direction = (1. if maximize else -1.) * g
    target = (1. if maximize else -1.) * value
    for j in range(30):
        updated = t + rate * direction
        next_value = objective(updated, empirical)[0]
        if (1. if maximize else -1.) * next_value >= target + 1e-4 * rate * (g @ g):
            return updated
        rate *= .5
    raise RuntimeError('Line search failed')


def metrics(t, empirical, teacher, teacher_w):
    p, q, means, decoder_means, ce = quantities(t)
    w, a, b = unpack(t)
    similarities = np.abs(w.T @ teacher_w) / (np.linalg.norm(w, axis=0)[:, None] * np.linalg.norm(teacher_w, axis=0)[None, :])
    alignment = max(similarities[0, 0] + similarities[1, 1], similarities[0, 1] + similarities[1, 0]) / K
    mismatch = (D - X @ decoder_means.T) / (2 * D)
    return np.array([np.sum(teacher * np.log(teacher / p)), -empirical @ np.log(p), np.sum(empirical[:, None] * q * ce) / D, np.sum(empirical[:, None] * q * mismatch), alignment, np.linalg.norm(w)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=12)
    parser.add_argument('--exposure-steps', type=int, default=1000)
    parser.add_argument('--practice-steps', type=int, default=9000)
    parser.add_argument('--examples', type=int, default=256)
    parser.add_argument('--dreams', type=int, default=2048)
    args = parser.parse_args()
    load_dotenv(Path.home() / '.env')
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_binary_rbm_search' / f'n{args.examples}_s{args.seeds}_e{args.exposure_steps}_p{args.practice_steps}'
    out.mkdir(parents=True, exist_ok=True)
    teacher_w = np.array([[1.2, 0], [.8, 0], [1., 0], [0, .9], [0, 1.1], [0, .7]])
    truth = pack(teacher_w, np.zeros(D), np.array([.2, -.1]))
    teacher = quantities(truth)[0]
    records, weights, exposure_records = [], [], []
    for seed in tqdm(range(args.seeds), desc='Independent finite datasets'):
        rng = np.random.default_rng(seed)
        counts = np.bincount(rng.choice(len(X), size=args.examples, p=teacher), minlength=len(X))
        empirical = counts / args.examples
        initial = pack(rng.normal(0, .15, size=(D, K)), np.zeros(D), np.zeros(K))
        exposure_history = [metrics(initial, empirical, teacher, teacher_w).tolist()]
        for j in range(args.exposure_steps):
            initial = step(initial, empirical, reconstruction, False)
            if (j + 1) % 10 == 0:
                exposure_history.append(metrics(initial, empirical, teacher, teacher_w).tolist())
        exposure_records.append(exposure_history)
        arms = {'likelihood': initial.copy(), 'sampled likelihood': initial.copy(), 'reconstruction': initial.copy(), 'self': initial.copy()}
        history = {name: [metrics(t, empirical, teacher, teacher_w).tolist()] for name, t in arms.items()}
        for j in range(args.practice_steps):
            current_likelihood, exact_gradient = likelihood(arms['likelihood'], empirical)
            arms['likelihood'] += .05 * exact_gradient
            assert likelihood(arms['likelihood'], empirical)[0] >= current_likelihood - 1e-12
            arms['sampled likelihood'] += .05 * sampled_likelihood_gradient(arms['sampled likelihood'], empirical, rng, args.dreams)
            arms['reconstruction'] = step(arms['reconstruction'], empirical, reconstruction, False)
            if (j + 1) % 10 == 0:
                for name, t in arms.items():
                    history[name].append(metrics(t, empirical, teacher, teacher_w).tolist())
        records.append(history)
        weights.append({name: t.tolist() for name, t in arms.items()})
    payload = {'args': vars(args), 'metric_order': ['population_KL', 'empirical_NLL', 'reconstruction_CE_per_bit', 'reconstruction_Hamming', 'motif_alignment', 'weight_norm'], 'teacher': truth.tolist(), 'curves': records, 'exposure_curves': exposure_records, 'final_weights': weights}
    (out / 'results.json').write_text(json.dumps(payload, indent=2))
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    times = np.arange(0, args.practice_steps + 1, 10)
    for ax, index in zip(axes.ravel(), range(6)):
        for name in arms:
            y = np.asarray([r[name] for r in records])[:, :, index]
            mean = y.mean(axis=0)
            error = y.std(axis=0) / np.sqrt(args.seeds)
            ax.plot(times, mean, label=name)
            ax.fill_between(times, mean - error, mean + error, alpha=.2)
        ax.set_xlabel('Offline updates')
        ax.set_ylabel(payload['metric_order'][index].replace('_', ' '))
    axes[0, 0].legend()
    figure = Path(__file__).resolve().parents[1] / 'tex/archive/figures/practice-binary-rbm-search.png'
    fig.savefig(figure)
    summary = {name: {'start': np.asarray([r[name][0] for r in records]).mean(axis=0).tolist(), 'finish': np.asarray([r[name][-1] for r in records]).mean(axis=0).tolist()} for name in arms}
    print(json.dumps(summary, indent=2))
    print(out / 'results.json')
    print(figure)


if __name__ == '__main__':
    main()
