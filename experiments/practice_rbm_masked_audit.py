"""Independent masked-reconstruction shortcut test for the binary RBM screen."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from practice_binary_rbm_search import D, K, NW, X, likelihood, metrics, quantities, unpack


FLIPS = np.arange(len(X))[:, None] ^ (1 << np.arange(D - 1, -1, -1))[None, :]


def masked_reconstruction(theta, empirical):
    w, a, b = unpack(theta)
    fields = X @ w + b
    hidden = np.tanh(fields)
    scores = np.concatenate(((X[:, :, None] * hidden[:, None, :]).reshape(len(X), NW), X, hidden), axis=1)
    logweights = X @ a + np.logaddexp(fields, -fields).sum(axis=1)
    delta = logweights[:, None] - logweights[FLIPS]
    conditional = -np.logaddexp(0, -delta)
    residual = np.exp(-np.logaddexp(0, delta))
    grad = np.einsum('x,xi,xij->j', empirical, residual, scores[:, None, :] - scores[FLIPS]) / D
    return float(empirical @ conditional.mean(axis=1)), grad


def optimize(theta, empirical, teacher, teacher_w, steps, rate):
    trajectory = [[0, *metrics(theta, empirical, teacher, teacher_w), masked_reconstruction(theta, empirical)[0]]]
    inverse_hessian = np.eye(theta.size)
    for step in range(1, steps + 1):
        value, gradient = masked_reconstruction(theta, empirical)
        if np.linalg.norm(gradient) < 1e-8:
            break
        direction = inverse_hessian @ gradient
        alpha = rate
        for line_step in range(40):
            proposed = theta + alpha * direction
            next_value, next_gradient = masked_reconstruction(proposed, empirical)
            if next_value >= value + 1e-4 * alpha * (gradient @ direction):
                break
            alpha *= 0.5
        assert line_step < 39
        displacement = proposed - theta
        change = gradient - next_gradient
        curvature = displacement @ change
        if curvature > 1e-12:
            transform = np.eye(theta.size) - np.outer(displacement, change) / curvature
            inverse_hessian = transform @ inverse_hessian @ transform.T + np.outer(displacement, displacement) / curvature
        theta = proposed
        assert next_value >= value - 1e-12
        row = [step, *metrics(theta, empirical, teacher, teacher_w), next_value]
        assert np.isfinite(row).all()
        trajectory.append(row)
        if step >= 100:
            window = np.array(trajectory[-21:])
            losses = np.stack((window[:, 2], -window[:, -1]))
            slopes = np.array([np.polyfit(np.log(window[:, 0]), np.log(y), 1)[0] for y in losses])
            if np.max(np.abs(slopes)) < 1e-6:
                break
    return theta, np.array(trajectory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--rate', type=float, default=1.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source.read_text())
    teacher_theta = np.array(source['teacher'])
    teacher = quantities(teacher_theta)[0]
    teacher_w = unpack(teacher_theta)[0]
    summaries = []
    for seed, final_weights in enumerate(tqdm(source['final_weights'], desc='Masked reconstruction audit')):
        rng = np.random.default_rng(seed)
        n = source['args']['examples']
        empirical = np.bincount(rng.choice(len(X), n, p=teacher), minlength=len(X)) / n
        initial = np.array(final_weights['self'])
        direction = rng.normal(size=initial.size)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-5
        finite_difference = (masked_reconstruction(initial + epsilon * direction, empirical)[0] - masked_reconstruction(initial - epsilon * direction, empirical)[0]) / (2 * epsilon)
        analytic = masked_reconstruction(initial, empirical)[1] @ direction
        assert abs(finite_difference - analytic) < 1e-8
        fitted, curve = optimize(initial.copy(), empirical, teacher, teacher_w, args.steps, args.rate)
        starts = [initial.copy()] + [rng.normal(0, .15, size=initial.size) for restart in range(4)]
        fitted_candidates = [optimize(start, empirical, teacher, teacher_w, args.steps, args.rate)[0] for start in starts]
        best = max(fitted_candidates, key=lambda candidate: masked_reconstruction(candidate, empirical)[0])
        ml = np.array(final_weights['likelihood'])
        result = dict(seed=seed, initial=metrics(initial, empirical, teacher, teacher_w).tolist(), masked=metrics(fitted, empirical, teacher, teacher_w).tolist(), multistart_masked=metrics(best, empirical, teacher, teacher_w).tolist(), ml=metrics(ml, empirical, teacher, teacher_w).tolist(), gradient_error=float(abs(finite_difference - analytic)), steps=int(curve[-1, 0]))
        summaries.append(result)
        np.savez_compressed(args.out / f'world-{seed:02d}.npz', curve=curve, fitted=fitted, multistart=best, initial=initial, empirical=empirical)
        (args.out / 'summary.json').write_text(json.dumps(dict(source=str(args.source), settings=dict(steps=args.steps, rate=args.rate), worlds=summaries), indent=2))
    fig, ax = plt.subplots()
    for seed in range(len(summaries)):
        curve = np.load(args.out / f'world-{seed:02d}.npz')['curve']
        ax.plot(curve[:, 0], curve[:, 1], alpha=0.25)
    ax.axhline(np.mean([s['initial'][0] for s in summaries]), label='Mean initial KL', color='black', linestyle=':')
    ax.axhline(np.mean([s['ml'][0] for s in summaries]), label='Mean practice ML KL', linestyle='--')
    ax.set(xlabel='Original-data masked reconstruction BFGS updates', ylabel='Population KL (nats)')
    ax.legend()
    fig.tight_layout()
    figure = Path(__file__).resolve().parents[1] / 'tex/archive/figures/practice-rbm-masked-audit.png'
    fig.savefig(figure)
    print(json.dumps({name: np.mean([s[name] for s in summaries], axis=0).tolist() for name in ['initial', 'masked', 'multistart_masked', 'ml']}, indent=2))
    print(args.out / 'summary.json')
    print(figure)


if __name__ == '__main__':
    main()
