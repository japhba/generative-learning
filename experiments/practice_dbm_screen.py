"""A fully coupled deep Boltzmann toy versus original-data reconstruction fits."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import trange

from practice_deep_prior_screen import kl, softmax, spins


D, M, K = 6, 4, 2
X, H = spins(D), spins(M)
SIZES = np.array([D * M, M * K, D, M, K])
SPLITS = np.cumsum(SIZES)[:-1]
P = SIZES.sum()
FLIPS = np.arange(len(X))[:, None] ^ (1 << np.arange(D))


def unpack(theta):
    w, u, a, b, c = np.split(theta, SPLITS)
    return w.reshape(D, M), u.reshape(M, K), a, b, c


def pack(w, u, a, b, c):
    return np.concatenate([w.ravel(), u.ravel(), a, b, c])


def model(theta):
    w, u, a, b, c = unpack(theta)
    hz = c + H @ u
    zmean = np.tanh(hz)
    log_h = H @ b + np.logaddexp(hz, -hz).sum(axis=1)
    logjoint = X @ w @ H.T + log_h[None, :]
    posterior_h = softmax(logjoint)
    maxlog = logjoint.max(axis=1)
    logf = X @ a + maxlog + np.log(np.exp(logjoint - maxlog[:, None]).sum(axis=1))
    px = softmax(logf)
    hmean = posterior_h @ H
    feature_w = (X[:, :, None] * hmean[:, None, :]).reshape(len(X), D * M)
    feature_u = posterior_h @ (H[:, :, None] * zmean[:, None, :]).reshape(len(H), M * K)
    feature_z = posterior_h @ zmean
    features = np.concatenate([feature_w, feature_u, X, hmean, feature_z], axis=1)
    return px, logf, features, posterior_h


def likelihood(theta, pdata):
    px, logf, features, posterior = model(theta)
    return float(pdata @ np.log(px)), (pdata - px) @ features


def pseudolikelihood(theta, pdata):
    px, logf, features, posterior = model(theta)
    contrast = logf[:, None] - logf[FLIPS]
    conditional = 1 / (1 + np.exp(-contrast))
    objective = -np.sum(pdata[:, None] * np.logaddexp(0, -contrast)) / D
    values = pdata[:, None] * (1 - conditional) / D
    coefficients = values.sum(axis=1)
    np.add.at(coefficients, FLIPS.ravel(), -values.ravel())
    return float(objective), coefficients @ features


def reconstruction(theta, pdata):
    w, u, a, b, c = unpack(theta)
    h1 = np.tanh(X @ w + b)
    z = np.tanh(h1 @ u + c)
    h2 = np.tanh(z @ u.T + b)
    scores = h2 @ w.T + a
    objective = np.sum(pdata[:, None] * (X * scores - np.logaddexp(scores, -scores))) / D
    ds4 = pdata[:, None] * (X - np.tanh(scores)) / D
    da = ds4.sum(axis=0)
    dw = ds4.T @ h2
    ds3 = (ds4 @ w) * (1 - h2**2)
    db = ds3.sum(axis=0)
    du = ds3.T @ z
    ds2 = (ds3 @ u) * (1 - z**2)
    dc = ds2.sum(axis=0)
    du += h1.T @ ds2
    ds1 = (ds2 @ u.T) * (1 - h1**2)
    db += ds1.sum(axis=0)
    dw += X.T @ ds1
    return float(objective), pack(dw, du, da, db, dc)


def gradient_audit():
    rng = np.random.default_rng(20105499)
    theta = rng.normal(0, .3, P)
    pdata = rng.dirichlet(np.ones(len(X)))
    errors = {}
    for name, fn in [('likelihood', likelihood), ('pseudolikelihood', pseudolikelihood), ('reconstruction', reconstruction)]:
        value, grad = fn(theta, pdata)
        estimates = []
        for j in range(P):
            delta = np.zeros(P)
            delta[j] = 1e-5
            estimates.append((fn(theta + delta, pdata)[0] - fn(theta - delta, pdata)[0]) / (2e-5))
        error = float(np.max(np.abs(grad - estimates)))
        assert error < 1e-7
        errors[name] = error
    return errors


def optimize(theta, pdata, ptrue, fn, steps, method):
    theta = theta.copy()
    moments, variances = np.zeros(P), np.zeros(P)
    history = []
    old_objective = -np.inf
    for step in range(steps + 1):
        objective, grad = fn(theta, pdata)
        if method == 'gradient':
            assert objective >= old_objective - 1e-10
        old_objective = objective
        if step % 40 == 0:
            px, _, _, posterior = model(theta)
            ll = likelihood(theta, pdata)[0]
            rec = reconstruction(theta, pdata)[0]
            pl = pseudolikelihood(theta, pdata)[0]
            history.append([step, kl(ptrue, px), ll, rec, pl, objective, np.linalg.norm(grad)])
            if step >= 2000:
                recent = np.array(history[-15:])
                slope = np.polyfit(np.log(recent[:, 0]), np.log(-recent[:, 5]), 1)[0]
                if abs(slope) < .0005 and np.linalg.norm(grad) < .002:
                    break
        if step == steps:
            break
        if method == 'gradient':
            theta += grad / P
        else:
            moments = .9 * moments + .1 * grad
            variances = .999 * variances + .001 * grad**2
            theta += .01 * moments / (1 - .9 ** (step + 1)) / (np.sqrt(variances / (1 - .999 ** (step + 1))) + 1e-8)
    return theta, np.array(history)


def run(seed, n, steps):
    rng = np.random.default_rng(seed)
    teacher = pack(rng.normal(0, .65, (D, M)), rng.normal(0, .8, (M, K)), rng.normal(0, .12, D), rng.normal(0, .12, M), rng.normal(0, .12, K))
    ptrue = model(teacher)[0]
    data = rng.choice(len(X), n, p=ptrue)
    pdata = np.bincount(data, minlength=len(X)) / n
    start = pack(rng.normal(0, .3, (D, M)), rng.normal(0, .3, (M, K)), np.arctanh(np.clip(pdata @ X, -.95, .95)), np.zeros(M), np.zeros(K))
    initial, initial_history = optimize(start, pdata, ptrue, reconstruction, steps, 'adam')
    out = dict(data=data, pdata=pdata, teacher=teacher, ptrue=ptrue, start=start, initial=initial, initial_history=initial_history)
    summary = dict(seed=seed, n=n, initial_kl=float(initial_history[-1, 1]), initial_grad=float(initial_history[-1, 6]), initial_steps=int(initial_history[-1, 0]))
    for name, fn, method in [('practice', likelihood, 'gradient'), ('pseudolikelihood', pseudolikelihood, 'adam'), ('reconstruction', reconstruction, 'adam')]:
        theta, history = optimize(initial, pdata, ptrue, fn, steps, method)
        out[name + '_theta'] = theta
        out[name + '_history'] = history
        summary[name] = dict(kl=float(history[-1, 1]), ll=float(history[-1, 2]), reconstruction=float(history[-1, 3]), pseudolikelihood=float(history[-1, 4]), grad=float(history[-1, 6]), steps=int(history[-1, 0]))
    return summary, out


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--steps', type=int, default=12000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | dict(out=str(args.out), time_uk=datetime.now(ZoneInfo('Europe/London')).isoformat(), gradient_errors=gradient_audit())
    (args.out / 'protocol.json').write_text(json.dumps(config, indent=2))
    summary = []
    for world in trange(args.worlds, desc='DBM worlds'):
        metrics, arrays = run(20105400 + world, args.n, args.steps)
        np.savez_compressed(args.out / f'world-{world:02d}.npz', **arrays)
        summary.append(metrics)
        (args.out / 'summary.json').write_text(json.dumps(dict(config=config, worlds=summary), indent=2))
    fig, ax = plt.subplots()
    grid = np.arange(0, args.steps + 1, 40)
    for name in ['practice', 'pseudolikelihood', 'reconstruction']:
        curves = []
        for world in range(args.worlds):
            history = np.load(args.out / f'world-{world:02d}.npz')[name + '_history']
            curves.append(np.interp(grid, history[:, 0], history[:, 1]))
        ax.plot(grid, np.mean(curves, axis=0), label=name)
    ax.set(xlabel='Updates after original-example reconstruction fit', ylabel='Population KL (nats)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / 'learning.png')
    print(json.dumps({name: np.mean([s[name]['kl'] for s in summary]) for name in ['practice', 'pseudolikelihood', 'reconstruction']}))


if __name__ == '__main__':
    main()
