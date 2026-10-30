"""Deep-prior shortcut controls and complete observed-space Gibbs-loop confirmation."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

from practice_deep_prior_screen import channel, kl, prior, softmax, spins


def source_fit(data, x, h):
    n, twice_d = data.shape
    d = twice_d // 2
    attenuation = np.sqrt(np.clip(np.mean(data[:, ::2] * data[:, 1::2], axis=0), 0.04, 0.96))
    eps = (1 - attenuation) / 2
    means = np.clip(data.reshape(n, d, 2).mean(axis=(0, 2)) / attenuation, -0.95, 0.95)
    b0 = np.arctanh(means)
    p0 = softmax(h @ b0)
    likelihood = channel(x, h, eps)
    indices = ((data + 1) // 2 @ (2 ** np.arange(twice_d))).astype(int)
    post0 = likelihood[indices] * p0
    post0 /= post0.sum(axis=1, keepdims=True)
    q = post0.mean(axis=0)
    return b0, p0, eps, likelihood, indices, post0, q


def tree_fit(q, h):
    d = h.shape[1]
    binary = (h + 1) // 2
    unary = np.array([[q[binary[:, j] == b].sum() for b in [0, 1]] for j in range(d)])
    pairs, edges = {}, []
    for i in range(d):
        for j in range(i + 1, d):
            joint = np.array([[q[(binary[:, i] == a) & (binary[:, j] == b)].sum() for b in [0, 1]] for a in [0, 1]])
            pairs[(i, j)] = joint
            mi = np.sum(joint * np.log(joint / (unary[i, :, None] * unary[j, None, :])))
            edges.append((mi, i, j))
    sets = list(range(d))
    logp = np.log(unary[np.arange(d)[None, :], binary]).sum(axis=1)
    selected = []
    for mi, i, j in sorted(edges, reverse=True):
        if sets[i] != sets[j]:
            selected.append((i, j))
            old, new = sets[j], sets[i]
            sets = [new if label == old else label for label in sets]
            joint = pairs[(i, j)]
            logp += np.log(joint[binary[:, i], binary[:, j]] / (unary[i, binary[:, i]] * unary[j, binary[:, j]]))
    p = np.exp(logp)
    assert abs(p.sum() - 1) < 1e-10
    return p, selected


def deconvolve(data, h, eps):
    proxy = data.reshape(len(data), h.shape[1], 2).mean(axis=2) / (1 - 2 * eps)
    raw = np.prod((1 + proxy[:, None, :] * h[None, :, :]) / 2, axis=2).mean(axis=0)
    assert abs(raw.sum() - 1) < 1e-10
    clipped = np.maximum(raw, 0)
    clipped /= clipped.sum()
    return clipped, raw


def controls(data, x, h, p0, q, likelihood):
    fractions = np.array([0, .05, .1, .2, .4, .6, .8, 1.0])
    scores = np.zeros((3, len(fractions)))
    for fold in range(4):
        mask = np.arange(len(data)) % 4 == fold
        _, pf, ef, lf, _, _, qf = source_fit(data[~mask], x, h)
        tf, _ = tree_fit(qf, h)
        df, _ = deconvolve(data[~mask], h, ef)
        val = ((data[mask] + 1) // 2 @ (2 ** np.arange(x.shape[1]))).astype(int)
        for family, estimate in enumerate([qf, tf, df]):
            for j, fraction in enumerate(fractions):
                px = lf @ ((1 - fraction) * estimate + fraction * pf)
                scores[family, j] += np.log(px[val]).sum()
    tree, edges = tree_fit(q, h)
    _, _, eps, _, _, _, _ = source_fit(data, x, h)
    deconv, raw = deconvolve(data, h, eps)
    selections = np.argmax(scores, axis=1)
    q_cv = (1 - fractions[selections[0]]) * q + fractions[selections[0]] * p0
    tree_cv = (1 - fractions[selections[1]]) * tree + fractions[selections[1]] * p0
    deconv_cv = (1 - fractions[selections[2]]) * deconv + fractions[selections[2]] * p0
    best = np.unravel_index(np.argmax(scores), scores.shape)
    best_p = [q_cv, tree_cv, deconv_cv][best[0]]
    return dict(aggregate=q, aggregate_cv=q_cv, tree=tree, tree_cv=tree_cv, deconv=deconv, deconv_cv=deconv_cv, direct_cv=best_p), dict(scores=scores, fractions=fractions, selected=selections, tree_edges=np.array(edges), deconv_raw=raw)


def train(q, h, b0, p0, eps, likelihood, indices, pxt, rng, mode, max_steps):
    d, k, batch = h.shape[1], 2, 512
    w, b, c = rng.normal(0, .03, (d, k)), b0.copy(), np.zeros(k)
    free_h = np.where(rng.random((batch, d)) < (1 + np.tanh(b0)) / 2, 1, -1)
    free_z = rng.choice([-1, 1], (batch, k))
    original_ll = np.log((likelihood @ p0)[indices]).mean()
    log_likelihood_data = np.log(likelihood[indices])
    curves, objective_history = [], []
    target_h = q @ h
    for step in range(max_steps + 1):
        p, zmean = prior(h, w, b, c)
        objective = q @ np.log(p)
        if step % 40 == 0:
            px = likelihood @ p
            certificate = q @ (np.log(p) - np.log(p0))
            ll = np.log(px[indices]).mean()
            assert ll - original_ll >= certificate - 1e-10
            post = likelihood[indices] * p
            post /= post.sum(axis=1, keepdims=True)
            reconstruction = np.sum(post * log_likelihood_data) / len(indices)
            zprob = (1 + zmean) / 2
            entropy = -np.sum(p[:, None] * (zprob * np.log(zprob) + (1 - zprob) * np.log(1 - zprob)))
            curves.append([step, kl(pxt, px), ll, certificate, objective, reconstruction, entropy, np.linalg.norm(w)])
            objective_history.append(-objective)
            if step >= 4000:
                recent = np.array(curves[-15:])
                slope = np.polyfit(np.log(recent[:, 0]), np.log(-recent[:, 4]), 1)[0]
                if abs(slope) < .0005:
                    break
        if step == max_steps:
            break
        positive_w = h.T @ (q[:, None] * zmean)
        positive_c = q @ zmean
        if mode == 'exact':
            negative_w = h.T @ (p[:, None] * zmean)
            negative_b, negative_c = p @ h, p @ zmean
            eta = .05
        else:
            for sweep in range(2):
                repeated = np.repeat(free_h, 2, axis=1)
                generated_x = repeated * np.where(rng.random(repeated.shape) < np.repeat(eps, 2), -1, 1)
                field = b + free_z @ w.T + .5 * np.log((1 - eps) / eps) * generated_x.reshape(batch, d, 2).sum(axis=2)
                free_h = np.where(rng.random(field.shape) < (1 + np.tanh(field)) / 2, 1, -1)
                field_z = c + free_h @ w
                free_z = np.where(rng.random(field_z.shape) < (1 + np.tanh(field_z)) / 2, 1, -1)
            negative_w = free_h.T @ np.tanh(c + free_h @ w) / batch
            negative_b = free_h.mean(axis=0)
            negative_c = np.tanh(c + free_h @ w).mean(axis=0)
            eta = .05 / (1 + step / 2000) ** .6
        w += eta * (positive_w - negative_w)
        b += eta * (target_h - negative_b)
        c += eta * (positive_c - negative_c)
    return dict(curves=np.array(curves), w=w, b=b, c=c, p=p), dict(kl=float(curves[-1][1]), likelihood_gain=float(ll - original_ll), certificate=float(certificate), steps=step, reconstruction=float(reconstruction), recognition_entropy=float(entropy))


def one_world(seed, n, modes, max_steps):
    rng = np.random.default_rng(seed)
    d, k = 6, 2
    h, x = spins(d), spins(2 * d)
    wt = rng.normal(0, .9, (d, k))
    bt, ct = rng.normal(0, .15, d), rng.normal(0, .15, k)
    pt, _ = prior(h, wt, bt, ct)
    pxt = channel(x, h, np.full(d, .10)) @ pt
    data = x[rng.choice(len(x), n, p=pxt)]
    b0, p0, eps, likelihood, indices, post0, q = source_fit(data, x, h)
    control_priors, control_details = controls(data, x, h, p0, q, likelihood)
    metrics = dict(seed=seed, n=n, initial_kl=kl(pxt, likelihood @ p0), initial_reconstruction=float(np.sum(post0 * np.log(likelihood[indices])) / n))
    arrays = dict(data=data, teacher_w=wt, teacher_b=bt, teacher_c=ct, teacher_p=pt, p0=p0, q=q, eps=eps, post0=post0, **control_details)
    for name, control in control_priors.items():
        metrics[name + '_kl'] = kl(pxt, likelihood @ control)
        arrays[name + '_p'] = control
    for mode in modes:
        fit, result = train(q, h, b0, p0, eps, likelihood, indices, pxt, np.random.default_rng(seed + 100000), mode, max_steps)
        metrics[mode] = result
        arrays.update({mode + '_' + key: value for key, value in fit.items()})
    return metrics, arrays


def plot(summary, root):
    for n in sorted(set(s['n'] for s in summary)):
        worlds = [s for s in summary if s['n'] == n]
        fig, ax = plt.subplots()
        steps = np.arange(0, 12001, 40)
        for mode in ['exact', 'sampled']:
            if mode in worlds[0]:
                trajectories = []
                for world in worlds:
                    arrays = np.load(root / f"n-{n}-seed-{world['seed']}.npz")
                    curve = arrays[mode + '_curves']
                    trajectories.append(np.interp(steps, curve[:, 0], curve[:, 1]))
                ax.plot(steps, np.mean(trajectories, axis=0), label=mode.capitalize() + ' deep prior')
        for name, label in [('initial', 'Independent prior'), ('aggregate_cv', 'CV shrunk aggregate'), ('tree_cv', 'CV shrunk tree'), ('deconv_cv', 'CV deconvolution')]:
            ax.axhline(np.mean([world[name + '_kl'] for world in worlds]), linestyle='--', label=label)
        ax.set(xlabel='Local learning updates', ylabel='Population KL (nats)', title=f'{n} initial examples; {len(worlds)} worlds')
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / f'learning-n-{n}.png')
        plt.close(fig)


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--samples', type=int, nargs='+', default=[128])
    parser.add_argument('--modes', nargs='+', default=['exact', 'sampled'])
    parser.add_argument('--max-steps', type=int, default=12000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | dict(out=str(args.out), time_uk=datetime.now(ZoneInfo('Europe/London')).isoformat(), seed_start=20105300)
    (args.out / 'protocol.json').write_text(json.dumps(config, indent=2))
    summary = []
    jobs = [(n, 20105300 + w) for n in args.samples for w in range(args.worlds)]
    for n, seed in tqdm(jobs, desc='Deep-prior confirmation'):
        metrics, arrays = one_world(seed, n, args.modes, args.max_steps)
        np.savez_compressed(args.out / f'n-{n}-seed-{seed}.npz', **arrays)
        summary.append(metrics)
        (args.out / 'summary.json').write_text(json.dumps(dict(config=config, worlds=summary), indent=2))
    plot(summary, args.out)
    for n in args.samples:
        worlds = [s for s in summary if s['n'] == n]
        result = {name: np.mean([s[name + '_kl'] for s in worlds]) for name in ['initial', 'aggregate', 'aggregate_cv', 'tree', 'tree_cv', 'deconv', 'deconv_cv', 'direct_cv']}
        result.update({mode: np.mean([s[mode]['kl'] for s in worlds]) for mode in args.modes})
        print(n, json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
