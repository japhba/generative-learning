"""Exact screen of a learned deep prior, including the saturated-prior shortcut."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import trange


def spins(d):
    return 2 * ((np.arange(2**d)[:, None] >> np.arange(d)) & 1) - 1


def softmax(a):
    e = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def prior(h, w, b, c):
    hz = c + h @ w
    lp = h @ b + np.logaddexp(hz, -hz).sum(axis=1)
    return softmax(lp), np.tanh(hz)


def channel(x, h, eps):
    match = x[:, None, :] == np.repeat(h, 2, axis=1)[None, :, :]
    e = np.repeat(eps, 2)
    return np.prod(np.where(match, 1 - e, e), axis=2)


def kl(p, q):
    return float(p @ (np.log(p) - np.log(q)))


def run(seed, d, k, n, steps):
    rng = np.random.default_rng(seed)
    h, x = spins(d), spins(2 * d)
    wt = rng.normal(0, 0.9, (d, k))
    bt, ct = rng.normal(0, 0.15, d), rng.normal(0, 0.15, k)
    pt, _ = prior(h, wt, bt, ct)
    eps_true = np.full(d, 0.10)
    pxt = channel(x, h, eps_true) @ pt
    indices = rng.choice(len(x), n, p=pxt)
    data = x[indices]
    cross = np.mean(data[:, ::2] * data[:, 1::2], axis=0)
    attenuation = np.sqrt(np.clip(cross, 0.04, 0.96))
    eps = (1 - attenuation) / 2
    means = np.clip(data.reshape(n, d, 2).mean(axis=(0, 2)) / attenuation, -0.95, 0.95)
    b0 = np.arctanh(means)
    p0 = softmax(h @ b0)
    likelihood = channel(x, h, eps)
    px0 = likelihood @ p0
    post0 = likelihood[indices] * p0
    post0 /= post0.sum(axis=1, keepdims=True)
    q = post0.mean(axis=0)
    original_ll = np.log(px0[indices]).mean()
    recon = np.einsum('nh,nh->', post0, np.log(likelihood[indices])) / n
    w, b, c = rng.normal(0, 0.03, (d, k)), b0.copy(), np.zeros(k)
    target_h = q @ h
    curves = []
    previous_q = -np.inf
    for step in range(steps + 1):
        p, zmean = prior(h, w, b, c)
        objective = q @ np.log(p)
        assert objective >= previous_q - 1e-11
        previous_q = objective
        if step % 20 == 0:
            px = likelihood @ p
            certificate = q @ (np.log(p) - np.log(p0))
            ll = np.log(px[indices]).mean()
            assert ll - original_ll >= certificate - 1e-10
            curves.append([step, kl(pxt, px), ll, certificate, np.linalg.norm(w), objective])
        dw = h.T @ (q[:, None] * zmean) - h.T @ (p[:, None] * zmean)
        db = target_h - p @ h
        dc = q @ zmean - p @ zmean
        if step == steps:
            break
        w += 0.05 * dw
        b += 0.05 * db
        c += 0.05 * dc
    qx = likelihood @ q
    metrics = dict(seed=seed, initial_kl=kl(pxt, px0), learned_kl=kl(pxt, px), saturated_kl=kl(pxt, qx), certificate=float(certificate), saturated_certificate=float(q @ (np.log(q) - np.log(p0))), reconstruction=float(recon), steps=step)
    return metrics, dict(curves=np.array(curves), data=data, teacher_w=wt, teacher_b=bt, teacher_c=ct, teacher_p=pt, learned_w=w, learned_b=b, learned_c=c, learned_p=p, q=q, p0=p0, eps=eps, post0=post0)


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--d', type=int, default=6)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, curves = [], []
    for world in trange(args.worlds, desc='Deep-prior worlds'):
        metrics, arrays = run(20105200 + world, args.d, args.k, args.n, args.steps)
        np.savez_compressed(args.out / f'world-{world:02d}.npz', **arrays)
        summary.append(metrics)
        curves.append(arrays['curves'])
        (args.out / 'summary.json').write_text(json.dumps(dict(config=vars(args) | {'out': str(args.out)}, worlds=summary), indent=2))
    curves = np.array(curves)
    fig, ax = plt.subplots()
    ax.plot(curves[0, :, 0], curves[:, :, 1].mean(axis=0), label='Structured deep-prior learning')
    ax.axhline(np.mean([s['initial_kl'] for s in summary]), label='Initial factorized prior', color='black', linestyle=':')
    ax.axhline(np.mean([s['saturated_kl'] for s in summary]), label='One-shot aggregated-posterior prior', linestyle='--')
    ax.set(xlabel='Local exact-gradient updates', ylabel='Population KL (nats)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / 'learning.png')
    print(json.dumps({key: float(np.mean([s[key] for s in summary])) for key in ['initial_kl', 'learned_kl', 'saturated_kl', 'certificate', 'saturated_certificate']}, indent=2))


if __name__ == '__main__':
    main()
