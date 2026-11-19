"""Deep Boltzmann machine x - h1 - h2, teacher and student of the same shape: what the negative phase does to each layer.

All updates are exact expected updates (enumeration over x and h1, h2 summed in closed form), so the arms differ only in
where the visible states of the negative phase come from:
  dream    fresh samples of the learner's own equilibrium distribution (the exact likelihood gradient)
  cd1/cd5  k block-Gibbs sweeps started at the teacher sample and its inferred code (contrastive divergence)
  twin     fresh samples of a twin trained the same way from another start, read by the learner's own recognition
  shallow  dream with the deep weights held at zero (the best single RBM)
  edge     no samples at all: every weight's own tanh(w), which is the exact model correlation on a tree
Units are +-1, no biases. Positive phase: teacher sample clamped, exact posterior over (h1, h2).
--block b gives every h1 unit its own b visible units (a tree; teacher and student alike), 0 connects all to all.
"""
import argparse
import itertools
import json
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


def spins(n):
    return np.array(list(itertools.product((-1., 1.), repeat=n)))


def softmax(logp, axis):
    p = np.exp(logp - logp.max(axis=axis, keepdims=True))
    return p / p.sum(axis=axis, keepdims=True)


def kl(p, q):
    keep = p > 0
    return float(np.sum(p[keep] * (np.log(p[keep]) - np.log(q[keep]))))


def mutual_information(joint):
    outer = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    keep = joint > 0
    return float(np.sum(joint[keep] * (np.log(joint[keep]) - np.log(outer[keep]))))


class Shape:
    def __init__(self, n, m, k):
        self.n, self.m, self.k = n, m, k
        self.X, self.H1, self.H2 = spins(n), spins(m), spins(k)
        self.outer1 = (self.H1[:, :, None] * self.H1[:, None, :]).reshape(len(self.H1), m * m)


def tables(s, w1, w2):
    """log p~(x, h1) with h2 summed out, p(h1 | x), p(x), and the fields on h2."""
    a = s.X @ w1 @ s.H1.T
    f2 = s.H1 @ w2
    logjoint = a + np.logaddexp(f2, -f2).sum(1)[None]
    top = logjoint.max(1)
    logpx = top + np.log(np.exp(logjoint - top[:, None]).sum(1))
    return a, f2, softmax(logjoint, 1), softmax(logpx, 0)


def clamped_stats(s, r, post, f2):
    """<x h1>, <h1 h2> and the law of h1 when x ~ r and (h1, h2) follow the learner's posterior."""
    pi = r @ post
    return s.X.T @ (r[:, None] * (post @ s.H1)), (s.H1 * pi[:, None]).T @ np.tanh(f2), pi


def gibbs_parts(s, a, f2, w2):
    px_h1 = softmax(a.T, 1)                                              # p(x | h1)
    ph2_h1 = softmax(f2 @ s.H2.T, 1)                                     # p(h2 | h1)
    q = softmax(a[:, None, :] + (s.H2 @ w2.T @ s.H1.T)[None], 2)         # p(h1' | x, h2)
    return px_h1, ph2_h1, q


def chain_stats(s, pi0, parts, sweeps):
    """Negative-phase statistics after `sweeps` block-Gibbs sweeps (x, h2 | h1) then (h1 | x, h2), h1 started from pi0."""
    px_h1, ph2_h1, q = parts
    pi = pi0
    for _ in range(sweeps):
        joint = np.einsum('a,ax,ab->xb', pi, px_h1, ph2_h1)
        pi = np.einsum('xb,xbc->c', joint, q)
    mean_h1 = q @ s.H1
    return np.einsum('xb,xi,xbm->im', joint, s.X, mean_h1), np.einsum('xb,xbm,bk->mk', joint, mean_h1, s.H2), pi


def deep_drive(s, arm, w1, pdata):
    """G in dW2/dt = G W2 for small deep weights, read off the arm's own update, and its largest growth rate."""
    probes = [gradient(s, arm, w1, 1e-4 * np.outer(np.eye(s.m)[b], np.eye(s.k)[0]), pdata)[1][:, 0] / 1e-4 for b in range(s.m)]
    return float(np.linalg.eigvals(np.array(probes).T).real.max())


def teacher_tables(s, w1, w2):
    a, f2, post, px = tables(s, w1, w2)
    ph2_h1 = softmax(f2 @ s.H2.T, 1)
    return dict(px=px, post=post, x_h2=px[:, None] * (post @ ph2_h1))


def evaluate(s, teacher, w1, w2, pdata):
    a, f2, post, px = tables(s, w1, w2)
    read2 = post @ softmax(f2 @ s.H2.T, 1)
    return dict(kl=kl(teacher['px'], px), kl_shallow=kl(teacher['px'], tables(s, w1, 0 * w2)[3]), norm1=float(np.linalg.norm(w1)), norm2=float(np.linalg.norm(w2)),
                drive_dream=deep_drive(s, 'dream', w1, pdata), drive_cd1=deep_drive(s, 'cd1', w1, pdata), drive_cd5=deep_drive(s, 'cd5', w1, pdata),
                reread=float(np.mean(px @ (post @ s.H1) ** 2)), read1=mutual_information((teacher['px'][:, None] * teacher['post']).T @ post), read2=mutual_information(teacher['x_h2'].T @ read2))


def gradient(s, arm, w1, w2, pdata, foreign=None):
    a, f2, post, px = tables(s, w1, w2)
    pos1, pos2, pi_data = clamped_stats(s, pdata, post, f2)
    if arm in ('dream', 'shallow'):
        neg1, neg2, _ = clamped_stats(s, px, post, f2)
    elif arm == 'twin':
        neg1, neg2, _ = clamped_stats(s, foreign, post, f2)
    elif arm == 'edge':
        neg1, neg2 = np.tanh(w1), np.tanh(w2)
    else:
        neg1, neg2, _ = chain_stats(s, pi_data, gibbs_parts(s, a, f2, w2), int(arm[2:]))
    return pos1 - neg1, (pos2 - neg2) * (arm != 'shallow'), px


def run(job):
    seed, arm, examples, args = job
    s = Shape(args['n'], args['m'], args['k'])
    rng = np.random.default_rng(seed)
    mask = (np.arange(s.n)[:, None] // args['block'] == np.arange(s.m)[None]) * 1. if args['block'] else np.ones((s.n, s.m))
    tw1, tw2 = mask * rng.normal(0, args['g1'], (s.n, s.m)), rng.normal(0, args['g2'], (s.m, s.k))
    teacher = teacher_tables(s, tw1, tw2)
    pdata = teacher['px'] if examples == 0 else np.bincount(rng.choice(len(s.X), examples, p=teacher['px']), minlength=len(s.X)) / examples
    starts = [(mask * rng.normal(0, args['init'], (s.n, s.m)), rng.normal(0, args['init'], (s.m, s.k))) for _ in range(2)]
    weights = [[w1.copy(), w2.copy() * (arm != 'shallow')] for w1, w2 in starts]
    history = [[] for _ in weights]
    for step in range(args['steps'] + 1):
        if step % args['every'] == 0 or step in args['early']:
            for j, (w1, w2) in enumerate(weights):
                history[j].append(dict(step=step) | evaluate(s, teacher, w1, w2, pdata))
        if not all(np.isfinite(w1).all() and np.abs(w1).max() < 50 for w1, _ in weights):
            break
        laws = [tables(s, w1, w2)[3] for w1, w2 in weights] if arm == 'twin' else [None, None]
        for j, (w1, w2) in enumerate(weights):
            g1, g2, _ = gradient(s, arm, w1, w2, pdata, laws[1 - j])
            weights[j] = [w1 + args['rate'] * mask * g1, w2 + args['rate'] * g2]
    keys = history[0][0].keys()
    px, post = teacher['px'], teacher['post']
    return dict(seed=seed, arm=arm, examples=examples, diverged=len(history[0]) < args['steps'] // args['every'], teacher_reread=float(np.mean(px @ (post @ s.H1) ** 2)), weights=[[w.tolist() for w in pair] for pair in weights],
                teacher=[tw1.tolist(), tw2.tolist()], curves=[{key: [row[key] for row in h] for key in keys} for h in history])


def audit(s):
    """A long data-started chain gives the exact gradient; the deep weights grow at the rate the code gap predicts."""
    rng = np.random.default_rng(0)
    w1, w2 = rng.normal(0, .5, (s.n, s.m)), rng.normal(0, .5, (s.m, s.k))
    pdata = rng.dirichlet(np.ones(len(s.X)))
    exact, long_chain = gradient(s, 'dream', w1, w2, pdata), gradient(s, 'cd400', w1, w2, pdata)
    chain_error = max(float(np.abs(exact[i] - long_chain[i]).max()) for i in range(2))
    assert chain_error < 1e-8, chain_error
    small = 1e-4 * rng.normal(size=(s.m, s.k))
    a, f2, post, px = tables(s, w1, 0 * w2)
    gap = ((pdata @ post - px @ post) @ s.outer1).reshape(s.m, s.m)
    linear_error = float(np.abs(gradient(s, 'dream', w1, small, pdata)[1] - gap @ small).max() / np.abs(gap @ small).max())
    assert linear_error < 1e-4, linear_error
    assert abs(deep_drive(s, 'dream', w1, pdata) - np.linalg.eigvalsh(gap)[-1]) < 1e-6
    return dict(chain_error=chain_error, linear_error=linear_error)


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--g1', type=float, default=.8)
    parser.add_argument('--g2', type=float, default=1.5)
    parser.add_argument('--init', type=float, default=.01)
    parser.add_argument('--rate', type=float, default=.05)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--every', type=int, default=200)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--examples', type=int, nargs='+', default=[0, 1024, 256])
    parser.add_argument('--arms', nargs='+', default=['dream', 'cd1', 'cd5', 'twin', 'shallow'])
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--tag', default='main')
    args = vars(parser.parse_args())
    args['early'] = [int(v) for v in np.unique(np.round(np.geomspace(1, args['every'], 24)))]
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_dbm_dream' / f"{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    checks = audit(Shape(args['n'], args['m'], args['k']))
    (out / 'protocol.json').write_text(json.dumps(args | checks, indent=2))
    jobs = [(20261119 + world, arm, examples, args) for examples in args['examples'] for world in range(args['worlds']) for arm in args['arms']]
    with Pool(args['workers']) as pool:
        results = list(tqdm(pool.imap_unordered(run, jobs), total=len(jobs), desc='runs'))
    (out / 'results.json').write_text(json.dumps(dict(args=args, checks=checks, runs=results)))
    print(out)


if __name__ == '__main__':
    main()
