"""Checks of the linear theory of the unlock on the block world (one check, three blocks of three symbols, one group of K = 9 states),
against the exact mean dynamics of block_flywheel.py. Own samples reach the reader only through the string.

1. Diffuse reader: the state-specific part of the reader's weights grows as d omega / dt = (eps q / K) C_x omega, C_x = cross-block
   covariance of the code under the samples the reader sees. Printed for several writers: largest eigenvalue of C_x, the estimate
   (alpha - 1) / q with alpha = n (q - 1) / (patterns - 1) for random patterns, and the growth rate measured in the exact dynamics.
2. Race: teacher strings read by a diffuse reader pull every writer state to uniform at rate eps (1 - share) / K, so the reader gains in
   total a factor exp(share q c_max / (2 (1 - share))). Measured against that.
3. After the latch: with some reader states sharp on valid strings, the teacher strings left for the other states are no longer uniform in
   pairs. Largest eigenvalue of that covariance, predicted and measured growth rate of the remaining states under teacher strings only.
4. Time to KL < 0.1 against the start amplitude of the reader: teacher strings only; a block of own samples of a writer that holds three
   strings (valid ones, or random ones), then teacher strings; reader and writer sharp from the start.

5. Two checks and two groups: whether each group ends on one check.

    python block_unlock_theory.py
"""
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from block_flywheel import EPS, FLOOR, Learner, World, measure, sm, step

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
W = World(1)
K, Q, NB = 9, 3, 3
VALID = W.S[W.valid]


def sharp(patterns, p=.99):
    """[len, 3, 3]: every pattern as block-wise distributions with probability p on its symbols."""
    P = np.full((len(patterns), NB, Q), (1 - p) / (Q - 1))
    P[np.arange(len(patterns))[:, None], np.arange(NB), np.array(patterns)] = p
    return P


def cross(p):
    """cross-block covariance [9, 9] of the one-hot code under a distribution p over the 27 strings."""
    H = W.H.reshape(len(p), -1)
    C = (H * p[:, None]).T @ H - np.outer(p @ H, p @ H)
    for b in range(NB):
        C[b * Q:(b + 1) * Q, b * Q:(b + 1) * Q] = 0
    return C


def learner(rng, a, writer=None, prior=None, reader=None):
    m = Learner(rng, W, K, a, a)
    if writer is not None:
        m.G = np.full((1, K, NB, Q), 1 / Q); m.G[0, :len(writer)] = writer
    if prior is not None:
        m.pi = np.array(prior, float)[None]
    if reader is not None:
        m.R[0, :len(reader)] = reader
    return m


def advance(m, eta, own, n, prior_rate=0.):
    for _ in range(n):
        dR, dG, share = step(m, eta, own, 'strict')
        norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-12))
        m.R, m.G, m.pi = norm(m.R + eta * dR), norm(m.G + eta * dG), norm(m.pi + eta * prior_rate * share)


def spread(m, states=slice(None)):
    """size of the state-specific part of the reader's weights (log scale) among the given states."""
    w = np.log(m.R[0, states])
    return float(np.sqrt(np.mean((w - w.mean(0)) ** 2)))


def rate(m, own, states=slice(None), eta=.03, grow=1e3, n_max=40000):
    """growth rate per sample of the state-specific part in the exact dynamics: slope of its log over the last third of the time it takes
    to grow by the factor `grow` (the reader starts at 1e-7, so it stays diffuse and the fastest direction has taken over)."""
    t, a = [0], [spread(m, states)]
    while a[-1] < grow * a[0] and len(t) < n_max / 20:
        advance(m, eta, own, 20); t.append(t[-1] + 20 * eta / EPS); a.append(spread(m, states))
    k = max(2, len(t) // 3)
    return float(np.polyfit(t[-k:], np.log(a[-k:]), 1)[0])


def part1(rng):
    print('1. diffuse reader, own samples only: growth rate of the reader per sample (x 1e-4)')
    star = [(z, z, z) for z in range(3)]
    sets = {'3 valid strings (z, z, z)': [star], '3 valid strings, random': [VALID[rng.choice(9, 3, replace=False)] for _ in range(8)], '3 random strings': [W.S[rng.choice(27, 3, replace=False)] for _ in range(8)],
            '6 random strings': [W.S[rng.choice(27, 6, replace=False)] for _ in range(8)], '9 random strings': [W.S[rng.choice(27, 9, replace=False)] for _ in range(8)], 'the 9 valid strings': [VALID]}
    out = {}
    for name, draws in sets.items():
        top, meas = [], []
        for pats in draws:
            n = len(pats)
            m = learner(rng, 1e-7, writer=sharp(pats, 1 - 1e-6), prior=[1 / n] * n + [0] * (K - n))
            top.append(np.linalg.eigvalsh(cross(m.write()[0].sum(0))).max())
            meas.append(rate(m, 1.))
        alpha = NB * (Q - 1) / (len(draws[0]) - 1)
        out[name] = dict(c_max=float(np.mean(top)), estimate=(alpha - 1) / Q, predicted=float(EPS * Q * np.mean(top) / K), measured=float(np.mean(meas)))
        print(f"  writer holds {name:26s} largest eigenvalue of C_x {np.mean(top):+.3f}   (alpha - 1) / q = {(alpha - 1) / Q:+.3f}   rate predicted {1e4 * EPS * Q * np.mean(top) / K:6.2f}   measured {1e4 * np.mean(meas):6.2f}")
    return out


def part2(rng):
    print('2. race against the wiping of the writer (writer holds (z, z, z), reader starts at 1e-4 on the same three strings): log of the total gain of the reader')
    out = {}
    for own in (.5, .7, .8):
        m = learner(rng, 0., writer=sharp([(z, z, z) for z in range(3)], 1 - 1e-6), prior=[1 / 3] * 3 + [0] * 6)
        m.R[0, :3] = sm(1e-4 * np.log(m.G[0, :3]))                                                 # the reader's first three states lean towards the writer's strings
        c = np.linalg.eigvalsh(cross(m.write()[0].sum(0))).max()
        a0 = spread(m); advance(m, .03, own, int(12 * K / (1 - own) / .03)); a1 = spread(m)             # 12 time constants of the wiping
        out[own] = dict(predicted=float(own * Q * c / (2 * (1 - own))), measured=float(np.log(a1 / a0)))
        print(f"  share of own samples {own:.1f}: predicted {out[own]['predicted']:.2f}   measured {out[own]['measured']:.2f}")
    return out


def part3(rng):
    print('3. teacher strings only, some reader states already sharp: growth rate of the other states per sample (x 1e-4)')
    out = {}
    cases = [(f'{n} sharp on valid strings', VALID[:n]) for n in (0, 1, 2, 3, 4, 6)] + [('1 sharp on a string that is not valid', np.array([[0, 0, 1]])), ('3 sharp on strings that are not valid', np.array([[0, 0, 1], [1, 2, 1], [2, 2, 0]]))]
    for name, pats in cases:
        n = len(pats)
        m = learner(rng, 1e-7, reader=sharp(pats) if n else None)
        y = m.read()[0]
        left = W.pT * (1 - y[:, :n].sum(1))                                                         # teacher strings left for the diffuse states
        Z, p = left.sum(), left / left.sum()
        mean = p @ W.H.reshape(27, -1)
        ev = np.linalg.eigvals(cross(p) @ np.diag(1 / mean)).real.max()
        pred = EPS * Z / (K - n) * ev
        out[name] = dict(c_max=float(np.linalg.eigvalsh(cross(p)).max()), left=float(Z), predicted=float(pred), measured=rate(m, 0., slice(n, None)))
        print(f"  {name:38s} share of teacher strings left {Z:.2f}   largest eigenvalue {out[name]['c_max']:+.3f}   rate predicted {1e4 * pred:6.2f}   measured {1e4 * out[name]['measured']:6.2f}")
    return out


def race(job):
    arm, a, seed, block = job
    rng = np.random.default_rng([seed, 5])
    eta, per = .3, .3 / EPS
    pats = VALID[rng.choice(9, 3, replace=False)] if arm == 'own_valid' else W.S[rng.choice(27, 3, replace=False)] if arm == 'own_random' else None
    m = Learner(np.random.default_rng([seed, 1]), W, K, *((3., 3.) if arm == 'sharp' else (a, a)))
    t = 0
    if pats is not None:
        m.G[0, :3], m.pi = sharp(pats), np.array([[1 / 3] * 3 + [0] * 6])
        advance(m, eta, 1., int(block / per)); t = block
        m.pi = np.full((1, K), 1 / K)
    while t < 2e6:
        now = measure(m)
        if now['kl'] < .1:
            return dict(arm=arm, a=a, seed=seed, block=block, t=t, kl=now['kl'], covered=now['covered'])
        advance(m, eta, 0., 20); t += 20 * per
    return dict(arm=arm, a=a, seed=seed, block=block, t=None, kl=now['kl'], covered=now['covered'])


def part4():
    amps, blocks = (.3, .1, .03, .01), (2000, 5000, 10000, 30000)
    jobs = [('teacher', a, s, 0) for a in amps for s in range(8)] + [('sharp', 0., s, 0) for s in range(8)] + [(arm, a, s, b) for arm in ('own_valid', 'own_random') for a in amps for b in blocks for s in range(8)]
    with Pool(28) as pool:
        res = pool.map(race, jobs)
    print('4. samples to KL < 0.1 (thousands, median of the starts that get there; own samples counted), and where the other starts end')
    out = {}
    labels = dict(teacher='teacher strings only', sharp='reader and writer sharp from the start', own_valid='own samples of a writer holding 3 valid strings, then teacher strings', own_random='own samples of a writer holding 3 random strings, then teacher strings')
    for arm in ('teacher', 'sharp', 'own_valid', 'own_random'):
        for a in (amps if arm != 'sharp' else (0.,)):
            for b in (blocks if arm.startswith('own') else (0,)):
                rs = [r for r in res if (r['arm'], r['a'], r['block']) == (arm, a, b)]
                done, stuck = [r['t'] for r in rs if r['t'] is not None], [r for r in rs if r['t'] is None]
                out[f'{arm} a={a} block={b}'] = dict(arm=arm, a=a, block=b, times=[r['t'] for r in rs], done=len(done), median=float(np.median(done)) if done else None, stuck_kl=[round(r['kl'], 2) for r in stuck], stuck_covered=[round(r['covered'], 2) for r in stuck])
                print(f"  {labels[arm]:70s} reader start {a:4.2f}  block {b // 1000:2d}k: {np.median(done) / 1000 if done else float('nan'):6.0f} k  ({len(done)}/8)   others end at KL {[round(r['kl'], 2) for r in stuck]}")
    return out


def two_checks(job):
    """two checks and two groups, every group connected to all six blocks: where the learner is after 600,000 samples."""
    arm, a, seed, block = job
    w2 = World(2)
    rng = np.random.default_rng([seed, 7])
    m = Learner(np.random.default_rng([seed, 1]), w2, K, a, a)
    eta, per = .3, .3 / EPS
    norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-12))
    if block:                                                                                   # the writer's group g holds three valid strings of check g and nothing on the other blocks
        m.G[:] = 1 / Q
        m.G[0, :3, :3], m.G[1, :3, 3:] = sharp(VALID[rng.choice(9, 3, replace=False)]), sharp(VALID[rng.choice(9, 3, replace=False)])
        m.pi = np.array([[1 / 3] * 3 + [0] * 6] * 2)
        for _ in range(int(block / per)):
            dR, dG, _ = step(m, eta, 1., 'strict'); m.R = norm(m.R + eta * dR)
        m.pi = np.full((2, K), 1 / K)
    for _ in range(int(600000 / per)):
        dR, dG, _ = step(m, eta, .5 if arm == 'handed' else 0., 'sleep'); m.R, m.G = norm(m.R + eta * dR), norm(m.G + eta * dG)
    d = np.log(m.R) - np.log(m.R).mean(1, keepdims=True)
    v = (d ** 2).sum((1, 3))
    return arm, a, block, float(measure(m)['kl']), (v[:, :3].sum(1) / v.sum(1)).tolist()         # last: share of each group's selectivity on the blocks of the first check


def part5():
    arms = (('teacher strings only', 'teacher', .1, 0), ('reader and writer sharp from the start', 'teacher', 3., 0), ('first 5,000 own samples of a writer with three valid strings per check', 'teacher', .1, 5000),
            ('first 10,000 own samples of a writer with three valid strings per check', 'teacher', .1, 10000), ('half own samples, writing state handed to the reader', 'handed', .1, 0))
    with Pool(28) as pool:
        res = pool.map(two_checks, [(arm, a, s, b) for _, arm, a, b in arms for s in range(8)])
    print('5. two checks, two groups, 600,000 samples')
    out = {}
    for label, arm, a, b in arms:
        rs = [r for r in res if r[:3] == (arm, a, b)]
        bound = sum(min(r[4]) < .1 and max(r[4]) > .9 for r in rs)
        out[label] = dict(kl=[r[3] for r in rs], done=sum(r[3] < .1 for r in rs), bound=bound)
        print(f"  {label:72s} KL < 0.1 in {out[label]['done']}/8 starts (median KL {np.median(out[label]['kl']):.2f}); each group on one check in {bound}/8")
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    numbers = dict(linear=part1(rng), race=part2(rng), cascade=part3(rng), times=part4(), two_checks=part5())
    json.dump(numbers, open(REPORTS / 'block-unlock-theory.json', 'w'), indent=1, default=str)
