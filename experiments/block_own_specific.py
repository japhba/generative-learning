"""Own samples against a twin's samples when both reach the reader only through the string: exact mean dynamics on the block world
(one check, one group of 9 states, the rule of order3_net.py; see block_flywheel.py and block_unlock_theory.py).

The reader starts weak and leans, state by state, towards the strings its own writer holds (shared start). The twin is a second learner
whose writer holds another code of the same kind. Two kinds of code:
  'latin':  nine strings (s1, s2, f(s1, s2)) for a Latin square f other than the teacher's, so every pair of blocks is uniform, as in
            teacher strings: own samples and teacher strings differ only at third order, and the drive on a diffuse reader is second order.
  'random': nine random strings: pairs of blocks are not uniform, and the drive is first order.
Printed: 1. the drive of own, twin and teacher strings on the reader along its own code (exponent in the start amplitude and size);
2. samples until the reader is sharp (most active state at 0.9 on the strings shown) with own strings and with the twin's strings;
3. samples until KL(teacher || own samples) < 0.1 for teacher strings only, and for a block of own or of twin strings (as long as the own
strings need to make the reader sharp) followed by teacher strings.

    python block_own_specific.py
"""
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from block_flywheel import EPS, Learner, World, measure, sm, step

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
W, K, Q, NB = World(1), 9, 3, 3
LATIN = [np.array(f) for f in itertools.product(itertools.permutations(range(3)), repeat=3) if all(len({f[r][c] for r in range(3)}) == 3 for c in range(3))]
TEACHER = next(i for i, f in enumerate(LATIN) if all(f[a][b] == (-a - b) % 3 for a in range(3) for b in range(3)))
norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-12))


def code(rng, kind, avoid=()):
    """nine strings [9, 3] in random order, and the index of the Latin square (-1 for random strings)."""
    if kind == 'latin':
        i = rng.choice([j for j in range(len(LATIN)) if j != TEACHER and j not in avoid])
        return np.array([(a, b, LATIN[i][a][b]) for a in range(3) for b in range(3)])[rng.permutation(9)], i
    return W.S[rng.choice(27, 9, replace=False)], -1


def sharp(patterns, p=.99):
    P = np.full((len(patterns), NB, Q), (1 - p) / (Q - 1))
    P[np.arange(len(patterns))[:, None], np.arange(NB), np.array(patterns)] = p
    return P


def make(strings, a):
    """learner whose writer holds the strings and whose reader leans towards them with amplitude a."""
    m = Learner(np.random.default_rng(0), W, K, 0., 0.)
    m.G[0] = sharp(strings)
    m.R[0] = sm(a * np.eye(Q)[strings])
    return m


def reach(m, p):
    return float(p @ m.read()[0].max(1))


def show(m, source, n, eta=.3):
    """n steps in which the reader sees only the samples of `source` (its own writer or a twin's)."""
    for _ in range(n):
        dR, _, _ = step(m, eta, 1., 'strict', source)
        m.R = norm(m.R + eta * dR)


def teach(m, t_max=2e6, eta=.3):
    t = 0
    while t < t_max:
        if measure(m)['kl'] < .1:
            return t
        for _ in range(20):
            dR, dG, _ = step(m, eta, 0., 'strict'); m.R, m.G = norm(m.R + eta * dR), norm(m.G + eta * dG)
        t += 20 * eta / EPS
    return None


def drive(m, p, direction):
    """expected change of the reader's log weights under strings with distribution p, along a direction [9, 3, 3], minus the same for noise."""
    def change(p):
        y = m.read()[0]
        return (np.einsum('n,nk,nbs->kbs', p, y, W.H) - np.einsum('n,nk->k', p, y)[:, None, None] * m.R[0]) / m.R[0]
    d = change(p) - change(np.full(27, 1 / 27))
    return float(np.sum(d * direction) / np.sum(direction * direction))


def job(args):
    kind, a, seed = args
    rng = np.random.default_rng([seed, 3])
    mine, i = code(rng, kind)
    theirs, _ = code(rng, kind, avoid=(i,))
    twin = make(theirs, a)
    out = dict(kind=kind, a=a, seed=seed, shared_with_twin=len({tuple(s) for s in mine} & {tuple(s) for s in theirs}), shared_with_teacher=int(W.valid[(mine * 3 ** np.arange(2, -1, -1)).sum(1)].sum()))
    per = .3 / EPS
    # samples until the reader is sharp on the strings it is shown
    for name, source in (('own', None), ('twin', twin)):
        m, p, n = make(mine, a), (make(mine, a) if source is None else twin).write()[0].sum(0), 0
        while reach(m, p) < .9 and n * per < 2e6:
            show(m, source, 5); n += 5
        out[f'sharp_{name}'] = n * per if reach(m, p) >= .9 else None
    block = int((out['sharp_own'] or 0) / per)
    # time to the solution
    out['teacher'] = teach(make(mine, a))
    for name, source in (('own', None), ('twin', twin)):
        m = make(mine, a); show(m, source, block)
        t = teach(m)
        out[f'{name}_then_teacher'] = None if t is None else t + block * per
    return out


if __name__ == '__main__':
    print('1. drive on the reader along its own code (reader amplitude a, writers sharp), minus noise')
    rng, amps = np.random.default_rng(0), np.array([.02, .04, .08, .16])
    for kind in ('latin', 'random'):
        rows = dict(own=[], twin=[], teacher=[])
        for _ in range(8):
            mine, i = code(rng, kind); theirs, _ = code(rng, kind, avoid=(i,))
            direction = np.eye(Q)[mine] - 1 / Q
            for name, p in (('own', make(mine, 1.).write()[0].sum(0)), ('twin', make(theirs, 1.).write()[0].sum(0)), ('teacher', W.pT)):
                rows[name].append([drive(make(mine, a), p, direction) for a in amps])
        for name, v in rows.items():
            v = np.mean(v, 0)
            print(f"  {kind:6s} codes, {name:7s} strings: exponent {np.polyfit(np.log(amps), np.log(np.abs(v)), 1)[0]:.2f}, at a = 0.16 {v[-1]:+.2e}")
    with Pool(28) as pool:
        res = pool.map(job, [(kind, a, s) for kind in ('latin', 'random') for a in (.3, .1, .03, .01) for s in range(8)])
    med = lambda v: (lambda d: f"{np.median(d) / 1000:7.0f}k ({len(d)}/8)" if d else '      - (0/8)')([x for x in v if x is not None])
    print('2. samples until the reader is sharp on the strings shown;  3. samples until KL < 0.1 (block counted)')
    for kind in ('latin', 'random'):
        for a in (.3, .1, .03, .01):
            rs = [r for r in res if r['kind'] == kind and r['a'] == a]
            print(f"  {kind:6s} codes, reader start {a:4.2f}:  sharp with own {med([r['sharp_own'] for r in rs])}  with twin {med([r['sharp_twin'] for r in rs])}   |   teacher only {med([r['teacher'] for r in rs])}"
                  f"   own block, then teacher {med([r['own_then_teacher'] for r in rs])}   twin block, then teacher {med([r['twin_then_teacher'] for r in rs])}")
    json.dump(res, open(REPORTS / 'block-own-specific.json', 'w'), indent=1)
