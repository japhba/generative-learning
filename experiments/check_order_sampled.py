"""One check over m blocks of q symbols (valid if the symbols sum to zero mod q), one winner-take-all group with one state per
valid string, sampled strings and states: how the time to the solution depends on the order of the check for teacher strings
alone and for the read-write loop with gain. With m blocks every subset of m - 1 blocks is uniform, so the teacher's drive on a
diffuse group is of order m - 1 in the tilt, while the loop's growth rate (eps / 2K)(sqrt(beta) - 1) does not depend on m.
Rule, reading and writing as in block_depth.py (one group, no levels).

    python check_order_sampled.py
"""
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
FLOOR = 1e-6
sm = lambda u: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(u - u.max(-1, keepdims=True)))


compact = lambda o: [compact(x) for x in o] if isinstance(o, list) else {k: compact(v) for k, v in o.items()} if isinstance(o, dict) else float(f'{o:.4g}') if isinstance(o, float) else o


def hebb(P, states, symbols, eps):
    K, q = P.shape[0], P.shape[-1]
    s1 = np.eye(K)[states]
    count = s1.sum(0)
    target = np.einsum('nk,niv->kiv', s1, np.eye(q)[symbols]) / np.maximum(count, 1)[:, None, None]
    keep = ((1 - eps) ** count)[:, None, None]
    return keep * P + (1 - keep) * target


def run(job):
    name, q, m, a, beta, own, mode, seed, t_max, eps = job
    rng, K = np.random.default_rng([seed, 2]), q ** (m - 1)
    R = sm(a * np.random.default_rng([seed, 1]).normal(size=(K, m, q))); G = R.copy()
    S = np.array(list(itertools.product(range(q), repeat=m)))
    valid = S.sum(1) % q == 0
    batch, n_own, blocks = 64, 0, np.arange(m)[None, :]
    draw = lambda logits: (logits + rng.gumbel(size=logits.shape)).argmax(-1)
    read = lambda s: draw(np.log(R + FLOOR)[:, blocks, s].sum(2).T)                              # [K, n, m] summed over blocks -> [n, K]

    def kl():
        Pw = sm(beta * np.log(G + FLOOR))
        p = np.exp(np.log(Pw)[:, blocks, S].sum(2)).mean(0)                                     # probability of every string
        return float(np.mean(np.log((1 / valid.sum()) / np.maximum(p[valid], 1e-12))))

    for i in range(int(t_max / batch) + 1):
        if i % 100 == 0 and kl() < .1:
            return dict(name=name, q=q, m=m, eps=eps, seed=seed, t=i * batch, kl=kl())
        if n_own < own * (i + 1):
            n_own += 1
            k = rng.integers(K, size=batch)
            s = draw(beta * np.log(G + FLOOR)[k])                                               # [batch, m]
            R = hebb(R, k if mode == 'kept' else read(s), s, eps)
        else:
            s = rng.integers(q, size=(batch, m)); s[:, -1] = (-s[:, :-1].sum(1)) % q
            k = read(s)
            R, G = hebb(R, k, s, eps), hebb(G, k, s, eps)
    return dict(name=name, q=q, m=m, eps=eps, seed=seed, t=None, kl=kl())


ARMS = {'teacher strings only': (1., 0., 'kept'), 'own, state kept, gain 1': (1., .5, 'kept'), 'own, state kept, gain 1.5': (1.5, .5, 'kept'), 'own, state kept, gain 2': (2., .5, 'kept'),
        'own, through the string, gain 2': (2., .5, 'string')}

if __name__ == '__main__':
    cases = ((2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4))
    jobs = [(name, q, m, .01, *ARMS[name], s, 2e7, .003) for q, m in cases for name in ARMS for s in range(8)]
    jobs.sort(key=lambda j: -(j[1] ** j[2]))
    with Pool(30) as pool:
        res = pool.map(run, jobs, chunksize=1)
    print('samples (thousands) until KL < 0.1, median of the starts that get there (number of 8); rate 0.003, start 0.01, up to 20 million samples')
    for q, m in cases:
        print(f'  {m} blocks of {q} symbols ({q ** (m - 1)} states, teacher drive of order {m - 1}):')
        for name in ARMS:
            rs = [r for r in res if (r['name'], r['q'], r['m']) == (name, q, m)]
            d = [r['t'] for r in rs if r['t'] is not None]
            print(f"      {name:34s} {np.median(d) / 1000 if d else float('nan'):8.0f}k ({len(d)})   KL of the others {np.round(sorted(r['kl'] for r in rs if r['t'] is None), 2)}")
    json.dump(compact(res), open(REPORTS / 'check-order-sampled.json', 'w'), indent=1)
