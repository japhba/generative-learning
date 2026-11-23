"""The stack of block_depth.py with sampled strings and states (no mean dynamics): batches of teacher strings and own samples, every
state drawn, the rule applied sample by sample at rate eps. Measurements come from the weights through block_depth.Stack.measure().

    python block_depth_sampled.py            (one to three levels, rate 0.003; numbers in reports/block-depth-sampled.json)
    python block_depth_sampled.py --extra    (smaller rates, silent groups; reports/block-depth-sampled-extra.json)
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from block_depth import CAUSES, CLASS, FLOOR, K, Stack, compact, sm

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
BY_CLASS = np.stack([np.nonzero(CLASS == m)[0] for m in range(3)])                               # [3 classes, 3 causes]


def teacher(rng, L, n):
    """n teacher strings as the level-1 causes [n, 3^(L-1)] (a level-1 cause = a valid triple of symbols)."""
    cause = rng.integers(9, size=(n, 1))
    for _ in range(L - 1):                                                                      # every cause fixes the classes of its three children; each child draws one of the 3 causes of its class
        cls = CAUSES[cause]                                                                     # [n, groups, 3]
        cause = BY_CLASS[cls, rng.integers(3, size=cls.shape)].reshape(n, -1)
    return CAUSES[cause]                                                                        # [n, level-1 groups, 3] symbols


def draw(rng, logits):
    return (logits + rng.gumbel(size=logits.shape)).argmax(-1)


def read(rng, m, values, theta=0.):
    """values [n, groups, 3] at the bottom -> list over levels of (child values, states read). A group passes its state upward only if
    the state's probability is above theta; otherwise it is silent (-1): the group above then neither reads nor learns from it."""
    out = []
    for l in range(m.L):
        logR = np.log(m.R[l] + FLOOR)                                                           # [groups, K, 3, v]
        g, i = np.arange(m.n[l])[None, :, None], np.arange(3)[None, None, :]
        logit = (logR[g, :, i, np.maximum(values, 0)] * (values >= 0)[..., None]).sum(2)        # [n, groups, 3, K] summed over the children that fired
        states = draw(rng, logit)
        out.append((values, states))
        if l + 1 < m.L:
            prob = np.take_along_axis(sm(logit), states[..., None], 2)[..., 0]
            values = np.where(prob > theta, states, -1).reshape(len(states), -1, 3)
    return out


def write(rng, m, n):
    """own samples from the top: list over levels of the states that wrote, and the symbols [n, level-1 groups, 3]."""
    wrote, states = [None] * m.L, rng.integers(K, size=(n, 1))
    for l in range(m.L - 1, -1, -1):
        wrote[l] = states
        lift = m.beta * (np.log(m.G[l] + FLOOR) - np.log(m.G[l].mean(1, keepdims=True) + FLOOR))    # [groups, K, 3, v]
        values = draw(rng, lift[np.arange(m.n[l])[None, :], states])                            # [n, groups, 3]
        states = values.reshape(n, -1)
    return wrote, values


def hebb(P, states, values, eps):
    """P [groups, K, 3, v]: for every sample, the active state of a group moves towards the values of its children that fired."""
    v = P.shape[-1]
    s1, fired = np.eye(K)[states], (values >= 0).astype(float)                                  # [n, groups, K], [n, groups, 3]
    count = np.einsum('ngk,ngi->gki', s1, fired)                                                # [groups, K, 3]
    target = np.einsum('ngk,ngiv->gkiv', s1, np.eye(v)[np.maximum(values, 0)] * fired[..., None]) / np.maximum(count, 1)[..., None]
    keep = ((1 - eps) ** count)[..., None]
    return keep * P + (1 - keep) * target


def run(job):
    name, L, a, beta, own, mode, seed, t_max, eps = job[:9]
    theta = job[9] if len(job) > 9 else 0.
    rng = np.random.default_rng([seed, 2])
    m = Stack(np.random.default_rng([seed, 1]), L, a, beta)
    batch, t, solved, curve = 64, 0, {}, []
    n_batches = int(t_max / batch)
    marks = set(np.unique(np.round(np.geomspace(1, n_batches, 50)).astype(int)).tolist()) | {0}
    n_own = 0
    for i in range(n_batches + 1):
        if i % 100 == 0 or i in marks:
            now = m.measure()
            for l in range(1, L + 1):
                if l not in solved and all(now[j]['kl'] < .1 for j in range(1, l + 1)):
                    solved[l] = i * batch
            if i in marks or len(solved) == L:
                curve.append(dict(samples=i * batch, **{f'{k}{l}': now[l][k] for l in now for k in ('kl', 'recover', 'reach_teacher', 'tilt')}))
            if len(solved) == L:
                break
        if n_own < own * (i + 1):                                                               # a batch of own samples
            n_own += 1
            wrote, symbols = write(rng, m, batch)
            back = read(rng, m, symbols, theta)
            for l in range(L):
                m.R[l] = hebb(m.R[l], wrote[l] if mode == 'kept' else back[l][1], back[l][0], eps)
        else:
            back = read(rng, m, teacher(rng, L, batch), theta)
            for l in range(L):
                m.R[l], m.G[l] = hebb(m.R[l], back[l][1], back[l][0], eps), hebb(m.G[l], back[l][1], back[l][0], eps)
    return dict(name=name, L=L, a=a, beta=beta, eps=eps, seed=seed, solved={str(k): v for k, v in solved.items()}, end={l: now[l]['kl'] for l in now}, curve=curve)


ARMS = {'teacher strings only': (1., 0., 'kept'), 'own, state kept, gain 1': (1., .5, 'kept'), 'own, state kept, gain 1.2': (1.2, .5, 'kept'), 'own, state kept, gain 1.5': (1.5, .5, 'kept'),
        'own, state kept, gain 2': (2., .5, 'kept'), 'own, through the string, gain 2': (2., .5, 'string')}

def extra():
    """one level at smaller rates, and two and three levels with silent groups (a state is passed upward only above probability 1/2)."""
    jobs = [(name, 1, .01, *ARMS[name], s, t_max, eps, 0.) for eps, t_max in ((.001, 4e6), (.0003, 14e6)) for name in ARMS for s in range(8)]
    jobs += [(name, L, .01, *ARMS[name], s, t_max, .003, .5) for L, t_max in ((2, 2e6), (3, 4e6)) for name in ARMS for s in range(8)]
    jobs.sort(key=lambda j: -j[7] * j[1])
    with Pool(30) as pool:
        res = pool.map(run, jobs, chunksize=1)
    for r, j in zip(res, jobs):
        r['theta'] = j[9]
    json.dump(compact(res), open(REPORTS / 'block-depth-sampled-extra.json', 'w'), separators=(',', ':'))


if __name__ == '__main__':
    import sys
    if '--extra' in sys.argv:
        extra(); sys.exit()
    eps_list = [float(x) for x in sys.argv[1:]] or [.003]
    jobs = [(name, L, a, *ARMS[name], s, {1: 1.5e6, 2: 3e6, 3: 5e6}[L], eps) for eps in eps_list for L in (3, 2, 1) for name in ARMS for a in (.1, .01) for s in range(8)]
    with Pool(30) as pool:
        res = pool.map(run, jobs, chunksize=1)
    for eps in eps_list:
        print(f'rate {eps}: samples (thousands) until every level up to l is within 0.1 nats per check; median of the starts that get there (number of 8)')
        for a in (.1, .01):
            for L in (1, 2, 3):
                for name in ARMS:
                    rs = [r for r in res if (r['name'], r['L'], r['a'], r['eps']) == (name, L, a, eps)]
                    cells = []
                    for l in range(1, L + 1):
                        d = [r['solved'][str(l)] for r in rs if str(l) in r['solved']]
                        cells.append(f"level {l}: {np.median(d) / 1000:6.0f}k ({len(d)})" if d else f"level {l}:       - (0)")
                    print(f"  start {a:4.2f}, {L} level(s), {name:34s} " + '   '.join(cells))
    json.dump(compact(res), open(REPORTS / f"block-depth-sampled{'-' + '-'.join(sys.argv[1:]) if sys.argv[1:] else ''}.json", 'w'), separators=(',', ':'))
