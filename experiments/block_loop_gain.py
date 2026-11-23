"""Read-write loop with gain on the block world (exact mean dynamics of block_flywheel.py): the writer draws more sharply than its
weights say (gain beta), reader and writer start weak on the same random tilt.

With the writing state kept for the reader ('sleep'), the reader's tilt relaxes to beta times the writer's at rate eps share / K and the
writer's tilt to the reader's at rate eps (1 - share) / K (teacher strings read by a diffuse reader), so the shared tilt grows at
rate (eps / 2K) (sqrt(beta) - 1) for share 1/2. Read through the string only ('strict'), own samples act at third order.
Printed: 1. that rate, predicted and measured; 2. samples until KL < 0.1 against the start amplitude, for teacher strings only and for
half own samples (sleep or strict, own or twin) at several gains; 3. the same with two checks and two groups.

4. How long the writing state has to be kept (python block_loop_gain.py --switch adds this part to the stored numbers).

    python block_loop_gain.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from block_flywheel import EPS, Learner, World, measure, step

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
K = 9
norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-12))
ARMS = {'teacher strings only': dict(own=0., mode='strict', beta=1.), 'own, state kept, gain 1': dict(own=.5, mode='sleep', beta=1.), 'own, state kept, gain 1.5': dict(own=.5, mode='sleep', beta=1.5),
        'own, state kept, gain 2': dict(own=.5, mode='sleep', beta=2.), 'own, state kept, gain 4': dict(own=.5, mode='sleep', beta=4.), 'own, through the string, gain 2': dict(own=.5, mode='strict', beta=2.),
        'own, through the string, gain 4': dict(own=.5, mode='strict', beta=4.), "twin's, through the string, gain 4": dict(own=.5, mode='twin', beta=4.)}


def go(job):
    name, C, a, seed, t_max = job
    arm, world = ARMS[name], World(C)
    m = Learner(np.random.default_rng([seed, 1]), world, K, a, a, beta=arm['beta'])
    twin = Learner(np.random.default_rng([seed, 2]), world, K, a, a, beta=arm['beta']) if arm['mode'] == 'twin' else None
    eta, t, curve = .3, 0, []
    while t <= t_max:
        now = measure(m); curve.append(dict(samples=t, kl=now['kl'], recover=now['recover'], reach_teacher=now['reach_teacher']))
        if now['kl'] < .1:
            return dict(name=name, C=C, a=a, seed=seed, t=t, kl=now['kl'], curve=curve)
        for _ in range(20):
            moves = [(m, step(m, eta, arm['own'], arm['mode'], twin))] + ([(twin, step(twin, eta, arm['own'], 'strict'))] if twin else [])
            for net, (dR, dG, _) in moves:
                net.R, net.G = norm(net.R + eta * dR), norm(net.G + eta * dG)
        t += 20 * eta / EPS
    return dict(name=name, C=C, a=a, seed=seed, t=None, kl=now['kl'], curve=curve)


def growth(beta, a=1e-6, n=3000, eta=.03):
    m = Learner(np.random.default_rng(0), World(1), K, a, a, beta=beta)
    spread = lambda: float(np.sqrt(np.mean((np.log(m.R[0]) - np.log(m.R[0]).mean(0)) ** 2)))
    t, v = [], []
    for i in range(n):
        dR, dG, _ = step(m, eta, .5, 'sleep'); m.R, m.G = norm(m.R + eta * dR), norm(m.G + eta * dG)
        if i % 50 == 0:
            t.append(i * eta / EPS); v.append(spread())
    k = len(t) // 3
    return float(np.polyfit(t[-k:], np.log(v[-k:]), 1)[0])


def switch(job):
    """the writing state is kept until reading an own sample gives the writing state with probability theta; after that own samples reach
    the reader through the string only ('string'), stop ('teacher') or go on as before ('kept')."""
    C, a, beta, theta, after, seed, t_max = job
    m = Learner(np.random.default_rng([seed, 1]), World(C), K, a, a, beta=beta)
    eta, t, phase, t_switch = .3, 0, 0, None
    while t <= t_max:
        now = measure(m)
        if phase == 0 and now['recover'] >= theta:
            phase, t_switch = 1, t
        if now['kl'] < .1:
            return dict(C=C, theta=theta, after=after, seed=seed, t=t, t_switch=t_switch, kl=now['kl'])
        own, mode = (.5, 'sleep') if phase == 0 else dict(string=(.5, 'strict'), teacher=(0., 'strict'), kept=(.5, 'sleep'))[after]
        for _ in range(20):
            dR, dG, _ = step(m, eta, own, mode); m.R, m.G = norm(m.R + eta * dR), norm(m.G + eta * dG)
        t += 20 * eta / EPS
    return dict(C=C, theta=theta, after=after, seed=seed, t=None, t_switch=t_switch, kl=now['kl'])


def part4():
    jobs = [(1, .03, 2., th, after, s, 1.5e6) for th in (.15, .2, .5, .9) for after in ('kept', 'string', 'teacher') for s in range(8)] + [(2, .1, 2., th, after, s, 6e5) for th in (.2, .5, .9) for after in ('kept', 'string', 'teacher') for s in range(8)]
    with Pool(30) as pool:
        res = pool.map(switch, jobs)
    print('4. gain 2, writing state kept until reading recovers it with probability theta (chance 0.11), then ...: starts solved of 8, median samples (thousands)')
    for C in (1, 2):
        for th in sorted({r['theta'] for r in res if r['C'] == C}):
            cells = []
            for after in ('kept', 'string', 'teacher'):
                d = [r['t'] for r in res if (r['C'], r['theta'], r['after']) == (C, th, after) and r['t'] is not None]
                cells.append(f"{after}: {len(d)}/8, {np.median(d) / 1000 if d else float('nan'):4.0f}k")
            print(f"  {C} check(s), theta {th:4.2f}:   " + '   '.join(cells))
    return res


if __name__ == '__main__':
    import sys
    if '--switch' in sys.argv:
        d = json.load(open(REPORTS / 'block-loop-gain.json')); d['switch'] = part4(); json.dump(d, open(REPORTS / 'block-loop-gain.json', 'w')); sys.exit()
    print('1. growth rate of the shared tilt per sample (x 1e-5), writing state kept, half own samples')
    rates = {}
    for beta in (1., 1.5, 2., 4.):
        rates[beta] = dict(predicted=EPS / (2 * K) * (np.sqrt(beta) - 1), measured=growth(beta))
        print(f"  gain {beta:3.1f}: predicted {1e5 * rates[beta]['predicted']:6.2f}   measured {1e5 * rates[beta]['measured']:6.2f}")
    amps = (.3, .1, .03, .01, .003)
    jobs = [(name, 1, a, s, 3e6) for name in ARMS for a in amps for s in range(8)] + [(name, 2, .1, s, 6e5) for name in ARMS for s in range(8)]
    with Pool(30) as pool:
        res = pool.map(go, jobs)
    print('2. one check: samples until KL < 0.1 in thousands (starts that get there of 8), by start amplitude ' + ' / '.join(str(a) for a in amps))
    for name in ARMS:
        cells = []
        for a in amps:
            d = [r['t'] for r in res if (r['name'], r['C'], r['a']) == (name, 1, a) and r['t'] is not None]
            cells.append(f"{np.median(d) / 1000:6.0f} ({len(d)})" if d else '     - (0)')
        print(f"  {name:36s} " + '  '.join(cells))
    print('3. two checks, two groups, start 0.1, 600,000 samples: starts solved of 8, median KL at the end')
    for name in ARMS:
        rs = [r for r in res if (r['name'], r['C']) == (name, 2)]
        print(f"  {name:36s} {sum(r['t'] is not None for r in rs)}/8   {np.median([r['kl'] for r in rs]):.2f}")
    json.dump(dict(rates={str(k): v for k, v in rates.items()}, runs=res), open(REPORTS / 'block-loop-gain.json', 'w'))
