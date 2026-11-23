"""Share of own samples at a fixed total (exact mean dynamics, one check, block_flywheel.py): KL(teacher || own samples) against the
number of samples for shares 0 ... 0.9 of own samples (writing state kept, writer gain beta), and how far own samples are from
teacher strings in single symbols and symbol pairs on the way.

    python block_loop_share.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from block_flywheel import EPS, Learner, World, measure, step

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
K, SHARES, T_MAX, EVERY = 9, (0., .1, .2, .3, .4, .5, .6, .7, .8, .9), 400000, 2000
norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-12))


compact = lambda o: [compact(x) for x in o] if isinstance(o, list) else {k: compact(v) for k, v in o.items()} if isinstance(o, dict) else float(f'{o:.4g}') if isinstance(o, float) else o


def go(job):
    share, beta, a, seed = job
    m = Learner(np.random.default_rng([seed, 1]), World(1), K, a, a, beta=beta)
    eta, out = .3, []
    for i in range(int(T_MAX / EVERY) + 1):
        now = measure(m); out.append(dict(samples=i * EVERY, kl=now['kl'], single=now['single'], pairs=now['pairs'], valid=now['valid'], recover=now['recover']))
        for _ in range(int(EVERY / (eta / EPS))):
            dR, dG, _ = step(m, eta, share, 'sleep'); m.R, m.G = norm(m.R + eta * dR), norm(m.G + eta * dG)
    return dict(share=share, beta=beta, a=a, seed=seed, curve=out)


if __name__ == '__main__':
    jobs = [(s, beta, .03, seed) for beta in (1.5, 2.) for s in SHARES for seed in range(8)]
    with Pool(8) as pool:
        res = pool.map(go, jobs)
    for beta in (1.5, 2.):
        print(f'gain {beta}, reader and writer start at 0.03: samples (thousands) until the median KL < 0.1, by share of own samples; largest deviation of own samples from uniform pairs on the way')
        for s in SHARES:
            rs = [r for r in res if r['share'] == s and r['beta'] == beta]
            kl = np.median([[c['kl'] for c in r['curve']] for r in rs], 0)
            first = next((i * EVERY for i, v in enumerate(kl) if v < .1), None)
            print(f"  share {s:.1f}: {first / 1000 if first is not None else float('nan'):5.0f}k   pairs off by at most {max(np.median([[c['pairs'] for c in r['curve']] for r in rs], 0)):.3f}, single symbols {max(np.median([[c['single'] for c in r['curve']] for r in rs], 0)):.3f}")
    json.dump(compact(res), open(REPORTS / 'block-loop-share.json', 'w'))
