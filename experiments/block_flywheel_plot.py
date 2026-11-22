"""Figure and numbers for block_flywheel.py: exact mean dynamics of one deep group on one check of the block world.

    MPLBACKEND=Agg python block_flywheel_plot.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from block_flywheel import run

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
WORLDS = 8
ARMS = dict(teacher={}, strict=dict(own=.5), sleep=dict(own=.5, mode='sleep'), twin=dict(own=.5, mode='twin'))
SETTINGS = {'posterior, reader weak, writer sharp': dict(a=.1, b=3.), 'posterior, both weak': dict(a=.1), 'posterior, both sharp': dict(a=3.),
            'covariance rule, reader weak, writer sharp': dict(a=.1, b=3., rule='cov'), 'match, both sharp': dict(a=3., response='match', samples=1600000, eta=1.)}


def job(args):
    setting, arm, seed = args
    return setting, arm, run(C=1, seed=seed, **SETTINGS[setting], **ARMS[arm])


if __name__ == '__main__':
    with Pool(28) as pool:
        out = pool.map(job, [(s, a, w) for s in SETTINGS for a in ARMS for w in range(WORLDS)])
    curves = {(s, a): [c for (s2, a2, c) in out if (s2, a2) == (s, a)] for s in SETTINGS for a in ARMS}
    arr = lambda s, a, k: np.array([[p[k] for p in c] for c in curves[s, a]])
    x = lambda s, a: np.maximum([p['samples'] for p in curves[s, a][0]], 100)
    first = lambda c: next((p['samples'] for p in c if p['kl'] < .1), None)

    def draw(ax, s, a, k, label, **kw):
        v = arr(s, a, k)
        line, = ax.plot(x(s, a), np.median(v, 0), label=label, **kw)
        ax.fill_between(x(s, a), np.quantile(v, .25, 0), np.quantile(v, .75, 0), color=line.get_color(), alpha=.15, lw=0)

    legend = lambda ax: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=1, frameon=False)
    names = dict(teacher='teacher strings only', strict='half own samples, read through the string', sleep='half own samples, writing state handed to the reader', twin="half a twin's samples, read through the string")
    colors = dict(teacher='C0', strict='C3', sleep='C4', twin='C2')
    fig, axes = plt.subplots(1, 3, figsize=(21, 8.5), constrained_layout=True)

    s = 'posterior, reader weak, writer sharp'
    for a in ARMS:
        draw(axes[0], s, a, 'kl', names[a], color=colors[a], ls='--' if a == 'sleep' else '-')
    axes[0].set(xscale='log', yscale='log', xlabel='samples (teacher + own)', ylabel='KL(teacher || own samples), nats', title='(a) weak reader, sharp writer on the same random code')
    legend(axes[0])

    for a in ('teacher', 'sleep'):
        draw(axes[1], s, a, 'reach_teacher', f'{names[a]}: teacher strings', color=colors[a])
        draw(axes[1], s, a, 'reach_own', f'{names[a]}: own samples', color=colors[a], ls=':')
    draw(axes[1], s, 'sleep', 'modes_valid', 'writing state handed to the reader: share of states whose string is valid', color='k', ls='--')
    axes[1].set(xscale='log', xlabel='samples (teacher + own)', ylabel='activity of the most active deep state\n(dashed: share of valid states)', title='(b) same start: how far strings reach the deep group')
    legend(axes[1])

    s = 'match, both sharp'
    for a in ('teacher', 'strict', 'sleep'):
        draw(axes[2], s, a, 'valid', f'{names[a]}: valid', color=colors[a])
        draw(axes[2], s, a, 'covered', f'{names[a]}: valid strings written', color=colors[a], ls=':')
    axes[2].set(xscale='log', xlabel='samples (teacher + own)', ylabel='share', title='(c) deep states that fire by absolute match, sharp shared start')
    legend(axes[2])

    for ext in ('png', 'svg'):
        fig.savefig(REPORTS / f'block-flywheel.{ext}')
    numbers = {s: {a: dict(first_below_01=[first(c) for c in curves[s, a]], end={k: float(np.median([c[-1][k] for c in curves[s, a]])) for k in curves[s, a][0][-1] if k != 'samples'},
                           start={k: float(np.median([c[0][k] for c in curves[s, a]])) for k in ('reach_teacher', 'reach_own', 'pairs')}) for a in ARMS} for s in SETTINGS}
    json.dump(numbers, open(REPORTS / 'block-flywheel.json', 'w'), indent=1)
    for s in SETTINGS:
        for a in ARMS:
            n = numbers[s][a]
            done = [f for f in n['first_below_01'] if f is not None]
            print(f"{s:44s} {a:8s} done {len(done)}/{WORLDS}  median samples to KL < 0.1: {np.median(done) / 1000 if done else float('nan'):6.0f} k   end: KL {n['end']['kl']:.2f} valid {n['end']['valid']:.2f} covered {n['end']['covered']:.2f}   "
                  f"start reach teacher {n['start']['reach_teacher']:.2f} own {n['start']['reach_own']:.2f}")
    print(REPORTS / 'block-flywheel.png')
