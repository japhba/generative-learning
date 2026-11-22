"""Three time courses of the order-three network runs, read against the README: how much of a teacher string the deep groups read,
how far own samples are from teacher strings in shallow and in check statistics, and what happens when the deep side of an own
sample is the reading of the own string instead of the units that wrote it. Uses the runs of order3_ladder.py and order3_net.py.

    MPLBACKEND=Agg python order3_spec_check_plot.py /var/tmp/jbauer/generative_order3
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
root = Path(sys.argv[1])
load = lambda d: json.load(open(root / d / 'results.json'))['results']
ladder, net = load('ladder_a') + load('ladder_b') + load('ladder_c'), load('net_core')


def draw(ax, rs, name, f, label, **kw):
    rs = [r for r in rs if r['name'] == name]
    v = np.array([[f(c) for c in r['curve']] for r in rs], float)
    x = np.maximum([c['samples'] for c in rs[0]['curve']], 32)
    line, = ax.plot(x, np.median(v, 0), label=label, **kw)
    ax.fill_between(x, np.quantile(v, .25, 0), np.quantile(v, .75, 0), color=line.get_color(), alpha=.15, lw=0)


legend = lambda ax: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=1, frameon=False)
fig, axes = plt.subplots(1, 3, figsize=(21, 8.5), constrained_layout=True)

ax = axes[0]
for name, label, color in (('watching', 'teacher strings only', 'C0'), ('ours_each', 'half own samples, started at each level in turn', 'C3'), ('ws_each', 'the same in classic wake-sleep', 'C4')):
    draw(ax, ladder, name, lambda c: c['read_mid_cov'], f'{label}: mid bits', color=color)
    draw(ax, ladder, name, lambda c: c['read_top_cov'], f'{label}: top bits', color=color, ls='--')
ax.set(xscale='log', xlim=(1e3, None), xlabel='samples (teacher + own)', ylabel='share of bits one deep group reads off a teacher string', title='(a) two learned levels: reading of teacher strings')
legend(ax)

ax = axes[1]
for which, color in (('single tokens', 'C0'), ('token pairs', 'C1'), ('level-1 checks', 'C2'), ('level-2 checks', 'C3')):
    draw(ax, ladder, 'ours_each', lambda c: c['mmd'][which], f'{which}, half own samples', color=color)
    draw(ax, ladder, 'watching', lambda c: c['mmd'][which], f'{which}, teacher strings only', color=color, ls='--')
ax.axhspan(-2, 2, color='gray', alpha=.15, lw=0)
ax.set(xscale='log', xlim=(1e3, None), xlabel='samples (teacher + own)', ylabel='MMD between own samples and teacher strings\n(in standard deviations between two teacher sets)', title='(b) two learned levels: own samples against teacher strings')
ax.set_yscale('symlog', linthresh=10); ax.set_ylim(-3, None)
legend(ax)

ax = axes[2]
for name, label, kw in (('watching', 'teacher strings only', dict(color='C0')), ('mix', 'half own samples, deep side = the units that wrote', dict(color='C3')),
                        ('mix_read', 'half own samples, deep side = reading of the own string', dict(color='C1', ls='--'))):
    draw(ax, net, name, lambda c: c['kl'], label, **kw)
ax.set(xscale='log', yscale='log', xlim=(1e3, None), xlabel='samples (teacher + own)', ylabel='KL(teacher || own samples), nats', title='(c) one learned level: where the deep side of an own sample comes from')
legend(ax)

for ext in ('png', 'svg'):
    fig.savefig(REPORTS / f'order3-spec-check.{ext}')
print(REPORTS / 'order3-spec-check.png')
