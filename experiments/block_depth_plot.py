"""Figure for block_depth.py, block_depth_sampled.py and check_order_sampled.py (numbers in reports/).

    MPLBACKEND=Agg python block_depth_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
EPS, K = .003, 9
legend = lambda ax: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=1, frameon=False)
fig, axes = plt.subplots(1, 3, figsize=(21, 8.5), constrained_layout=True)

ax = axes[0]
d = json.load(open(REPORTS / 'block-depth.json'))
x = np.linspace(.15, 4, 200)
ax.plot(x, 1e5 * EPS / (2 * K) * (np.sqrt(x) - 1), color='gray', ls=':', label='linear theory')
for beta, color in ((1.5, 'C1'), (2., 'C3'), (4., 'C5')):
    rows = [t for t in d['threshold'] if t['beta'] == beta]
    ax.plot([t['beta'] * t['fidelity'] for t in rows], [1e5 * t['measured'] for t in rows], 'o', color=color, label=f'exact mean dynamics, gain {beta:g}')
ax.axhline(0, color='k', lw=.5); ax.axvline(1, color='k', lw=.5)
ax.set(xlabel='gain of the writer x fidelity of the loop one level below', ylabel='growth rate of the level-2 tilt (1e-5 per sample)', title='(a) no sampling noise: when the loop of a level opens')
legend(ax)

ax = axes[1]
res = json.load(open(REPORTS / 'block-depth-sampled.json'))
arms = (('teacher strings only', 'teacher strings only', dict(color='C0')), ('own, state kept, gain 1', 'half own samples, gain 1', dict(color='C4')), ('own, state kept, gain 1.2', 'half own samples, gain 1.2', dict(color='C2')),
        ('own, state kept, gain 1.5', 'half own samples, gain 1.5', dict(color='C1')), ('own, state kept, gain 2', 'half own samples, gain 2', dict(color='C3')))
for j, (name, label, kw) in enumerate(arms):
    rs = [r for r in res if (r['name'], r['L'], r['a']) == (name, 3, .01)]
    med = []
    for l in (1, 2, 3):
        ts = [r['solved'][str(l)] for r in rs if str(l) in r['solved']]
        med.append(np.median(ts) if len(ts) >= 4 else np.nan)
        ax.plot([l + .04 * (j - 2)] * len(ts), ts, '.', color=kw['color'], alpha=.35)
    ax.plot((1, 2, 3), med, marker='o', label=f"{label} (all three levels solved in {sum('3' in r['solved'] for r in rs)} of 8 starts)", **kw)
ax.set(yscale='log', xticks=(1, 2, 3), xlabel='level', ylabel='samples until all levels up to this one are solved', title='(b) sampled strings, three levels: teacher strings alone are fastest')
legend(ax)

ax = axes[2]
o = json.load(open(REPORTS / 'check-order-sampled.json'))
for name, label, kw in (('teacher strings only', 'teacher strings only', dict(color='C0')), ('own, state kept, gain 1', 'half own samples, gain 1', dict(color='C4')), ('own, state kept, gain 1.5', 'half own samples, gain 1.5', dict(color='C1')),
                        ('own, state kept, gain 2', 'half own samples, gain 2', dict(color='C3'))):
    med = []
    for m in (3, 4, 5, 6):
        ts = [r['t'] for r in o if (r['name'], r['q'], r['m']) == (name, 2, m) and r['t'] is not None]
        med.append(np.median(ts) if len(ts) >= 4 else np.nan)
        ax.plot([m] * len(ts), ts, '.', color=kw['color'], alpha=.35)
    ax.plot((3, 4, 5, 6), med, marker='o', label=label, **kw)
ax.set(yscale='log', xticks=(3, 4, 5, 6), xlabel='number of bits in the parity check (one check, one group)', ylabel='samples until the check is solved', title='(c) sampled strings, one level: wider checks')
legend(ax)

for ext in ('png', 'svg'):
    fig.savefig(REPORTS / f'block-depth.{ext}')
print(REPORTS / 'block-depth.png')
