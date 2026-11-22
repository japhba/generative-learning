"""Figure for block_unlock_theory.py (numbers in reports/block-unlock-theory.json).

    python block_unlock_theory.py && MPLBACKEND=Agg python block_unlock_theory_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
d = json.load(open(REPORTS / 'block-unlock-theory.json'))
legend = lambda ax: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=1, frameon=False)
fig, axes = plt.subplots(1, 3, figsize=(21, 8.5), constrained_layout=True)

ax = axes[0]
for key, label, color in (('linear', 'diffuse reader, own samples of writers holding 3 to 9 strings', 'C3'), ('cascade', 'teacher strings only, 0 to 6 reader states already sharp on valid strings', 'C0')):
    rows = [v for k, v in d[key].items() if 'not valid' not in k]
    ax.plot([1e4 * v['predicted'] for v in rows], [1e4 * v['measured'] for v in rows], 'o', color=color, label=label)
ax.plot([0, 7], [0, 7], color='gray', ls=':')
ax.set(xlabel='growth rate of the reader, linear theory (1e-4 per sample)', ylabel='measured in the exact dynamics (1e-4 per sample)', title='(a) linear theory against the exact dynamics')
legend(ax)

ax = axes[1]
times = d['times']
amps = sorted({v['a'] for v in times.values() if v['arm'] == 'teacher'})
def line(arm, block, label, **kw):
    rows = [next(v for v in times.values() if (v['arm'], v['a'], v['block']) == (arm, a, block)) for a in amps]
    ax.plot(amps, [np.median([t for t in r['times'] if t is not None]) for r in rows], marker='o', label=label, **kw)
    for r in rows:
        ts = [t for t in r['times'] if t is not None]
        ax.plot([r['a']] * len(ts), ts, '.', color=kw['color'], alpha=.35)
line('teacher', 0, 'teacher strings only', color='C0')
for block, color in ((2000, 'C1'), (5000, 'C3'), (10000, 'C4')):
    line('own_valid', block, f'first {block:,} own samples of a writer holding 3 valid strings', color=color)
line('own_random', 10000, 'first 10,000 own samples of a writer holding 3 random strings', color='C2', ls='--')
ax.axhline(times['sharp a=0.0 block=0']['median'], color='k', ls=':', label='reader and writer sharp from the start')
ax.set(xscale='log', yscale='log', xlabel='start amplitude of the reader', ylabel='samples until KL(teacher || own samples) < 0.1 nats', title='(b) time to the solution (starts that get there)')
legend(ax)

ax = axes[2]
blocks = sorted({v['block'] for v in times.values() if v['arm'] == 'own_valid'})
for a, color in zip(amps, ('C0', 'C1', 'C2', 'C3')):
    for arm, ls in (('own_valid', '-'), ('own_random', '--')):
        ax.plot(blocks, [next(v for v in times.values() if (v['arm'], v['a'], v['block']) == (arm, a, b))['done'] / 8 for b in blocks], marker='o', color=color, ls=ls, label=f"reader start {a}, writer holds 3 {'valid' if arm == 'own_valid' else 'random'} strings")
ax.set(xscale='log', xlabel='number of own samples before the teacher strings', ylabel='share of starts that reach the solution', title='(c) too many own samples leave every reader state on the three strings')
legend(ax)

for ext in ('png', 'svg'):
    fig.savefig(REPORTS / f'block-unlock-theory.{ext}')
print(REPORTS / 'block-unlock-theory.png')
