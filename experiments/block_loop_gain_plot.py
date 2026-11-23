"""Figure for block_loop_gain.py (numbers in reports/block-loop-gain.json).

    MPLBACKEND=Agg python block_loop_gain_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
d = json.load(open(REPORTS / 'block-loop-gain.json'))
runs = [r for r in d['runs'] if r['C'] == 1]
amps = sorted({r['a'] for r in runs})
legend = lambda ax: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=1, frameon=False)
fig, axes = plt.subplots(1, 3, figsize=(21, 8.5), constrained_layout=True)

ax = axes[0]
gains = sorted(float(k) for k in d['rates'])
ax.plot(gains, [1e5 * d['rates'][str(g)]['measured'] for g in gains], 'o', color='C3', label='exact dynamics')
g = np.linspace(1, 4, 100)
ax.plot(g, 1e5 * .003 / 18 * (np.sqrt(g) - 1), color='gray', ls=':', label='linear theory')
ax.set(xlabel='gain of the writer', ylabel='growth rate of the shared tilt (1e-5 per sample)', title='(a) read-write loop at a weak start, writing state kept')
legend(ax)

ax = axes[1]
styles = {'teacher strings only': dict(color='C0'), 'own, state kept, gain 1': dict(color='C4', ls=':'), 'own, state kept, gain 1.5': dict(color='C1'), 'own, state kept, gain 2': dict(color='C3'), 'own, state kept, gain 4': dict(color='C5'),
          'own, through the string, gain 2': dict(color='C3', ls='--'), "twin's, through the string, gain 4": dict(color='C2', ls='--')}
labels = {'teacher strings only': 'teacher strings only', 'own, state kept, gain 1': 'half own samples, writing state kept, gain 1', 'own, state kept, gain 1.5': 'the same, gain 1.5', 'own, state kept, gain 2': 'the same, gain 2',
          'own, state kept, gain 4': 'the same, gain 4', 'own, through the string, gain 2': 'half own samples through the string only, gain 2', "twin's, through the string, gain 4": "half a twin's samples through the string, gain 4"}
for name, kw in styles.items():
    med = []
    for a in amps:
        ts = [r['t'] for r in runs if r['name'] == name and r['a'] == a and r['t'] is not None]
        med.append(np.median(ts) if len(ts) >= 4 else np.nan)
        ax.plot([a] * len(ts), ts, '.', color=kw['color'], alpha=.3)
    ax.plot(amps, med, marker='o', label=labels[name], **kw)
ax.set(xscale='log', yscale='log', xlabel='start amplitude of reader and writer', ylabel='samples until KL(teacher || own samples) < 0.1 nats', title='(b) time to the solution')
legend(ax)

ax = axes[2]
for name, color in (('teacher strings only', 'C0'), ('own, state kept, gain 2', 'C3'), ('own, through the string, gain 2', 'C1')):
    rs = [r for r in runs if r['name'] == name and r['a'] == .03]
    n = min(len(r['curve']) for r in rs)
    t = np.maximum([c['samples'] for c in rs[0]['curve'][:n]], 1000)
    for key, ls in (('recover', '-'), ('reach_teacher', ':')):
        v = np.array([[c[key] for c in r['curve'][:n]] for r in rs])
        full = {'own, state kept, gain 2': 'half own samples, writing state kept, gain 2'}.get(name, labels[name])
        ax.plot(t, np.median(v, 0), color=color, ls=ls, label=f"{full}: {'own sample read as the state that wrote it' if key == 'recover' else 'most active state on a teacher string'}")
ax.set(xscale='log', xlabel='samples (teacher + own)', ylabel='probability', title='(c) start amplitude 0.03: what reading gives')
legend(ax)

for ext in ('png', 'svg'):
    fig.savefig(REPORTS / f'block-loop-gain.{ext}')
print(REPORTS / 'block-loop-gain.png')
