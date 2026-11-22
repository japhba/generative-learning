"""Figure and summary numbers for order3_ladder.py (two learned levels) and the one-level control of order3_deep.py.

    MPLBACKEND=Agg python order3_ladder_plot.py /var/tmp/jbauer/generative_order3
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
res = load('ladder_a') + load('ladder_b') + load('ladder_c')
one = load('deep_draws8')                                                                        # one learned level, two-level world
stack = {d: load(d) for d in ('deep_stack', 'deep_stack_staged') if (root / d / 'results.json').exists()}
total = lambda c: c['kl1'] + c['kl2']


def curve(name, f, rs=res):
    rs = [r for r in rs if r['name'] == name]
    n = min(len(r['curve']) for r in rs)
    v = np.array([[f(c) if callable(f) else c[f] for c in r['curve'][:n]] for r in rs], float)
    return np.array([c['samples'] for c in rs[0]['curve'][:n]]), np.median(v, 0), np.quantile(v, .25, 0), np.quantile(v, .75, 0), v


def draw(ax, name, f, label, rs=res, **kw):
    x, m, lo, hi, _ = curve(name, f, rs)
    line, = ax.plot(np.maximum(x, 32), m, label=label, **kw)
    ax.fill_between(np.maximum(x, 32), lo, hi, color=line.get_color(), alpha=.15, lw=0)


end = lambda name, f, rs=res: float(curve(name, f, rs)[1][-1])
legend = lambda ax, **kw: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=1, frameon=False, **kw)
fig, axes = plt.subplots(2, 3, figsize=(21, 15), constrained_layout=True)

ax = axes[0, 0]
for name, label, kw in (('shallow_watching', 'teacher strings only', dict(color='C0')), ('shallow_first_ours', 'then half own samples, deep units drawn independently', dict(color='C3')),
                        ('shallow_first_ours_read', 'then half own samples, deep units set by reading a fresh teacher string', dict(color='C1', ls='--'))):
    draw(ax, name, 'kl1', label, rs=one, **kw)
ax.axvline(65536, color='gray', ls=':')
ax.set(xscale='log', yscale='log', xlabel='samples (teacher + own)', ylabel='KL(teacher || own samples), nats', title='(a) one learned level: what the deep units of an own sample are')
legend(ax)

ax = axes[0, 1]
for name, label, kw in (('watching', 'teacher strings only', dict(color='C0')), ('ours', 'half own samples, written from the top', dict(color='C3')),
                        ('ws', 'classic wake-sleep (own samples written from the top)', dict(color='C4', ls='--')), ('ours_late131072', 'written from the top, upper level starts after 131,072 samples', dict(color='C2'))):
    draw(ax, name, total, label, **kw)
ax.set(xscale='log', yscale='log', xlabel='samples (teacher + own)', ylabel='level-1 + level-2 KL, nats', title='(b) two learned levels: own samples written from the top')
legend(ax)

ax = axes[0, 2]
for name, label, kw in (('watching', 'teacher strings only', dict(color='C0')), ('ours_each', 'own samples started at each level in turn', dict(color='C3')),
                        ('ws_each', 'the same in classic wake-sleep (teacher strings teach the writer only)', dict(color='C4', ls='--')),
                        ('ours_mid', 'own samples started at the lower level only', dict(color='C1')), ('ours_each_twin', "a twin's samples, started at each level in turn", dict(color='C2', ls=':'))):
    draw(ax, name, total, label, **kw)
ax.set(xscale='log', yscale='log', xlabel='samples (teacher + own)', ylabel='level-1 + level-2 KL, nats', title='(c) two learned levels: own samples started at each level')
legend(ax)

ax = axes[1, 0]
arms = (('watching', 'teacher\nstrings only'), ('first65536_ours', 'own samples\nfrom the top'), ('first65536_ours_mid', 'own samples\nfrom the\nlower level'), ('first65536_ours_each', 'from each\nlevel; teacher\nstrings teach\nthe reader too'),
        ('first65536_ws_each', 'from each\nlevel; only they\nteach the\nreader'))
keys = (('read_mid_dup', 'mid bits read by two groups', 'C1'), ('read_top_dup', 'top bits read by two groups', 'C4'))
for j, (key, label, color) in enumerate(keys):
    for i, (name, _) in enumerate(arms):
        v = curve(name, key)[4][:, -1]
        ax.bar(i + (j - .5) * .38, np.median(v), .36, color=color, alpha=.6, label=label if i == 0 else None)
        ax.scatter(np.full(len(v), i + (j - .5) * .38) + np.linspace(-.1, .1, len(v)), v, color='k', alpha=.5, marker='.')
ax.set_xticks(range(len(arms)), [l for _, l in arms])
ax.set(ylabel='share at the end', title='(d) after 65,536 teacher strings on both levels: which duplicates go')
ax.legend(frameon=False)

ax = axes[1, 1]
starts = (0, 32768, 65536, 131072, 262144)
for tag, label, color in (('ours', 'teacher strings teach reader and writer', 'C3'), ('ws', 'classic wake-sleep', 'C4')):
    v = np.array([curve(tag if s == 0 else f'{tag}_late{s}', total)[4][:, -1] for s in starts])    # [start, world]
    ax.plot(range(len(starts)), np.median(v, 1), color=color, label=label)
    for i in range(len(starts)):
        ax.scatter(np.full(v.shape[1], i) + np.linspace(-.12, .12, v.shape[1]) + (.02 if tag == 'ws' else -.02), v[i], color=color, alpha=.5, marker='.')
ax.axhline(end('watching', total), color='C0', ls=':', label='teacher strings only')
ax.set_xticks(range(len(starts)), ['0'] + [f'{s:,}' for s in starts[1:]])
ax.set(yscale='log', xlabel='samples before the upper level starts to learn', ylabel='level-1 + level-2 KL at the end, nats', title='(e) own samples written from the top: when the upper level may start')
legend(ax)

ax = axes[1, 2]
for name, label, color in (('ours_each', 'teacher strings teach reader and writer', 'C3'), ('ws_each', 'classic wake-sleep', 'C4')):
    draw(ax, name, 'read_mid_cov', f'{label}: mid bits read', color=color)
    draw(ax, name, 'read_top_cov', f'{label}: top bits read', color=color, ls='--')
ax.set(xscale='log', xlim=(4000, None), xlabel='samples (teacher + own)', ylabel='share of bits a deep group reads off a teacher string', title='(f) own samples started at each level: reading of the two levels')
legend(ax)

fig.savefig(REPORTS / 'order3-ladder.png', dpi=110)
fig.savefig(REPORTS / 'order3-ladder.svg')

KEYS = ('kl1', 'kl2', 'valid1', 'checks2', 'read_mid_cov', 'read_mid_dup', 'read_top_cov', 'read_top_dup', 'trip_mid', 'trip_top')
summary = dict(start={k: float(curve('watching', k)[1][0]) for k in ('kl1', 'kl2')}, arms={}, one_level={}, stack={})
for name in sorted({r['name'] for r in res}):
    x, m = curve(name, total)[:2]
    summary['arms'][name] = dict(worlds=sum(r['name'] == name for r in res), end={k: end(name, k) for k in KEYS}, end_mmd={k: float(np.median([r['curve'][-1]['mmd'][k] for r in res if r['name'] == name])) for k in res[0]['curve'][0]['mmd']},
                                 total_per_world=[round(total(r['curve'][-1]), 2) for r in res if r['name'] == name], first_below_1=int(x[np.argmax(m < 1)]) if (m < 1).any() else None, samples=x.tolist(), total=np.round(m, 3).tolist(),
                                 read_mid=np.round(curve(name, 'read_mid_cov')[1], 3).tolist(), read_top=np.round(curve(name, 'read_top_cov')[1], 3).tolist())
for name in sorted({r['name'] for r in one}):
    x, m = curve(name, 'kl1', one)[:2]
    summary['one_level'][name] = dict(worlds=sum(r['name'] == name for r in one), end={k: end(name, k, one) for k in ('kl1', 'read1_cov', 'read1_dup')}, first_below_1=int(x[np.argmax(m < 1)]) if (m < 1).any() else None, samples=x.tolist(), kl1=np.round(m, 3).tolist())
for d, rs in stack.items():
    for name in sorted({r['name'] for r in rs}):
        summary['stack'][f'{d}/{name}'] = dict(worlds=sum(r['name'] == name for r in rs), end={k: end(name, k, rs) for k in ('kl1', 'kl2', 'read1_cov', 'read2_cov', 'read2_part', 'read3_cov', 'usage2')},
                                               end_mmd_single=float(np.median([r['curve'][-1]['mmd']['single tokens'] for r in rs if r['name'] == name])))
json.dump(summary, open(REPORTS / 'order3-ladder.json', 'w'), indent=1)
for name, v in summary['arms'].items():
    print(f"{name:24s} " + '  '.join(f'{k} {v["end"][k]:.2f}' for k in KEYS) + f"  below 1 nat from {v['first_below_1']}")
for name, v in summary['one_level'].items():
    print(f"one level: {name:24s} " + '  '.join(f'{k} {x:.2f}' for k, x in v['end'].items()) + f"  below 1 nat from {v['first_below_1']}")
for name, v in summary['stack'].items():
    print(f"stack: {name:36s} " + '  '.join(f'{k} {x:.2f}' for k, x in v['end'].items()) + f"  single-token z {v['end_mmd_single']:.0f}")
