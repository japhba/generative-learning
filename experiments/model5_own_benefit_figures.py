"""Figures and numbers for the model 5 follow-up (model5_own_benefit.py), from /var/tmp/jbauer/model5_lateral_verdict/own_benefit_*.npz and efficiency.npz.
Writes <out>/model5-own-block.svg/.png, <out>/model5-own-depth.svg/.png and <out>/own_benefit_numbers.json.   Usage: MPLBACKEND=Agg python model5_own_benefit_figures.py <out>"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

D = Path('/var/tmp/jbauer/model5_lateral_verdict'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#898781', '#0b0b0b'
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']); W, H = plt.rcParams['figure.figsize']; nums = {}


def band(ax, x, y, color, label, **kw):                                                                                            # median over worlds (axis 1) with the interquartile range
    q1, med, q3 = np.percentile(y, [25, 50, 75], 1); ax.plot(x, med, color=color, label=label, **kw); ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0)


def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', transparent=True); fig.savefig(OUT / f'{name}.png', dpi=150, facecolor='white'); plt.close(fig); print(OUT / f'{name}.png')


def med(x): return float(np.median(x))


# 1. one block: what the classes are worth, and what each verdict gets of it
b = np.load(D / 'own_benefit_bound.npz'); s = np.load(D / 'own_benefit_sparse.npz'); e = np.load(D / 'efficiency.npz'); fig, ax = plt.subplots(constrained_layout=True); ARM = 'eta1'
band(ax, b['n'], b['table'], GREY, 'counted table', ls='--', marker='o'); band(ax, e['n'], np.minimum(e['practice0.8'], 20), ORANGE, 'practice, verdict on the written pair', marker='o')
band(ax, s['n'], np.minimum(s[ARM][:, :, -1, 0], 20), BLUE, 'practice, pooled verdict', marker='o'); band(ax, b['n'], b['found'], AQUA, 'clustering of the pair counts', marker='o')
band(ax, b['n'], b['oracle'], INK, 'table over the true classes', ls=':', marker='o')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_ylim(top=6); ax.set_xlabel('teacher pairs'); ax.set_ylabel('population loss (nats)'); ax.legend(loc='lower left'); save(fig, 'model5-own-block')
nums['block'] = {str(int(n)): dict(table=med(b['table'][i]), oracle=med(b['oracle'][i]), found=med(b['found'][i]), found_right=med(b['right'][i]), shown=med(b['shown'][i])) for i, n in enumerate(b['n'])}
for i, n in enumerate(s['n']): nums['block'][str(int(n))] |= {arm: med(s[arm][i, :, -1, 0]) for arm in s.files if arm not in ('n', 'own')} | dict(pooled_classes=med(s[ARM][i, :, -1, 2]), pooled_worst=float(s[ARM][i, :, -1, 0].max()))
for i, n in enumerate(e['n']): nums['block'][str(int(n))] |= dict(note=med(e['practice0.8'][i]), recall=med(e['recall'][i]))
kl = np.median(s[ARM][list(s['n']).index(4096), :, :, 0], 0); nums['speed'] = dict(own=[int(x) for x in s['own']], kl4096=kl.round(4).tolist(), below_table=int(s['own'][np.argmax(kl < med(b['table'][list(b['n']).index(4096)]))]))

# 2. three levels at a fixed budget of strings
d = np.load(D / 'own_benefit_depth.npz'); B = d['total']; fs = d['shares']; mix = np.median(d['mix'], 2); flat, clu, ora = (np.median(d[k], 1) for k in ('flat', 'clustered', 'oracle')); fig, axes = plt.subplots(1, 2, figsize=(2 * W, H), constrained_layout=True)
ax = axes[0]; band(ax, B, d['flat'], GREY, 'teacher strings only (table of strings)', ls='--', marker='o'); best = d['mix'][mix.argmin(0), np.arange(len(B))]
band(ax, B, best, BLUE, 'teacher and own strings, best share', marker='o'); band(ax, B, d['clustered'], AQUA, 'teacher strings only, classes by clustering', marker='o'); band(ax, B, d['oracle'], INK, 'teacher strings only, true classes given', ls=':', marker='o')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_ylim(bottom=1e-4); ax.set_xlabel('strings in total (teacher and own)'); ax.set_ylabel('population loss of strings (nats)')
grid = np.vstack([flat, mix]); ax = axes[1]; im = ax.imshow(grid, origin='lower', aspect='auto', cmap=SEQ, norm=LogNorm(np.nanmin(grid), np.nanmax(grid)))
ax.set_xticks(range(len(B)), [f'$2^{{{int(np.log2(t))}}}$' for t in B]); ax.set_yticks(range(len(fs)), ['0 (table of strings)'] + [f'{f:g}' for f in fs[1:]]); ax.plot(range(len(B)), grid.argmin(0), 'o', mfc='white', mec='black', label='best share for this total')
ax.set_xlabel('strings in total (teacher and own)'); ax.set_ylabel('share of own strings'); fig.colorbar(im, ax=ax, label='population loss of strings (nats)'); hl = [a.get_legend_handles_labels() for a in axes]; fig.legend(hl[0][0] + hl[1][0], hl[0][1] + hl[1][1], loc='outside lower center', ncols=3); save(fig, 'model5-own-depth')
nums['depth'] = dict(total=[int(t) for t in B], shares=[float(f) for f in fs], flat=flat.round(4).tolist(), mix=mix.round(4).tolist(), clustered=clu.round(4).tolist(), oracle=ora.round(5).tolist(), best_share=[float(fs[i]) for i in grid.argmin(0)],
                     best_mix=mix.min(0).round(4).tolist(), best_own_share=[float(fs[1 + i]) for i in mix.argmin(0)], mix_wins_worlds=[int((d['mix'][mix.argmin(0)[j], j] < d['flat'][j]).sum()) for j in range(len(B))], worlds=int(d['flat'].shape[1]), names=np.median(d['names'], 2).round(1).tolist(), named=np.median(d['named'], 2).round(3).tolist())
(OUT / 'own_benefit_numbers.json').write_text(json.dumps(nums, indent=1)); print(OUT / 'own_benefit_numbers.json')
