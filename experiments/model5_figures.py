"""Figures and numbers for the model 5 note, from /var/tmp/jbauer/model5_lateral_verdict/*.npz (model5_lateral_verdict.py, model5_depth.py).
Writes <out>/model5-*.svg, <out>/model5-*.png and <out>/numbers.json.   Usage: MPLBACKEND=Agg python model5_figures.py <out>"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

from model5_lateral_verdict import World, table_kl

D = Path('/var/tmp/jbauer/model5_lateral_verdict'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#898781'
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']); W, H = plt.rcParams['figure.figsize']; nums = {}


def band(ax, x, y, color, label, **kw):                                                                                            # median over worlds with the interquartile range
    q1, med, q3 = np.percentile(y, [25, 50, 75], 0); ax.plot(x, med, color=color, label=label, **kw); ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0)


def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', transparent=True); fig.savefig(OUT / f'{name}.png', dpi=150, facecolor='white'); plt.close(fig); print(OUT / f'{name}.png')


def med(x): return float(np.median(x))


def table_curve(ns, worlds=16):                                                                                                    # loss of the counted table after n teacher pairs, median over worlds
    out = []
    for seed in range(worlds):
        w = World(seed=seed); rng = np.random.default_rng(900 + seed); M = np.zeros((w.N, w.N)); row, done = [], 0
        for n in ns: k, kp = w.teacher(rng, int(n) - done); np.add.at(M, (k, kp), 1); done = int(n); row.append(table_kl(w, M))
        out.append(row)
    return np.array(out)


# 1. what M and each property of the prior do
d = np.load(D / 'ablation.npz'); keys = list(d['keys']); x = d['check']; arms = [('full', 'veto, spread, shared pools', BLUE, '-'), ('no_veto', 'no veto (M ignored)', ORANGE, '-'), ('no_spread', 'no spread', AQUA, '-'), ('private', 'no reuse (private pools)', YELLOW, '-')]
fig, axes = plt.subplots(1, 3, figsize=(2.4 * W, 0.95 * H), constrained_layout=True)
for ax, (key, name) in zip(axes, (('kl', 'population loss (nats)'), ('whole', 'classes held by one unit'), ('new', 'unshown valid pairs written\n(relative to the population)'))):
    for arm, label, col, ls in arms: band(ax, x, d[arm][:, 1:, keys.index(key)], col, label, ls=ls)
    if key == 'kl': ax.axhline(med(d['table']), color=GREY, ls='--', label='counted table'); ax.set_yscale('log')
    ax.set_xscale('log'); ax.set_xlabel('own samples'); ax.set_ylabel(name)
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=5); save(fig, 'model5-ablation'); K = keys.index
nums['ablation'] = dict(table=med(d['table']), recall=med(d['recall']), full=med(d['full'][:, -1, K('kl')]), full_new=med(d['full'][:, -1, K('new')]), full_whole_worlds=int((d['full'][:, -1, K('whole')] == 1).sum()),
                        full_whole_min=float(d['full'][:, -1, K('whole')].min()), gen=med(d['generator_credit'][:, -1, K('kl')]), no_veto=med(d['no_veto'][:, -1, K('kl')]), no_spread=med(d['no_spread'][:, -1, K('kl')]),
                        no_spread_new=med(d['no_spread'][:, -1, K('new')]), private=med(d['private'][:, -1, K('kl')]), private_new=med(d['private'][:, -1, K('new')]), start=med(d['full'][:, 0, K('kl')]),
                        worlds=len(d['table']), own=int(x[-1]), full_valid=med(d['full'][:, -1, K('valid')]))

# 2. population loss against the size of the teacher set
d = np.load(D / 'efficiency.npz'); n = d['n']; fig, ax = plt.subplots(constrained_layout=True)
for key, label, col, ls in (('table', 'counted table', GREY, '--'), ('bottleneck', 'likelihood gradient, v units', YELLOW, '-'), ('recall', 'recall through M, no own samples', AQUA, '-'), ('practice0.8', 'practice', BLUE, '-')):
    band(ax, n, np.minimum(d[key], 20).T, col, label, ls=ls, marker='o')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_ylim(top=6); ax.set_xlabel('teacher pairs'); ax.set_ylabel('population loss (nats)'); ax.legend(); save(fig, 'model5-efficiency')
nums['efficiency'] = {str(int(a)): {k: med(d[k][i]) for k in d.files if k != 'n'} | dict(locked=int((d['whole0.8'][i] == 1).sum()), worse=int((d['practice0.8'][i] > d['table'][i]).sum())) for i, a in enumerate(n)}

# 3. total samples and own share
d = np.load(D / 'heatmap.npz'); kl = np.median(d['kl'], 1); tab = table_curve(d['total']); grid = np.vstack([np.median(tab, 0), np.full(kl.shape[1], np.nan), kl]); fig, ax = plt.subplots(constrained_layout=True)
im = ax.imshow(grid, origin='lower', aspect='auto', cmap=SEQ, norm=LogNorm(np.nanmin(grid), np.nanmax(grid))); ax.set_xticks(range(0, len(d['total']), 2), [f'$2^{{{int(np.log2(t))}}}$' for t in d['total'][::2]])
ax.set_yticks([0] + list(range(2, 2 + len(d['shares']))), ['counted table'] + [f'{s:g}' for s in d['shares']]); ax.plot(range(kl.shape[1]), 2 + kl.argmin(0), 'o', mfc='white', mec='black', label='best share of the learner')
ax.set_xlabel('samples in total (teacher and own)'); ax.set_ylabel('share of own samples'); ax.legend(loc='upper left'); fig.colorbar(im, label='population loss (nats)')
save(fig, 'model5-heatmap'); nums['heatmap'] = dict(shares=[float(s) for s in d['shares']], total=[int(t) for t in d['total']], kl=kl.round(4).tolist(), best=[float(d['shares'][i]) for i in kl.argmin(0)], table=np.median(tab, 0).round(5).tolist(),
                                                    table_wins=int((np.median(tab, 0) < kl.min(0)).sum()))

# 4. whose samples
d = np.load(D / 'source.npz'); fig, ax = plt.subplots(constrained_layout=True); tab = table_curve(4096 + d['check'])
for y, label, col, ls in ((d['own'], 'practice on own samples', BLUE, '-'), (d['twin'], "practice on a twin's samples", ORANGE, '-'), (tab, 'as many fresh teacher pairs, counted', GREY, '--')): band(ax, d['check'], y, col, label, ls=ls)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('further samples after 4,096 teacher pairs'); ax.set_ylabel('population loss (nats)'); ax.legend(); save(fig, 'model5-source')
nums['source'] = {k: med(d[k][:, -1]) for k in ('own', 'twin', 'teacher')} | dict(twin_kl=med(d['twin_kl']), n=int(d['check'][-1]), table_end=med(tab[:, -1]), twin_best=float(np.median(d['twin'], 0).min()), teacher_best=float(np.median(d['teacher'], 0).min()),
                                                                                own_wins=int((d['own'][:, -1] < d['twin'][:, -1]).sum()), cross=int(d['check'][np.argmax(np.median(d['own'], 0) < np.median(d['twin'], 0))]))

# 5. the likelihood-gradient learner
g = json.loads((D / 'gradient.json').read_text()); starts = [('noise0.1', 'weak'), ('noise1.0', 'moderate'), ('random', 'sharp')]; fig, axes = plt.subplots(1, 2, figsize=(1.7 * W, 0.85 * H), constrained_layout=True)
rng = np.random.default_rng(0); pos, lab = [], []
for i, (st, name) in enumerate(starts):
    for j, th in enumerate((0, 1)):
        for a, (arm, col) in enumerate((('wake', GREY), ('practice', BLUE))):
            y = np.array(g[f'{st}|{th}|{arm}']['kl']); xx = 2.6 * i + j + 0.3 * (a - 0.5); axes[0].plot(xx + 0.08 * rng.uniform(-1, 1, len(y)), y, '.', color=col, alpha=0.5); axes[0].plot([xx - 0.12, xx + 0.12], [np.median(y)] * 2, color=col)
        pos.append(2.6 * i + j); lab.append(f'{name}\n' + ('none' if th == 0 else 'on'))
axes[0].set_xticks(pos, lab); axes[0].set_yscale('log'); axes[0].set_ylabel('population loss (nats)'); axes[0].set_xlabel('random start, threshold'); axes[0].plot([], [], color=GREY, label='teacher pairs only'); axes[0].plot([], [], color=BLUE, label='half own samples'); axes[0].legend()
for i, (st, name) in enumerate(starts):
    r = g[f'{st}|1|practice']
    for a, (key, col) in enumerate((('el_teacher', GREY), ('el_own', BLUE))): y = np.array(r[key]); xx = i + 0.3 * (a - 0.5); axes[1].plot(xx + 0.08 * rng.uniform(-1, 1, len(y)), y, '.', color=col, alpha=0.5); axes[1].plot([xx - 0.12, xx + 0.12], [np.median(y)] * 2, color=col)
axes[1].set_xticks(range(3), [s[1] for s in starts]); axes[1].set_xlabel('random start (threshold on)'); axes[1].set_ylabel('share of samples above the threshold'); axes[1].plot([], [], color=GREY, label='teacher pairs'); axes[1].plot([], [], color=BLUE, label='own samples'); axes[1].legend()
save(fig, 'model5-gradient'); nums['gradient'] = {k: {q: med(v[q]) for q in ('kl', 'valid', 'purity', 'el_teacher', 'el_own')} | dict(locked=int((np.array(v['kl']) < 0.05).sum())) for k, v in g.items()}

# 6. three levels
d = np.load(D / 'depth.npz'); fig, ax = plt.subplots(constrained_layout=True)
for key, label, col, ls in (('flat', 'table of whole strings', GREY, '--'), ('first', 'all strings first, then practice', ORANGE, '-'), ('between', 'practice between two halves of the strings', BLUE, '-')): band(ax, d['n'], d[key].T, col, label, ls=ls, marker='o')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('teacher strings'); ax.set_ylabel('population loss of strings (nats)'); ax.legend(); save(fig, 'model5-depth')
nums['depth'] = {str(int(a)): {k: med(d[k][i]) for k in d.files if k != 'n'} for i, a in enumerate(d['n'])}
(OUT / 'numbers.json').write_text(json.dumps(nums, indent=1)); print(OUT / 'numbers.json')
