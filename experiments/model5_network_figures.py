"""Figures and numbers for the network runs of model 5 (model5_network_baseline.py, model5_network_practice.py), from /var/tmp/jbauer/model5_lateral_verdict/network_*.npz.
Writes <out>/model5-network-{watching,practice,budget,online,local,localrule}.svg/.png and <out>/network_numbers.json.   Usage: MPLBACKEND=Agg python model5_network_figures.py <out>"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

from model5_lateral_verdict import World
from model5_network_baseline import groups
from model5_network_practice import LR, Net
from model5_own_benefit import class_table, kl

D = Path('/var/tmp/jbauer/model5_lateral_verdict'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#898781', '#0b0b0b'
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']); W, H = plt.rcParams['figure.figsize']; nums = {}


def band(ax, x, y, color, label, **kw):                                                                                            # median over worlds (axis 0) with the interquartile range
    q1, med, q3 = np.percentile(y, [25, 50, 75], 0); ax.plot(x, med, color=color, label=label, **kw); ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0)


def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', transparent=True); fig.savefig(OUT / f'{name}.png', dpi=150, facecolor='white'); plt.close(fig); print(OUT / f'{name}.png')


def med(x): return float(np.median(x))


# 1. watching, the learning curve: what the output writes and what the hidden layer holds, step by step (2,048 teacher pairs, 8 worlds, weight decay 1e-3)
steps = np.unique(np.round(np.geomspace(1, 600, 40)).astype(int)); curves = []
for seed in range(1, 9):
    w = World(seed=seed); rng = np.random.default_rng(2000 + seed); k, kp = w.teacher(rng, 2048); val = rng.random(2048) < 0.15; Q, Qv = np.zeros((w.N, w.N)), np.zeros((w.N, w.N)); np.add.at(Q, (k[~val], kp[~val]), 1); np.add.at(Qv, (k[val], kp[val]), 1)
    net = Net(w.N, np.random.default_rng(seed)); row = []
    for t in range(1, 601):
        lp = net.logp()
        if t in steps: row.append([-(Qv / Qv.sum() * lp).sum(), -(Q / Q.sum() * lp).sum(), kl(w, np.exp(lp)), class_table(w, Q + Qv, groups(net.P[0], w.v), groups(net.P[1], w.v))])
        net.step(np.exp(lp) - Q / Q.sum(), LR, 1e-3)
    curves.append(row)
curves = np.array(curves); fig, axes = plt.subplots(1, 2, figsize=(1.8 * W, 0.9 * H), constrained_layout=True)
band(axes[0], steps, curves[:, :, 1], GREY, 'teacher pairs it is trained on', ls='--'); band(axes[0], steps, curves[:, :, 0], INK, 'held-out teacher pairs'); axes[0].set_ylabel('cross-entropy (nats)')
band(axes[1], steps, curves[:, :, 2], ORANGE, 'what the output writes'); band(axes[1], steps, curves[:, :, 3], BLUE, 'the hidden layer read out'); axes[1].set_ylabel('population loss (nats)'); axes[1].set_yscale('log'); stop = int(steps[np.median(curves[:, :, 0], 0).argmin()])
for ax in axes: ax.axvline(stop, color=GREY, lw=0.8, ls=':'); ax.set_xscale('log'); ax.set_xlabel('training step'); ax.legend()
save(fig, 'model5-network-watching'); nums['watching'] = dict(stop=stop, out_at_stop=med(curves[:, list(steps).index(stop), 2]), hidden_at_stop=med(curves[:, list(steps).index(stop), 3]), out_end=med(curves[:, -1, 2]), hidden_end=med(curves[:, -1, 3]),
                                                             hidden_first=int(steps[np.argmax(np.median(curves[:, :, 3], 0) < 0.05)]))

# 2. practice against as many fresh teacher pairs
d = np.load(D / 'network_practice_curves.npz'); ns = list(d['ns']); P = d['P']; totals = list(d['totals']); show = [n for n in ns if n >= 2048]; fig, axes = plt.subplots(1, len(show), figsize=(len(show) * 0.95 * W, H), sharey=True, constrained_layout=True)
ARMS = (('own', 'own proposals, judged', BLUE, '-'), ('own_reader', 'the same, reader learns too', MAGENTA, '-'), ('twin', "a twin's proposals, judged", ORANGE, '-'), ('draws', 'pairs drawn from the judge, used as teacher pairs', AQUA, '-'))
for ax, n in zip(axes, show):
    i = ns.index(n); fr = np.array([d['fresh'][totals.index(n + int(c))] for c in P]); band(ax, P[1:], fr[1:, :, 0].T, GREY, 'fresh teacher pairs', ls='--', marker='o')
    for arm, label, col, ls in ARMS: band(ax, P[1:], d[arm][i][:, 1:, 0], col, label, ls=ls, marker='o')
    ax.axhline(med(d['own'][i][:, 0, 4]), color=INK, ls=':', label="the judge's own estimate"); ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel(f'further samples after {n:,} teacher pairs')
axes[0].set_ylabel('population loss of the writer (nats)'); fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=3); save(fig, 'model5-network-practice')
nums['curves'] = {str(n): dict(P=[int(c) for c in P], fresh=[med(d['fresh'][totals.index(n + int(c))][:, 0]) for c in P], table=[med(d['fresh'][totals.index(n + int(c))][:, 1]) for c in P],
                               **{arm: np.median(d[arm][ns.index(n)][:, :, 0], 0).round(4).tolist() for arm in d['arms']}, judge=med(d['own'][ns.index(n)][:, 0, 4]), invalid=med(d['own'][ns.index(n)][:, 0, 5]), refused=med(d['own'][ns.index(n)][:, 0, 6]),
                               tau=med(d['own'][ns.index(n)][:, 0, 7]), asked_own=np.median(d['own'][ns.index(n)][:, :, 3], 0).round(3).tolist(), asked_twin=np.median(d['twin'][ns.index(n)][:, :, 3], 0).round(3).tolist(),
                               judge_reader=np.median(d['own_reader'][ns.index(n)][:, :, 4], 0).round(4).tolist(), invalid_reader=np.median(d['own_reader'][ns.index(n)][:, :, 5], 0).tolist(), refused_reader=np.median(d['own_reader'][ns.index(n)][:, :, 6], 0).tolist(),
                               own_wins=[int((d['own'][ns.index(n)][:, j, 0] < d['fresh'][totals.index(n + int(c))][:, 0]).sum()) for j, c in enumerate(P)], twin_wins=[int((d['twin'][ns.index(n)][:, j, 0] < d['own'][ns.index(n)][:, j, 0]).sum()) for j in range(len(P))],
                               worlds=int(d['own'].shape[1])) for n in ns}

# 3. a fixed budget split between teacher pairs and own proposals
if (D / 'network_practice_budget.npz').exists():
    d = np.load(D / 'network_practice_budget.npz'); B = d['total']; fs = d['shares']; arm = str(d['arms'][0]); mix = np.median(d[arm][:, :, :, 0], 2); fresh = np.median(d['fresh'][:, :, 0], 1); grid = np.vstack([fresh, mix]); fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap=SEQ, norm=LogNorm(grid.min(), grid.max())); ax.set_xticks(range(len(B)), [f'$2^{{{int(np.log2(t))}}}$' for t in B]); ax.set_yticks(range(len(fs)), ['0 (teacher pairs only)'] + [f'{f:g}' for f in fs[1:]])
    ax.plot(range(len(B)), grid.argmin(0), 'o', mfc='white', mec='black', label='best share for this total'); ax.set_xlabel('samples in total (teacher pairs and own proposals)'); ax.set_ylabel('share of own proposals'); fig.colorbar(im, label='population loss of the writer (nats)')
    fig.legend(loc='outside lower center'); save(fig, 'model5-network-budget')
    nums['budget'] = dict(total=[int(t) for t in B], shares=[float(f) for f in fs], fresh=fresh.round(4).tolist(), table=np.median(d['fresh'][:, :, 1], 1).round(4).tolist(), mix=mix.round(4).tolist(), best_share=[float(fs[i]) for i in grid.argmin(0)],
                          wins=[int((d[arm][mix[:, j].argmin(), j, :, 0] < d['fresh'][j, :, 0]).sum()) for j in range(len(B))], worlds=int(d['fresh'].shape[1]), **{f'mix_{a}': np.median(d[str(a)][:, :, :, 0], 2).round(4).tolist() for a in d['arms'][1:]})
# 4. the writer learns through its weights only (no memory of judged pairs)
def practice_panels(d, show, arms, name, kept=None):
    ns = list(d['ns']); P = d['P']; totals = list(d['totals']); fig, axes = plt.subplots(1, len(show), figsize=(len(show) * 0.95 * W, H), sharey=True, constrained_layout=True)
    for ax, n in zip(axes, show):
        i = ns.index(n); fr = np.array([d['fresh'][totals.index(n + int(c))] for c in P]); band(ax, P[1:], fr[1:, :, 0].T, GREY, 'fresh teacher pairs, same network', ls='--', marker='o'); ax.plot(P[1:], np.median(fr[1:, :, 1], 1), color=GREY, ls=':', marker='.', label='fresh teacher pairs, counted table')
        for arm, label, col, ls in arms: band(ax, P[1:], d[arm][i][:, 1:, 0], col, label, ls=ls, marker='o')
        ax.axhline(med(d[arms[0][0]][i][:, 0, 4]), color=INK, ls=':', label="the judge's own estimate"); ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel(f'further samples after {n:,} teacher pairs')
    axes[0].set_ylabel('population loss of the writer (nats)'); fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=4); save(fig, name)
    return {str(n): dict(P=[int(c) for c in P], fresh=[med(d['fresh'][totals.index(n + int(c))][:, 0]) for c in P], table=[med(d['fresh'][totals.index(n + int(c))][:, 1]) for c in P], judge=med(d[arms[0][0]][ns.index(n)][:, 0, 4]), invalid=med(d[arms[0][0]][ns.index(n)][:, 0, 5]),
                         **{arm: np.median(d[arm][ns.index(n)][:, :, 0], 0).round(4).tolist() for arm in d['arms']}, **{f'{arm}_apart': np.median(d[arm][ns.index(n)][:, :, 8], 0).round(3).tolist() for arm in d['arms']},
                         **{f'{arm}_wins': [int((d[arm][ns.index(n)][:, j, 0] < d['fresh'][totals.index(n + int(c))][:, 0]).sum()) for j, c in enumerate(P)] for arm in d['arms']},
                         **{f'{arm}_wins_table': [int((d[arm][ns.index(n)][:, j, 0] < d['fresh'][totals.index(n + int(c))][:, 1]).sum()) for j, c in enumerate(P)] for arm in d['arms']}, worlds=int(d[arms[0][0]].shape[1])) for n in ns}


d = np.load(D / 'network_practice_curves_online.npz')
nums['online'] = practice_panels(d, [4096, 8192, 16384], (('online', 'own proposals, weights only', MAGENTA, '-'), ('online_local', 'the same, only the two written codes move', AQUA, '-'), ('online_twin', "a twin's proposals, weights only", ORANGE, '-'), ('online_top', 'hidden layer frozen', YELLOW, '-'),
                                                           ('own', 'own proposals, judged counts kept', BLUE, '-')), 'model5-network-online')

# 5. the network trained by a local rule: plateau at small weights, hidden layer and output
d = np.load(D / 'network_practice_local.npz'); keys = list(d['keys']); fig, axes = plt.subplots(1, 2, figsize=(1.8 * W, 0.9 * H), sharey=True, constrained_layout=True); x = np.arange(1, d['curves'].shape[2] + 1)
for j, (ax, title) in enumerate(zip(axes, ('the hidden layer read out', 'what the output writes'))):
    for i, init in enumerate(d['inits']): band(ax, x, d['curves'][i][:, :, j], SEQ(0.25 + 0.25 * i), f'random start of size {init:g}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('passes over the teacher pairs'); ax.set_title(title)
axes[0].set_ylabel('population loss (nats)'); axes[0].legend(); save(fig, 'model5-network-local')
nums['local'] = {str(int(n)): {k: med(d['stats'][i][:, keys.index(k)]) for k in keys} for i, n in enumerate(d['ns'])} | dict(first=[int(np.argmax(np.median(d['curves'][i][:, :, 0], 0) < 0.05)) + 1 for i in range(len(d['inits']))], inits=[float(i) for i in d['inits']], worlds=int(d['stats'].shape[1]))
if (D / 'network_practice_curves_localrule.npz').exists():
    d = np.load(D / 'network_practice_curves_localrule.npz'); nums['localrule'] = practice_panels(d, [4096, 8192, 16384], (('online_local', 'own proposals, only the two written codes move', AQUA, '-'), ('online', 'own proposals, normalised step', MAGENTA, '-')), 'model5-network-localrule')

b = np.load(D / 'network_baseline.npz'); nums['baseline'] = {str(int(n)): {k: med(b[k][i]) for k in ('table', 'net16', 'hidden16', 'oracle', 'right16')} | dict(all_right=int((b['right16'][i] == 1).sum()), worlds=int(b['table'].shape[1])) for i, n in enumerate(b['n'])}
(OUT / 'network_numbers.json').write_text(json.dumps(nums, indent=1)); print(OUT / 'network_numbers.json')
