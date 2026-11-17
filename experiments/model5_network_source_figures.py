"""Figures and numbers for section 10 of the model 5 note: whose proposals the writer practises on (one block), and three levels with networks.
From /var/tmp/jbauer/model5_lateral_verdict/network_practice_curves_{source,source_local,mix,mix_local}.npz and network_depth_{watching,watching_local,saddle,curves}.npz.
Also the tied-reader twin test (twin_tied.npz). Writes <out>/model5-network-{source,parts,depth-watching,depth-practice}.svg/.png, <out>/model5-twin-tied.svg/.png and <out>/source_numbers.json.   Usage: MPLBACKEND=Agg python model5_network_source_figures.py <out>"""

import json
from pathlib import Path
import sys

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

D = Path('/var/tmp/jbauer/model5_lateral_verdict'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#898781', '#0b0b0b'; W, H = plt.rcParams['figure.figsize']; nums = {}


def band(ax, x, y, color, label, **kw):                                                                                            # median over worlds (axis 0) with the interquartile range
    q1, med, q3 = np.percentile(y, [25, 50, 75], 0); ax.plot(x, med, color=color, label=label, **kw); ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0)


def save(fig, name): fig.savefig(OUT / f'{name}.svg', transparent=True); fig.savefig(OUT / f'{name}.png', dpi=150, facecolor='white'); plt.close(fig); print(OUT / f'{name}.png')


def med(x, axis=None): return np.median(x, axis).round(4).tolist()


def wins(a, b): return int((a < b).sum())                                                                                           # worlds in which a is below b


def parts(x): return -np.log(x[..., 1]), x[..., 0] + np.log(x[..., 1])                                                               # loss = -log(valid share) + loss among the valid strings


# 1. one block: the writer's loss by proposal source
src = {'bp': [np.load(D / 'network_practice_curves_source.npz'), np.load(D / 'network_practice_curves_mix.npz')], 'local': [np.load(D / 'network_practice_curves_mix_local.npz'), np.load(D / 'network_practice_curves_source_local.npz')]}
names = {'bp': dict(own='online', twin='online_twinsame', other='online_twin', judge='online_judge', mix='online_mix', mixtwin='online_mix_twinsame', uniform='online_uniform', reward='online_reward', rewardtwin='online_reward_twinsame'),
         'local': dict(own='online_local', twin='online_local_twinsame', judge='online_local_judge', mix='online_local_mix', mixtwin='online_local_mix_twinsame', uniform='online_local_uniform')}
def arm(kind, key, n):                                                                                                              # worlds x checkpoints x scores
    for d in src[kind]:
        if names[kind][key] in d['arms']: return d[names[kind][key]][list(d['ns']).index(n)]
LINES = (('own', 'own proposals', BLUE), ('twin', "a twin's proposals (same teacher pairs, other random start)", ORANGE), ('judge', "drawn from the judge's estimate (no writer)", AQUA), ('uniform', 'drawn uniformly from all pairs', YELLOW))
fig, axes = plt.subplots(1, 2, figsize=(1.9 * W, H), sharey=True, constrained_layout=True); n = 8192
for ax, kind, title in zip(axes, ('bp', 'local'), ('backpropagation, normalised step', 'local rule, local step')):
    d = src[kind][0]; P = d['P']; totals = list(d['totals']); fr = np.array([d['fresh'][totals.index(n + int(c))] for c in P]); band(ax, P[1:], fr[1:, :, 0].T, GREY, 'fresh teacher pairs instead', ls='--', marker='o')
    for key, label, col in LINES: band(ax, P[1:], arm(kind, key, n)[:, 1:, 0], col, label, marker='o')
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel(f'proposals after {n:,} teacher pairs'); ax.set_title(title)
axes[0].set_ylabel('population loss of the writer (nats)'); fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=2); save(fig, 'model5-network-source')
for kind in src:
    d = src[kind][0]; P = [int(c) for c in d['P']]; totals = list(d['totals']); nums[kind] = {}
    for n in (4096, 8192, 16384):
        e = dict(P=P, fresh=[med(d['fresh'][totals.index(n + c)][:, 0]) for c in P], table=[med(d['fresh'][totals.index(n + c)][:, 1]) for c in P])
        for key in names[kind]:
            a = arm(kind, key, n)
            if a is None: continue
            e[key] = med(a[:, :, 0], 0); e[key + '_valid'] = med(a[:, :, 1], 0); e[key + '_holes'] = med(a[:, :, 2], 0); e[key + '_wrong'] = med(parts(a)[0], 0); e[key + '_uneven'] = med(parts(a)[1], 0)
            if key != 'own': e[key + '_behind_own'] = [wins(arm(kind, 'own', n)[:, j, 0], a[:, j, 0]) for j in range(len(P))]            # worlds in which own proposals leave the lower loss
        e['mix_below_mixtwin'] = [wins(arm(kind, 'mix', n)[:, j, 0], arm(kind, 'mixtwin', n)[:, j, 0]) for j in range(len(P))]; e['worlds'] = int(len(arm(kind, 'own', n))); nums[kind][str(n)] = e
d = src['bp'][0]; a = d['online_reward'][list(d['ns']).index(2048)]; nums['collapse'] = dict(loss=med(a[:, -1, 0]), holes=med(a[:, -1, 2]), worlds=int((a[:, -1, 2] > 0.9).sum()))                                        # accept-or-refuse rule after 2,048 teacher pairs: worlds in which the writer ends on a few pairs
nums['overlap'] = {kind: {str(n): med(d[f"start_{names[kind]['twin']}"][list(d['ns']).index(n)], 0) for n in (4096, 8192, 16384)} for kind, d in (('bp', src['bp'][0]), ('local', src['local'][0]))}       # share of the learner's invalid mass the twin writes too; correlation of log-probabilities of the valid pairs

# 2. three levels: what watching gives (loss by teacher strings), and the order in which the two levels form (by pass)
wf = {'local': np.load(D / 'network_depth_watching_local.npz'), 'backprop': np.load(D / 'network_depth_watching.npz')}; ns = wf['local']['ns']; sd = np.load(D / 'network_depth_saddle.npz'); fig, axes = plt.subplots(1, 2, figsize=(1.9 * W, H), constrained_layout=True); ax = axes[0]
wk = lambda r, k: wf[r][r][:, :, list(wf[r]['keys']).index(k)]                                                                       # teacher-set sizes x worlds
band(ax, ns, wk('local', 'table').T, GREY, 'counted table of strings', ls='--', marker='o'); band(ax, ns, wk('local', 'writes').T, ORANGE, 'local rule, what it writes', marker='o'); band(ax, ns, wk('backprop', 'writes').T, MAGENTA, 'backpropagation, what it writes', marker='o')
band(ax, ns, wk('local', 'hidden').T, BLUE, 'local rule, hidden layers read out', marker='o'); band(ax, ns, wk('backprop', 'hidden').T, AQUA, 'backpropagation, hidden layers read out', marker='o'); band(ax, ns, wk('local', 'oracle').T, INK, 'table over the true symbols', ls=':')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('teacher strings'); ax.set_ylabel('population loss (nats)'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncols=2); ax = axes[1]; ep = np.arange(1, sd['curves'].shape[2] + 1); LS = ('-', '--', '-.', ':')
for i, ls in enumerate(LS): ax.plot(ep, np.median(sd['curves'][i, :, :, 0], 0), color=BLUE, ls=ls); ax.plot(ep, np.median(sd['curves'][i, :, :, 1], 0), color=ORANGE, ls=ls)
proxy = [Line2D([], [], color=BLUE, label='classes (level 1)'), Line2D([], [], color=ORANGE, label='symbols (level 2)')] + [Line2D([], [], color=GREY, ls=ls, label=f'random start of size {init:g}') for init, ls in zip(sd['inits'], LS)]
ax.set_xscale('log'); ax.set_xlabel(f"passes over {int(sd['n']):,} teacher strings (local rule)"); ax.set_ylabel('share held in one group of hidden vectors'); ax.legend(handles=proxy, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncols=2); save(fig, 'model5-network-depth-watching')
nums['watching'] = {r: {k: med(wk(r, k), 1) for k in wf[r]['keys']} for r in wf} | dict(ns=[int(x) for x in ns], worlds=int(wk('local', 'writes').shape[1]))
nums['passes'] = dict(n=int(sd['n']), inits=[float(x) for x in sd['inits']], classes=[int(np.argmax(np.median(c[:, :, 0], 0) >= 0.95)) + 1 for c in sd['curves']], symbols=[int(np.argmax(np.median(c[:, :, 1], 0) >= 0.95)) + 1 for c in sd['curves']], end=[med(c[:, -1, :2], 0) for c in sd['curves']])

# 3. three levels: practice by proposal source against fresh teacher strings
c = np.load(D / 'network_depth_curves.npz'); cn = list(c['ns']); P = c['P']; totals = list(c['totals']); fig, axes = plt.subplots(1, len(cn), figsize=(len(cn) * 0.95 * W, H), sharey=True, constrained_layout=True)
LINES3 = (('own', 'own proposals', BLUE, '-'), ('twin', "a twin's proposals", ORANGE, '-'), ('judge', "drawn from the judge's estimate (no writer)", AQUA, '-'), ('mix', 'half own, half from the judge', BLUE, '--'), ('mixtwin', "half the twin's, half from the judge", ORANGE, '--'), ('shallow', "the writer's blocks, combined at random", YELLOW, '-'))
for ax, n in zip(axes, cn):
    i = cn.index(n); fr = np.array([c['fresh'][totals.index(n + int(x))] for x in P]); band(ax, P[1:], fr[1:, :, 0].T, GREY, 'fresh teacher strings instead (local rule)', ls='--', marker='o')
    for key, label, col, ls in LINES3: band(ax, P[1:], c[key][i][:, 1:, 0], col, label, ls=ls, marker='o')
    ax.axhline(np.median(wk('backprop', 'writes')[list(ns).index(n)]), color=MAGENTA, ls=':', label='backpropagation on the same teacher strings, no further samples'); ax.axhline(np.median(c['extra_own'][i][:, 0]), color=INK, ls=':', label="the judge's own estimate")
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel(f'proposals after {n:,} teacher strings')
axes[0].set_ylabel('population loss of the writer (nats)'); fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncols=2); save(fig, 'model5-network-depth-practice')
nums['depth'] = {}
for n in cn:
    i = cn.index(n); e = dict(P=[int(x) for x in P], fresh=[med(c['fresh'][totals.index(n + int(x))][:, 0]) for x in P], readout=[med(c['fresh'][totals.index(n + int(x))][:, 1]) for x in P], table=[med(c['fresh'][totals.index(n + int(x))][:, 2]) for x in P], worlds=int(c['own'].shape[1]))
    for key in c['arms']:
        a = c[key][i]; e[key] = med(a[:, :, 0], 0); e[key + '_valid'] = med(a[:, :, 1], 0); e[key + '_holes'] = med(a[:, :, 2], 0); e[key + '_wrong'] = med(parts(a)[0], 0); e[key + '_uneven'] = med(parts(a)[1], 0); e[key + '_top'] = med(a[:, :, 5], 0)
        if key != 'own': e[key + '_behind_own'] = [wins(c['own'][i][:, j, 0], a[:, j, 0]) for j in range(len(P))]
        e[key + '_below_fresh'] = [wins(a[:, j, 0], c['fresh'][totals.index(n + int(x))][:, 0]) for j, x in enumerate(P)]
    e['mix_below_mixtwin'] = [wins(c['mix'][i][:, j, 0], c['mixtwin'][i][:, j, 0]) for j in range(len(P))]; x = c['extra_own'][i]; e |= dict(pE=med(x[:, 0]), invalid=med(x[:, 1]), refused=med(x[:, 2]), reader=med(x[:, 6])); x = c['extra_twin'][i]; e |= dict(overlap=med(x[:, 7]), corr=med(x[:, 8]), twin_before=med(x[:, 9]), twin_after=med(x[:, 10]))
    nums['depth'][str(n)] = e

# 4. what each source repairs: the loss split into the invalid mass and the unevenness among the valid strings
fig, axes = plt.subplots(2, 2, figsize=(1.9 * W, 1.5 * H), sharex='col', constrained_layout=True); n = 8192; d = src['local'][0]
cols = (('one block, local rule', d['P'], lambda key: arm('local', key, n)), ('three levels, local rule', P, lambda key: c[key][cn.index(n)]))
for j, (title, x, get) in enumerate(cols):
    for key, label, col, ls in (('own', 'own proposals', BLUE, '-'), ('twin', "a twin's proposals", ORANGE, '-'), ('judge', "drawn from the judge's estimate", AQUA, '-'), ('mix', 'half own, half from the judge', BLUE, '--'), ('mixtwin', "half the twin's, half from the judge", ORANGE, '--')):
        a = get(key); band(axes[0, j], x, parts(a)[0], col, label, ls=ls, marker='o'); band(axes[1, j], x, parts(a)[1], col, label, ls=ls, marker='o')
    axes[0, j].set_title(title); axes[1, j].set_xlabel(f'proposals after {n:,} teacher samples')
    for ax in axes[:, j]: ax.set_xscale('symlog', base=2, linthresh=float(x[1])); ax.set_yscale('log')
axes[0, 0].set_ylabel('invalid mass:  $-\\log$(valid share)  (nats)'); axes[1, 0].set_ylabel('unevenness among the valid (nats)'); fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncols=3); save(fig, 'model5-network-parts')

# 5. a reader tied to its own writer (model5_twin_tied.py): practice on own samples against a twin's, in the two table learners
t = np.load(D / 'twin_tied.npz'); g = t['gradient']; marks = t['marks']; gr = t['grower']; fig, axes = plt.subplots(1, 3, figsize=(2.6 * W, 0.95 * H), constrained_layout=True); ARM3 = (('no practice', GREY, '--'), ('own samples', BLUE, '-'), ("a twin's samples", ORANGE, '-'))
for i, (label, col, ls) in enumerate(ARM3): band(axes[0], marks, g[1, 0, i, :, :, 0], col, label, ls=ls, marker='o')
axes[0].set_xscale('symlog', linthresh=100); axes[0].set_yscale('log'); axes[0].set_xlabel('wake batches (units from a sharp random start)'); axes[0].set_ylabel('population loss (nats)'); axes[0].legend()
for key, label, col in ((3, 'own samples', BLUE), (4, "a twin's samples", ORANGE), (5, 'teacher pairs', INK)): band(axes[1], marks, g[1, 0, 1, :, :, key], col, label, marker='o')
axes[1].set_xscale('symlog', linthresh=100); axes[1].set_xlabel('wake batches (same learner, practising on own samples)'); axes[1].set_ylabel('share the reader explains above chance'); axes[1].legend()
for i, (label, col, ls) in enumerate((('own samples', BLUE, '-'), ("a twin's samples, same teacher pairs", ORANGE, '-'), ("a twin's samples, other teacher pairs", ORANGE, ':'))): band(axes[2], t['check'], gr[i, :, :, 0], col, label, ls=ls)
axes[2].set_xscale('log', base=2); axes[2].set_yscale('log'); axes[2].set_xlabel('practice samples (units that start as copies of the codes)'); axes[2].set_ylabel('population loss (nats)'); axes[2].legend(); save(fig, 'model5-twin-tied')
nums['tied'] = dict(worlds=int(g.shape[3]), marks=[int(x) for x in marks], loss={st: {arm: med(g[a, 0, i, :, :, 0], 0) for i, arm in enumerate(t['arms'])} for a, st in enumerate(t['starts'])}, own_below_twin={st: [wins(g[a, 0, 1, :, j, 0], g[a, 0, 2, :, j, 0]) for j in range(len(marks))] for a, st in enumerate(t['starts'])},
                    none_below_twin={st: wins(g[a, 0, 0, :, -1, 0], g[a, 0, 2, :, -1, 0]) for a, st in enumerate(t['starts'])}, own_below_none={st: [wins(g[a, 0, 1, :, j, 0], g[a, 0, 0, :, j, 0]) for j in range(len(marks))] for a, st in enumerate(t['starts'])},
                    reach={st: med(g[a, 0, 1, :, :, 3:6], 0) for a, st in enumerate(t['starts'])}, grower=dict(own=med(gr[0, :, :, 0], 0), twin_same=med(gr[1, :, :, 0], 0), twin_other=med(gr[2, :, :, 0], 0), check=[int(x) for x in t['check']], own_wins=[wins(gr[0, :, -1, 0], gr[i, :, -1, 0]) for i in (1, 2)],
                    twin_kl=med(t['grower_twin_kl'], 1), mid_wins=wins(gr[0, :, list(t['check']).index(524288), 0], gr[1, :, list(t['check']).index(524288), 0])))
(OUT / 'source_numbers.json').write_text(json.dumps(nums, indent=1)); print(OUT / 'source_numbers.json')
