"""Figures for model5_cut_mechanism.py. Reads OUT/cut_mechanism.npz, writes OUT/figures/model5-cut-units.{svg,png} (the units of one world over training), model5-cut-probes.{svg,png}
(what one sample of each kind does to each level) and model5-cut-gating.{svg,png} (the top unit waits for the level below; reader and writer), and OUT/cut_mechanism_numbers.json.
Usage: python model5_cut_mechanism_figures.py [world]"""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model5_lateral_verdict import OUT

d = np.load(f'{OUT}/cut_mechanism.npz'); res, prb, tun, abl, jud, ck = d['res'], d['prb'], d['tun'], d['abl'], d['jud'], d['checks']; keys, rules, arms = list(d['keys']), list(d['rules']), list(d['arms']); W = len(res); x = np.maximum(ck, 0.5)
B1, B2 = ((0.0625, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9)), ((0.5, 0.55), (0.55, 0.7), (0.7, 0.9)); L1, L2 = ('6 to 10%', '10 to 30%', '30 to 60%', '60 to 90%'), ('50 to 55%', '55 to 70%', '70 to 90%'); WSP = arms.index('WSP')
(OUT / 'figures').mkdir(exist_ok=True)


def binned(ai, rule, level, bins=None, cond=None):                                                                                 # median change of valid<level> per 1,000 samples under one rule, over the states whose valid<level> lies in a bin
    v = res[:, ai, :, level]; p = 1000 * prb[:, ai, :, rules.index(rule), level]; ok = np.ones_like(v, bool) if cond is None else cond; return [float(np.median(p[(v >= lo) & (v < hi) & ok])) for lo, hi in (bins or (B1, B2)[level - 1])]


def save(fig, name): [fig.savefig(OUT / 'figures' / f'{name}.{e}') for e in ('svg', 'png')]; print(OUT / 'figures' / f'{name}.png')


# the units of one world
w = int(sys.argv[1]) if len(sys.argv) > 1 else 3; T = (0, 16, 32, 64, 128, 2048); ROWS = ('value written\nby each unit u', 'units chosen by z\nat the first cut', 'units chosen by z\nat the second cut', 'unit named for a\nteacher pair of value a')
fig, ax = plt.subplots(4, len(T), figsize=plt.figaspect(0.62) * 1.5, constrained_layout=True)
for j, t in enumerate(T):
    for i in range(4):
        M = tun[w, WSP, list(ck).index(t), i]; a = ax[i, j]; a.imshow(M if i < 3 else M.T, cmap='Greys', vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
        if i == 0: a.set_title(f'round {t:,}')
        if j == 0: a.set_ylabel(ROWS[i], rotation=0, ha='right', va='center')
save(fig, 'model5-cut-units')

# what one sample of each kind does to each level, in the states of the learner with practice
fig, ax = plt.subplots(2, 3, figsize=plt.figaspect(0.55) * 1.7, constrained_layout=True)
SETS = ((('wake', 'a teacher string, read by the reader', 'C0', '-'), ('wake_tied', 'a teacher string, read as well as possible', 'C0', '--'), ('practice', 'an own sample, judged', 'C3', '-')),
        (('accepted', 'own samples the judge accepts', 'C2', '-'), ('refused', 'own samples the judge refuses', 'C3', '-')),
        (('practice', 'own sample, credited to the units that wrote it', 'C3', '-'), ('own_read', 'own sample, credited by reading', 'C1', '-'), ('twin_read', "a twin's sample, credited by reading", 'C4', '-')))
for lev in (1, 2):
    for j, S in enumerate(SETS):
        a = ax[lev - 1, j]; lab = (L1, L2)[lev - 1]
        for rule, name, col, ls in S: a.plot(range(len(lab)), binned(WSP, rule, lev), ls, color=col, marker='o', label=name)
        a.axhline(0, color='0.6', lw=0.8); a.set_xticks(range(len(lab))); a.set_xticklabels(lab); a.set_xlabel(f'written strings that obey level {lev}'); a.set_ylabel(f'change of that share per 1,000 samples')
        if lev == 1: a.set_title(('Teacher string or own sample', 'Accepted or refused', 'Whose sample, credited how')[j])
        if lev == 2: a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=False)
save(fig, 'model5-cut-probes')

# the top unit waits for the level below; what reader and writer agree on
fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(0.42) * 1.25, constrained_layout=True); G = B1 + ((0.9, 1.01),); GL = L1 + ('over 90%',)
for ai, lab, col in ((arms.index('WS'), 'states of the wake-sleep learner', 'C0'), (WSP, 'states of the learner with practice', 'C3')):
    v1, v2 = res[:, ai, :, 1], res[:, ai, :, 2]; p = 1000 * prb[:, ai, :, rules.index('practice'), 2]; y = [np.median(p[(v1 >= lo) & (v1 < hi) & (v2 < 0.6)]) if ((v1 >= lo) & (v1 < hi) & (v2 < 0.6)).sum() >= 5 else np.nan for lo, hi in G]; ax[0].plot(range(len(G)), y, color=col, marker='o', label=lab)
ax[0].set_xticks(range(len(G))); ax[0].set_xticklabels(GL); ax[0].set_xlabel('written strings that obey level 1'); ax[0].set_ylabel('change of the level-2 share per 1,000 own samples'); ax[0].set_title('The verdict reaches the top unit through level 1')
for k, lab, col in (('agree1_teacher', 'teacher strings, level 1', 'C0'), ('agree1_own', 'own samples, level 1', 'C3'), ('agree2_teacher', 'teacher strings, level 2', 'C0'), ('agree2_own', 'own samples, level 2', 'C3')): ax[1].plot(x, np.median(res[:, WSP, :, keys.index(k)], 0), '--' if '2' in k else '-', color=col, label=lab)
ax[1].plot(x, (np.median(res[:, WSP, :, 2], 0) - 0.5) / 0.5, ':', color='k', label='level 2 as written (0: chance)'); ax[1].set_xscale('log', base=2); ax[1].set_xlabel('rounds'); ax[1].set_ylabel('reader names the unit the writer would'); ax[1].set_title('Reading follows writing at level 2')
for a in ax: a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), frameon=False)
save(fig, 'model5-cut-gating')

# numbers
def reach_t(L, thr): b = L > thr; ok = np.array([b[i:].all() for i in range(len(L))]); return float(ck[ok.argmax()]) if ok.any() else float('inf')
num = dict(worlds=W, world_shown=w, bins1=L1, bins2=L2, probes={}, gating={}, agree={}, order={})
for ai, arm in enumerate(arms):
    lossb = lambda rule: [float(np.median(1000 * prb[:, ai, :, rules.index(rule), 0][(res[:, ai, :, 1] >= lo) & (res[:, ai, :, 1] < hi)])) for lo, hi in B1]                                 # change of the loss per 1,000 samples, states binned by valid1
    num['probes'][arm] = {rule: dict(level1=binned(ai, rule, 1), level2=binned(ai, rule, 2), loss=lossb(rule), level2_positive=[float(np.mean(prb[:, ai, :, rules.index(rule), 2][(res[:, ai, :, 2] >= lo) & (res[:, ai, :, 2] < hi)] > 0)) for lo, hi in B2]) for rule in rules}
    v1, v2 = res[:, ai, :, 1], res[:, ai, :, 2]; p = 1000 * prb[:, ai, :, rules.index('practice'), 2]; num['gating'][arm] = [float(np.median(p[(v1 >= lo) & (v1 < hi) & (v2 < 0.6)])) if ((v1 >= lo) & (v1 < hi) & (v2 < 0.6)).sum() >= 5 else None for lo, hi in G]
    num['agree'][arm] = {int(r): {k: float(np.median(res[:, ai, list(ck).index(r), keys.index(k)])) for k in keys if k.startswith(('agree', 'valid', 'sel', 'read'))} for r in (0, 16, 32, 64, 128, 256, 512, 2048)}
    A = tun[:, ai, -1, 0]; TA, TB = tun[:, ai, -1, 1], tun[:, ai, -1, 2]; bound = (res[:, ai, -1, 1] > 0.9) & (res[:, ai, -1, 2] > 0.9); ent = lambda M: -(M * np.log(M + 1e-300)).sum(-1).mean(-1)                          # entropy of the rows of a table, mean over rows
    num['order'][arm] = dict(bound=int(bound.sum()), duplicates=int(sum(len(set(a.argmax(1))) < 4 for a in A)), duplicates_among_stuck=int(sum(len(set(a.argmax(1))) < 4 for a in A[~bound])), stuck=int((~bound).sum()),
                             purity=float(np.median(A.max(1).mean(1)[bound])), code_first=int((ent(TA) < ent(TB))[bound].sum()), code_second=int((ent(TA) > ent(TB))[bound].sum()), sharp_rows=float(np.median(np.minimum(ent(TA), ent(TB))[bound])), two_rows=float(np.median(np.maximum(ent(TA), ent(TB))[bound]) / np.log(2)),
                             t1=float(np.median([reach_t(r[:, 1], 0.5) for r in res[:, ai]])), t2=float(np.median([reach_t(r[:, 2], 0.75) for r in res[:, ai]])),
                             units_at=[float(np.median((tun[:, ai, list(ck).index(r), 0].max(2) > 0.8).sum(1))) for r in (16, 32, 64, 128, 2048)])
num['ablation'] = {f'{es:g}': {arm: dict(t1=float(np.median([reach_t(r[:, 1], 0.5) for r in abl[:, si, ai]])), t2=float(np.median([reach_t(r[:, 2], 0.75) for r in abl[:, si, ai]])), bound=int(((abl[:, si, ai, -1, 1] > 0.9) & (abl[:, si, ai, -1, 2] > 0.9)).sum()), loss=float(np.median(abl[:, si, ai, -1, 0]))) for ai, arm in enumerate(arms)} for si, es in enumerate(d['sleeps'])}
num['judge'] = {int(n): dict(zip(('shown', 'valid_accepted', 'invalid_accepted', 'unseen_valid_accepted'), np.median(jud[:, i], 0).tolist())) for i, n in enumerate(d['sizes'])}
json.dump(num, open(OUT / 'cut_mechanism_numbers.json', 'w'), indent=1); print(json.dumps(num['order'], indent=0)); print(num['ablation']); print(num['judge']); print(num['gating'])
