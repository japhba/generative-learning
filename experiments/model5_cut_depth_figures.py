"""Figure for model5_cut_depth.py: two levels of the writer's own units, practice against the same amount of wake and sleep. Reads OUT/cut_depth_table.npz (the upper rule is a table; wake on fresh
teacher strings), writes OUT/figures/model5-cut-depth.{svg,png} and OUT/cut_depth_numbers.json.   Usage: python model5_cut_depth_figures.py"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model5_lateral_verdict import OUT

d = np.load(f'{OUT}/cut_depth_table.npz'); res, ck, arms, uses, ns = d['res'][:, :, 0], d['checks'], list(d['arms']), d['uses'], [int(n) for n in d['ns']]; N0 = ns.index(4096)   # res[n, world, arm, check, key]; keys: loss, valid1, valid2, read1, read2
LAB = dict(W='wake only', WS31='wake and sleep, 3 : 1', WS='wake and sleep, 1 : 1', Wtied="wake only, read by the writer's own posterior", WSP='wake, sleep and practice, 1 : 1')
COL = dict(W='0.5', WS31='C0', WS='C0', Wtied='C2', WSP='C3', WSPread='C1', WSPtwin='C4'); STY = dict(WS31='--')


def samples(a): return np.maximum(ck * 64 * uses[arms.index(a)], 32)                                                               # the start is drawn at half a batch


def reach(L, thr): b = L > thr; ok = np.array([b[i:].all() for i in range(len(L))]); return ck[ok.argmax()] if ok.any() else np.inf   # first check from which the score stays above thr


def times(ni, a, k1, k2, th1, th2): X = res[ni, :, arms.index(a)]; u = 64 * uses[arms.index(a)]; return np.array([reach(x[:, k1], th1) for x in X]) * u, np.array([reach(x[:, k2], th2) for x in X]) * u


def at(ni, a, t, key=0): x = np.log(samples(a)); return np.array([np.interp(np.log(t), x, l) for l in res[ni, :, arms.index(a), :, key]])


fig, ax = plt.subplots(2, 2, figsize=plt.figaspect(0.85) * 1.6, constrained_layout=True)
for a in ('W', 'WS31', 'WS', 'Wtied', 'WSP'):
    L = res[N0, :, arms.index(a), :, 0]; ax[0, 0].plot(samples(a), np.median(L, 0), STY.get(a, '-'), color=COL[a], label=LAB[a])
    if a in ('WS', 'WSP'): ax[0, 0].fill_between(samples(a), *np.percentile(L, [25, 75], 0), color=COL[a], alpha=0.15, lw=0)
ax[0, 0].set_yscale('log'); ax[0, 0].set_ylabel('loss at the cuts (nats)'); ax[0, 0].set_title('Practice against wake and sleep')
for k, key, ttl, a_ in ((1, 2, 'Writing, level by level', ax[0, 1]), (3, 4, 'Reading teacher strings, level by level', ax[1, 0])):
    for a in ('WS', 'WSP'):
        X = res[N0, :, arms.index(a)]; a_.plot(samples(a), np.median(X[:, :, k], 0), color=COL[a], label=f'{LAB[a]}: level 1'); a_.plot(samples(a), np.median(X[:, :, key], 0), '--', color=COL[a], label=f'{LAB[a]}: level 2')
    a_.set_title(ttl)
ax[0, 1].set_ylabel('written strings that obey it'); ax[1, 0].set_ylabel('information of the named unit')
for a, lab in (('WSP', 'own samples, credited to the units that wrote them'), ('WSPread', 'own samples, credited by reading'), ('WSPtwin', "a twin's samples, credited by reading")): ax[1, 1].plot(samples(a), np.median(res[N0, :, arms.index(a), :, 2], 0), color=COL[a], label=lab)
ax[1, 1].set_ylabel('written strings that obey level 2'); ax[1, 1].set_title('Whose samples')
for a in ax.ravel(): a.set_xscale('log', base=2); a.set_xlabel('samples used (teacher strings and own samples)'); a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)
(OUT / 'figures').mkdir(exist_ok=True); [fig.savefig(OUT / 'figures' / f'model5-cut-depth.{e}') for e in ('svg', 'png')]; print(OUT / 'figures' / 'model5-cut-depth.png')

num = {}
for n in ns:
    i = ns.index(n); num[n] = dict(arms={}, at={})
    for a in arms:
        X = res[i, :, arms.index(a)]
        if np.isnan(X).all(): continue
        t1, t2 = times(i, a, 1, 2, 0.5, 0.75); r1, r2 = times(i, a, 3, 4, 0.5, 0.5); bound = (X[:, -1, 1] > 0.9) & (X[:, -1, 2] > 0.9)
        num[n]['arms'][a] = dict(t1=float(np.median(t1)), t2=float(np.median(t2)), reach1=int(np.isfinite(t1).sum()), reach2=int(np.isfinite(t2).sum()), level2_not_before=int((t2 >= t1)[np.isfinite(t2)].sum()), r1=float(np.median(r1)), r2=float(np.median(r2)),
                                 read2_not_before=int((r2 >= r1)[np.isfinite(r2)].sum()), reach_r2=int(np.isfinite(r2).sum()), bound=int(bound.sum()), good=int((bound & (X[:, -1, 0] < 0.2)).sum()), final=float(np.median(X[:, -1, 0])), final_mean=float(X[:, -1, 0].mean()))
    for t in (4096, 8192, 32768, 131072, 262144):
        ws = np.min([at(i, a, t) for a in ('WS31', 'WS', 'WS13')], 0); p = at(i, 'WSP', t); num[n]['at'][t] = dict(best_ws=float(np.median(ws)), wsp=float(np.median(p)), wtied=float(np.median(at(i, 'Wtied', t))), w=float(np.median(at(i, 'W', t))), worlds_below_ws=int((p < ws).sum()), ratio=float(np.median(ws / p)))
json.dump(num, open(OUT / 'cut_depth_numbers.json', 'w'), indent=1)
for n in ns:
    print(n, {t: (round(v['best_ws'], 3), round(v['wsp'], 3), v['worlds_below_ws'], round(v['ratio'], 2)) for t, v in num[n]['at'].items()})
    for a in ('WS', 'WS31', 'WS13', 'WSP', 'Wtied', 'WSPread', 'WSPtwin', 'twin'): print('   ', a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in num[n]['arms'][a].items()})
