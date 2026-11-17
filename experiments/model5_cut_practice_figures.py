"""Figure for model5_cut_practice.py: practice against the same amount of wake and sleep (README, Baselines, 17 Nov). Reads OUT/cut_practice_fresh.npz (wake on fresh teacher strings),
writes OUT/figures/model5-cut-practice.{svg,png} and OUT/cut_practice_numbers.json.   Usage: python model5_cut_practice_figures.py"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model5_lateral_verdict import OUT

d = np.load(f'{OUT}/cut_practice_fresh.npz'); res, ck, arms, uses, ns = d['res'][:, :, 0], d['checks'], list(d['arms']), d['uses'], [int(n) for n in d['ns']]; N0 = ns.index(1024)   # res[n, world, arm, check, key], first setting of the steps
LAB = dict(W='wake only', WS31='wake and sleep, 3 : 1', WS='wake and sleep, 1 : 1', WS13='wake and sleep, 1 : 3', Wtied="wake only, read by the writer's own posterior", WSP='wake, sleep and practice, 1 : 1', WSP13='wake, sleep and practice, 1 : 3',
           WSPread='own samples, credited by reading', WSPtwin="a twin's samples, credited by reading")
COL = dict(W='0.5', WS31='C0', WS='C0', WS13='C0', Wtied='C2', WSP='C3', WSP13='C3', WSPread='C1', WSPtwin='C4'); STY = dict(WS='--', WS13=':', WSP13='--')


def samples(a): return np.maximum(ck * 64 * uses[arms.index(a)], 32)                                                               # the start is drawn at half a batch


def at(ni, a, t, key=0):                                                                                                           # every world's score after t samples (interpolated in log samples)
    x = np.log(samples(a)); return np.array([np.interp(np.log(t), x, l) for l in res[ni, :, arms.index(a), :, key]])


fig, ax = plt.subplots(2, 2, figsize=plt.figaspect(0.85) * 1.6, constrained_layout=True)
for a in ('W', 'WS31', 'WS', 'WS13', 'Wtied', 'WSP', 'WSP13'):
    L = res[N0, :, arms.index(a), :, 0]; ax[0, 0].plot(samples(a), np.median(L, 0), STY.get(a, '-'), color=COL[a], label=LAB[a])
    if a in ('WS31', 'WSP'): ax[0, 0].fill_between(samples(a), *np.percentile(L, [25, 75], 0), color=COL[a], alpha=0.15, lw=0)
ax[0, 0].set_yscale('log'); ax[0, 0].set_ylabel('loss of the pair at the cut (nats)'); ax[0, 0].set_title('Practice against wake and sleep')
for a in ('WS', 'WSP'):
    ax[0, 1].plot(samples(a), np.median(res[N0, :, arms.index(a), :, 1], 0), color=COL[a], label=f'{LAB[a]}: written pairs that agree'); ax[0, 1].plot(samples(a), np.median(res[N0, :, arms.index(a), :, 2], 0), ':', color=COL[a], label=f'{LAB[a]}: teacher pairs read')
ax[0, 1].set_ylabel('share'); ax[0, 1].set_title('Writing and reading')
for T, ls, mk in ((8192, '-', 'o'), (131072, '--', 's')):
    best = [np.median(np.min([at(i, a, T) for a in ('WS31', 'WS', 'WS13')], 0)) for i in range(len(ns))]; ax[1, 0].plot(ns, best, ls, color='C0', marker=mk, label=f'wake and sleep, best split, after {T:,} samples')
    ax[1, 0].plot(ns, [np.median(at(i, 'WSP', T)) for i in range(len(ns))], ls, color='C3', marker=mk, label=f'wake, sleep and practice, 1 : 1, after {T:,} samples')
ax[1, 0].plot(ns, np.median(d['aux'][:, :, 3], 1), ':', color='k', label="the judge's own loss")
ax[1, 0].set_xscale('log', base=2); ax[1, 0].set_yscale('log'); ax[1, 0].set_xlabel('stored teacher strings'); ax[1, 0].set_ylabel('loss of the pair at the cut (nats)'); ax[1, 0].set_title('The window by size of the teacher set')
for a in ('WSP', 'WSPread', 'WSPtwin'): ax[1, 1].plot(samples(a), np.median(res[N0, :, arms.index(a), :, 0], 0), color=COL[a], label='own samples, credited to the unit that wrote them' if a == 'WSP' else LAB[a])
ax[1, 1].set_yscale('log'); ax[1, 1].set_ylabel('loss of the pair at the cut (nats)'); ax[1, 1].set_title('Whose samples')
for a in (ax[0, 0], ax[0, 1], ax[1, 1]): a.set_xscale('log', base=2); a.set_xlabel('samples used (teacher strings and own samples)')
for a in ax.ravel(): a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)
(OUT / 'figures').mkdir(exist_ok=True); [fig.savefig(OUT / 'figures' / f'model5-cut-practice.{e}') for e in ('svg', 'png')]; print(OUT / 'figures' / 'model5-cut-practice.png')

num = {}
for n in ns:
    i = ns.index(n); num[n] = {}
    for t in (2048, 4096, 8192, 16384, 65536, 262144):
        ws = np.min([at(i, a, t) for a in ('WS31', 'WS', 'WS13')], 0); p = at(i, 'WSP', t); num[n][t] = dict(best_ws=float(np.median(ws)), wsp=float(np.median(p)), wsp13=float(np.median(at(i, 'WSP13', t))), wtied=float(np.median(at(i, 'Wtied', t))), w=float(np.median(at(i, 'W', t))),
            read=float(np.median(at(i, 'WSPread', t))), twin=float(np.median(at(i, 'WSPtwin', t))), worlds_below_ws=int((p < ws).sum()), ratio=float(np.median(ws / p)), worlds_below_wtied=int((p < at(i, 'Wtied', t)).sum()), worlds_read_below_twin=int((at(i, 'WSPread', t) < at(i, 'WSPtwin', t)).sum()))
json.dump(dict(numbers=num, judge_loss=dict(zip(map(int, ns), np.median(d['aux'][:, :, 3], 1).tolist())), judge_accepts_wrong=dict(zip(map(int, ns), np.median(d['aux'][:, :, 1], 1).tolist()))), open(OUT / 'cut_practice_numbers.json', 'w'), indent=1)
for n in (512, 1024): print(n, {t: (round(v['best_ws'], 3), round(v['wsp'], 3), v['worlds_below_ws'], round(v['read'], 3), round(v['twin'], 3), v['worlds_read_below_twin']) for t, v in num[n].items()})
