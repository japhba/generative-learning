"""Figure for model5_cut_twin.py: own samples against a twin's at the same stage, two levels. Reads OUT/cut_twin.npz, writes OUT/figures/model5-cut-twin.{svg,png} and OUT/cut_twin_numbers.json.
Usage: python model5_cut_twin_figures.py"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model5_lateral_verdict import OUT

d = np.load(f'{OUT}/cut_twin.npz'); res, ck, arms, ns = d['res'], d['checks'], list(d['arms']), [int(n) for n in d['ns']]; N0 = ns.index(4096); x = np.maximum(ck, 0.5)   # res[n, world, arm, check, key]
LAB = dict(own_cause='own samples, credited to the units that wrote them', own_read='own samples, credited by reading', cross_read="a twin's samples, twin at the same stage", ahead_read="a twin's samples, twin ahead"); COL = dict(own_cause='C3', own_read='C1', cross_read='C4', ahead_read='C5')


def cross(L, thr):                                                                                                                 # the round at which the score last rises through thr (interpolated in log rounds); inf if it ends below
    if L[-1] <= thr: return np.inf
    i = np.flatnonzero(L <= thr)[-1]; return float(np.exp(np.interp(thr, [L[i], L[i + 1]], np.log([x[i], x[i + 1]]))))


fig, ax = plt.subplots(2, 2, figsize=plt.figaspect(0.85) * 1.6, constrained_layout=True); ax = ax.ravel()
for k, a_, ttl in ((1, ax[0], 'Writing, level 1'), (2, ax[1], 'Writing, level 2')):
    for a in arms: a_.plot(x, np.median(res[N0, :, arms.index(a), :, k], 0), color=COL[a], label=LAB[a])
    a_.set_title(ttl); a_.set_ylabel('written strings that obey it')
X = res[N0, :, arms.index('cross_read')]; ax[2].plot(x, np.median(X[:, :, 5], 0), color='k', label='pair of codes at a cut (level 1)'); ax[2].plot(x, np.median(X[:, :, 6], 0), '--', color='k', label='pair of cut values (level 2)')
ax[2].set_title('How far apart the two learners write'); ax[2].set_ylabel('total variation distance')
for k, lab, col in ((9, 'own samples', 'C1'), (11, "the twin's samples", 'C4')): ax[3].plot(x, np.median(X[:, :, k], 0), color=col, label=f'{lab}: level 1'); ax[3].plot(x, np.median(X[:, :, k + 1], 0), '--', color=col, label=f'{lab}: level 2')
ax[3].set_title("What the reader recovers of a sample's causes"); ax[3].set_ylabel('information of the named unit')
for a in ax: a.set_xscale('log', base=2); a.set_xlabel('rounds (one judged batch of 64 each)'); a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False)
(OUT / 'figures').mkdir(exist_ok=True); [fig.savefig(OUT / 'figures' / f'model5-cut-twin.{e}') for e in ('svg', 'png')]; print(OUT / 'figures' / 'model5-cut-twin.png')

num = {}
for n in ns:
    i = ns.index(n); T = {a: np.array([[cross(w[:, 1], 0.5), cross(w[:, 2], 0.75)] for w in res[i, :, arms.index(a)]]) for a in arms}; num[n] = {}
    for a in arms:
        X = res[i, :, arms.index(a)]; bound = (X[:, -1, 1] > 0.9) & (X[:, -1, 2] > 0.9); ok = np.isfinite(T[a]).all(1) & np.isfinite(T['own_read']).all(1); r = T[a][ok] / T['own_read'][ok]
        num[n][a] = dict(t1=float(np.median(T[a][:, 0])), t2=float(np.median(T[a][:, 1])), bound=int(bound.sum()), ratio1=float(np.median(r[:, 0])), ratio2=float(np.median(r[:, 1])), ratio2_iqr=[float(q) for q in np.percentile(r[:, 1], [25, 75])],
                         later1=int((r[:, 0] > 1).sum()), later2=int((r[:, 1] > 1).sum()), worlds=int(ok.sum()), stage2=float(np.median((T[a][:, 1] - T[a][:, 0])[ok])))
    Xc = res[i, :, arms.index('cross_read')]; num[n]['tv'] = {int(r): [float(np.median(Xc[:, list(ck).index(r), 5])), float(np.median(Xc[:, list(ck).index(r), 6]))] for r in (0, 16, 32, 64, 128, 256, 512, 2048)}; num[n]['reach'] = {int(r): [float(v) for v in np.median(Xc[:, list(ck).index(r), 9:13], 0)] for r in (0, 16, 32, 64, 128, 256, 512, 2048)}
json.dump(num, open(OUT / 'cut_twin_numbers.json', 'w'), indent=1)
for n in ns:
    print(n, 'tv at the cut, tv of the values:', {r: [round(v, 2) for v in t] for r, t in num[n]['tv'].items()}); print('   reach own 1, own 2, twin 1, twin 2:', {r: [round(v, 2) for v in t] for r, t in num[n]['reach'].items()})
    for a in arms: print('   ', a, {k: ([round(q, 2) for q in v] if isinstance(v, list) else round(v, 2) if isinstance(v, float) else v) for k, v in num[n][a].items()})
