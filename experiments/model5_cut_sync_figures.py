"""Figure for model5_cut_sync.py: what is left of practice's lead, and of own samples' lead over a twin's, when reader and writer are in sync. Reads OUT/cut_sync.npz and OUT/cut_sync_decay01.npz
(the same with a wake step that shrinks), writes OUT/figures/model5-cut-sync.{svg,png} and OUT/cut_sync_numbers.json.   Usage: python model5_cut_sync_figures.py"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model5_lateral_verdict import OUT

d = np.load(f'{OUT}/cut_sync.npz'); res, ck, syncs, arms, keys, uses = d['res'], d['checks'], list(d['syncs']), list(d['arms']), list(d['keys']), d['uses']; dd = np.load(f'{OUT}/cut_sync_decay01.npz')['res']   # res[world, sync, arm, check, key]
ORDER = [s for s in syncs if s != 'exact'] + ['exact']; XL = ORDER[:-1] + ['exact']; XLAB = "reader's sleep step (exact: the writer's own posterior)"; x = np.maximum(ck, 0.5); W = res.shape[0]
LAB = dict(wake='no practice', own_cause='own samples, credited to the units that wrote them', own_read='own samples, credited by reading', twin_read="a twin's samples, credited by reading"); COL = dict(wake='C0', own_cause='C3', own_read='C1', twin_read='C4')
SRC = dict(teacher='teacher strings', own='own samples', twin="a twin's samples")


def cross(L, thr):                                                                                                                 # the round at which the score last rises through thr (interpolated in log rounds); inf if it ends below
    if L[-1] <= thr: return np.inf
    i = np.flatnonzero(L <= thr)[-1]; return float(np.exp(np.interp(thr, [L[i], L[i + 1]], np.log([x[i], x[i + 1]]))))


def times(R, s, a): X = R[:, syncs.index(s), arms.index(a)]; u = 64 * uses[syncs.index(s), arms.index(a)]; return np.array([cross(w[:, 1], 0.5) for w in X]) * u, np.array([cross(w[:, 2], 0.75) for w in X]) * u   # samples until level 1, level 2 is half written


def bound(R, s, a): X = R[:, syncs.index(s), arms.index(a), -1]; return (X[:, 1] > 0.9) & (X[:, 2] > 0.9)


def samples(s, a): return np.maximum(ck * 64 * uses[syncs.index(s), arms.index(a)], 32)


def at(R, s, a, n): return np.array([np.interp(np.log(n), np.log(samples(s, a)), l) for l in R[:, syncs.index(s), arms.index(a), :, 0]])


fig, ax = plt.subplots(2, 2, figsize=plt.figaspect(0.85) * 1.6, constrained_layout=True); ax = ax.ravel(); pos = np.arange(len(ORDER))
for a in ('wake', 'own_cause'):
    T = np.array([[np.median(t) for t in times(res, s, a)] for s in ORDER]); lab = LAB[a] if a == 'wake' else 'with practice'; ax[0].plot(pos, T[:, 0], 'o-', color=COL[a], label=f'{lab}: level 1'); ax[0].plot(pos, T[:, 1], 'o--', color=COL[a], label=f'{lab}: level 2')
ax[0].set_yscale('log'); ax[0].set_xticks(pos, XL); ax[0].set_xlabel(XLAB); ax[0].set_ylabel('samples until the level is half written'); ax[0].set_title('When each level is written')
for i, a in enumerate(('wake', 'own_read', 'twin_read')): ax[1].bar(pos + (i - 1) * 0.27, [bound(res, s, a).sum() for s in ORDER], 0.27, color=COL[a], label=LAB[a])
ax[1].set_xticks(pos, XL); ax[1].set_xlabel(XLAB); ax[1].set_ylabel(f'worlds bound at both levels (of {W})'); ax[1].set_title('Where the learner ends')
for a in ('wake', 'own_read', 'twin_read'):
    L = res[:, syncs.index('exact'), arms.index(a), :, 0]; ax[2].plot(samples('exact', a), np.median(L, 0), color=COL[a], label=LAB[a]); ax[2].fill_between(samples('exact', a), *np.percentile(L, [25, 75], 0), color=COL[a], alpha=0.15, lw=0)
    ax[2].plot(samples('exact', a), np.median(dd[:, 0, arms.index(a), :, 0], 0), ':', color=COL[a])
ax[2].plot([], [], ':', color='0.3', label='the same with a wake step that shrinks'); ax[2].set_xscale('log', base=2); ax[2].set_yscale('log'); ax[2].set_xlabel('samples used'); ax[2].set_ylabel('loss at the cuts (nats)'); ax[2].set_title("Reader = the writer's own posterior")
X = res[:, syncs.index('0.003'), arms.index('own_cause')]
for src, col in (('teacher', 'k'), ('own', 'C1'), ('twin', 'C4')): ax[3].plot(x, np.median(X[:, :, keys.index(f'agree1_{src}')], 0), color=col, label=f'{SRC[src]}: level 1'); ax[3].plot(x, np.median(X[:, :, keys.index(f'agree2_{src}')], 0), '--', color=col, label=f'{SRC[src]}: level 2')
ax[3].set_xscale('log', base=2); ax[3].set_xlabel('rounds'); ax[3].set_ylabel('reader names the unit the writer would'); ax[3].set_title('Whose strings the reader reads (sleep step 0.003)')
for a in ax: a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), frameon=False)
(OUT / 'figures').mkdir(exist_ok=True); fig.savefig(OUT / 'figures' / 'model5-cut-sync.svg', transparent=True); fig.savefig(OUT / 'figures' / 'model5-cut-sync.png', dpi=150, facecolor='white')

num = dict(worlds=W, syncs={})
for s in ORDER:
    e = num['syncs'][s] = {}
    for a in arms: t1, t2 = times(res, s, a); e[a] = dict(t1=float(np.median(t1)), t2=float(np.median(t2)), bound=int(bound(res, s, a).sum()), final=float(np.median(res[:, syncs.index(s), arms.index(a), -1, 0])), at={str(n): float(np.median(at(res, s, a, n))) for n in (8192, 32768, 131072)})
    for lev in (0, 1):
        o, tw, oc, w = (times(res, s, a)[lev] for a in ('own_read', 'twin_read', 'own_cause', 'wake')); ok, ok2, ok3 = np.isfinite(o) & np.isfinite(tw), np.isfinite(o) & np.isfinite(oc), np.isfinite(w) & np.isfinite(oc)
        e[f'level{lev + 1}'] = dict(twin_over_own=float(np.median(tw[ok] / o[ok])), twin_later=int((tw[ok] > o[ok]).sum()), twin_worlds=int(ok.sum()), read_over_cause=float(np.median(o[ok2] / oc[ok2])), read_later=int((o[ok2] > oc[ok2]).sum()), wake_over_practice=float(np.median(w[ok3] / oc[ok3])), wake_later=int((w[ok3] > oc[ok3]).sum()), wake_worlds=int(ok3.sum()))
for tag, R in (('constant', res), ('shrinking', dd)):
    e = num[f'exact_{tag}'] = {a: dict(bound=int(((R[:, 0, arms.index(a), -1, 1] > 0.9) & (R[:, 0, arms.index(a), -1, 2] > 0.9)).sum()), at={str(n): float(np.median([np.interp(np.log(n), np.log(samples('exact', a)), l) for l in R[:, 0, arms.index(a), :, 0]])) for n in (8192, 32768, 131072)}) for a in arms}
    for n in (8192, 32768, 131072): lw, lo = (np.array([np.interp(np.log(n), np.log(samples('exact', a)), l) for l in R[:, 0, arms.index(a), :, 0]]) for a in ('wake', 'own_read')); e[f'own_below_wake_{n}'] = int((lo < lw).sum())
X = res[:, syncs.index('0.003')]; num['agree'] = {a: {str(r): {k: float(np.median(X[:, arms.index(a), list(ck).index(r), keys.index(k)])) for k in keys[7:]} for r in (32, 64, 128, 256, 512, 2048)} for a in ('wake', 'own_cause')}
json.dump(num, open(OUT / 'cut_sync_numbers.json', 'w'), indent=1); print(OUT / 'figures' / 'model5-cut-sync.png')
