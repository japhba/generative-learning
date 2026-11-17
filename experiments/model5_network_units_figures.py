"""Figure and numbers for section 11 of the model 5 note: a network learner with latent units in the writer (model5_network_units.py).
From /var/tmp/jbauer/model5_lateral_verdict/network_units.npz and network_units_scan.npz. Writes <out>/model5-network-units.svg/.png and <out>/units_numbers.json.   Usage: MPLBACKEND=Agg python model5_network_units_figures.py <out>"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

D = Path('/var/tmp/jbauer/model5_lateral_verdict'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#898781', '#0b0b0b'; W, H = plt.rcParams['figure.figsize']


def band(ax, x, y, color, label, **kw):                                                                                            # median over worlds (axis 0) with the interquartile range
    q1, med, q3 = np.percentile(y, [25, 50, 75], 0); ax.plot(x, med, color=color, label=label, **kw); ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0)


def med(x, axis=None): return np.median(x, axis).round(4).tolist()


d = np.load(D / 'network_units.npz'); R, F, ns, cfg, arms, check = d['res'], d['ref'], [int(n) for n in d['ns']], list(d['cfg']), list(d['arms']), d['check']; rk = list(d['refkeys']); c = cfg.index; a = arms.index   # R: sizes x learners x worlds x arms x checkpoints x scores
HEADS = (('comp16', 'units recruited from the hidden layer, no own samples', AQUA), ('tied16', "own units, reader = the writer's own posterior", ORANGE), ('raw16', 'own units, reader trained by sleep on the codes', YELLOW), ('trunk16', 'own units, reader trained by sleep on the hidden layer', BLUE))
ARMS = (('replay', 'more watching', GREY), ('own', 'practice on own samples', BLUE), ('own_replay', 'practice and watching in turn', AQUA), ('twin', "practice on a twin's samples", ORANGE), ('blind', 'practice, judge accepts everything', MAGENTA))
fig, axes = plt.subplots(2, 2, figsize=(1.9 * W, 2.1 * H), constrained_layout=True); ax = axes[0, 0]
band(ax, ns, F[:, 0, :, rk.index('table')].T, GREY, 'counted table of the teacher pairs', ls='--', marker='o'); band(ax, ns, F[:, 0, :, rk.index('classes')].T, INK, 'table over the true classes', ls=':', marker='o')
for name, label, col in HEADS: band(ax, ns, R[:, c(name), :, a('replay'), 0, 0].T, col, label, marker='o')
ax.plot(ns, np.median(F[:, c('comp16'), :, rk.index('limit')], 1), color=AQUA, ls='--', label='the same, in the limit of small wake steps')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('teacher pairs'); ax.set_ylabel('population loss of the writer (nats)'); ax.set_title('after watching'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22))
n = 4096; i = ns.index(n); ax = axes[0, 1]
for name, label, col in ARMS: band(ax, check[1:], R[i, c('comp16'), :, a(name), 1:, 0], col, label, marker='o')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel(f'further samples after {n:,} teacher pairs'); ax.set_ylabel('population loss of the writer (nats)'); ax.set_title('units from the hidden layer'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22))
for ax, key, ylabel in zip(axes[1], (0, 1), ('population loss of the writer (nats)', 'valid share of what the writer writes')):
    for name, label, col in ARMS: band(ax, check[1:], R[i, c('trunk16'), :, a(name), 1:, key], col, label, marker='o')
    ax.set_xscale('log', base=2); ax.set_xlabel(f'further samples after {n:,} teacher pairs'); ax.set_ylabel(ylabel); ax.set_title('own units, reader trained by sleep')
axes[1, 0].set_yscale('log'); fig.canvas.draw(); r = fig.canvas.get_renderer(); boxes = [(f'axes {j}', x.get_window_extent(r)) for j, x in enumerate(axes.ravel())] + [(f'legend {j}', x.get_legend().get_window_extent(r)) for j, x in enumerate(axes.ravel()) if x.get_legend()] + [(f'label {j}', t.get_window_extent(r)) for j, x in enumerate(axes.ravel()) for t in (x.xaxis.label, x.yaxis.label, x.title)]
fb = fig.bbox; print('figure', fb.width, fb.height, '| outside the figure:', [nm for nm, bb in boxes if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fb.width or bb.y1 > fb.height], '| overlaps:', [(p, q) for j, (p, bp) in enumerate(boxes) for q, bq in boxes[j + 1:] if bp.overlaps(bq) and not (p.startswith('label') and q.startswith('label'))])
fig.savefig(OUT / 'model5-network-units.svg', transparent=True); fig.savefig(OUT / 'model5-network-units.png', dpi=150, facecolor='white'); plt.close(fig); print(OUT / 'model5-network-units.png')

nums = dict(worlds=int(R.shape[2]), ns=ns, check=[int(x) for x in check], ref={k: med(F[:, 0, :, rk.index(k)], 1) for k in ('table', 'classes', 'hidden', 'judge', 'invalid', 'refused')}, units=med(F[:, c('comp16'), :, rk.index('units')], 1), limit=med(F[:, c('comp16'), :, rk.index('limit')], 1))
for name in cfg:
    e = dict(watch={k: med(R[:, c(name), :, 0, 0, j], 1) for j, k in enumerate(d['keys'])}, above_table=[int((R[i, c(name), :, 0, 0, 0] > F[i, 0, :, 0]).sum()) for i in range(len(ns))])
    for arm in arms: e[arm] = {k: [med(R[i, c(name), :, a(arm), :, j], 0) for i in range(len(ns))] for j, k in enumerate(d['keys'])}
    e['own_below_replay'] = [int((R[i, c(name), :, a('own'), -1, 0] < R[i, c(name), :, a('replay'), -1, 0]).sum()) for i in range(len(ns))]; e['mix_below_replay'] = [int((R[i, c(name), :, a('own_replay'), -1, 0] < R[i, c(name), :, a('replay'), -1, 0]).sum()) for i in range(len(ns))]
    e['own_below_twin'] = [int((R[i, c(name), :, a('own'), -1, 0] < R[i, c(name), :, a('twin'), -1, 0]).sum()) for i in range(len(ns))]; nums[name] = e
sc = np.load(D / 'network_units_scan.npz'); C, S, wc = sc['curve'], sc['res'], sc['wcheck']                                                # C: readers x worlds x watching checkpoints x scores; S: readers x worlds x arms x checkpoints x scores
nums['scan'] = dict(worlds=int(C.shape[1]), check=[int(x) for x in sc['check']], wcheck=[int(x) for x in wc], heads={str(h): dict(watch=med(C[i, :, :, 0], 0), plateau=int(np.median([wc[np.argmax(C[i, w, :, 0] < 1.3)] for w in range(C.shape[1])])),   # plateau: first checkpoint with a loss below 1.3 nats
                    **{str(arm): dict(loss=med(S[i, :, j, :, 0], 0), valid=med(S[i, :, j, :, 1], 0), below=int((S[i, :, j, -1, 0] < S[i, :, 0, -1, 0]).sum())) for j, arm in enumerate(sc['arms'])}) for i, h in enumerate(sc['heads'])})
assert (C[:, :, -1, 0] < 1.3).all()
(OUT / 'units_numbers.json').write_text(json.dumps(nums))
