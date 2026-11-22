"""Figure and summary numbers for order3_overlap.py.

    MPLBACKEND=Agg python order3_overlap_plot.py /var/tmp/jbauer/generative_order3/overlap_main/results.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
res = [r for path in sys.argv[1:] for r in json.load(open(path))['results']]


def get(c, key):
    for k in key.split('/'):
        c = c[k]
    return c


def curve(name, key):
    rs = [r for r in res if r['name'] == name]
    n = min(len(r['curve']) for r in rs)
    v = np.array([[get(c, key) for c in r['curve'][:n]] for r in rs], float)
    return np.array([c['samples'] for c in rs[0]['curve'][:n]]), np.median(v, 0), np.quantile(v, .25, 0), np.quantile(v, .75, 0)


def draw(ax, name, key, label, **kw):
    x, m, lo, hi = curve(name, key)
    x = np.maximum(x, 32)
    line, = ax.plot(x, m, label=label, **kw)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.15, lw=0)


end = lambda name, key, at=-1: float(curve(name, key)[1][at])
comps = lambda name: next(r['cfg']['comps'] for r in res if r['name'] == name)
legend = lambda ax: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=1, frameon=False)
BLANK, ALONE = np.log(27), np.log(27 / 2.5)                               # nats per component: uniform tokens; every check written by its own randomly chosen check unit

fig, axes = plt.subplots(2, 4, figsize=(32, 17), constrained_layout=True)

ax = axes[0, 0]
c = comps('practice')
for name, label, kw in (('practice', 'practice: own samples', dict(color='C3', lw=2.5)), ('watching', 'teacher strings, same reader and rule', dict(color='C0', lw=2.5)), ('twin', "a twin's samples", dict(color='C2')),
                        ('store_teacher', 'teacher strings, best unit stores the string', dict(color='k', ls=':'))):
    draw(ax, name, 'kl', label, **kw)
ax.axhline(c * BLANK, color='gray', lw=.8); ax.text(40, c * BLANK * 1.01, 'uniform tokens', color='gray')
ax.axhline(c * ALONE, color='gray', lw=.8, ls='--'); ax.text(40, c * ALONE * 1.01, 'check units alone (each check written on its own)', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples'); ax.set_ylabel('KL(teacher || learner), nats')
ax.set_title(f'(a) {6 * c} tokens, three overlapping checks per six tokens:\nunits compete per component when writing')
legend(ax)

ax = axes[0, 1]
draw(ax, 'practice', 'valid', 'components valid in a practice sample', color='C3', lw=2.5)
draw(ax, 'practice', 'kept', 'check units that fire again on the own string', color='C1')
draw(ax, 'practice', 'per_check_valid', 'components valid if every check is written by a different unit', color='C4', ls='--')
draw(ax, 'watching', 'valid', 'components valid, after teacher strings instead', color='C0')
ax.set_xscale('log'); ax.set_xlabel('own samples'); ax.set_ylabel('share'); ax.set_ylim(0, 1)
ax.set_title('(b) inside practice')
legend(ax)

ax = axes[0, 2]
for key, kw in (('single tokens', dict(color='C0')), ('token pairs', dict(color='C1')), ('token checks', dict(color='C3', lw=2.5))):
    draw(ax, 'practice', f'mmd/{key}', f'{key}, 1,024 units', **kw)
for name, ls, lab in (('units256_practice', ':', '256 units'), ('units4096_practice', '--', '4,096 units')):
    x, m, lo, hi = curve(name, 'mmd/token pairs')
    ax.plot(np.maximum(x, 32), m, color='C1', ls=ls, label=f'token pairs, {lab} ({6 * comps(name)} tokens)')
ax.axhspan(-3, 3, color='gray', alpha=.15, lw=0)
ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=10); ax.set_ylim(-5, None); ax.set_xlabel('own samples'); ax.set_ylabel('MMD$^2$ own vs teacher, in s.d. of teacher vs teacher')
ax.set_title('(c) can a statistic tell practice samples\nfrom teacher strings? (1,024 samples each; band: no)')
legend(ax)

ax = axes[0, 3]
for name, label, kw in (('practice', 'practice', dict(color='C3', lw=2.5)), ('watching', 'teacher strings, same reader and rule', dict(color='C0', lw=2.5)), ('store_teacher', 'teacher strings, best unit stores the string', dict(color='k', ls=':'))):
    draw(ax, name, 'kl_one_unit', label, **kw)
ax.axhline(c * BLANK, color='gray', lw=.8); ax.text(40, c * BLANK * 1.03, 'uniform tokens', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples'); ax.set_ylabel('KL(teacher || learner), nats')
ax.set_title('(d) the same learners\nif one unit writes the whole string')
legend(ax)

ax = axes[1, 0]
ca = comps('apart_practice')
draw(ax, 'apart_practice', 'kl', 'practice', color='C3', lw=2.5)
draw(ax, 'apart_watching', 'kl', 'teacher strings, same reader and rule', color='C0', lw=2.5)
ax.axhline(ca * 2 * np.log(3), color='gray', lw=.8); ax.text(40, ca * 2 * np.log(3) * .93, 'uniform tokens', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples'); ax.set_ylabel('KL(teacher || learner), nats'); ax.set_ylim(0, None)
ax.set_title('(e) control: checks that share no token\n(the first layer rejects nothing a deep unit writes)')
legend(ax)

ax = axes[1, 1]
widths = [(3, 'width3'), (9, 'width9'), (27, 'width27'), (comps('practice'), None)]
tok = [6 * w for w, _ in widths]
ax.plot(tok, [max(end(f'{n}_watching' if n else 'watching', 'teacher_fires', 0), 1e-4) for _, n in widths], 'o-', color='C0', label='teacher strings that fire a deep unit, at the start\n(lowest point: none of 2,048)')
ax.plot(tok, [max(end(f'{n}_practice' if n else 'practice', 'writer_fires', 0), 1e-4) for _, n in widths], 'o-', color='C3', label='own samples that fire their writer, at the start')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('tokens'); ax.set_ylabel('share of strings')
ax2 = ax.twinx()
ax2.plot(tok, [end(f'{n}_watching' if n else 'watching', 'kl') / w for w, n in widths], 's--', color='C0', label='KL per component after teacher strings (right)')
ax2.plot(tok, [end(f'{n}_practice' if n else 'practice', 'kl') / w for w, n in widths], 's--', color='C3', label='KL per component after practice (right)')
ax2.axhline(ALONE, color='gray', lw=.8, ls='--'); ax2.set_ylabel('KL per component, nats'); ax2.set_ylim(0, None)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=1, frameon=False)
ax.set_title('(f) width')

ax = axes[1, 2]
cs = comps('eps0.1_practice')
for name, label, color in (('eps0.03_practice', 'rate 0.03', 'C0'), ('eps0.1_practice', 'rate 0.1', 'C2'), ('width27_practice', 'rate 0.3', 'C3'), ('eps1.0_practice', 'rate 1 (the weights are the last read-back)', 'C4')):
    draw(ax, name, 'kl', label, color=color, lw=2)
ax.axhline(cs * ALONE, color='gray', lw=.8, ls='--')
ax.set_xscale('log'); ax.set_xlabel('own samples'); ax.set_ylabel('KL(teacher || learner), nats'); ax.set_ylim(0, None)
ax.set_title(f'(g) rate of the Hebb rule ({6 * cs} tokens)')
legend(ax)

ax = axes[1, 3]
ks = [(64, 'units64_practice'), (256, 'units256_practice'), (1024, 'width27_practice'), (4096, 'units4096_practice')]
ax.plot([k for k, _ in ks], [end(n, 'kl') for _, n in ks], 's-', color='C3', label='KL after practice, nats')
ax.plot([k for k, _ in ks], [end(n, 'mmd/token pairs', 0) for _, n in ks], 'o-', color='C1', label='token-pair statistic of practice samples at the start, in s.d.')
ax.axhspan(0, 3, color='gray', alpha=.15, lw=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('deep units')
ax.set_title(f'(h) number of deep units ({6 * cs} tokens)')
legend(ax)
fig.savefig(REPORTS / 'order3-overlap.png', dpi=100, facecolor='white')
fig.savefig(REPORTS / 'order3-overlap.svg')

out = {}
for name in sorted({r['name'] for r in res}):
    rs = [r for r in res if r['name'] == name]
    keys = ['valid', 'kept', 'silent', 'writer_fires', 'other_fires', 'teacher_fires', 'kl', 'kl_one_unit', 'per_check_valid', 'holders', 'sites_sharp', 'held_valid']
    x, kl, _, _ = curve(name, 'kl')
    out[name] = dict(worlds=len(rs), comps=rs[0]['cfg']['comps'], units=rs[0]['cfg']['units'], samples=int(x[-1]), start={k: end(name, k, 0) for k in keys}, end={k: end(name, k) for k in keys},
                     mmd_start={k: end(name, f'mmd/{k}', 0) for k in rs[0]['curve'][0]['mmd']}, mmd_end={k: end(name, f'mmd/{k}') for k in rs[0]['curve'][0]['mmd']},
                     curve=dict(samples=x.tolist(), kl=kl.tolist(), kl_one_unit=curve(name, 'kl_one_unit')[1].tolist(), valid=curve(name, 'valid')[1].tolist(), teacher_fires=curve(name, 'teacher_fires')[1].tolist(), sites_sharp=curve(name, 'sites_sharp')[1].tolist()))
json.dump(out, open(REPORTS / 'order3-overlap.json', 'w'), indent=1)
for name, o in out.items():
    print(f"{name:18s} C {o['comps']:2d} K {o['units']:4d} n {o['samples']:7d}  valid {o['start']['valid']:.3f}->{o['end']['valid']:.3f}  kl {o['start']['kl']:6.1f}->{o['end']['kl']:6.1f}  one-unit {o['start']['kl_one_unit']:6.1f}->{o['end']['kl_one_unit']:6.1f}  per-check {o['end']['per_check_valid']:.3f}  "
          f"wr {o['start']['writer_fires']:.2f}->{o['end']['writer_fires']:.2f} t_fires {o['start']['teacher_fires']:.4f}->{o['end']['teacher_fires']:.4f} sharp {o['end']['sites_sharp']:.2f}  mmd {[round(v, 1) for v in o['mmd_start'].values()]} -> {[round(v, 1) for v in o['mmd_end'].values()]}")
print(REPORTS / 'order3-overlap.png')
