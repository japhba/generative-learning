"""Figure and summary numbers for order3_practice.py (and the single-block grid of order3_paper_checks.py).

    MPLBACKEND=Agg python order3_practice_plot.py /var/tmp/jbauer/generative_order3/run_main/results.json
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
scan = [s for path in sys.argv[1:] for s in json.load(open(path)).get('judge_scan', [])]
paper = json.load(open(REPORTS / 'order3-paper-checks.json'))
Q = 3


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


def end(name, key, at=-1):
    return float(curve(name, key)[1][at])


def blocks(name):
    return next(r['cfg']['blocks'] for r in res if r['name'] == name)


fig, axes = plt.subplots(3, 3, figsize=(26, 27), constrained_layout=True)
legend = lambda ax, **kw: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=1, frameon=False, **kw)

# (a) the per-check rule on one block
ax = axes[0, 0]
etas, lams = sorted({r['eta'] for r in paper['time_to_valid']}), sorted({r['lam'] for r in paper['time_to_valid']})
grid = np.array([[next(r['invalid']['median'] for r in paper['time_to_valid'] if r['eta'] == e and r['lam'] == l) for l in lams] for e in etas])
im = ax.imshow(np.log10(np.where(np.isfinite(grid), grid, np.nan)), origin='lower', aspect='auto', cmap='viridis_r')
for i, e in enumerate(etas):
    for j, l in enumerate(lams):
        d0 = np.log1p(e) / Q + (1 - 1 / Q) * np.log1p(-l)
        ax.text(j, i, f"{grid[i, j]:.0f}\n({d0:+.2f})" if np.isfinite(grid[i, j]) else f"never\n({d0:+.2f})", ha='center', va='center', color='w' if np.isfinite(grid[i, j]) and grid[i, j] > 100 else 'k')
ax.set_xticks(range(len(lams)), lams); ax.set_yticks(range(len(etas)), etas)
ax.set_xlabel('decay of a failed check'); ax.set_ylabel('reinforcement of a passed check')
ax.set_title('(a) one block alone: samples until a wrong triple\nhas become a valid one (in brackets: mean change\nof a log weight at a diffuse block)')
fig.colorbar(im, ax=ax, label='log10 samples')

# (b) first level, generator in which the units compete per check
ax = axes[0, 1]
b = blocks('practice')
for name, label, kw in (('practice', 'practice: own samples, credit by reading', dict(color='C3', lw=2.5)), ('practice_writer_credit', 'practice, writer credited (tag)', dict(color='C1', ls='--')),
                        ('watching', 'teacher strings, same reader and rule', dict(color='C0', lw=2.5)), ('twin', "a twin's samples", dict(color='C2')),
                        ('store_teacher', 'teacher strings, best unit stores the string', dict(color='k', ls=':')), ('practice_free_world', 'practice in a world without a rule (nothing is rejected)', dict(color='C4', ls='-.'))):
    draw(ax, name, 'kl', label, **kw)
ax.axhline(b * np.log(Q), color='gray', lw=.8); ax.text(40, b * np.log(Q) * 1.03, 'blank deep layer', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples'); ax.set_ylabel('KL(teacher || learner), nats')
ax.set_title(f'(b) first level, {3 * b} tokens:\nunits compete per check when writing')
legend(ax)

# (c) the same learners if one unit writes the whole string
ax = axes[0, 2]
for name, label, kw in (('practice', 'practice', dict(color='C3', lw=2.5)), ('watching', 'teacher strings, same reader and rule', dict(color='C0', lw=2.5)), ('store_teacher', 'teacher strings, best unit stores the string', dict(color='k', ls=':'))):
    draw(ax, name, 'kl_one_unit', label, **kw)
ax.axhline(b * np.log(Q), color='gray', lw=.8); ax.text(40, b * np.log(Q) * 1.08, 'blank deep layer', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples'); ax.set_ylabel('KL(teacher || learner), nats')
ax.set_title('(c) the same learners\nif one unit writes the whole string')
legend(ax)

# (d) surface statistics of the practice samples
ax = axes[1, 0]
for key, kw in (('single tokens', dict(color='C0')), ('token pairs', dict(color='C1')), ('token checks', dict(color='C3', lw=2.5))):
    draw(ax, 'practice', f'mmd_practice/{key}', f'{key}, 1,024 units', **kw)
for name, ls, lab in (('units256_practice', ':', '256 units'), ('units4096_practice', '--', '4,096 units'), ('softstart_practice', '-.', '1,024 units, softer start')):
    x, m, lo, hi = curve(name, 'mmd_practice/token pairs')
    ax.plot(np.maximum(x, 32), m, color='C1', ls=ls, label=f'token pairs, {lab} (81 tokens)')
ax.axhspan(-3, 3, color='gray', alpha=.15, lw=0)
ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=10); ax.set_xlabel('own samples'); ax.set_ylabel('MMD$^2$ own vs teacher, in s.d. of teacher vs teacher')
ax.set_ylim(-5, None)
ax.set_title('(d) first level: can a statistic tell own samples\nfrom teacher strings? (1,024 samples each; band: no)')
legend(ax)

# (e) second level
ax = axes[1, 1]
b2 = blocks('level2_practice')
for name, label, kw in (('level2_practice', 'practice: own samples, credit by reading', dict(color='C3', lw=2.5)), ('level2_watching', 'teacher strings, same reader and rule', dict(color='C0', lw=2.5)),
                        ('level2_twin', "a twin's samples", dict(color='C2')), ('level2_store_teacher', 'teacher strings, best unit stores the string', dict(color='k', ls=':'))):
    draw(ax, name, 'kl', label, **kw)
ax.axhline(b2 / 3 * np.log(Q), color='gray', lw=.8); ax.text(40, b2 / 3 * np.log(Q) * 1.03, 'blank deep layer', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples'); ax.set_ylabel('KL(teacher || learner), nats')
ax.set_title(f'(e) second level, on the code of {b2} finished blocks:\njudged by checks over three blocks')
legend(ax)

# (f) second level: surface statistics
ax = axes[1, 2]
for key, kw in (('single tokens', dict(color='C0')), ('token pairs', dict(color='C1')), ('token checks', dict(color='C2')), ('block pairs', dict(color='C4')), ('block checks', dict(color='C3', lw=2.5))):
    draw(ax, 'level2_practice', f'mmd_practice/{key}', key, **kw)
ax.axhspan(-3, 3, color='gray', alpha=.15, lw=0)
ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=10); ax.set_xlabel('own samples'); ax.set_ylabel('MMD$^2$ own vs teacher, in s.d. of teacher vs teacher')
ax.set_ylim(-5, None)
ax.set_title('(f) second level: which statistic tells own samples\nfrom teacher strings')
legend(ax)

# (g) width
ax = axes[2, 0]
widths = [(3, 'width3'), (9, 'width9'), (27, 'width27'), (blocks('practice'), None)]
tok = [3 * w for w, _ in widths]
fires = [max(end(f'{n}_watching' if n else 'watching', 'teacher_fires', 0), 1e-4) for _, n in widths]
ax.plot(tok, fires, 'o-', color='C0', label='teacher strings that fire a deep unit, at the start\n(lowest point: none of 2,048)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('tokens (width of a deep unit)'); ax.set_ylabel('share of teacher strings')
ax2 = ax.twinx()
ax2.plot(tok, [end(f'{n}_watching' if n else 'watching', 'kl') / w for w, n in widths], 's--', color='C0', label='KL per block after teacher strings (right)')
ax2.plot(tok, [end(f'{n}_practice' if n else 'practice', 'kl') / w for w, n in widths], 's-', color='C3', label='KL per block after practice (right)')
ax2.axhline(np.log(Q), color='gray', lw=.8)
ax2.set_ylabel('KL per block, nats')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=1, frameon=False)
ax.set_title('(g) width: the threshold shuts teacher strings\nout of wide units only')

# (h) step sizes in the full learner
ax = axes[2, 1]
names = [r['name'] for r in res if r['name'].startswith('step_')]
etas, lams = sorted({float(n.split('_')[1][3:]) for n in names}), sorted({float(n.split('_')[2][3:]) for n in names})
val = lambda e, l: end(f'step_eta{e}_lam{l}', 'sites_held')
grid = np.array([[val(e, l) for l in lams] for e in etas])
im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=27)
for i, e in enumerate(etas):
    for j, l in enumerate(lams):
        ax.text(j, i, f"{grid[i, j]:.0f}\n({np.log1p(e) / Q + (1 - 1 / Q) * np.log1p(-l):+.2f})", ha='center', va='center', color='k' if grid[i, j] > 16 else 'w')
ax.set_xticks(range(len(lams)), lams); ax.set_yticks(range(len(etas)), etas)
ax.set_xlabel('decay of a failed check'); ax.set_ylabel('reinforcement of a passed check')
ax.set_title('(h) full learner, 81 tokens: checks a unit still holds\nafter practice, of 27 (9 were right at the start;\nin brackets as in a)')
fig.colorbar(im, ax=ax, label='checks held per unit')

# (i) attempts at units that are local
ax = axes[2, 2]
for name, label, color in (('sparse_lam0.2', 'units wired to a third of the tokens, decay 0.2', 'C1'), ('sparse_lam0.4', 'the same, decay 0.4', 'C3'), ('groups_coactive', 'groups of units written together', 'C4')):
    draw(ax, name, 'kl', f'{label}: KL', color=color, lw=2.5)
    draw(ax, name, 'writer_fires', None, color=color, ls=':')
ax.axhline(27 * np.log(Q), color='gray', lw=.8); ax.text(40, 27 * np.log(Q) * 1.05, 'blank deep layer', color='gray')
ax.plot([], [], color='k', ls=':', label='dotted: share of writers that fire on their own sample\n(same axis, 0 to 1)')
ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1); ax.set_xlabel('own samples'); ax.set_ylabel('KL(teacher || learner), nats')
ax.set_ylim(0, None)
ax.set_title('(i) 81 tokens: attempts at local units')
legend(ax, **{})
fig.savefig(REPORTS / 'order3-practice.png', dpi=110, facecolor='white')
fig.savefig(REPORTS / 'order3-practice.svg')

# numbers
out = {}
for name in sorted({r['name'] for r in res}):
    rs = [r for r in res if r['name'] == name]
    keys = ['check_pass', 'all_pass', 'kl', 'kl_one_unit', 'writer_fires', 'other_fires', 'teacher_fires', 'teacher_units', 'sites_held', 'held_valid', 'inside_valid', 'sharp_outside', 'holders_per_site', 'sites_without', 'coactive']
    x, kl, _, _ = curve(name, 'kl')
    half = (kl[0] + kl[-1]) / 2
    out[name] = dict(worlds=len(rs), blocks=rs[0]['cfg']['blocks'], samples=int(x[-1]), start={k: end(name, k, 0) for k in keys}, end={k: end(name, k) for k in keys},
                     samples_to_half_kl=int(x[np.argmax(kl <= half)]) if kl[-1] < kl[0] else None,
                     mmd_practice_start={k: end(name, f'mmd_practice/{k}', 0) for k in rs[0]['curve'][0]['mmd_practice']}, mmd_practice_end={k: end(name, f'mmd_practice/{k}') for k in rs[0]['curve'][0]['mmd_practice']},
                     mmd_generator_end={k: end(name, f'mmd/{k}') for k in rs[0]['curve'][0]['mmd']}, judge_exact=float(np.mean([r['judge']['sites_exact'] for r in rs])), judge_strings=rs[0]['judge']['n_strings'],
                     kl_curve=dict(samples=x.tolist(), kl=kl.tolist(), kl_one_unit=curve(name, 'kl_one_unit')[1].tolist(), check_pass=curve(name, 'check_pass')[1].tolist(), teacher_fires=curve(name, 'teacher_fires')[1].tolist()))
out['judge_scan'] = {f"level {lv}, {n} strings": float(np.mean([s['sites_exact'] and s['allowed_exact'] for s in scan if s['level'] == lv and s['strings'] == n])) for lv in (1, 2) for n in sorted({s['strings'] for s in scan if s['level'] == lv})}
json.dump(out, open(REPORTS / 'order3-practice.json', 'w'), indent=1)
for name, o in out.items():
    if name != 'judge_scan':
        print(f"{name:26s} B {o['blocks']:3d} n {o['samples']:7d}  pass {o['start']['check_pass']:.3f}->{o['end']['check_pass']:.3f}  kl {o['start']['kl']:6.1f}->{o['end']['kl']:6.1f} (half {o['samples_to_half_kl']})  one-unit {o['start']['kl_one_unit']:6.1f}->{o['end']['kl_one_unit']:6.1f}  "
              f"wr {o['end']['writer_fires']:.2f} t_fires {o['start']['teacher_fires']:.4f}->{o['end']['teacher_fires']:.4f}  held {o['end']['sites_held']:.2f} valid {o['end']['held_valid']:.2f} outside {o['end']['sharp_outside']:.1f}  "
              f"mmdP {[round(v, 1) for v in o['mmd_practice_start'].values()]} -> {[round(v, 1) for v in o['mmd_practice_end'].values()]}  gen {[round(v, 1) for v in o['mmd_generator_end'].values()]}")
print(out['judge_scan'])
print(REPORTS / 'order3-practice.png')
