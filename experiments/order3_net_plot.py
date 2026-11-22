"""Figure and summary numbers for order3_net.py.

    MPLBACKEND=Agg python order3_net_plot.py /var/tmp/jbauer/generative_order3
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
root = Path(sys.argv[1])
load = lambda d: json.load(open(root / d / 'results.json'))['results']
res = load('net_core') + load('net_more') + load('net_budget')
after = load('net_after')                                                                        # the arms after 65,536 teacher strings, more evaluation points
SETS = (('324 tokens,\nsame rate', ('net_wide',)), ('216 tokens,\nrate 0.001', ('net_w36',)), ('324 tokens,\nrate 0.0005', ('net_w54',)), ('three values\nper token', ('net_q3', 'net_ws_q_3_states_3_samples_524288')),
        ('sharp random\nstart', ('net_rob_init_2.0', 'net_ws_init_2.0')), ('1.5 x deep\ngroups', ('net_rob_groups_81', 'net_ws_groups_81')), ('rate 0.01', ('net_rob_eps_0.01', 'net_ws_eps_0.01')))
other = {tag: sum((load(d) for d in dirs), []) for tag, dirs in SETS if all((root / d / 'results.json').exists() for d in dirs)}


def get(c, key):
    for k in key.split('/'):
        c = c[k]
    return c


def curve(name, key, rs=None):
    rs = [r for r in (res if rs is None else rs) if r['name'] == name]
    n = min(len(r['curve']) for r in rs)
    v = np.array([[get(c, key) for c in r['curve'][:n]] for r in rs], float)
    return np.array([c['samples'] for c in rs[0]['curve'][:n]]), np.nanmedian(v, 0), np.nanquantile(v, .25, 0), np.nanquantile(v, .75, 0)


def draw(ax, name, key, label, rs=None, **kw):
    x, m, lo, hi = curve(name, key, rs)
    x = np.maximum(x, 32)
    line, = ax.plot(x, m, label=label, **kw)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=.15, lw=0)


end = lambda name, key, at=-1, rs=None: float(curve(name, key, rs)[1][at])
legend = lambda ax, **kw: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=1, frameon=False, **kw)
cfg = next(r['cfg'] for r in res if r['name'] == 'watching')
C, K = cfg['comps'], cfg['groups']
START = end('watching', 'kl', 0)

fig = plt.figure(figsize=(32, 17), constrained_layout=True)
gs = fig.add_gridspec(2, 4)
axes = np.array([[fig.add_subplot(gs[i, j]) if (i, j) != (0, 2) else None for j in range(4)] for i in range(2)])

ax = axes[0, 0]
for name, label, kw in (('watching', 'teacher strings only', dict(color='C0', lw=2.5)), ('watch_sleep', 'half own samples, they teach the reader', dict(color='C3', lw=2.5)),
                        ('mix', 'half own samples, they teach reader and writer', dict(color='C1')), ('ws', 'classic wake-sleep (teacher strings teach the writer only)', dict(color='C4', ls='--')),
                        ('mix_g', 'half own samples, they teach the writer only', dict(color='C5', ls=':')), ('ws_twin', "classic wake-sleep on a twin's samples", dict(color='C2')),
                        ('practice', 'own samples only', dict(color='gray', ls=':'))):
    draw(ax, name, 'kl', label, **kw)
ax.axhline(START, color='gray', lw=.8); ax.text(40, START * 1.01, 'first layer alone', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples (teacher strings + own samples)'); ax.set_ylabel('KL(teacher || learner), nats'); ax.set_ylim(0, None)
ax.set_title(f'(a) {6 * C} binary tokens, {K} deep groups of two units')
legend(ax)

ax = axes[0, 1]
for name, color, lab in (('watching', 'C0', 'teacher strings only'), ('watch_sleep', 'C3', 'half own samples')):
    draw(ax, name, 'tokens_owned', f'{lab}: tokens that a deep group writes', color=color, lw=2.5)
    draw(ax, name, 'tokens_shared', f'{lab}: tokens that two or more groups write', color=color, ls='--')
ax.axhline(1 - np.exp(-1), color='gray', lw=.8); ax.text(40, 1 - np.exp(-1) + .01, 'groups choosing tokens at random: 1 - 1/e', color='gray')
ax.axhline(1 - 2 * np.exp(-1), color='gray', lw=.8, ls='--'); ax.text(40, 1 - 2 * np.exp(-1) + .01, '1 - 2/e', color='gray')
ax.set_xscale('log'); ax.set_xlabel('samples'); ax.set_ylabel('share of the shared tokens'); ax.set_ylim(0, 1.02)
ax.set_title('(b) what the deep groups hold')
legend(ax)

sub = gs[0, 2].subgridspec(1, 2)
r = next(r for r in after if r['name'] == 'first65536_mix' and r['seed'] == 0)
at = next(i for i, c in enumerate(r['curve']) if c['samples'] == r['cfg']['watch_first'])
last = np.array(r['curve'][-1]['loadings'])
rows = np.argsort(last.argmax(1))
for j, (i, title) in enumerate(((at, f"(c) one world, after\n{r['cfg']['watch_first']:,} teacher strings"), (-1, 'then own samples in turn\nwith teacher strings'))):
    a = fig.add_subplot(sub[0, j])
    im = a.imshow(np.array(r['curve'][i]['loadings'])[rows], vmin=0, vmax=1, cmap='Greys', aspect='auto', interpolation='nearest')
    a.set_xlabel('shared token'); a.set_title(title)
    if j == 0:
        a.set_ylabel('deep group')
fig.colorbar(im, ax=a, label='how much the group tells about the token\n(1: its two units stand for the two values)')

ax = axes[0, 3]
shares = [(0, 'watching'), (1 / 32, 'sleep_own0.03125'), (1 / 16, 'sleep_own0.0625'), (1 / 8, 'sleep_own0.125'), (1 / 4, 'sleep_own0.25'), (1 / 2, 'watch_sleep'), (3 / 4, 'sleep_own0.75'), (7 / 8, 'sleep_own0.875'), (15 / 16, 'sleep_own0.9375')]
x = curve('watching', 'kl')[0]
keep = x >= 3000
grid = np.array([curve(n, 'kl')[1][keep] for _, n in shares])
im = ax.pcolormesh(np.arange(keep.sum() + 1), np.arange(len(shares) + 1), grid, norm=LogNorm(.25, START), cmap='viridis_r')
ax.set_xticks(np.arange(keep.sum()) + .5, [f'{v / 1000:.0f}k' for v in x[keep]], rotation=60); ax.set_yticks(np.arange(len(shares)) + .5, ['0', '1/32', '1/16', '1/8', '1/4', '1/2', '3/4', '7/8', '15/16'])
ax.set_xlabel('samples (teacher strings + own samples)'); ax.set_ylabel('share of own samples')
ax.set_title('(d) budget: own samples teach the reader,\nteacher strings reader and writer')
fig.colorbar(im, ax=ax, label='KL(teacher || learner), nats')

ax = axes[1, 0]
for n, ls in ((256, ':'), (1024, '--'), (4096, '-')):
    draw(ax, f'set{n}_watching', 'kl', f'{n:,} stored teacher strings, shown again and again', color='C0', ls=ls, lw=2)
    draw(ax, f'set{n}_mix', 'kl', f'{n:,} stored teacher strings in turn with own samples', color='C3', ls=ls, lw=2)
draw(ax, 'watching', 'kl', 'fresh teacher strings only', color='k', lw=1)
ax.set_xscale('log'); ax.set_xlabel('samples (teacher strings shown + own samples)'); ax.set_ylabel('KL(teacher || learner), nats'); ax.set_ylim(0, None)
ax.set_title('(e) a stored teacher set')
legend(ax)

ax = axes[1, 1]
for key, color in (('single tokens', 'C0'), ('token pairs', 'C1'), ('token checks', 'C3')):
    draw(ax, 'watching', f'mmd/{key}', f'{key}, teacher strings only', color=color, lw=2.5)
    draw(ax, 'watch_sleep', f'mmd/{key}', f'{key}, half own samples', color=color, ls='--')
draw(ax, 'set1024_watching', 'mmd/single tokens', 'single tokens, 1,024 stored teacher strings only', color='C0', ls=':')
ax.axhspan(-3, 3, color='gray', alpha=.15, lw=0)
ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=10); ax.set_ylim(-5, None); ax.set_xlabel('samples'); ax.set_ylabel('MMD$^2$ own vs teacher, in s.d. of teacher vs teacher')
ax.set_title('(f) can a statistic tell own samples from teacher strings?\n(1,024 samples each; band: no)')
legend(ax)

ax = axes[1, 2]
first = 65536
for name, label, kw in (('watching', 'teacher strings go on', dict(color='C0', lw=2.5)), (f'first{first}_mix', 'own samples in turn with teacher strings', dict(color='C3', lw=2.5)),
                        (f'first{first}_mix_twin', "a twin's samples in turn with teacher strings", dict(color='C2')), (f'first{first}_offline', 'own samples only', dict(color='C1', lw=2)),
                        (f'first{first}_offline_intended', 'own samples only, not read back by the first layer', dict(color='C1', ls=':')), (f'first{first}_offline_read', 'own samples only, credited by reading', dict(color='C5', ls='--'))):
    draw(ax, name, 'kl', label, rs=after, **kw)
ax.axvline(first, color='gray', lw=.8)
ax.set_xlim(3e4, 262144); ax.set_ylim(0, 1.6 * START); ax.set_xlabel('samples'); ax.set_ylabel('KL(teacher || learner), nats')
ax.set_title(f'(g) after {first:,} teacher strings')
legend(ax)

ax = axes[1, 3]
sets = [(f'{6 * C} tokens', res)] + list(other.items())
w = .38
for i, (tag, rs) in enumerate(sets):
    for j, (name, color) in enumerate((('watching', 'C0'), ('watch_sleep', 'C3'))):
        v = [r['curve'][-1]['valid'] for r in rs if r['name'] == name]
        ax.bar(i + (j - .5) * w, np.median(v), w, color=color, alpha=.6, label=('teacher strings only', 'half own samples, they teach the reader')[j] if i == 0 else None)
        ax.scatter(np.full(len(v), i + (j - .5) * w), v, color='k', s=12, zorder=3)
ax.set_xticks(range(len(sets)), [t for t, _ in sets]); ax.set_ylabel('valid components in own samples at the end'); ax.set_ylim(0, 1.02)
ax.set_title('(h) other sizes and settings (dots: worlds)')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1, frameon=False)
fig.savefig(REPORTS / 'order3-net.png', dpi=100, facecolor='white')
fig.savefig(REPORTS / 'order3-net.svg')

out, keys = {}, ['valid', 'kl', 'coverage', 'own_read', 'own_trip', 'teacher_trip', 'tokens_owned', 'tokens_shared', 'groups_free', 'group_pr', 'usage']
for tag, rs in [('main', res)] + list(other.items()):
    for name in sorted({r['name'] for r in rs}):
        rr = [r for r in rs if r['name'] == name]
        x, kl, _, _ = curve(name, 'kl', rs)
        out[f'{tag}/{name}'] = dict(worlds=len(rr), cfg={k: rr[0]['cfg'][k] for k in ('q', 'comps', 'groups', 'states', 'eps', 'init', 'own', 'watch_first', 'set', 'samples')}, end={k: end(name, k, rs=rs) for k in keys},
                                    kl_worlds=sorted(round(r['curve'][-1]['kl'], 3) for r in rr), mmd_end={k: end(name, f'mmd/{k}', rs=rs) for k in rr[0]['curve'][0]['mmd']},
                                    curve=dict(samples=x.tolist(), kl=kl.tolist(), valid=curve(name, 'valid', rs)[1].tolist(), tokens_owned=curve(name, 'tokens_owned', rs)[1].tolist(), tokens_shared=curve(name, 'tokens_shared', rs)[1].tolist(),
                                               own_trip=curve(name, 'own_trip', rs)[1].tolist(), teacher_trip=curve(name, 'teacher_trip', rs)[1].tolist()))
json.dump(out, open(REPORTS / 'order3-net.json', 'w'), indent=1)
for name, o in out.items():
    print(f"{name:45s} n {o['cfg']['samples']:7d}  valid {o['end']['valid']:.2f}  kl {o['end']['kl']:6.2f}  owned {o['end']['tokens_owned']:.2f} shared {o['end']['tokens_shared']:.2f} free {o['end']['groups_free']:.2f}  trips own {o['end']['own_trip']:.2f} teacher {o['end']['teacher_trip']:.2f}  mmd {[round(v, 1) for v in o['mmd_end'].values()]}")
print(REPORTS / 'order3-net.png')
