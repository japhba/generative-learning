"""Figure of the tied-binding study: KL curves in the tied and free worlds, solved counts, budget maps by share of own samples.

    MPLBACKEND=Agg python tied_binding_plot.py --dir /var/tmp/jbauer/tied_binding --out ../tex/figures/tied_binding
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--dir'); ap.add_argument('--out'); ap.add_argument('--main', default='main_v2.json'); ap.add_argument('--shares', default='shares.json')
ap.add_argument('--stage', type=float, default=1e6, help='sample count where level 2 starts (0: no stage)'); ap.add_argument('--probe', type=float, default=1e6, help='sample count of panel f')
a = ap.parse_args()
main, shares = json.load(open(f'{a.dir}/{a.main}')), json.load(open(f'{a.dir}/{a.shares}'))   # main_v2: main.json with the twin arm rerun as a practising twin
STYLE = {'watching (teacher only)': ('k', '-'), 'watching, exact posterior': ('0.45', '-'), 'watching + inhibition (c 3)': ('0.45', '--'),
         'classic wake-sleep': ('C0', '-'), 'sleep only': ('C9', '-'), 'practice (sleep + verdict)': ('C3', '-'),
         'practice, strict credit': ('C3', ':'), 'twin samples (credit by reading)': ('C1', ':'), 'equal-validity foreign (credit by reading)': ('C4', ':')}
LABEL = {'watching + inhibition (c 3)': 'watching + lateral inhibition', 'twin samples (credit by reading)': "a twin's samples", 'equal-validity foreign (credit by reading)': 'equal-validity foreign samples'}
fig, ax = plt.subplots(2, 3, figsize=(19, 10.5), layout='constrained')
for col, world in enumerate(('tied', 'free')):
    for arm, (c, ls) in STYLE.items():
        rs = [r for r in main if r['arm'] == arm and r['world'] == world]
        x = np.array([p['samples'] for p in rs[0]['curve']])
        y = np.median([[max(p['kl'], 1e-3) for p in r['curve']] for r in rs], 0)
        ax[0, col].plot(x[1:], y[1:], color=c, ls=ls, label=LABEL.get(arm, arm))
    if a.stage: ax[0, col].axvline(a.stage, color='0.8', lw=1)
    if world == 'tied': ax[0, col].axhline(np.log(3), color='0.8', lw=1, ls='--')
    ax[0, col].set(xscale='log', yscale='log', xlabel='samples (teacher and own counted alike)', ylabel='KL(teacher || writer), nats (median of 32)',
                   title=('(a) a level above ties the two checks' if world == 'tied' else '(b) the two checks are independent'))
ax[0, 0].legend(fontsize='small', loc='lower left')

arms = list(STYLE) + ['practice, verdict only', 'offline practice after watching', 'offline practice, novelty-weighted verdict', 'watching + inhibition']
arms = [x for x in arms if any(r['arm'] == x for r in main)]
solved = {w: [sum(r['curve'][-1]['kl'] < .1 for r in main if r['arm'] == arm and r['world'] == w) for arm in arms] for w in ('tied', 'free')}
yy = np.arange(len(arms))
ax[0, 2].barh(yy - .2, solved['tied'], .4, color='C3', label='level above (tied)')
ax[0, 2].barh(yy + .2, solved['free'], .4, color='0.6', label='no level above (free)')
ax[0, 2].set_yticks(yy); ax[0, 2].set_yticklabels([LABEL.get(x, x).replace(' (c 3)', '') + (' (rate x10)' if x == 'watching + inhibition' else '') for x in arms], fontsize='small')
ax[0, 2].invert_yaxis(); ax[0, 2].set(xlabel='worlds solved (KL < 0.1 at 2M samples) of 32', title='(c) which learners solve which world'); ax[0, 2].legend(fontsize='small')

def heat(axh, base, title):
    sh = [s for s in (0, .125, .25, .5, .75, .875) if s in (0, .5) or any(r['arm'] == f'{base} @ {s}' for r in shares)]
    rows = []
    for s in sh:
        if s == 0: rs = [r for r in main if r['arm'] == 'watching (teacher only)' and r['world'] == 'tied']
        elif s == .5: rs = [r for r in main if r['arm'] == base and r['world'] == 'tied']
        else: rs = [r for r in shares if r['arm'] == f'{base} @ {s}' and r['world'] == 'tied']
        rows.append(np.median([[p['kl'] for p in r['curve']] for r in rs], 0))
    x = np.array([p['samples'] for p in main[0]['curve']])
    xe = np.sqrt(x[1:-1] * x[2:]); xe = np.concatenate([[x[1] ** 1.5 / x[2] ** .5], xe, [x[-1] ** 1.5 / x[-2] ** .5]])
    im = axh.pcolormesh(xe, np.arange(len(sh) + 1), np.log10(np.maximum(np.array(rows)[:, 1:], 1e-3)), shading='auto', vmin=-3, vmax=.6, cmap='viridis')
    axh.set(xscale='log', xlabel='samples', ylabel='share of own samples', title=title, yticks=np.arange(len(sh)) + .5, yticklabels=[str(s) for s in sh])
    if a.stage: axh.axvline(a.stage, color='w', lw=1)
    return im
heat(ax[1, 0], 'practice (sleep + verdict)', '(d) practice, tied world: log10 median KL')
im = heat(ax[1, 1], 'classic wake-sleep', '(e) classic wake-sleep, tied world: log10 median KL')
fig.colorbar(im, ax=ax[1, 1], label='log10 median KL (nats)')

stage = []
for arm in ('watching (teacher only)', 'watching, exact posterior', 'classic wake-sleep', 'sleep only', 'practice (sleep + verdict)'):
    rs = [r for r in main if r['arm'] == arm and r['world'] == 'tied']
    xs = [p['samples'] for p in rs[0]['curve']]; i1 = max(i for i, v in enumerate(xs) if v <= a.probe)
    stage.append(([r['curve'][i1]['bind'] for r in rs], [r['curve'][i1]['kl'] for r in rs]))
for j, (b, k) in enumerate(stage):
    jit = np.random.default_rng(j).uniform(-.15, .15, len(b))
    ax[1, 2].scatter(np.full(len(b), j) + jit, k, s=14, color=['k', '0.45', 'C0', 'C9', 'C3'][j])
ax[1, 2].axhline(np.log(3), color='0.6', ls='--', lw=1)
ax[1, 2].set_xticks(range(5)); ax[1, 2].set_xticklabels(['watching', 'exact\nposterior', 'classic\nwake-sleep', 'sleep\nonly', 'practice'])
ax[1, 2].set(ylabel=f'KL at {a.probe:.0e} samples, nats', title=('(f) before level 2 learns' if a.stage else '(f) while level 2 is still slow') + ': dashed = best possible without it')
fig.savefig(a.out + '.png', dpi=150, facecolor='white', bbox_inches='tight'); fig.savefig(a.out + '.svg', bbox_inches='tight')
print(a.out + '.png')
