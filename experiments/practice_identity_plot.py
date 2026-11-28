"""Figure for the practice identity: exact check at writers flat on the valid set, and sampled times to the solution on the order-m world
against the prediction of the identity.

    MPLBACKEND=Agg python practice_identity_plot.py --exact /var/tmp/jbauer/practice_identity/exact.json --sampled /var/tmp/jbauer/practice_identity/sampled.json --out ../tex/figures/practice_identity
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--exact'); ap.add_argument('--sampled'); ap.add_argument('--out')
a = ap.parse_args()
ex, sa = json.load(open(a.exact)), json.load(open(a.sampled))
med = lambda rs: np.median([r['t'] for r in rs if r['t'] is not None]) if any(r['t'] is not None for r in rs) else np.nan
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

f = ex['flat']
ax[0].plot([.3, .7], [.3, .7], color='0.7', lw=1, zorder=0, label='identity line')
for (q, m), mk in zip(((2, 3), (2, 4), (2, 5), (3, 3), (3, 4)), 'osD^v'):
    rs = [r for r in f if (r['q'], r['m']) == (q, m)]
    ax[0].scatter([r['pS'] for r in rs], [r['ratio'] for r in rs], marker=mk, label=f'q = {q}, m = {m}', s=28)
ax[0].set(xlabel='pass rate of own samples p(S)', ylabel='|practice drive| / |teacher drive, exact posterior|', title='(a) writers flat on the valid set (exact)')
ax[0].legend(fontsize='small')

arms = ['teacher only', 'verdict + sleep', 'verdict, strict credit', 'sleep only']
cols = dict(zip(arms, ['k', 'C1', 'C2', 'C0']))
for arm in arms:
    y = [med([r for r in sa if (r['arm'], r['q'], r['m'], r['a']) == (arm, 2, m, .01)]) for m in (3, 4, 5)]
    ax[1].plot((3, 4, 5), np.array(y) / 1e3, 'o-', color=cols[arm], label=arm)
T = np.array([med([r for r in sa if (r['arm'], r['q'], r['m'], r['a']) == ('teacher only', 2, m, .01)]) for m in (3, 4, 5)])
ax[1].plot((3, 4, 5), T / (.5 + .5 * .5) / 1e3, ':', color='C1', label='identity: teacher time / (1/2 + p(S)/2)')
ax[1].plot((3, 4, 5), 2 * T / 1e3, ':', color='C0', label='own samples worth nothing to the writer: 2 x teacher time')
ax[1].set(yscale='log', xlabel='order m of the check (q = 2)', ylabel='samples to KL < 0.1, thousands (median of 8)', title='(b) weak start, half of the samples own', xticks=(3, 4, 5))
ax[1].legend(fontsize='small')

starts = (.01, .3, 1., 4.)
for arm in arms:
    y = [med([r for r in sa if (r['arm'], r['q'], r['m'], r['a']) == (arm, 3, 3, s)]) for s in starts]
    ax[2].plot(starts, np.array(y) / 1e3, 'o-', color=cols[arm], label=arm)
ax[2].set(xscale='log', yscale='log', xlabel='amplitude of the shared random start', ylabel='samples to KL < 0.1, thousands (median)', title='(c) q = 3, m = 3, by start amplitude')
fig.tight_layout()
fig.savefig(a.out + '.png', dpi=150, facecolor='white'); fig.savefig(a.out + '.svg')
print(a.out + '.png')
