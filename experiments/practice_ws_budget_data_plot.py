"""Does generation still pay when the supplied data are scarce? Figures and paired contrasts for practice_ws_budget.py
runs with --Ns (a fixed set of N supplied examples, reused), next to the fresh-data runs (each example used once)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from practice_ws_budget_plot import axes_style

ROOT = Path('/var/tmp/jbauer/generative_practice_disentangle/budget')
LINES = [('ws', '0.7', '#2a78d6', '-', 'wake-sleep, 70% supplied'), ('ws', '0.9', '#2a78d6', '--', 'wake-sleep, 90% supplied'),
         ('ws', '1.0', '#2a78d6', ':', 'wake-sleep, no generation'), ('ae', '1.0', '#1baf7a', '-', 'autoencoder, no generation'),
         ('ae', '0.9', '#1baf7a', '--', 'autoencoder + sleep, 90% supplied'), ('ws_teach', '1.0', '#eb6834', '-', 'exact posteriors, no generation')]
N_PARAMS = 12 * 6 + 12 + 6 * 4 + 6  # W, c, V, a of the learner's two-layer model
FRESH = 2 ** 15  # where the fresh-data column is drawn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--capped', nargs='+', default=sorted(str(p) for p in ROOT.glob('run_dataN_*')))
    ap.add_argument('--fresh', nargs='+', default=[str(ROOT / 'run_20261105_001836'), str(ROOT / 'run_ae_20261105_135239')])
    ap.add_argument('--curves_N', default='128,2048'); ap.add_argument('--out', default='/nfs/nhome/live/jbauer/generative-learning/reports/ws-budget-data.png'); a = ap.parse_args()
    d = {}  # (N, prior, learner, x) -> list over worlds of curves; N = 0 for fresh data
    for run in a.capped:
        for f in Path(run).glob('N*/curves.json'):
            j = json.load(open(f))
            for prior, block in j['curves'].items():
                for l, byx in block.items():
                    for x, ws in byx.items(): d[j['args']['N'], prior, l, x] = [ws[w] for w in sorted(ws, key=int)]
    for run in a.fresh:
        for prior, block in json.load(open(Path(run) / 'curves.json'))['curves'].items():
            for l, byx in block.items():
                for x, ws in byx.items(): d[0, prior, l, x] = [ws[w] for w in sorted(ws, key=int)]
    Ns = sorted({k[0] for k in d if k[0]}); priors = sorted({k[1] for k in d}); cN = [int(v) for v in a.curves_N.split(',')]
    final = lambda N, p, l, x, k='kl': np.array([c[-1][k] for c in d[N, p, l, x]])
    fig, axes = plt.subplots(len(priors), 2 + len(cN), figsize=(5 * (2 + len(cN)), 4.3 * len(priors)), squeeze=False, layout='constrained')
    for i, p in enumerate(priors):
        mass = np.array([final(N, p, 'ws', '0.7', 'mass_seen').mean() for N in Ns]); seen = np.array([final(N, p, 'ws', '0.7', 'configs_seen').mean() for N in Ns])
        ax = axes[i, 0]
        for l, x, col, ls, lab in LINES:
            ax.plot(Ns, [final(N, p, l, x).mean() for N in Ns], color=col, ls=ls, lw=2, marker='o', ms=4, label=lab)
            ax.plot([Ns[-1], FRESH], [final(Ns[-1], p, l, x).mean(), final(0, p, l, x).mean()], color=col, ls=':', lw=1); ax.plot(FRESH, final(0, p, l, x).mean(), 'o', color=col, ms=4)
        n = np.geomspace(Ns[0], Ns[-1], 50); ax.plot(n, N_PARAMS / (2 * n), color='#52514e', lw=1, label='parameters / 2N (asymptotic error of the exact fit)')
        ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator(Ns + [FRESH])); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: 'fresh' if v == FRESH else f'$2^{{{int(round(np.log2(v)))}}}$\n{mass[Ns.index(int(round(v)))]:.0%}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}' if f'{v:.0e}'[0] in '235' else '')); ax.grid(alpha=0.25, lw=0.5)
        ax.set(xlabel='distinct supplied examples N (prior mass of the latent configurations they visit)', ylabel='population KL after $2^{21}$ examples [nats]'); ax.set_title(f'{p} prior', loc='left')
        ax = axes[i, 1]
        for l, x, col, ls, lab in LINES: ax.plot(Ns, [(final(N, p, l, x) - final(N, p, l, x, 'kl_train')).mean() for N in Ns], color=col, ls=ls, lw=2, marker='o', ms=4)
        ax.axhline(0, color='#52514e', lw=0.8); ax.set_xscale('log', base=2); ax.xaxis.set_major_locator(FixedLocator(Ns)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'$2^{{{int(round(np.log2(v)))}}}$')); ax.grid(alpha=0.25, lw=0.5)
        ax.set(xlabel='distinct supplied examples N', ylabel='KL on new samples − KL on the training set [nats]')
        for j, N in enumerate(cN):
            ax = axes[i, 2 + j]; P = np.array([q['P'] for q in d[N, p, 'ws', '0.7'][0]])
            for l, x, col, ls, lab in LINES: ax.plot(P[1:], np.array([[q['kl'] for q in c] for c in d[N, p, l, x]]).mean(0)[1:], color=col, ls=ls, lw=2)
            axes_style(ax, P); ax.set(xlabel='total examples (supplied presentations + generated)', ylabel='population KL [nats]'); ax.set_title(f'N = {N} distinct supplied examples', loc='left')
        if i == 0: axes[i, 0].legend(frameon=False, fontsize='small')
    fig.savefig(a.out, dpi=130); print(a.out)
    rng = np.random.default_rng(0); rows = []
    def pair(N, p, A, Bm):
        diff = final(N, p, *A) - final(N, p, *Bm); b = rng.choice(diff, (10000, len(diff))).mean(1)
        return dict(diff=float(diff.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], lower_in=int((diff < 0).sum()), n=len(diff))
    for p in priors:
        for N in Ns + [0]:
            r = dict(prior=p, N=N, means={f'{l}@{x}': float(final(N, p, l, x).mean()) for l, x, *_ in LINES})
            if N: r.update(mass_seen=float(final(N, p, 'ws', '0.7', 'mass_seen').mean()), configs_seen=float(final(N, p, 'ws', '0.7', 'configs_seen').mean()),
                           ws_x={x: float(final(N, p, 'ws', x).mean()) for x in ['0.5', '0.7', '0.9', '1.0']})
            r['contrasts'] = {f'{A[0]}@{A[1]} - {Bm[0]}@{Bm[1]}': pair(N, p, A, Bm) for A, Bm in [(('ws', '0.7'), ('ae', '1.0')), (('ws', '0.9'), ('ae', '1.0')), (('ws', '0.7'), ('ws_teach', '1.0')), (('ae', '0.9'), ('ae', '1.0'))]}
            rows.append(r)
            print(f"{p:5s} N={N if N else 'fresh':>5} " + (f"mass {r['mass_seen']:.2f} configs {r['configs_seen']:.1f}  " if N else '') + ' '.join(f'{k} {v:.3f}' for k, v in r['means'].items()))
            for k, c in r['contrasts'].items(): print(f"      {k:22s} {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] lower in {c['lower_in']}/{c['n']}")
            if N: print('      ws by fraction supplied: ' + ' '.join(f'{x}: {v:.3f}' for x, v in r['ws_x'].items()))
    json.dump(rows, open(Path(a.capped[0]) / 'data_paired.json', 'w'), indent=1); print(Path(a.capped[0]) / 'data_paired.json')


if __name__ == '__main__': main()
