"""Figures for practice_ws_budget.py: KL heatmaps over (fraction supplied, total examples) and learning curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
import numpy as np

BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']  # palette.md sequential steps 100..700
SEQ = LinearSegmentedColormap.from_list('seq', BLUE)
DIV = LinearSegmentedColormap.from_list('div', ['#184f95', '#6da7ec', '#f0efec', '#ec7a7a', '#a52a2a'])  # blue <-> red, gray midpoint
CAT = {'ws': '#2a78d6', 'ws_teach': '#eb6834'}
INK, INK2 = '#0b0b0b', '#52514e'
NAMES = {'ws': 'wake-sleep', 'ws_teach': 'both parts trained on supplied examples'}


def pct(x): return f'{100 * x:g}'


def axes_style(ax, P, logy=True):
    """Totals as powers of two on a log axis, plain numbers on the KL axis."""
    ax.set_xscale('log', base=2); ax.xaxis.set_major_locator(FixedLocator([p for p in P if p > 0 and np.log2(p) % 2 == 0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'$2^{{{int(round(np.log2(v)))}}}$')); ax.xaxis.set_minor_locator(NullLocator())
    if logy: ax.set_yscale('log'); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}' if f'{v:.0e}'[0] in '235' else ''))
    ax.grid(alpha=0.25, lw=0.5)


def load(run):
    d = json.load(open(Path(run) / 'curves.json')); xs = d['xs']; out = {}
    for prior, block in d['curves'].items():
        for learner, byx in block.items():
            runs = [byx[str(x)] for x in xs]  # one dict (world -> curve) per x
            Ps = [p['P'] for p in next(iter(runs[0].values()))]
            kl = np.array([[[p['kl'] for p in curve] for curve in r.values()] for r in runs])  # (x, world, P)
            out[prior, learner] = dict(P=np.array(Ps), kl=kl)
    return d, xs, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--outdir', default='/nfs/nhome/live/jbauer/generative-learning/reports')
    ap.add_argument('--P_min', type=int, default=256); a = ap.parse_args()
    d, xs, data = load(a.run); priors = list(d['curves']); learners = [l for l in NAMES if (priors[0], l) in data]
    xs = np.array(xs); P = data[priors[0], learners[0]]['P']; rows = P >= a.P_min; Pr = P[rows]

    # heatmaps: mean KL over worlds, one panel per prior x learner, plus the log ratio of the two learners
    fig, axes = plt.subplots(len(priors), len(learners) + 1, figsize=(5.4 * (len(learners) + 1), 4.8 * len(priors)), squeeze=False, layout='constrained')
    allm = np.concatenate([data[p, l]['kl'].mean(1)[:, rows].ravel() for p in priors for l in learners])
    norm = LogNorm(allm.min(), allm.max())
    for i, prior in enumerate(priors):
        for j, learner in enumerate(learners):
            M = data[prior, learner]['kl'].mean(1)[:, rows].T  # (P, x)
            ax = axes[i, j]; im = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(Pr) + 1), M, cmap=SEQ, norm=norm, edgecolors='face')
            best = M.argmin(1); ax.plot(best + 0.5, np.arange(len(Pr)) + 0.5, 'o', mfc='none', mec=INK, label='lowest KL in the row')
            ax.set_title(f'{prior} prior: {NAMES[learner]}', loc='left')
            ax.set_xticks(np.arange(len(xs)) + 0.5, [pct(x) for x in xs], rotation=90); ax.set_yticks(np.arange(len(Pr)) + 0.5, [f'$2^{{{int(np.log2(p))}}}$' for p in Pr])
            ax.set_xlabel('% of examples supplied (rest generated)'); ax.set_ylabel('total examples')
            if i == 0 and j == 0: ax.legend(loc='lower left', frameon=False)
        cb = fig.colorbar(im, ax=axes[i, :len(learners)].tolist(), label='population KL(world || model) [nats]', shrink=0.9)
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}' if f'{v:.0e}'[0] in '235' else ''))
        if len(learners) == 2:
            Rm = np.log10(data[prior, learners[1]]['kl'].mean(1) / data[prior, learners[0]]['kl'].mean(1))[:, rows].T
            lim = np.abs(Rm).max(); ax = axes[i, -1]
            im2 = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(Pr) + 1), Rm, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), edgecolors='face')
            ax.set_title(f'{prior} prior: baseline vs wake-sleep', loc='left'); fig.colorbar(im2, ax=ax, label='log10 KL ratio (blue: baseline lower)', shrink=0.9)
            ax.set_xticks(np.arange(len(xs)) + 0.5, [pct(x) for x in xs], rotation=90); ax.set_yticks(np.arange(len(Pr)) + 0.5, [f'$2^{{{int(np.log2(p))}}}$' for p in Pr])
            ax.set_xlabel('% of examples supplied (rest generated)')
    f1 = Path(a.outdir) / 'ws-budget-heatmap.png'; fig.savefig(f1, dpi=130); plt.close(fig)

    # learning curves: selected fractions per learner; envelopes; the best fraction against the total
    show = [1.0, 0.95, 0.9, 0.8, 0.5, 0.2]; shades = BLUE[::-1][:len(show)]
    fig, axes = plt.subplots(len(priors), len(learners) + 2, figsize=(5 * (len(learners) + 2), 4.2 * len(priors)), squeeze=False)
    for i, prior in enumerate(priors):
        for j, learner in enumerate(learners):
            ax = axes[i, j]; kl = data[prior, learner]['kl']
            for x, col in zip(show, shades):
                k = int(np.flatnonzero(np.isclose(xs, x))[0]); m, s = kl[k].mean(0), kl[k].std(0) / np.sqrt(kl.shape[1])
                ax.plot(P[1:], m[1:], color=col, lw=2, label=f'{pct(x)}% supplied')
            ax.axhline(kl[0, :, 0].mean(), color=INK2, ls=':', lw=1, label='start')
            ax.set(xlabel='total examples', ylabel='population KL [nats]'); ax.set_title(f'{prior} prior: {NAMES[learner]}', loc='left'); axes_style(ax, P)
            if i == 0 and j == 0: ax.legend(frameon=False)
        ax = axes[i, len(learners)]
        for learner in learners:
            kl = data[prior, learner]['kl']; m = kl.mean(1)
            ax.plot(P[1:], m.min(0)[1:], color=CAT[learner], lw=2, label=f'{NAMES[learner]}, best split')
            ax.plot(P[1:], m[-1][1:], color=CAT[learner], lw=2, ls='--', label=f'{NAMES[learner]}, all supplied')
        ax.set(xlabel='total examples', ylabel='population KL [nats]'); ax.set_title(f'{prior} prior: best split vs none generated', loc='left'); axes_style(ax, P)
        if i == 0: ax.legend(frameon=False)
        ax = axes[i, len(learners) + 1]
        for learner in learners:
            m = data[prior, learner]['kl'].mean(1); ax.plot(P[rows], 100 * xs[m[:, rows].argmin(0)], 'o-', color=CAT[learner], lw=2, ms=8, label=NAMES[learner])
        ax.set(xlabel='total examples', ylabel='% supplied at the lowest KL', ylim=(-3, 103)); ax.set_title(f'{prior} prior: best split', loc='left'); axes_style(ax, P[rows], logy=False)
        if i == 0: ax.legend(frameon=False)
    fig.tight_layout(); f2 = Path(a.outdir) / 'ws-budget-curves.png'; fig.savefig(f2, dpi=130, bbox_inches='tight'); plt.close(fig)

    # numbers for the write-up: per prior, learner and total, the best fraction, its KL and the KL with all supplied
    summ = {}
    for (prior, learner), v in data.items():
        m, se = v['kl'].mean(1), v['kl'].std(1) / np.sqrt(v['kl'].shape[1])
        summ[f'{prior}/{learner}'] = [dict(P=int(p), best_x=float(xs[m[:, c].argmin()]), best_kl=float(m[:, c].min()), best_se=float(se[m[:, c].argmin(), c]),
                                           all_supplied_kl=float(m[-1, c]), all_supplied_se=float(se[-1, c]), half_kl=float(m[int(np.flatnonzero(np.isclose(xs, 0.5))[0]), c]))
                                      for c, p in enumerate(v['P']) if p >= a.P_min]
    json.dump(summ, open(Path(a.run) / 'summary.json', 'w'), indent=1)
    print(f1); print(f2); print(Path(a.run) / 'summary.json')


if __name__ == '__main__': main()
