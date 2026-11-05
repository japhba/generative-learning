"""Learning curves for practice_ws_explore.py: KL, validity and coverage against training steps, per supplied-set size."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

COL = {'ws': '#2a78d6', 'ws_teach': '#eb6834', 'ws_explore': '#1baf7a', 'ws_explore_iw': '#eda100', 'ws_fresh': '#52514e'}  # palette.md slots 1-4, fresh = secondary ink
NAME = {'ws': 'wake-sleep', 'ws_teach': 'no generation (both parts on supplied)', 'ws_explore': 'wake-sleep + exploration with own check',
        'ws_explore_iw': 'same, kept codes weighted 1/p', 'ws_fresh': 'reference: unlimited fresh supplied examples'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--out', default='/nfs/nhome/live/jbauer/generative-learning/reports/ws-explore-curves.png'); a = ap.parse_args()
    d = json.load(open(Path(a.run) / 'results.json')); res = d['results']; Ns = sorted(res, key=int)
    keys = [('kl', 'population KL [nats]', True), ('validity', 'prior mass on valid configurations', False), ('coverage', 'valid configurations covered', False)]
    fig, axes = plt.subplots(len(Ns), 3, figsize=(15, 3.8 * len(Ns)), squeeze=False, layout='constrained')
    for i, N in enumerate(Ns):
        distinct = np.mean([v['distinct'] for v in next(iter(res[N].values())).values()])
        for j, (k, lab, logy) in enumerate(keys):
            ax = axes[i, j]
            for cond in [c for c in COL if c in res[N]]:
                runs = list(res[N][cond].values()); steps = np.array([p['step'] for p in runs[0]['curve']]); m = np.array([[p[k] for p in r['curve']] for r in runs]).mean(0)
                ax.plot(steps[1:], m[1:], color=COL[cond], lw=2, ls='--' if cond == 'ws_fresh' else '-', label=NAME[cond])
            if k == 'coverage': ax.axhline(distinct / 144, color='#0b0b0b', ls=':', lw=1, label='share of configurations in the supplied set')
            ax.set_xscale('log'); ax.set(xlabel='training steps', ylabel=lab); ax.grid(alpha=0.25, lw=0.5)
            if logy: ax.set_yscale('log'); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}' if f'{v:.0e}'[0] in '25' else ''))
            else: ax.set_ylim(0, 1.02)
            ax.set_title(f'N = {N} supplied examples ({distinct:.0f} distinct of 144)' if j == 0 else '', loc='left')
        if i == 0: axes[i, 2].legend(frameon=False, loc='lower right')
    fig.savefig(a.out, dpi=130); print(a.out)


if __name__ == '__main__': main()
