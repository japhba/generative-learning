"""Where should the dreams' latents come from? Learning curves, exploration measures and paired contrasts for the dream
variants of practice_ws_budget.py (ws, ws_replay, ws_replay_wide, ws_uniform)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from practice_ws_budget_plot import axes_style

COL = {'ws': '#2a78d6', 'ws_replay': '#eb6834', 'ws_replay_wide': '#1baf7a', 'ws_uniform': '#eda100'}  # palette.md slots 1-4
NAME = {'ws': 'dreams from the learned prior', 'ws_replay': 'replay of the codes given to supplied data',
        'ws_replay_wide': 'replay, spread matched in x', 'ws_uniform': 'uniform latents'}
KEYS = [('kl', 'population KL [nats]', True), ('dream_latent_entropy', 'entropy of dream latents [nats]', False), ('world_to_dream_nn', 'world sample to nearest dream', False)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--x', default='0.7')
    ap.add_argument('--out', default='/nfs/nhome/live/jbauer/generative-learning/reports/ws-dreams.png'); a = ap.parse_args()
    d = json.load(open(Path(a.run) / 'curves.json'))['curves']; priors = list(d); rng = np.random.default_rng(0); rows = []
    arr = lambda prior, l, x, k: np.array([[p[k] for p in c] for c in d[prior][l][x].values()])  # (world, checkpoint)
    fig, axes = plt.subplots(len(priors), 3, figsize=(16, 4.3 * len(priors)), squeeze=False, layout='constrained')
    for i, prior in enumerate(priors):
        P = np.array([p['P'] for p in next(iter(d[prior]['ws'][a.x].values()))])
        for j, (k, lab, logy) in enumerate(KEYS):
            ax = axes[i, j]
            for l in COL:
                m = arr(prior, l, a.x, k).mean(0); ax.plot(P[1:], m[1:], color=COL[l], lw=2, label=NAME[l])
            ax.set(xlabel='total examples', ylabel=lab); axes_style(ax, P, logy=logy)
            ax.set_title(f'{prior} prior, {float(a.x):.0%} supplied' if j == 0 else '', loc='left')
        if i == 0: axes[i, 0].legend(frameon=False)
    fig.savefig(a.out, dpi=130); print(a.out)
    for prior in priors:
        P = np.array([p['P'] for p in next(iter(d[prior]['ws']['0.7'].values()))])
        for x in sorted(d[prior]['ws'], key=float):
            for Pv in [2 ** 13, 2 ** 16, 2 ** 21]:
                c = int(np.flatnonzero(P == Pv)[0]); base = arr(prior, 'ws', x, 'kl')[:, c]
                for l in ['ws_replay', 'ws_replay_wide', 'ws_uniform']:
                    diff = arr(prior, l, x, 'kl')[:, c] - base; b = rng.choice(diff, (10000, len(diff))).mean(1)
                    rows.append(dict(prior=prior, x=float(x), P=Pv, variant=l, kl=float(arr(prior, l, x, 'kl')[:, c].mean()), kl_prior=float(base.mean()), diff=float(diff.mean()),
                                     ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], lower_in=int((diff < 0).sum()), n=len(diff),
                                     entropy=float(arr(prior, l, x, 'dream_latent_entropy')[:, c].mean()), entropy_prior=float(arr(prior, 'ws', x, 'dream_latent_entropy')[:, c].mean()),
                                     nn=float(arr(prior, l, x, 'world_to_dream_nn')[:, c].mean()), nn_prior=float(arr(prior, 'ws', x, 'world_to_dream_nn')[:, c].mean())))
                    r = rows[-1]; print(f"{prior:5s} {float(x):.0%} P=2^{int(np.log2(Pv)):2d} {l:15s} KL {r['kl']:.3f} vs prior dreams {r['kl_prior']:.3f}: {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] lower in {r['lower_in']}/{r['n']}"
                                        f" | latent entropy {r['entropy']:.2f} vs {r['entropy_prior']:.2f} | nn {r['nn']:.3f} vs {r['nn_prior']:.3f}")
    json.dump(rows, open(Path(a.run) / 'dreams_paired.json', 'w'), indent=1)


if __name__ == '__main__': main()
