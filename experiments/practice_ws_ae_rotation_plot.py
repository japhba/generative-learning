"""Figure for practice_ws_ae_rotation.py: the two-cause rotation plot with the autoencoder added.
Left: the plane after 500 updates for wake-sleep and the baseline autoencoder (world samples, generator axes, the four
points W z the generator can produce). Right: angle of the generator's axes from the world's over updates, 16 seeds."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path('/var/tmp/jbauer/generative_practice_disentangle/ae_floor/two_cause.json')
OUT = Path(__file__).resolve().parents[1] / 'reports/ws-ae-rotation-2d.png'
COL = {'ws': '#2a78d6', 'frozen': '#008300', 'ae': '#eb6834', 'ae_fast': '#eda100'}
LAB = {'ws': 'wake–sleep', 'frozen': 'recognition frozen', 'ae': 'autoencoder (baseline, recognizer rate ×0.05)', 'ae_fast': 'autoencoder, recognizer rate ×1'}


def plane(ax, X, snap, col, title):
    W, c = np.array(snap['W']), np.array(snap['c'])
    ax.scatter(*X.T, s=4, color='0.75', zorder=1)
    for z in [(1, 1), (1, -1), (-1, 1), (-1, -1)]: ax.plot(*(W @ np.array(z) + c), 'o', color=col, mec='white', zorder=3)
    for i in range(2): ax.annotate('', xy=W[:, i] + c, xytext=c, arrowprops=dict(arrowstyle='-|>', color=col, lw=2), zorder=4)
    ax.axhline(0, color='0.85', lw=0.8, zorder=0); ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)


def main():
    d = json.loads(SRC.read_text()); X = np.array(d['X0']); L = d['learners']
    fig = plt.figure(figsize=(14, 4.4)); gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.15, 2.2])
    plane(fig.add_subplot(gs[0]), X, L['ws']['snaps0']['0'], '0.3', 'start (all learners)\nturned by 30°')
    plane(fig.add_subplot(gs[1]), X, L['ws']['snaps0']['500'], COL['ws'], 'wake–sleep\nafter 500 updates')
    plane(fig.add_subplot(gs[2]), X, L['ae']['snaps0']['500'], COL['ae'], 'autoencoder (baseline)\nafter 500 updates')
    ax = fig.add_subplot(gs[4]); t = np.arange(len(L['ws']['gen'][0]))
    for k in ['ws', 'frozen', 'ae', 'ae_fast']:
        g = np.abs(np.array(L[k]['gen'])); m, se = g.mean(0), g.std(0, ddof=1) / np.sqrt(len(g))
        ax.plot(t, m, color=COL[k], lw=2, label=LAB[k]); ax.fill_between(t, m - se, m + se, color=COL[k], alpha=0.2, lw=0)
    ax.set_xscale('symlog', linthresh=10); ax.set_xlim(0, t[-1]); ax.set_ylim(0, 32); ax.set_xlabel('update (two batches of 32 each)')
    ax.set_ylabel('angle of generator axes from world (°)'); ax.legend(frameon=False, loc='lower left'); ax.grid(alpha=0.3)
    ax.set_title('two causes, 256 stored samples, 16 seeds (mean ± s.e.)')
    fig.tight_layout(); fig.savefig(OUT, dpi=150); print(OUT)


if __name__ == '__main__': main()
