"""Figures and numbers for practice_ws_ae_floor.py and practice_ws_depth.py.

ws-ae-floor.png    deep world: KL over examples for wake-sleep and the autoencoder family (fresh examples, and 128 distinct
                   examples reused), the generator's angle from the world, and the error split into decoder and prior
ws-factorial.png   the same learners in a world with independent causes, next to the deep world
ws-depth.png       bottom and top layer alignment over updates, from three starts
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

ROOT = Path('/var/tmp/jbauer/generative_practice_disentangle')
REPORTS = Path(__file__).resolve().parents[1] / 'reports'
COL = {'ws:x=0.7': '#2a78d6', 'ws:x=1': '#1baf7a', 'ae:top=mf:s=0.05': '#eb6834', 'ae:top=mf:s=1': '#eda100', 'ae:top=bin:s=1': '#e87ba4',
       'ae:top=mf:s=1:wd=0.01': '#e34948', 'vae:s=1': '#6250d6'}
LAB = {'ws:x=0.7': 'wake–sleep, 30% dreamed', 'ws:x=1': 'wake–sleep, no dreams', 'ae:top=mf:s=0.05': 'autoencoder (report baseline)',
       'ae:top=mf:s=1': 'autoencoder, recognizer rate ×20', 'ae:top=bin:s=1': '+ binary top code', 'ae:top=mf:s=1:wd=0.01': '+ weight decay',
       'vae:s=1': 'VAE (autoencoder + KL to prior)'}


def latest(pattern, prior=None):
    runs = [p for p in ROOT.glob(pattern) if prior is None or prior in json.loads((p / 'curves.json').read_text())['curves']]
    return sorted(runs, key=lambda p: p.name[-15:])[-1]


def load(run):
    j = json.loads((run / 'curves.json').read_text()); prior = next(iter(j['curves']))
    return j, {spec: [ws[w] for w in sorted(ws, key=int)] for spec, ws in j['curves'][prior].items()}


def arr(runs, key):
    P = np.array([p['P'] for p in runs[0] if key in p]); return P, np.array([[p[key] for p in r if key in p] for r in runs])


def band(ax, P, v, col, label, ls='-'):
    m, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v)); ax.plot(P, m, ls, color=col, lw=2, label=label); ax.fill_between(P, m - se, m + se, color=col, alpha=0.18, lw=0)


def plain_log(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); axis.set_minor_formatter(NullFormatter())


def kl_panel(ax, C, specs, title):
    for s in specs:
        P, v = arr(C[s], 'kl'); keep = P >= 2 ** 8; band(ax, P[keep], v[:, keep], COL[s], LAB[s])
    ax.set_xscale('log', base=2); ax.set_yscale('log'); plain_log(ax.yaxis, [0.1, 0.2, 0.5, 1, 2, 4]); ax.set_xlabel('examples P'); ax.set_ylabel('KL(world ‖ model), nats'); ax.set_title(title); ax.grid(alpha=0.3, which='both')


def floor_fig(deep, n128):
    jd, C = load(deep); _, Cn = load(n128); specs = [s for s in COL if s in C]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    kl_panel(ax[0], C, specs, f"fresh examples, {len(C[specs[0]])} worlds")
    kl_panel(ax[1], Cn, [s for s in COL if s in Cn], '128 distinct examples, reused')
    for s in specs:
        P, v = arr(C[s], 'gen_world'); band(ax[2], P[P >= 2 ** 8], v[:, P >= 2 ** 8], COL[s], LAB[s])
    ax[2].set_xscale('log', base=2); ax[2].set_xlabel('examples P'); ax[2].set_ylabel('angle of generator axes from world (°)'); ax[2].set_title('rotation of the latent axes'); ax[2].grid(alpha=0.3)
    y = np.arange(len(specs))[::-1]
    for i, s in zip(y, specs):
        f = np.array([[r[-1]['kl'], r[-1]['kl_refit']] for r in C[s]])
        ax[3].plot(f[:, 1].mean(), i, 'o', mfc='white', mec=COL[s], mew=2, ms=9); ax[3].plot(f[:, 0].mean(), i, 'o', color=COL[s], ms=9)
        ax[3].plot([f[:, 1].mean(), f[:, 0].mean()], [i, i], color=COL[s], lw=2)
    ax[3].set_yticks(y); ax[3].set_yticklabels([LAB[s] for s in specs]); ax[3].set_xscale('log'); plain_log(ax[3].xaxis, [0.1, 0.2, 0.5, 1, 2]); ax[3].set_xlabel(f"KL after 2^{int(np.log2(jd['args']['P_max']))} fresh examples, nats")
    ax[3].set_title('filled: own prior; open: decoder kept,\nprior refitted by maximum likelihood'); ax[3].grid(alpha=0.3, which='both')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=len(specs), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); out = REPORTS / 'ws-ae-floor.png'; fig.savefig(out, dpi=150); print(out)
    return C, Cn


def factorial_fig(deep, fact):
    _, Cd = load(deep); _, Cf = load(fact); specs = [s for s in COL if s in Cf]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    kl_panel(ax[0], Cd, specs, 'deep world: two top causes correlate the six'); kl_panel(ax[1], Cf, specs, 'factorial world: six independent causes')
    for s in specs:
        for C, ls in [(Cd, '--'), (Cf, '-')]:
            P, v = arr(C[s], 'gen_world'); band(ax[2], P[P >= 2 ** 8], v[:, P >= 2 ** 8], COL[s], LAB[s] if ls == '-' else None, ls)
    ax[2].set_xscale('log', base=2); ax[2].set_xlabel('examples P'); ax[2].set_ylabel('angle of generator axes from world (°)'); ax[2].set_title('rotation (solid: factorial, dashed: deep)'); ax[2].grid(alpha=0.3)
    lo = min(a.get_ylim()[0] for a in ax[:2]); hi = max(a.get_ylim()[1] for a in ax[:2]); [a.set_ylim(lo, hi) for a in ax[:2]]
    h, l = ax[1].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=len(specs), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); out = REPORTS / 'ws-factorial.png'; fig.savefig(out, dpi=150); print(out)
    return Cd, Cf


def t_frac(steps, v, f, k=7):
    """First update at which a curve (moving average over k grid points) has covered a fraction f of its drop from the start
    to its final level (mean of the last 10 points)."""
    vs = np.convolve(np.pad(v, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid'); v0, vf = vs[:3].mean(), vs[-10:].mean()
    return steps[np.argmax(vs <= v0 - f * (v0 - vf))]


def depth_fig(run):
    j = json.loads((run / 'curves.json').read_text()); C = {s: [r[w] for w in sorted(r, key=int)] for s, r in j['curves'].items()}
    titles = {'rotated': "report's start: bottom rotated, second-order matched", 'random': 'random start in both layers'}
    steps = np.array([p['step'] for p in C['rotated'][0]]); keep = steps >= 1; V = lambda s, k: np.array([[p[k] for p in r] for r in C[s]])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for a, s in zip(ax, ['rotated', 'random']):
        band(a, steps[keep], V(s, 'gen_world')[:, keep], '#2a78d6', 'bottom layer: generator vs world')
        band(a, steps[keep], V(s, 'top_world')[:, keep], '#eb6834', 'top layer: top weights vs world')
        band(a, steps[keep], V('oracle_bottom', 'top_world')[:, keep], '0.5', "top layer, bottom fixed at the world's", '--')
        for k, col in [('gen_world', '#2a78d6'), ('top_world', '#eb6834')]:
            a.plot(np.median([t_frac(steps, v, 0.9) for v in V(s, k)]), 8, '^', color=col, ms=10, clip_on=False)
        a.set_xscale('log'); a.set_xlabel('update (one wake and one sleep batch of 32)'); a.set_title(titles[s]); a.grid(alpha=0.3); a.set_ylim(7, 66)
    ax[0].set_ylabel('angle from the world (°)'); ax[0].legend(frameon=False, loc='lower left')
    fig.text(0.5, 0.005, 'triangles: median update at which each layer has covered 90% of its improvement; 12 worlds, fresh samples, mean ± s.e.', ha='center')
    fig.tight_layout(rect=(0, 0.04, 1, 1)); out = REPORTS / 'ws-depth.png'; fig.savefig(out, dpi=150); print(out)
    for s in C:
        for k in ['gen_world', 'top_world', 'corr_err']:
            if s == 'oracle_bottom' and k == 'gen_world': continue
            t = {f: np.array([t_frac(steps, v, f) for v in V(s, k)]) for f in [0.25, 0.5, 0.9]}
            print(f"{s:14s} {k:10s} " + '  '.join(f"t{int(100 * f)} median {np.median(x):6.0f}" for f, x in t.items()) + f"   {np.mean([r[0][k] for r in C[s]]):.2f} -> {np.mean([r[-1][k] for r in C[s]]):.2f}")
    for s in ['rotated', 'random']:
        for f in [0.25, 0.5, 0.9]:
            tb, tt = (np.array([t_frac(steps, v, f) for v in V(s, k)]) for k in ['gen_world', 'top_world'])
            to = np.array([t_frac(steps, v, f) for v in V('oracle_bottom', 'top_world')])
            print(f"  {s} t{int(100 * f)}: top after bottom in {(tt > tb).sum()}/{len(tt)} worlds (median ratio {np.median(tt / np.maximum(tb, 1)):.2f}); top after oracle-bottom top in {(tt > to).sum()}/{len(tt)} (median ratio {np.median(tt / np.maximum(to, 1)):.1f})")


def numbers(C, label):
    print(f'--- {label}')
    for s, runs in C.items():
        f = lambda k: np.array([r[-1][k] for r in runs])
        P, kl = arr(runs, 'kl'); i13 = int(np.flatnonzero(P == 2 ** 13)[0]); i16 = int(np.flatnonzero(P == 2 ** 16)[0])
        print(f"{s:24s} kl@2^13 {kl[:, i13].mean():.3f} kl@2^16 {kl[:, i16].mean():.3f} kl@end {f('kl').mean():.3f} (min over P {kl.mean(0).min():.3f}) refit {f('kl_refit').mean():.3f} gen {f('gen_world').mean():.1f} "
              f"vnorm {f('v_norm').mean():.1f} sat_top {f('sat_top').mean():.2f} aeL_bot {f('ae_loss_bottom').mean():.3f} aeL_top {f('ae_loss_top').mean():.3f}" + (f" top {f('top_world').mean():.1f}" if 'top_world' in runs[0][-1] else ''))
    ref = np.array([r[-1]['kl'] for r in C['ws:x=0.7']])
    for s in C:
        if s != 'ws:x=0.7':
            d = np.array([r[-1]['kl'] for r in C[s]]) - ref; print(f'  {s:24s} minus ws70 at end: {d.mean():+.3f}, ws70 lower in {(d > 0).sum()}/{len(d)} worlds')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--deep', type=Path, default=None); ap.add_argument('--fact', type=Path, default=None); ap.add_argument('--n128', type=Path, default=None); ap.add_argument('--depth', type=Path, default=None)
    ap.add_argument('--only', default='floor,factorial,depth'); a = ap.parse_args()
    deep = (a.deep or latest('ae_floor/run_N0_2*', 'deep')) if a.only != 'depth' else None
    if 'floor' in a.only: n128 = a.n128 or latest('ae_floor/run_N128_*'); C, Cn = floor_fig(deep, n128); numbers(C, f'deep fresh {deep}'); numbers(Cn, f'deep N=128 {n128}')
    if 'factorial' in a.only:
        fact = a.fact or latest('ae_floor/run_N0_2*', 'factorial')
        _, Cf = factorial_fig(deep, fact); numbers(Cf, f'factorial fresh {fact}')
    if 'depth' in a.only: depth_fig(a.depth or latest('depth/run_N0_2*'))


if __name__ == '__main__': main()
