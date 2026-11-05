"""Where does generation start to pay, and does that threshold scale with the latent space? For practice_ws_budget.py runs
with --Ns at several K (run_dataK<K>_ws_*, run_dataK<K>_ae_*): the advantage of generation (wake-sleep without generation
minus with 70% supplied; autoencoder minus wake-sleep) against N, N / 2^K (latent configurations) and N / parameters."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path('/var/tmp/jbauer/generative_practice_disentangle/budget')
COL = {4: '#2a78d6', 6: '#eb6834', 8: '#1baf7a'}  # palette.md slots 1-3
CONTRASTS = [(('ws', '1.0'), 'advantage of generation within wake-sleep\n(no generation − 70% supplied) [nats]'), (('ae', '1.0'), 'advantage over the autoencoder\n(autoencoder − wake-sleep 70% supplied) [nats]')]


def n_params(K, D=12, H=4): return D * K + D + K * H + K  # W, c, V, a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/nfs/nhome/live/jbauer/generative-learning/reports/ws-budget-K.png'); a = ap.parse_args()
    d = {}  # (K, N, prior, learner, x) -> final checkpoints over worlds
    for run in sorted(ROOT.glob('run_dataK*')):
        for f in run.glob('N*/curves.json'):
            j = json.load(open(f)); K, N = j['args']['K'], j['args']['N']
            for prior, block in j['curves'].items():
                for l, byx in block.items():
                    for x, ws in byx.items(): d[K, N, prior, l, x] = [ws[w][-1] for w in sorted(ws, key=int)]
    Ks = sorted({k[0] for k in d}); Ns = sorted({k[1] for k in d}); priors = sorted({k[2] for k in d}); rng = np.random.default_rng(0); rows = []
    kl = lambda K, N, p, l, x: np.array([c['kl'] for c in d[K, N, p, l, x]])
    for K in Ks:
        for p in priors:
            for N in Ns:
                r = dict(K=K, prior=p, N=N, configs=2 ** K, params=n_params(K), configs_seen=float(np.mean([c['configs_seen'] for c in d[K, N, p, 'ws', '0.7']])),
                         mass_seen=float(np.mean([c['mass_seen'] for c in d[K, N, p, 'ws', '0.7']])), **{f'kl_{l}@{x}': float(kl(K, N, p, l, x).mean()) for l, x in [('ws', '0.7'), ('ws', '1.0'), ('ae', '1.0')]})
                for (l, x), _ in CONTRASTS:
                    diff = kl(K, N, p, l, x) - kl(K, N, p, 'ws', '0.7'); b = rng.choice(diff, (10000, len(diff))).mean(1)
                    r[f'adv_{l}'] = dict(mean=float(diff.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], positive_in=int((diff > 0).sum()), n=len(diff))
                rows.append(r)
    get = lambda K, p, key: [next(r for r in rows if r['K'] == K and r['prior'] == p and r['N'] == N)[key] for N in Ns]
    fig, axes = plt.subplots(len(priors) * 2, 3, figsize=(15, 4 * 2 * len(priors)), squeeze=False, layout='constrained')
    for i, p in enumerate(priors):
        for c, ((l, _), lab) in enumerate(CONTRASTS):
            for j, (scale, xl) in enumerate([(lambda K: 1, 'distinct supplied examples N'), (lambda K: 2 ** K, 'N / latent configurations $2^K$'), (lambda K: n_params(K), 'N / parameters of the model')]):
                ax = axes[2 * i + c, j]
                for K in Ks:
                    adv = get(K, p, f'adv_{l}'); xs = np.array(Ns) / scale(K)
                    ax.plot(xs, [v['mean'] for v in adv], color=COL[K], lw=2, marker='o', ms=4, label=f'K = {K} ({2 ** K} configurations, {n_params(K)} parameters)')
                    ax.fill_between(xs, [v['ci'][0] for v in adv], [v['ci'][1] for v in adv], color=COL[K], alpha=0.15, lw=0)
                ax.axhline(0, color='#52514e', lw=0.8); ax.axvline(1, color='#52514e', lw=0.8, ls=':') if j else None
                ax.set_xscale('log', base=2); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.grid(alpha=0.25, lw=0.5)
                ax.set(xlabel=xl, ylabel=lab if j == 0 else ''); ax.set_title(f'{p} prior' if j == 0 else '', loc='left')
        if i == 0: axes[0, 0].legend(frameon=False, fontsize='small')
    fig.savefig(a.out, dpi=130); print(a.out)
    for K in Ks:
        for p in priors:
            print(f'--- K = {K} ({2 ** K} configurations, {n_params(K)} parameters), {p} prior')
            for r in [r for r in rows if r['K'] == K and r['prior'] == p]:
                g, e = r['adv_ws'], r['adv_ae']
                print(f"N={r['N']:5d} configs seen {r['configs_seen']:5.1f} mass {r['mass_seen']:.2f} | KL ws70 {r['kl_ws@0.7']:.3f} ws-nogen {r['kl_ws@1.0']:.3f} ae {r['kl_ae@1.0']:.3f}"
                      f" | gen adv {g['mean']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] {g['positive_in']}/{g['n']} | vs AE {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] {e['positive_in']}/{e['n']}")
            for key in ['adv_ws', 'adv_ae']:  # smallest N from which the advantage is positive (CI above 0) at every larger N
                ok = [r['N'] for r in rows if r['K'] == K and r['prior'] == p and all(q[key]['ci'][0] > 0 for q in rows if q['K'] == K and q['prior'] == p and q['N'] >= r['N'])]
                print(f'    threshold {key}: N* = {min(ok) if ok else None}' + (f'  (N*/2^K = {min(ok) / 2 ** K:g}, N*/params = {min(ok) / n_params(K):.2f})' if ok else ''))
    json.dump(rows, open(ROOT / 'dataK_summary.json', 'w'), indent=1)


if __name__ == '__main__': main()
