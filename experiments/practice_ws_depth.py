"""Is the deep layer of the two-layer wake-sleep learner (toy model 2) learned later than the shallow one?

Practice as in practice_ws_layers.py 'two_layer' (deep world: two top causes set the probabilities of six causes; learner with
H = 4 top units; one wake and one sleep batch of 32 per update), with the alignment of each layer to the world recorded on a
log-spaced grid of updates:
  bottom  gen_world: mean angle between the generator's columns w_i and the world's columns a_j they are matched to
  top     top_world: the top weights V re-expressed in world order (through the bottom's current matching) against the
          world's top loadings B, matched column by column; corr_err: the learner's prior correlations, in world order,
          against the world's
Starts:
  rotated        the report's start: bottom generator second-order matched but rotated, top layer small random weights
  random         generic random bottom generator (column norm init_scale), recognizer fitted to it by sleep; top as above
  oracle_bottom  bottom generator fixed at the world's A and recognizer fixed at the world readout (logistic regression of
                 the causes on world samples); only the top layer learns. Its timing is the top layer's own speed, so a top
                 layer that lags further behind in the joint learner is waiting for the bottom.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, initial_learner, mixture_logpdf
from practice_disentangle_prior import top_layer
from practice_ws_layers import logistic, match, unit

OUT = Path('/var/tmp/jbauer/generative_practice_disentangle/depth')


def bern(U, r): return np.where(r.random(U.shape) < 1 / (1 + np.exp(-2 * U)), 1.0, -1.0)


def run(task):
    w, start, args = task
    K, D, B, H, sig, lr, lrs = args['K'], args['D'], args['batch'], args['H'], args['sigma'], args['lr'], args['lr'] * args['sleep_lr_scale']
    rng = np.random.default_rng([args['seed'], w]); world = World(K, D, sig, 'deep', rng)
    Xdata, _ = world.sample_x(max(args['N'], 1024), rng); Xdata = Xdata[:args['N']]; Xt, Zt = world.sample_x(4000, rng)
    true_lp = mixture_logpdf(Xt, world.A, np.zeros(D), world)
    L0 = initial_learner(world, rng, 20000, 0.02, B, 'random' if start == 'random' else 'rotated', args['init_scale'])
    r = np.random.default_rng([args['seed'], w, ['rotated', 'random', 'oracle_bottom'].index(start)])
    L = Learner(L0.W, L0.c, L0.R, L0.b); V, a = 0.1 * r.standard_normal((K, H)), np.zeros(K); R2, b2 = np.zeros((H, K)), np.zeros(H)
    if start == 'oracle_bottom':
        Rw = logistic(Xt, Zt); L = Learner(world.A, np.zeros(D), Rw[:, :D], Rw[:, D])
    Sw = world.Sigma / np.outer(np.sqrt(np.diag(world.Sigma)), np.sqrt(np.diag(world.Sigma))); off = ~np.eye(K, dtype=bool)
    grid = set(np.unique(np.round(np.geomspace(1, args['steps'], args['points'])).astype(int)).tolist()) | {0}; curve = []
    for t in range(args['steps'] + 1):
        if t in grid:
            own = top_layer(world, V, a); C = unit(L.W).T @ world.A; p, s = match(C)
            Vw = np.zeros_like(V); Vw[p] = V * s[:, None]; Ct = unit(world.params['B']).T @ unit(Vw); q, _ = match(Ct)
            d = np.sqrt(np.diag(own.Sigma)); corr = np.zeros((K, K)); corr[np.ix_(p, p)] = own.Sigma / np.outer(d, d) * np.outer(s, s)
            curve.append(dict(step=t, kl=float(np.mean(true_lp - mixture_logpdf(Xt, L.W, L.c, own))),
                              gen_world=float(np.degrees(np.arccos(np.minimum(np.abs(C[np.arange(K), p]), 1))).mean()),
                              top_world=float(np.degrees(np.arccos(np.minimum(np.abs(Ct[np.arange(len(Ct)), q]), 1))).mean()),
                              corr_err=float(np.sqrt(((corr - Sw)[off] ** 2).mean())), v_norm=float(np.linalg.norm(V))))
        if t == args['steps']: break
        X = Xdata[r.integers(0, len(Xdata), B)] if args['N'] else world.sample_x(B, r)[0]
        Z = L.sample_q(X, r); Gq = bern(Z @ R2.T + b2, r)  # wake: read upward, delta rules on both generators
        if start != 'oracle_bottom': E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / B; L.c += lr * E.mean(0)
        Ez = Z - np.tanh(Gq @ V.T + a); V += lr * Ez.T @ Gq / B; a += lr * Ez.mean(0)
        Gd = np.where(r.random((B, H)) < 0.5, 1.0, -1.0); Zd = bern(Gd @ V.T + a, r)  # sleep: dream downward, delta rules on both recognizers
        if start != 'oracle_bottom':
            Xd = Zd @ L.W.T + L.c + sig * r.standard_normal((B, D)); G1 = Zd - np.tanh(L.logits(Xd)); L.R += lrs * G1.T @ Xd / B; L.b += lrs * G1.mean(0)
        G2 = Gd - np.tanh(Zd @ R2.T + b2); R2 += lrs * G2.T @ Zd / B; b2 += lrs * G2.mean(0)
    return w, start, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--starts', default='rotated,random,oracle_bottom'); ap.add_argument('--worlds', type=int, default=12)
    ap.add_argument('--steps', type=int, default=40000); ap.add_argument('--points', type=int, default=120)
    ap.add_argument('--N', type=int, default=0, help='stored samples; 0 = a fresh batch every update')
    ap.add_argument('--K', type=int, default=6); ap.add_argument('--D', type=int, default=12); ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--H', type=int, default=4); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--init_scale', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=0.02); ap.add_argument('--sleep_lr_scale', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=28); ap.add_argument('--tag', default='')
    args = vars(ap.parse_args())
    out = OUT / f"run_N{args['N']}{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(w, s, args) for s in args['starts'].split(',') for w in range(args['worlds'])]; curves = {}
    with ProcessPoolExecutor(args['workers']) as ex:
        for w, start, curve in tqdm(ex.map(run, tasks), total=len(tasks), desc='runs'): curves.setdefault(start, {})[w] = curve
    json.dump(dict(args=args, curves=curves), open(out / 'curves.json', 'w'))
    for start, runs in curves.items():
        print(f'{start:14s} ' + '  '.join(f"{k} {np.mean([runs[w][0][k] for w in runs]):.2f}->{np.mean([runs[w][-1][k] for w in runs]):.2f}" for k in ['kl', 'gen_world', 'top_world', 'corr_err']))
    print(out)


if __name__ == '__main__': main()
