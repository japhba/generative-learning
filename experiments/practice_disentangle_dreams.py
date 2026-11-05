"""Where should the dreams' latents come from, when the learner's prior is fixed and right (the first toy)?

Same world, misaligned start, stored data and budget as practice_disentangle.py (one wake batch on stored data and one
dream batch per step). Only the latents of the dreams differ:
  prior    z from the learner's own (fixed, true) prior: plain wake-sleep
  replay   z = the recogniser's current reading of a stored example: dreams that only re-render current explanations
  uniform  z uniform and independent: exploration that ignores the prior
  none     no dreams (a second wake batch instead): recognition frozen
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, evaluate, initial_learner, mixture_logpdf

CONDS = ['prior', 'replay', 'uniform', 'none']


def run(task):
    prior, w, cond, a = task
    rng = np.random.default_rng([a['seed'], w]); world = World(a['K'], a['D'], a['sigma'], prior, rng)
    Xdata, _ = world.sample_x(a['N'], rng); Xt, Zt = world.sample_x(4000, rng); true_lp = mixture_logpdf(Xt, world.A, np.zeros(a['D']), world)
    L0 = initial_learner(world, rng, 20000, 0.02, a['batch'], 'rotated'); L = Learner(L0.W, L0.c, L0.R, L0.b)
    r = np.random.default_rng([a['seed'], w, CONDS.index(cond)]); B, lr, lrs = a['batch'], a['lr'], a['lr'] * a['sleep_lr_scale']; curve = []
    every = a['steps'] // a['points']
    for t in range(a['steps'] + 1):
        if t % every == 0:
            e = evaluate(L, world, Xt, Zt, true_lp)
            Zc = world.sample_z(2048, r) if cond in ('prior', 'none') else L.sample_q(Xdata[r.integers(0, len(Xdata), 2048)], r) if cond == 'replay' else np.where(r.random((2048, a['K'])) < .5, 1., -1.)
            cnt = np.unique(Zc, axis=0, return_counts=True)[1] / 2048
            curve.append(dict(step=t, kl=e['kl'], acc_sym=e['acc_sym'], align=e['align'], dream_latent_entropy=float(-(cnt * np.log(cnt)).sum())))
        if t == a['steps']: break
        L.wake(Xdata[r.integers(0, len(Xdata), B)], r, lr)
        if cond == 'none': L.wake(Xdata[r.integers(0, len(Xdata), B)], r, lr); continue
        Z = world.sample_z(B, r) if cond == 'prior' else L.sample_q(Xdata[r.integers(0, len(Xdata), B)], r) if cond == 'replay' else np.where(r.random((B, a['K'])) < .5, 1., -1.)
        X = Z @ L.W.T + L.c + world.sigma * r.standard_normal((B, world.D)); G = Z - np.tanh(L.logits(X))
        L.R += lrs * G.T @ X / B; L.b += lrs * G.mean(0)
    return prior, w, cond, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--priors', default='factorial,chain,deep'); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--points', type=int, default=40); ap.add_argument('--K', type=int, default=6); ap.add_argument('--D', type=int, default=12)
    ap.add_argument('--sigma', type=float, default=0.5); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--sleep_lr_scale', type=float, default=10.0); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--workers', type=int, default=28)
    a = vars(ap.parse_args()); out = Path('/var/tmp/jbauer/generative_practice_disentangle/dreams') / f"run_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(p, w, c, a) for p in a['priors'].split(',') for w in range(a['worlds']) for c in CONDS]; res = {}
    with ProcessPoolExecutor(a['workers']) as ex:
        for p, w, c, curve in tqdm(ex.map(run, tasks), total=len(tasks), desc='runs'): res.setdefault(p, {}).setdefault(c, {})[w] = curve
    json.dump(dict(args=a, results=res), open(out / 'results.json', 'w'))
    rng = np.random.default_rng(0)
    for p in res:
        fin = {c: np.array([res[p][c][w][-1]['kl'] for w in range(a['worlds'])]) for c in CONDS}
        for c in CONDS:
            d = fin[c] - fin['prior']; b = rng.choice(d, (10000, len(d))).mean(1)
            print(f"{p:9s} {c:8s} final KL {fin[c].mean():.3f}  vs prior dreams {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] lower in {(d < 0).sum()}/{len(d)}"
                  f"  binding acc {np.mean([res[p][c][w][-1]['acc_sym'] for w in range(a['worlds'])]):.2f}  dream latent entropy {np.mean([res[p][c][w][-1]['dream_latent_entropy'] for w in range(a['worlds'])]):.2f}")
    print(out)


if __name__ == '__main__': main()
