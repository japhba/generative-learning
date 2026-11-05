"""Budget heatmap for the two-layer binary wake-sleep learner (toy model 2, practice_ws_layers.py 'two_layer').

A learner gets P examples in total. A fraction x are supplied examples (fresh world samples, each used once) and 1 - x
are generated examples (ancestral dreams g -> z -> x of its own two-layer model, each used once). Examples come in
batches of 32. Supplied and generated batches are interleaved so that after t batches exactly ceil(t x) were supplied,
starting with a supplied one; with a constant learning rate the learner after P examples is therefore the budget-P
learner, and one run per (world, learner, x) gives the whole column of the heatmap and its learning curve.

Learners (same world, test set and rotated start; the top layer starts from the same small random weights):
  ws        wake-sleep. Supplied examples train the generators (wake delta rules on recognised codes), generated ones
            the recognizers (sleep delta rules).
  ws_teach  the README baseline: supplied examples train both parts at every step. The generators learn by the same
            wake rule; the recognizers by posterior teaching, a delta rule towards the exact posterior means of the
            current generative model (the teacher-forcing analogue: the exact gradient of log p(x) needs exactly these
            posteriors). Generated examples still train the recognizers by sleep, so x = 1 is a learner with no
            generation at all.

Metric: population KL(world || model) in nats, on 4000 held-out world samples (exact sum over latents), the same for
every learner and x.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, evaluate, initial_learner, mixture_logpdf, posterior
from practice_disentangle_prior import top_layer

OUT = Path('/var/tmp/jbauer/generative_practice_disentangle/budget')
LEARNERS = ['ws', 'ws_teach']
XS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]


def bern(U, r): return np.where(r.random(U.shape) < 1 / (1 + np.exp(-2 * U)), 1.0, -1.0)  # p(+1) = sigmoid(2 U)


def run(task):
    prior, w, learner, xi, args = task
    x = XS[xi]; K, D, B, H, sig = args['K'], args['D'], args['batch'], args['H'], args['sigma']
    lr, lrs = args['lr'], args['lr'] * args['sleep_lr_scale']
    rng = np.random.default_rng([args['seed'], w]); world = World(K, D, sig, prior, rng)
    Xt, Zt = world.sample_x(4000, rng); true_lp = mixture_logpdf(Xt, world.A, np.zeros(D), world)
    L0 = initial_learner(world, rng, 20000, 0.02, B, 'rotated')
    V = 0.1 * np.random.default_rng([args['seed'], w, 99]).standard_normal((K, H)); a = np.zeros(K)
    R2, b2 = np.zeros((H, K)), np.zeros(H); L = Learner(L0.W, L0.c, L0.R, L0.b)
    r = np.random.default_rng([args['seed'], w, LEARNERS.index(learner), xi]); Gall = np.array(list(product([-1.0, 1.0], repeat=H)))
    nb = args['P_max'] // B; cps = {0} | {2 ** k // B for k in range(int(math.log2(B)), int(math.log2(args['P_max'])) + 1)}
    curve = []
    for t in range(nb + 1):
        if t in cps:
            e = evaluate(L, top_layer(world, V, a), Xt, Zt, true_lp)
            curve.append(dict(P=t * B, supplied=math.ceil(t * x) * B, **{k: e[k] for k in ['kl', 'align', 'acc_matched', 'acc_own']}))
        if t == nb: break
        if math.ceil((t + 1) * x) > math.ceil(t * x):  # a supplied batch
            X = world.sample_x(B, r)[0]; Z = L.sample_q(X, r); Gq = bern(Z @ R2.T + b2, r)
            if learner == 'ws_teach':  # posterior targets from the current model, before this step's updates
                Ez = posterior(X, L.W, L.c, top_layer(world, V, a)) @ world.Z
                lg = -np.logaddexp(0, -2 * (Gall @ V.T + a)[None] * Z[:, None, :]).sum(2)  # log p(z|g), (B, 2^H), uniform p(g)
                Eg = np.exp(lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)) @ Gall
                G1 = Ez - np.tanh(L.logits(X)); G2 = Eg - np.tanh(Z @ R2.T + b2)
                L.R += lrs * G1.T @ X / B; L.b += lrs * G1.mean(0); R2 += lrs * G2.T @ Z / B; b2 += lrs * G2.mean(0)
            E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / B; L.c += lr * E.mean(0)  # wake, bottom layer
            Ez2 = Z - np.tanh(Gq @ V.T + a); V += lr * Ez2.T @ Gq / B; a += lr * Ez2.mean(0)  # wake, top layer
        else:  # a generated batch: one ancestral dream per example trains both recognizers (sleep)
            Gd = np.where(r.random((B, H)) < 0.5, 1.0, -1.0); Zd = bern(Gd @ V.T + a, r); Xd = Zd @ L.W.T + L.c + sig * r.standard_normal((B, D))
            G1 = Zd - np.tanh(L.logits(Xd)); L.R += lrs * G1.T @ Xd / B; L.b += lrs * G1.mean(0)
            G2 = Gd - np.tanh(Zd @ R2.T + b2); R2 += lrs * G2.T @ Zd / B; b2 += lrs * G2.mean(0)
    return prior, w, learner, xi, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--priors', default='chain,deep'); ap.add_argument('--learners', default=','.join(LEARNERS))
    ap.add_argument('--worlds', type=int, default=12); ap.add_argument('--P_max', type=int, default=2 ** 21)
    ap.add_argument('--K', type=int, default=6); ap.add_argument('--D', type=int, default=12); ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--H', type=int, default=4); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.02); ap.add_argument('--sleep_lr_scale', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=28); ap.add_argument('--tag', default='')
    args = vars(ap.parse_args())
    out = OUT / f"run{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(p, w, l, xi, args) for p in args['priors'].split(',') for w in range(args['worlds']) for l in args['learners'].split(',') for xi in range(len(XS))]
    tasks.sort(key=lambda t: -t[3])  # supplied-heavy runs are the slowest for ws_teach; start them first
    curves = {}
    with ProcessPoolExecutor(args['workers']) as ex:
        for prior, w, learner, xi, curve in tqdm(ex.map(run, tasks), total=len(tasks), desc='runs'):
            curves.setdefault(prior, {}).setdefault(learner, {}).setdefault(str(XS[xi]), {})[w] = curve
    json.dump(dict(args=args, xs=XS, curves=curves), open(out / 'curves.json', 'w'))
    print(out)


if __name__ == '__main__': main()
