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
  ae        the autoencoding baseline (README): supplied examples train both parts end-to-end by reconstruction error.
            Bottom layer: binary autoencoder on x (mean-field codes tanh(R x + b), expected squared error, backprop into
            R and W; practice_disentangle.ae_step 'bae'). Top layer: the same on the recognised codes z (codes
            tanh(R2 z + b2), log-loss of z under sigmoid(2 (V m + a))). Generated examples train the recognizers by sleep,
            so x = 1 is a pure autoencoder that never generates.

  ws_replay       ws whose bottom-layer dreams take their latents from a replay of the codes recognition gave to the last
                  1,024 supplied examples: no exploration of latent configurations beyond the current explanations of the data
  ws_replay_wide  the same, with the rendering noise of those dreams inflated so that they spread as far in observation
                  space (total variance) as dreams from the prior: observation-space exploration matched, latent exploration not
  ws_uniform      ws whose bottom-layer dreams take uniform independent latents: maximal latent exploration, ignoring the prior
  (in all three the top layer still sleeps on its own prior dreams; checkpoints also record the dream latents' entropy,
  the dreams' total variance in x, and the mean distance from held-out world samples to the nearest dream)

Metric: population KL(world || model) in nats, on 4000 held-out world samples (exact sum over latents), the same for
every learner and x.

--Ns caps the data: supplied batches are then drawn with replacement from a fixed set of N world samples (the first N of
one pool per world, so the sets are nested across N), each reused many times, while generated examples stay unlimited.
Checkpoints then also record kl_train (the same KL on the training set; kl - kl_train measures memorisation) and how much
of the latent distribution the N examples visit (distinct latent configurations, their true prior mass).
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

from practice_disentangle import Learner, World, ae_step, evaluate, initial_learner, mixture_logpdf, posterior
from practice_disentangle_prior import top_layer

OUT = Path('/var/tmp/jbauer/generative_practice_disentangle/budget')
LEARNERS = ['ws', 'ws_teach', 'ae', 'ws_replay', 'ws_replay_wide', 'ws_uniform']
DREAMS = {'ws_replay': 'replay', 'ws_replay_wide': 'replay_wide', 'ws_uniform': 'uniform'}  # bottom-layer dream latents; plain ws dreams from its prior
XS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]


def bern(U, r): return np.where(r.random(U.shape) < 1 / (1 + np.exp(-2 * U)), 1.0, -1.0)  # p(+1) = sigmoid(2 U)


def run(task):
    prior, w, learner, xi, args = task
    x = XS[xi]; K, D, B, H, sig = args['K'], args['D'], args['batch'], args['H'], args['sigma']
    lr, lrs = args['lr'], args['lr'] * args['sleep_lr_scale']
    rng = np.random.default_rng([args['seed'], w]); world = World(K, D, sig, prior, rng)
    Xt, Zt = world.sample_x(4000, rng); true_lp = mixture_logpdf(Xt, world.A, np.zeros(D), world)
    L0 = initial_learner(world, rng, 20000, 0.02, B, 'rotated')
    N = args['N']
    if N:  # a fixed set of N supplied examples, reused
        Xpool, Zpool = world.sample_x(args['N_pool'], np.random.default_rng([args['seed'], w, 7])); Xpool, Zpool = Xpool[:N], Zpool[:N]
        Xtr = Xpool[:4000]; true_lp_tr = mixture_logpdf(Xtr, world.A, np.zeros(D), world)
        seen = (world.Z[:, None, :] == np.unique(Zpool, axis=0)[None]).all(2).any(1); cover = dict(configs_seen=int(seen.sum()), mass_seen=float(world.p[seen].sum()))
    V = 0.1 * np.random.default_rng([args['seed'], w, 99]).standard_normal((K, H)); a = np.zeros(K)
    R2, b2 = np.zeros((H, K)), np.zeros(H); L = Learner(L0.W, L0.c, L0.R, L0.b)
    r = np.random.default_rng([args['seed'], w, LEARNERS.index(learner), xi]); Gall = np.array(list(product([-1.0, 1.0], repeat=H)))
    nb = args['P_max'] // B; cps = {0} | {2 ** k // B for k in range(int(math.log2(B)), int(math.log2(args['P_max'])) + 1)}
    curve = []; dreams = DREAMS.get(learner, 'prior'); buf = np.zeros((0, K)); sig_w = sig
    def bottom_latents(n, own=None):
        """Latents of the bottom layer's dreams: the learned prior, a replay of the codes given to supplied examples, or uniform."""
        if dreams == 'uniform': return np.where(r.random((n, K)) < 0.5, 1.0, -1.0)
        if dreams.startswith('replay') and len(buf): return buf[r.integers(0, len(buf), n)]
        return bern(np.where(r.random((n, H)) < 0.5, 1.0, -1.0) @ V.T + a, r)
    for t in range(nb + 1):
        if dreams == 'replay_wide' and t % 50 == 0 and len(buf) > 1:  # inflate rendering noise so the dreams spread as far in x as prior dreams
            Sp = top_layer(world, V, a).Sigma; Sr = np.cov(buf.T); sig_w = math.sqrt(sig ** 2 + max(0., np.trace(L.W @ (Sp - Sr) @ L.W.T)) / D)
        if t in cps:
            tw = top_layer(world, V, a); e = evaluate(L, tw, Xt, Zt, true_lp)
            if N: e.update(kl_train=float(np.mean(true_lp_tr - mixture_logpdf(Xtr, L.W, L.c, tw))), **cover)
            Zc = bottom_latents(4096); Xc = Zc @ L.W.T + L.c + (sig_w if dreams == 'replay_wide' else sig) * r.standard_normal((4096, D))
            counts = np.unique(Zc, axis=0, return_counts=True)[1] / 4096; nn = np.sqrt(((Xt[:1000, None, :] - Xc[None]) ** 2).sum(2).min(1))
            curve.append(dict(P=t * B, supplied=math.ceil(t * x) * B, **{k: e[k] for k in ['kl', 'align', 'acc_matched', 'acc_own', 'kl_train', 'configs_seen', 'mass_seen'] if k in e},
                              dream_latent_entropy=float(-(counts * np.log(counts)).sum()), dream_x_var=float(np.trace(np.cov(Xc.T))), world_to_dream_nn=float(nn.mean())))
        if t == nb: break
        if math.ceil((t + 1) * x) > math.ceil(t * x):  # a supplied batch
            X = Xpool[r.integers(0, N, B)] if N else world.sample_x(B, r)[0]; Z = L.sample_q(X, r); Gq = bern(Z @ R2.T + b2, r)
            if learner == 'ws_teach':  # posterior targets from the current model, before this step's updates
                Ez = posterior(X, L.W, L.c, top_layer(world, V, a)) @ world.Z
                lg = -np.logaddexp(0, -2 * (Gall @ V.T + a)[None] * Z[:, None, :]).sum(2)  # log p(z|g), (B, 2^H), uniform p(g)
                Eg = np.exp(lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)) @ Gall
                G1 = Ez - np.tanh(L.logits(X)); G2 = Eg - np.tanh(Z @ R2.T + b2)
                L.R += lrs * G1.T @ X / B; L.b += lrs * G1.mean(0); R2 += lrs * G2.T @ Z / B; b2 += lrs * G2.mean(0)
            if learner == 'ae':  # end-to-end reconstruction in both layers, no recognised samples in the generator update
                ae_step(L, X, world, lr, args['ae_lr_scale'], 'bae')
                m2 = np.tanh(Z @ R2.T + b2); E2 = Z - np.tanh(m2 @ V.T + a); g2 = (E2 @ V) * (1 - m2 ** 2)
                V += lr * E2.T @ m2 / B; a += lr * E2.mean(0); R2 += lr * args['ae_lr_scale'] * g2.T @ Z / B; b2 += lr * args['ae_lr_scale'] * g2.mean(0)
                continue
            if dreams.startswith('replay'): buf = np.concatenate([buf, Z])[-1024:]  # the explanations recognition gave to supplied examples
            E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / B; L.c += lr * E.mean(0)  # wake, bottom layer
            Ez2 = Z - np.tanh(Gq @ V.T + a); V += lr * Ez2.T @ Gq / B; a += lr * Ez2.mean(0)  # wake, top layer
        else:  # a generated batch: one ancestral dream per example trains both recognizers (sleep)
            Gd = np.where(r.random((B, H)) < 0.5, 1.0, -1.0); Zd = bern(Gd @ V.T + a, r)
            Zb = Zd if dreams == 'prior' else bottom_latents(B)  # the top layer always sleeps on its own prior dreams
            Xd = Zb @ L.W.T + L.c + (sig_w if dreams == 'replay_wide' else sig) * r.standard_normal((B, D))
            G1 = Zb - np.tanh(L.logits(Xd)); L.R += lrs * G1.T @ Xd / B; L.b += lrs * G1.mean(0)
            G2 = Gd - np.tanh(Zd @ R2.T + b2); R2 += lrs * G2.T @ Zd / B; b2 += lrs * G2.mean(0)
    return prior, w, learner, xi, N, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--priors', default='chain,deep'); ap.add_argument('--learners', default=','.join(LEARNERS))
    ap.add_argument('--worlds', type=int, default=12); ap.add_argument('--P_max', type=int, default=2 ** 21)
    ap.add_argument('--K', type=int, default=6); ap.add_argument('--D', type=int, default=12); ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--H', type=int, default=4); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.02); ap.add_argument('--sleep_lr_scale', type=float, default=10.0); ap.add_argument('--ae_lr_scale', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=28); ap.add_argument('--tag', default='')
    ap.add_argument('--xs', default='', help='comma-separated fractions supplied to run (default: all of XS)')
    ap.add_argument('--Ns', default='0', help='comma-separated numbers of distinct supplied examples, reused; 0 = fresh world samples, each used once')
    args = vars(ap.parse_args()); Ns = [int(v) for v in args['Ns'].split(',')]; args['N_pool'] = max(Ns)
    out = OUT / f"run{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(p, w, l, xi, dict(args, N=N)) for N in Ns for p in args['priors'].split(',') for w in range(args['worlds']) for l in args['learners'].split(',') for xi in range(len(XS)) if not args['xs'] or any(abs(XS[xi] - float(v)) < 1e-9 for v in args['xs'].split(','))]
    tasks.sort(key=lambda t: -t[3])  # supplied-heavy runs are the slowest for ws_teach; start them first
    curves = {}
    with ProcessPoolExecutor(args['workers']) as ex:
        for prior, w, learner, xi, N, curve in tqdm(ex.map(run, tasks), total=len(tasks), desc='runs'):
            curves.setdefault(N, {}).setdefault(prior, {}).setdefault(learner, {}).setdefault(str(XS[xi]), {})[w] = curve
    for N in Ns:  # one curves.json per N (in N<N>/ when several)
        d = out if len(Ns) == 1 else out / f'N{N}'; d.mkdir(exist_ok=True); json.dump(dict(args=dict(args, N=N), xs=XS, curves=curves[N]), open(d / 'curves.json', 'w'))
    print(out)


if __name__ == '__main__': main()
