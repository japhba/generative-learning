"""Underspecifying teacher, exploring learner: can generation fill in what a few supplied examples leave open?

World: K = 10 binary causes, uniform over the 144 configurations of a hard-core chain (no two neighbouring causes both
-1), rendered x = A z + sigma eps. The teacher supplies a fixed set of N examples (N = 16 shows about 15 of the 144
configurations), reused throughout. Every local rule is visible in them (each allowed pattern of each pair of causes
appears), but most whole configurations are not.

Learner: the two-layer binary wake-sleep model of toy model 2, with H = 6 top causes (64 mixture components, enough to
memorise the supplied configurations). It starts with the true decoder and a recognizer fitted by sleep to its own
initial, nearly independent prior, so the question is only how its prior over causes is filled in.

Conditions (one supplied batch and one dream batch of 32 per step):
  ws          wake on the supplied set, sleep on dreams
  ws_teach    no generation: both parts trained on the supplied set (wake, and posterior teaching of the recognizers)
  ws_explore  ws plus one practice batch per step: dream g -> z from the top layer, flip each cause with probability
              eps (exploration), render, recognise, and keep the dream if its recognised code passes the learner's own
              check; the top layer then learns from the kept codes by the same wake rule. The check is a co-occurrence
              table of the pair patterns in the recognised codes of the supplied examples (Hebbian counts), accepting a
              code only if each of its pair patterns has been seen with frequency >= tau.
  ws_explore_iw  ws_explore with kept codes weighted by 1 / p_model(z) (normalised in the batch), against mode collapse
  ws_fresh    reference: wake on fresh world samples every step (unlimited teacher), sleep on dreams

Scores: population KL(world || model) on held-out world samples; the model prior's mass on valid configurations
(validity); the share of the 144 valid configurations given at least half their uniform mass (coverage).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, mixture_logpdf, posterior
from practice_disentangle_prior import top_layer

OUT = Path('/var/tmp/jbauer/generative_practice_disentangle/explore')
CONDS = ['ws', 'ws_teach', 'ws_explore', 'ws_explore_iw', 'ws_fresh']


def bern(U, r): return np.where(r.random(U.shape) < 1 / (1 + np.exp(-2 * U)), 1.0, -1.0)


def hardcore_world(K, D, sigma, rng):
    w = World(K, D, sigma, 'factorial', rng); valid = ~((w.Z[:, 1:] == -1) & (w.Z[:, :-1] == -1)).any(1)
    w.valid = valid; w.logp = np.where(valid, -np.log(valid.sum()), -np.inf); w.p = np.exp(w.logp)
    w.mu = w.p @ w.Z; w.Sigma = (w.Z - w.mu).T @ ((w.Z - w.mu) * w.p[:, None]); w.prior = 'hardcore'; return w


def scores(L, own, world, Xt, true_lp):
    kl = float(np.mean(true_lp - mixture_logpdf(Xt, L.W, L.c, own)))
    p = own.p; nv = world.valid.sum()
    return dict(kl=kl, validity=float(p[world.valid].sum()), coverage=float((p[world.valid] >= 0.5 / nv).mean()),
                entropy_valid=float(-(p[world.valid] * np.log(np.maximum(p[world.valid], 1e-300))).sum()))


def run(task):
    w, N, cond, args = task
    K, D, B, H, sig = args['K'], args['D'], args['batch'], args['H'], args['sigma']; lr, lrs = args['lr'], args['lr'] * args['sleep_lr_scale']
    rng = np.random.default_rng([args['seed'], w]); world = hardcore_world(K, D, sig, rng)
    Xt, _ = world.sample_x(4000, rng); true_lp = mixture_logpdf(Xt, world.A, np.zeros(D), world)
    Xs, Zs = world.sample_x(N, np.random.default_rng([args['seed'], w, N]))  # the supplied set
    r = np.random.default_rng([args['seed'], w, N, CONDS.index(cond)]); Gall = np.array(list(product([-1.0, 1.0], repeat=H)))
    V, a = 0.1 * np.random.default_rng([args['seed'], w, 99]).standard_normal((K, H)), np.zeros(K); R2, b2 = np.zeros((H, K)), np.zeros(H)
    L = Learner(world.A, np.zeros(D), np.linalg.pinv(world.A) / sig ** 2 * 0.25, np.zeros(K))
    own = top_layer(world, V, a)
    for _ in range(args['init_sleep']):  # recognizer fitted to the true decoder under the learner's own initial prior
        Zd = own.sample_z(B, r); Xd = Zd @ L.W.T + L.c + sig * r.standard_normal((B, D)); G = Zd - np.tanh(L.logits(Xd)); L.R += 0.02 * G.T @ Xd / B; L.b += 0.02 * G.mean(0)
    counts = np.zeros((K, K, 2, 2)); seen = 0; iu = np.triu_indices(K, 1); kept = []
    cps = sorted({0} | {int(v) for v in np.unique(np.round(np.logspace(1, np.log10(args['steps']), args['points'])))}); curve = []
    for t in range(args['steps'] + 1):
        own = top_layer(world, V, a) if (t in cps or cond in ('ws_teach', 'ws_explore_iw')) else None
        if t in cps: curve.append(dict(step=t, kept=float(np.mean(kept[-200:])) if kept else 0.0, **scores(L, own, world, Xt, true_lp)))
        if t == args['steps']: break
        X = world.sample_x(B, r)[0] if cond == 'ws_fresh' else Xs[r.integers(0, N, B)]
        Z = L.sample_q(X, r); Gq = bern(Z @ R2.T + b2, r)
        if cond == 'ws_teach':
            Ez = posterior(X, L.W, L.c, own) @ world.Z
            lg = -np.logaddexp(0, -2 * (Gall @ V.T + a)[None] * Z[:, None, :]).sum(2); Eg = np.exp(lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)) @ Gall
            G1 = Ez - np.tanh(L.logits(X)); G2 = Eg - np.tanh(Z @ R2.T + b2)
            L.R += lrs * G1.T @ X / B; L.b += lrs * G1.mean(0); R2 += lrs * G2.T @ Z / B; b2 += lrs * G2.mean(0)
        if cond.startswith('ws_explore'):  # the check: pair patterns of the recognised (mean) codes of supplied examples
            S = (np.sign(L.mean_code(X)) > 0).astype(int)
            for i, j in zip(*iu): np.add.at(counts[i, j], (S[:, i], S[:, j]), 1)
            seen += B
        E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / B; L.c += lr * E.mean(0)
        Ez2 = Z - np.tanh(Gq @ V.T + a); V += lr * Ez2.T @ Gq / B; a += lr * Ez2.mean(0)
        if cond != 'ws_teach':  # sleep: one ancestral dream per example trains both recognizers
            Gd = np.where(r.random((B, H)) < 0.5, 1.0, -1.0); Zd = bern(Gd @ V.T + a, r); Xd = Zd @ L.W.T + L.c + sig * r.standard_normal((B, D))
            G1 = Zd - np.tanh(L.logits(Xd)); L.R += lrs * G1.T @ Xd / B; L.b += lrs * G1.mean(0)
            G2 = Gd - np.tanh(Zd @ R2.T + b2); R2 += lrs * G2.T @ Zd / B; b2 += lrs * G2.mean(0)
        if cond.startswith('ws_explore'):  # practice: explore around own dreams, keep what passes the check, learn the prior from it
            Gd = np.where(r.random((B, H)) < 0.5, 1.0, -1.0); Zd = bern(Gd @ V.T + a, r); Zd = np.where(r.random(Zd.shape) < args['eps'], -Zd, Zd)
            Xd = Zd @ L.W.T + L.c + sig * r.standard_normal((B, D)); Zr = np.sign(L.mean_code(Xd)); Zr[Zr == 0] = 1; S = (Zr > 0).astype(int)
            freq = counts[iu[0], iu[1], S[:, iu[0]], S[:, iu[1]]] / max(seen, 1)  # (B, pairs)
            ok = (freq >= args['tau']).all(1); kept.append(float(ok.mean()))
            if ok.any():
                Zk = Zr[ok]; wts = np.ones(len(Zk))
                if cond == 'ws_explore_iw':
                    idx = ((Zk > 0).astype(int) * (2 ** np.arange(K - 1, -1, -1))).sum(1)  # row index in world.Z (product order, -1 before +1)
                    wts = 1 / np.maximum(own.p[idx], 1e-12); wts = wts / wts.mean()
                Gk = bern(Zk @ R2.T + b2, r); Ek = (Zk - np.tanh(Gk @ V.T + a)) * wts[:, None]
                V += lr * Ek.T @ Gk / B; a += lr * Ek.sum(0) / B
    return w, N, cond, curve, int(len(set(map(tuple, Zs.astype(int)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--conds', default=','.join(CONDS)); ap.add_argument('--Ns', default='16,32,64'); ap.add_argument('--worlds', type=int, default=8)
    ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--points', type=int, default=24)
    ap.add_argument('--K', type=int, default=10); ap.add_argument('--D', type=int, default=20); ap.add_argument('--sigma', type=float, default=0.4)
    ap.add_argument('--H', type=int, default=6); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--init_sleep', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=0.02); ap.add_argument('--sleep_lr_scale', type=float, default=10.0)
    ap.add_argument('--eps', type=float, default=0.1); ap.add_argument('--tau', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=28); ap.add_argument('--tag', default='')
    args = vars(ap.parse_args())
    out = OUT / f"run{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(w, N, c, args) for N in map(int, args['Ns'].split(',')) for w in range(args['worlds']) for c in args['conds'].split(',')]
    res = {}
    with ProcessPoolExecutor(args['workers']) as ex:
        for w, N, cond, curve, distinct in tqdm(ex.map(run, tasks), total=len(tasks), desc='runs'):
            res.setdefault(str(N), {}).setdefault(cond, {})[w] = dict(curve=curve, distinct=distinct)
    json.dump(dict(args=args, results=res), open(out / 'results.json', 'w'))
    for N, block in res.items():
        for cond, runs in block.items():
            last = [v['curve'][-1] for v in runs.values()]
            print(f"N={N:>3s} {cond:14s} " + '  '.join(f"{k} {np.mean([l[k] for l in last]):.3f}" for k in ['kl', 'validity', 'coverage', 'entropy_valid', 'kept']) + f"  distinct supplied {np.mean([v['distinct'] for v in runs.values()]):.1f}")
    print(out)


if __name__ == '__main__': main()
