"""Weight snapshots for the wake-sleep report (toy model 2): one- and two-layer wake-sleep on the binary world and its
linear-Gaussian twin, with every weight saved along training so the report can animate the rotation of the latent space
and the alignment of recognition with generation, layer by layer.

Conditions (same world, stored data, rotated start and budget: one wake and one sleep batch of 32 per step):
  two_layer    binary z with a learned top layer g in {-1,+1}^H, p(z_i=+1|g) = sigmoid(2 (V g + a)_i); V by the wake delta
               rule on recognised codes, its recognizer R2 by the sleep delta rule (practice_disentangle_prior.py, top_layer)
  independent  one layer, independent binary latents with learned frequencies (no correlations can be represented)
  true_prior   one layer, the world's prior fixed and known (the original toy)
  gaussian     linear-Gaussian twin of true_prior: z ~ N(mu, Sigma) with the world prior's mean and covariance, recognizer
               N(R x + b, diag s2), the same delta rules with the identity in place of tanh (Neal & Dayan 1997)

Alignment is measured as angles between weight vectors. Sleep is logistic (Gaussian: linear) regression of dreamed causes
on dreams, so its fixed point for the current generator, R*(W), is fitted at every snapshot; the recognizer's lag is the
angle between R and R*(W). The world readout is the same regression on foreign samples and their true causes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import permutations, product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, evaluate, initial_learner, mixture_logpdf
from practice_disentangle_prior import top_layer

OUT = Path('/var/tmp/jbauer/generative_practice_disentangle/layers')
CONDS = ['two_layer', 'two_layer_h2', 'independent', 'true_prior', 'gaussian']


def match(C):
    """Best one-to-one assignment of the rows of |C| to its columns (rows <= columns), by enumeration: columns, signs."""
    k, m = C.shape; P = np.array(list(permutations(range(m), k)))
    p = P[np.abs(C)[np.arange(k), P].sum(1).argmax()]
    return p, np.sign(C[np.arange(k), p])


def unit(M, axis=0): return M / np.maximum(np.linalg.norm(M, axis=axis, keepdims=True), 1e-12)
def angles(U, V): return np.degrees(np.arccos(np.clip(np.abs((unit(U, 1) * unit(V, 1)).sum(1)), 0, 1)))  # row by row


def logistic(F, Y, w=None, ridge=1e-3, iters=25):
    """Readout weights r (one row per target) of p(y=+1|f) = sigmoid(2 (r.f + b)) for +-1 targets, by Newton's method."""
    n, d = F.shape; F1 = np.hstack([F, np.ones((n, 1))]); w = np.full(n, 1 / n) if w is None else w / w.sum(); out = []
    for y in ((Y + 1) / 2).T:
        beta = np.zeros(d + 1)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-F1 @ beta)); g = F1.T @ (w * (y - p)) - ridge * beta; Hs = (F1 * (w * p * (1 - p))[:, None]).T @ F1 + ridge * np.eye(d + 1)
            beta += np.linalg.solve(Hs, g)
        out.append(beta / 2)
    return np.array(out)  # (targets, d + 1): weights, then bias


def gaussian_kl(W, c, mu, Sigma, world, Xtest, true_lp):
    C = W @ Sigma @ W.T + world.sigma ** 2 * np.eye(world.D); d = Xtest - (W @ mu + c); _, logdet = np.linalg.slogdet(C)
    return float(np.mean(true_lp + 0.5 * (np.einsum('ni,ij,nj->n', d, np.linalg.inv(C), d) + logdet + world.D * np.log(2 * np.pi))))


def run(task):
    prior, w, cond, args = task
    rng = np.random.default_rng([args['seed'], w]); world = World(args['K'], args['D'], args['sigma'], prior, rng)
    Xdata, _ = world.sample_x(max(args['N'], 1024), rng); Xdata = Xdata[:args['N']]; Xt, Zt = world.sample_x(4000, rng)
    true_lp = mixture_logpdf(Xt, world.A, np.zeros(args['D']), world)
    L0 = initial_learner(world, rng, 20000, 0.02, args['batch'], 'rotated')  # same start for every condition (as in practice_disentangle_prior.py)
    r = np.random.default_rng([args['seed'], w, CONDS.index(cond)])
    K, D, B, lr, lrs, sig = world.K, world.D, args['batch'], args['lr'], args['lr'] * args['sleep_lr_scale'], world.sigma
    H = 2 if cond == 'two_layer_h2' else args['H']; gauss = cond == 'gaussian'
    L = Learner(L0.W, L0.c, L0.R, L0.b); h = np.zeros(K)
    V, a = 0.1 * r.standard_normal((K, H)), np.zeros(K); R2, b2 = np.zeros((H, K)), np.zeros(H)
    mu, Sigma = world.mu, world.Sigma; chol = np.linalg.cholesky(Sigma); s2 = np.full(K, 0.1)
    # world readouts: the regression a perfectly aligned learner's sleep would converge to, one row per world cause
    Rw = (np.linalg.lstsq(np.hstack([Xt, np.ones((len(Xt), 1))]), Zt, rcond=None)[0].T if gauss else logistic(Xt, Zt))[:, :D]
    if gauss:  # Gaussian recognizer fitted to the rotated start by the same number of sleep steps
        L.R = np.linalg.pinv(L0.W) * 0.25; L.b = np.zeros(K)
        for _ in range(20000):
            Z = mu + r.standard_normal((B, K)) @ chol.T; X = Z @ L.W.T + L.c + sig * r.standard_normal((B, D))
            G = Z - (X @ L.R.T + L.b); L.R += 0.02 * G.T @ X / B; L.b += 0.02 * G.mean(0); s2 += 0.02 * ((G ** 2).mean(0) - s2)
    Gall = np.array(list(product([-1.0, 1.0], repeat=H)))
    every = args['steps'] // args['frames']; snaps, curve = {}, []
    for t in range(args['steps'] + 1):
        own = top_layer(world, V, a) if cond.startswith('two_layer') else top_layer(world, np.zeros((K, 1)), h) if cond == 'independent' else world
        if t % every == 0:
            if gauss:
                m = Xt @ L.R.T + L.b; r0 = np.random.default_rng(12345); Zo = mu + r0.standard_normal((2000, K)) @ chol.T
                Xo = Zo @ L.W.T + L.c + sig * r0.standard_normal((2000, D)); p, s = match(unit(L.W).T @ world.A)
                ev = dict(kl=gaussian_kl(L.W, L.c, mu, Sigma, world, Xt, true_lp), acc_matched=float(np.mean(np.sign(m) * s == Zt[:, p])), acc_own=float(np.mean(np.sign(Xo @ L.R.T + L.b) == np.sign(Zo))))
                Rstar = np.linalg.solve(L.W.T @ L.W / sig ** 2 + np.linalg.inv(Sigma), L.W.T / sig ** 2)
            else:
                e = evaluate(L, own, Xt, Zt, true_lp); ev = {k: e[k] for k in ['kl', 'acc_matched', 'acc_own']}
                r0 = np.random.default_rng(12345); Zo = own.sample_z(8192, r0); Xo = Zo @ L.W.T + L.c + sig * r0.standard_normal((8192, D))
                Rstar = logistic(Xo, Zo)[:, :D]
            C = unit(L.W).T @ world.A; p, s = match(C)  # learner unit i renders world cause p[i] with sign s[i]
            ev |= dict(gen_world=float(np.degrees(np.arccos(np.minimum(np.abs(C[np.arange(K), p]), 1))).mean()), rec_lag=float(angles(L.R, Rstar).mean()), rec_world=float(angles(L.R, Rw[p]).mean()), ideal_world=float(angles(Rstar, Rw[p]).mean()))
            Sl = Sigma if gauss else own.Sigma; d = np.sqrt(np.diag(Sl)); corr = np.zeros((K, K)); corr[np.ix_(p, p)] = Sl / np.outer(d, d) * np.outer(s, s)  # prior correlations, world order
            dw = np.sqrt(np.diag(Sigma)); off = ~np.eye(K, dtype=bool); ev['corr_err'] = float(np.sqrt(((corr - Sigma / np.outer(dw, dw))[off] ** 2).mean()))
            snap = dict(W=L.W, R=L.R, Rstar=Rstar, corr=corr, perm=p, sign=s)
            if cond.startswith('two_layer'):  # top layer: R2 against the sleep fixed point R2*(V), fitted on the enumerated joint of (g, z)
                lz = -np.logaddexp(0, -2 * (Gall @ V.T + a)[:, None, :] * world.Z[None]).sum(2)  # log p(z|g), (2^H, 2^K)
                Zr, Gr = np.tile(world.Z, (len(Gall), 1)), np.repeat(Gall, len(world.Z), 0)  # every (g, z) pair, weighted by p(g) p(z|g)
                R2star = logistic(Zr, Gr, w=np.exp(lz).ravel())[:, :K]
                wts = np.linalg.norm(R2star, axis=1); ev |= dict(top_lag=float(np.average(angles(R2, R2star), weights=wts)) if np.abs(R2).sum() else 90.0, top_strength=np.linalg.norm(V, axis=0).tolist())
                ev['top_acc_own'] = float(np.mean(np.sign(Zr @ R2.T + b2) == Gr, axis=1) @ (np.exp(lz).ravel() / len(Gall)))
                Vw = np.zeros_like(V); Vw[p] = V * s[:, None]  # the top layer's weights in world order
                snap |= dict(V=V, R2=R2, R2star=R2star, Vw=Vw)
                if prior == 'deep':
                    Ct = unit(world.params['B']).T @ unit(Vw); q, _ = match(Ct); ev['top_world'] = float(np.degrees(np.arccos(np.minimum(np.abs(Ct[np.arange(len(Ct)), q]), 1))).mean())
            curve.append(dict(step=t, **ev))
            for k, v in snap.items(): snaps.setdefault(k, []).append(np.array(v, copy=True))
        if t == args['steps']: break
        X = Xdata[r.integers(0, len(Xdata), B)] if args['N'] else world.sample_x(B, r)[0]
        if gauss:  # wake: sample a Gaussian code, delta rule on the generator; sleep: delta rule on the recognizer
            Z = X @ L.R.T + L.b + np.sqrt(s2) * r.standard_normal((B, K)); E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / B; L.c += lr * E.mean(0)
            Zd = mu + r.standard_normal((B, K)) @ chol.T; Xd = Zd @ L.W.T + L.c + sig * r.standard_normal((B, D))
            G = Zd - (Xd @ L.R.T + L.b); L.R += lrs * G.T @ Xd / B; L.b += lrs * G.mean(0); s2 += lrs * ((G ** 2).mean(0) - s2)
            continue
        Z = L.sample_q(X, r); E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / B; L.c += lr * E.mean(0)  # wake, bottom layer
        if cond == 'independent': h += lr * (Z.mean(0) - np.tanh(h))
        if cond.startswith('two_layer'):  # wake, top layer: infer g from the code, delta rule on V; sleep, top recognizer
            Gq = np.where(r.random((B, H)) < 1 / (1 + np.exp(-2 * (Z @ R2.T + b2))), 1.0, -1.0); Ez = Z - np.tanh(Gq @ V.T + a)
            V += lr * Ez.T @ Gq / B; a += lr * Ez.mean(0)
            Gd = np.where(r.random((B, H)) < 0.5, 1.0, -1.0); Zd = np.where(r.random((B, K)) < 1 / (1 + np.exp(-2 * (Gd @ V.T + a))), 1.0, -1.0)
            Eg = Gd - np.tanh(Zd @ R2.T + b2); R2 += lrs * Eg.T @ Zd / B; b2 += lrs * Eg.mean(0)
        L.sleep(own, r, lrs, B)  # sleep, bottom layer: dream z from the current prior, render, delta rule on the recognizer
    world_arrays = dict(A=world.A, Sigma=Sigma, Rw=Rw, B=world.params.get('B', np.zeros((K, 2))))
    return prior, w, cond, curve, {k: np.array(v) for k, v in snaps.items()}, world_arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--priors', default='chain,deep'); ap.add_argument('--conds', default=','.join(CONDS))
    ap.add_argument('--worlds', type=int, default=8); ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--K', type=int, default=6); ap.add_argument('--D', type=int, default=12); ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--N', type=int, default=256, help='stored foreign samples; 0 = a fresh batch every step'); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.02); ap.add_argument('--sleep_lr_scale', type=float, default=10.0); ap.add_argument('--H', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=24); ap.add_argument('--tag', default='')
    args = vars(ap.parse_args())
    out = OUT / f"run_N{args['N']}{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(p, w, c, args) for p in args['priors'].split(',') for w in range(args['worlds']) for c in args['conds'].split(',')]
    curves, arrays = {}, {}
    with ProcessPoolExecutor(args['workers']) as ex:
        for prior, w, cond, curve, snaps, world in tqdm(ex.map(run, tasks), total=len(tasks), desc='runs'):
            curves.setdefault(prior, {}).setdefault(cond, []).append(curve)
            arrays |= {f'{prior}/{cond}/{w}/{k}': v for k, v in snaps.items()} | {f'{prior}/world/{w}/{k}': v for k, v in world.items()}
    np.savez_compressed(out / 'snapshots.npz', **arrays)
    json.dump(dict(args=args, curves=curves), open(out / 'curves.json', 'w'))
    keys = ['kl', 'gen_world', 'rec_lag', 'rec_world', 'ideal_world', 'acc_own', 'acc_matched', 'corr_err', 'top_lag', 'top_acc_own', 'top_world']
    for prior, block in curves.items():
        for cond, runs in block.items():
            print(f'{prior:6s} {cond:13s} ' + '  '.join(f'{k} {np.mean([c[0][k] for c in runs]):.3g}->{np.mean([c[-1][k] for c in runs]):.3g}' for k in keys if k in runs[0][0]))
    print(out)


if __name__ == '__main__': main()
