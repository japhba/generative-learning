"""Why the autoencoder baseline of toy model 2 plateaus, whether regularising it closes the gap to wake-sleep, and whether
the phenomenology needs the world's correlated prior.

Same setting as practice_ws_budget.py (two-layer learner, K = 6 causes, H = 4 top units, D = 12, sigma = 0.5, rotated start,
fresh supplied examples each used once, batches of 32). A learner is given as 'kind:key=value:...':

  ws:x=0.7          wake-sleep, 70% supplied batches (wake) interleaved with 30% dreamed ones (sleep); x=1 never dreams
  ae:top=mf:s=0.05  the autoencoder baseline of practice_ws_budget.py: bottom layer a binary autoencoder with the expected
                    squared error under its factorial code (practice_disentangle.ae_step 'bae'); top layer a deterministic
                    autoencoder of the sampled codes z (mean-field top code tanh(R2 z + b2), log-loss of z). s scales the
                    recognizers' learning rate (0.05 in the baseline, inherited from the first toy).
  ae:top=bin        top layer with the expected log-loss under its factorial top code q(g|z), summed exactly over the 2^H top
                    configurations: the top-layer analogue of the bottom's binary autoencoder (stochastic binary code)
  vae               a two-layer variational autoencoder: the same network trained end to end on the evidence lower bound,
                    E_q[log p(x|z)] + E_q[log p(z|g) + log p(g)] + H[q(z|x)] + H[q(g|z)], all expectations summed exactly
                    (bottom: practice_disentangle.ae_step 'vae' with log p(z) replaced by the top layer's bound for each z)
  wd=...            L2 weight decay on W, R, V, R2 (any kind)

--N caps the data: supplied batches are then drawn with replacement from a fixed set of N world samples.

Checkpoints (2 per octave of examples) record the population KL(world || model) in nats, the angle between the generator's
axes and the world's (gen_world, as in practice_ws_layers.py), the top layer's angle to the world's top causes
(top_world, two-layer world only), the size of the top weights, how saturated the codes are, and (from 2^10 examples on) the
KL with the learner's decoder kept and its prior replaced by the maximum-likelihood prior over the 64 latent
configurations (kl_refit: how much of the error is the decoder's). At the end, every learner's parameters are also scored
on the autoencoder's own objective (ae_loss_bottom: expected squared error per sample; ae_loss_top: log-loss of the sampled
codes under the deterministic top autoencoder), and the final weights are saved.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
from itertools import product
import json
import math
import zlib
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_disentangle import Learner, World, ae_step, initial_learner, mixture_logpdf, posterior
from practice_disentangle_prior import top_layer
from practice_ws_layers import match, unit

OUT = Path('/var/tmp/jbauer/generative_practice_disentangle/ae_floor')


def bern(U, r): return np.where(r.random(U.shape) < 1 / (1 + np.exp(-2 * U)), 1.0, -1.0)  # p(+1) = sigmoid(2 U)


def parse(spec):
    kind, *kv = spec.split(':'); d = dict(kind=kind, x=1.0, top='mf', s=0.05, wd=0.0)
    for item in kv: k, v = item.split('='); d[k] = v if k == 'top' else float(v)
    return d


def top_terms(Z, V, a, R2, b2, Gall):
    """Top-layer pieces for codes Z (n, K): mean top code m, q(g|z) over all top configurations, log p(z|g)."""
    m = np.clip(np.tanh(Z @ R2.T + b2), -1 + 1e-7, 1 - 1e-7); logf = np.log((1 + Gall[None] * m[:, None, :]) / 2)  # (n, 2^H, H)
    U = Gall @ V.T + a; ll = -np.logaddexp(0, -2 * U[None] * Z[:, None, :]).sum(2)  # (n, 2^H)
    return m, logf, np.exp(logf.sum(2)), U, ll


def top_elbo(Z, V, a, R2, b2, Gall, entropy):
    """E_q(g|z)[log p(z|g)] (+ H[q(g|z)] + log p(g) if entropy): the top layer's bound on log p(z), per row of Z."""
    m, logf, q, U, ll = top_terms(Z, V, a, R2, b2, Gall); out = (q * ll).sum(1)
    if entropy: out = out - (q * logf.sum(2)).sum(1) - Gall.shape[1] * np.log(2)
    return out


def top_step(Z, V, a, R2, b2, Gall, lr, lr_r, entropy):
    """Gradient ascent on the mean of top_elbo over the codes Z, exact over the top configurations."""
    m, logf, q, U, ll = top_terms(Z, V, a, R2, b2, Gall); n = len(Z)
    Ez = Z[:, None, :] - np.tanh(U)[None]  # (n, 2^H, K): d log p(z|g) / dU
    dV = np.einsum('ng,ngk,gh->kh', q, Ez, Gall) / n; da = np.einsum('ng,ngk->k', q, Ez) / n
    dq = 0.5 * Gall[None] * np.exp(logf.sum(2, keepdims=True) - logf)  # dq(g)/dm_j, (n, 2^H, H)
    gm = np.einsum('ngh,ng->nh', dq, ll) - (np.arctanh(m) if entropy else 0)
    ga = gm * (1 - m ** 2)
    return V + lr * dV, a + lr * da, R2 + lr_r * ga.T @ Z / n, b2 + lr_r * ga.mean(0)


def ae_losses(L, V, a, R2, b2, X, r):
    """The baseline autoencoder's objectives at any parameters: expected squared error of x under the factorial code (per
    sample) and log-loss of a sampled code z under the deterministic top autoencoder (nats per sample)."""
    m = np.tanh(L.logits(X)); E = X - m @ L.W.T - L.c; bottom = float(((E ** 2).sum(1) + ((1 - m ** 2) * (L.W ** 2).sum(0)).sum(1)).mean())
    Z = L.sample_q(X, r); m2 = np.tanh(Z @ R2.T + b2); top = float(np.logaddexp(0, -2 * (m2 @ V.T + a) * Z).sum(1).mean())
    return bottom, top


def em_prior(world, X, W, c, iters=100):
    """Maximum-likelihood prior over the latent configurations for a fixed decoder (EM on X)."""
    w = copy.copy(world); p = np.full(len(world.Z), 1 / len(world.Z))
    for _ in range(iters): w.logp = np.log(p + 1e-300); p = posterior(X, W, c, w).mean(0)
    w.logp = np.log(p + 1e-300); w.p = p; return w


def run(task):
    prior, w, spec, args = task
    cfg = parse(spec); kind, x = cfg['kind'], cfg['x']; K, D, B, H, sig = args['K'], args['D'], args['batch'], args['H'], args['sigma']
    lr, lrs, lr_r, wd = args['lr'], args['lr'] * args['sleep_lr_scale'], args['lr'] * cfg['s'], cfg['wd']
    rng = np.random.default_rng([args['seed'], w]); world = World(K, D, sig, prior, rng)  # the same worlds as practice_ws_budget.py
    Xt, Zt = world.sample_x(4000, rng); true_lp = mixture_logpdf(Xt, world.A, np.zeros(D), world)
    L0 = initial_learner(world, rng, 20000, 0.02, B, 'rotated')
    Xfit = world.sample_x(8000, np.random.default_rng([args['seed'], w, 11]))[0]
    if args['N']: Xpool = world.sample_x(args['N'], np.random.default_rng([args['seed'], w, 7]))[0]  # a fixed set of N supplied examples, reused
    V = 0.1 * np.random.default_rng([args['seed'], w, 99]).standard_normal((K, H)); a = np.zeros(K)
    R2, b2 = np.zeros((H, K)), np.zeros(H); L = Learner(L0.W, L0.c, L0.R, L0.b)
    r = np.random.default_rng([args['seed'], w, zlib.crc32(spec.encode())]); Gall = np.array(list(product([-1.0, 1.0], repeat=H)))
    nb = args['P_max'] // B; cps = sorted({0} | {int(round(2 ** (k / 2))) // B for k in range(2 * int(math.log2(B)), 2 * int(math.log2(args['P_max'])) + 1)})
    curve = []
    for t in range(nb + 1):
        if t in cps:
            tw = top_layer(world, V, a); kl = float(np.mean(true_lp - mixture_logpdf(Xt, L.W, L.c, tw)))
            C = unit(L.W).T @ world.A; p, s = match(C)
            e = dict(P=t * B, kl=kl, gen_world=float(np.degrees(np.arccos(np.minimum(np.abs(C[np.arange(K), p]), 1))).mean()),
                     v_norm=float(np.linalg.norm(V)), sat=float(np.abs(np.tanh(L.logits(Xt))).mean()), sat_top=float(np.abs(np.tanh(L.sample_q(Xt, r) @ R2.T + b2)).mean()))
            if prior == 'deep':
                Vw = np.zeros_like(V); Vw[p] = V * s[:, None]; Ct = unit(world.params['B']).T @ unit(Vw); q, _ = match(Ct)
                e['top_world'] = float(np.degrees(np.arccos(np.minimum(np.abs(Ct[np.arange(len(Ct)), q]), 1))).mean())
            if t * B >= 2 ** 10 and math.log2(t * B).is_integer():
                e['kl_refit'] = float(np.mean(true_lp - mixture_logpdf(Xt, L.W, L.c, em_prior(world, Xfit, L.W, L.c))))
            curve.append(e)
        if t == nb: break
        if math.ceil((t + 1) * x) > math.ceil(t * x):  # a supplied batch
            X = Xpool[r.integers(0, args['N'], B)] if args['N'] else world.sample_x(B, r)[0]; Z = L.sample_q(X, r)
            if kind == 'ws':  # wake: delta rules on the generators from recognised codes
                Gq = bern(Z @ R2.T + b2, r); E = X - Z @ L.W.T - L.c; L.W += lr * E.T @ Z / B; L.c += lr * E.mean(0)
                Ez2 = Z - np.tanh(Gq @ V.T + a); V += lr * Ez2.T @ Gq / B; a += lr * Ez2.mean(0)
            elif kind == 'ae' and cfg['top'] == 'mf':  # the baseline, exactly as in practice_ws_budget.py
                ae_step(L, X, world, lr, cfg['s'], 'bae')
                m2 = np.tanh(Z @ R2.T + b2); E2 = Z - np.tanh(m2 @ V.T + a); g2 = (E2 @ V) * (1 - m2 ** 2)
                V += lr * E2.T @ m2 / B; a += lr * E2.mean(0); R2 += lr_r * g2.T @ Z / B; b2 += lr_r * g2.mean(0)
            elif kind == 'ae':  # binary top code
                ae_step(L, X, world, lr, cfg['s'], 'bae'); V, a, R2, b2 = top_step(Z, V, a, R2, b2, Gall, lr, lr_r, False)
            elif kind == 'vae':  # bottom: ELBO with the top layer's bound as log p(z); top: its ELBO on the sampled codes
                wt = copy.copy(world); wt.logp = top_elbo(world.Z, V, a, R2, b2, Gall, True)
                ae_step(L, X, wt, lr, cfg['s'], 'vae'); V, a, R2, b2 = top_step(Z, V, a, R2, b2, Gall, lr, lr_r, True)
            if wd: L.W -= lr * wd * L.W; L.R -= lr * wd * L.R; V -= lr * wd * V; R2 -= lr * wd * R2
        else:  # a dreamed batch trains both recognizers (sleep)
            Gd = np.where(r.random((B, H)) < 0.5, 1.0, -1.0); Zd = bern(Gd @ V.T + a, r); Xd = Zd @ L.W.T + L.c + sig * r.standard_normal((B, D))
            G1 = Zd - np.tanh(L.logits(Xd)); L.R += lrs * G1.T @ Xd / B; L.b += lrs * G1.mean(0)
            G2 = Gd - np.tanh(Zd @ R2.T + b2); R2 += lrs * G2.T @ Zd / B; b2 += lrs * G2.mean(0)
    lb, lt = ae_losses(L, V, a, R2, b2, Xt, np.random.default_rng(5)); curve[-1] |= dict(ae_loss_bottom=lb, ae_loss_top=lt)
    return prior, w, spec, curve, dict(W=L.W, c=L.c, R=L.R, b=L.b, V=V, a=a, R2=R2, b2=b2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--priors', default='deep'); ap.add_argument('--learners', default='ws:x=0.7,ws:x=1,ae:top=mf:s=0.05')
    ap.add_argument('--worlds', type=int, default=12); ap.add_argument('--P_max', type=int, default=2 ** 21)
    ap.add_argument('--K', type=int, default=6); ap.add_argument('--D', type=int, default=12); ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--H', type=int, default=4); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.02); ap.add_argument('--sleep_lr_scale', type=float, default=10.0)
    ap.add_argument('--N', type=int, default=0, help='distinct supplied examples, reused; 0 = fresh world samples, each used once')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=28); ap.add_argument('--tag', default='')
    args = vars(ap.parse_args())
    out = OUT / f"run_N{args['N']}{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(p, w, l, args) for p in args['priors'].split(',') for l in args['learners'].split(',') for w in range(args['worlds'])]
    tasks.sort(key=lambda t: t[2].startswith('ws'))  # the slow (enumerating) learners first
    curves, weights = {}, {}
    with ProcessPoolExecutor(args['workers']) as ex:
        for prior, w, spec, curve, wts in tqdm(ex.map(run, tasks), total=len(tasks), desc='runs'):
            curves.setdefault(prior, {}).setdefault(spec, {})[w] = curve; weights |= {f'{prior}/{spec}/{w}/{k}': v for k, v in wts.items()}
    json.dump(dict(args=args, curves=curves), open(out / 'curves.json', 'w')); np.savez_compressed(out / 'weights.npz', **weights)
    keys = ['kl', 'kl_refit', 'gen_world', 'top_world', 'v_norm', 'sat', 'sat_top', 'ae_loss_bottom', 'ae_loss_top']
    for prior, block in curves.items():
        for spec, runs in block.items():
            fin = [runs[w][-1] for w in sorted(runs)]
            print(f'{prior:9s} {spec:28s} ' + '  '.join(f'{k} {np.mean([f[k] for f in fin]):.3f}' for k in keys if k in fin[0]))
    print(out)


if __name__ == '__main__': main()
