"""Model 5, practice against wake and sleep in the smallest world with an agreement (model5_cut_check.py: codes k1 k2 | k3 k4, a two-valued feature of the class of k2 equals one of the class of k3).
README, Baselines (17 Nov, relaxed): in some transient window of training, own samples used for Practice have to be more sample-efficient than the same amount of wake and/or sleep in the same learner.
Learner (separate weights, small random starts):
  from the stored teacher strings only, as in model5_cut_check.py: block net and cut net (local rule); the judge of the agreement p_E(k2, k3) (teacher pairs at the cut spread over the cut net's
     hidden-layer neighbours); the writer's block part P_B(k, k') (teacher blocks counted in the block units, smoothing by the held-out strings). No arm changes these.
  the writer's own unit u (U values) for the agreement adds to the top-down input of the two codes at the cut:
     p(k2 | u) = softmax(log m_R + D1[u]),  p(k3 | u) = softmax(log m_L + D2[u]),  p(u) = softmax(PI),  m_R, m_L: what the block part writes at these places;  k1 ~ P_B(. | k2),  k4 ~ P_B(. | k3)
  the reader's naming layer for u: q(u | k2, k3) = softmax(GAIN (W2_u . x_R[k2] + W3_u . x_L[k3]) + c_u) on the cut net's hidden vectors x
Rules (delta rules, one step per sample):
  wake      a teacher string (drawn from the stored set; --fresh: a new one from the teacher, with reader, judge and block part unchanged), u drawn from the naming layer:  D1[u] += eta_w (e_k2 - p(. | u)),  D2[u] likewise,  PI += eta_w (e_u - p(u))
  sleep     an own sample with the u that wrote it: the naming layer moves towards u (step eta_s)
  practice  an own sample (u, k2, k3): delta = log p(k2, k3) - log p_E(k2, k3), cut at +-CLIP; a refused pair counts as FLOOR x the mean positive answer and is only ever moved down.
            The wake step at u with factor d = -(delta - running mean) and step eta_p. Averaged over own samples this is the gradient of KL(p_writer(k2, k3) || p_E).
Arms (one round = one batch of 64 of every kind of sample the arm uses; samples used = teacher strings read + samples written, counted alike):
  W         wake only                                          WS       wake, sleep on own samples (classic wake-sleep)
  WSP       wake, own samples used for sleep and for practice   WP       wake, practice, no sleep (the naming layer stays as it started)
  SP        no wake: own samples for sleep and practice        Wtied    wake only, u drawn from the writer's own posterior (no separate reader; reference)
  WSPread   wake, sleep on own samples, practice on further own samples credited to the unit the naming layer reads
  WSPtwin   the same with practice on the samples of a twin (same teacher strings, other random starts of every part, running WSP itself); its own curve is stored as arm 'twin'
  WS31, WS13, WSP31, WSP13   WS and WSP with three wake batches and one own batch per round, and with one and three
Scores (exact): loss = KL(p*(k2, k3) || p(k2, k3)), the part of the string's population loss that the arms can change (the rest is the block part); agree = share of written pairs that agree;
  reading = information of the read u about the feature of a teacher pair, in units of log 2 (0: none, 1: all).
Steps: --etas wake,sleep,practice,prior; prior 1: p(u) learned with the wake and practice steps, 0: p(u) fixed uniform. --run scan: wake-sleep with and without practice over a grid of steps (--gw, --gs, --gp) and priors (--priors).
Writes OUT/cut_practice[_scan][_tag].npz.   Usage: python model5_cut_practice.py [--run grid|scan] [--worlds 16] [--procs 16] [--ns 1024] [--U 2] [--etas 0.2,0.05,0.2,0] [--etas2 0.1,0.05,0.2,0] [--fresh] [--rounds 2048] [--tag x]
--run grid: all arms at --etas, and W, WS, WSP, Wtied at --etas2."""

import argparse
import os
from multiprocessing import Pool
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm

from model5_cut_check import RHO, Cut, units
from model5_lateral_verdict import OUT, draw
from model5_network_practice import D, bandwidth, evidence, local_net
from model5_network_units import normed
from model5_own_benefit import ALPHAS

INIT, BATCH, GAIN, FLOOR, CLIP = 0.1, 64, 4.0, 1e-3, 5.0
ARMS = ('W', 'WS', 'WSP', 'WP', 'SP', 'Wtied', 'WSPread', 'WSPtwin', 'WS31', 'WS13', 'WSP31', 'WSP13', 'twin'); KEYS = ('loss', 'agree', 'reading')
SPLIT = dict(WS31=('WS', 3, 1), WS13=('WS', 1, 3), WSP31=('WSP', 3, 1), WSP13=('WSP', 1, 3)); USES = dict(W=1, WS=2, WSP=2, WP=2, SP=1, Wtied=1, WSPread=3, WSPtwin=3, WS31=4, WS13=4, WSP31=4, WSP13=4, twin=2)   # SPLIT: arm, wake batches and own batches per round; USES: batches of samples per round


def checks(rounds): return np.concatenate([[0], np.unique(np.round(2 ** np.arange(0, np.log2(rounds) + 0.01, 0.5)).astype(int))])


def reader(c, S, val, seed):                                                                                                       # everything that comes from the teacher strings alone
    w = c.w; cw = SimpleNamespace(N=w.N, v=RHO, valid=c.agree, pstar=c.agree / c.agree.sum()); left, right = np.concatenate([S[:, 0], S[:, 2]]), np.concatenate([S[:, 1], S[:, 3]])
    netB, _, _, QB, QvB, _ = local_net(w, left, right, np.concatenate([val, val]), np.random.default_rng(seed)); netC, _, _, QC, QvC, _ = local_net(cw, S[:, 1], S[:, 2], val, np.random.default_rng(seed + 1))
    uL, uR = (units(normed(X), shown, bandwidth(netB, QB, QvB)) for X, shown in ((netB.P[0], left), (netB.P[1], right))); size = np.outer(np.bincount(uL)[uL], np.bincount(uR)[uR])
    def table(M, al): C = np.zeros((uL.max() + 1, uR.max() + 1)); np.add.at(C, (uL[:, None], uR[None, :]), M); return ((C + al) / (C + al).sum())[np.ix_(uL, uR)] / size
    al = min(ALPHAS, key=lambda a: -(QvB * np.log(table(QB, a))).sum())                                                           # smoothing of the block part by the held-out blocks
    return SimpleNamespace(pE=evidence(netC, QC + QvC, bandwidth(netC, QC, QvC)), PB=table(QB + QvB, al), XR=normed(netC.P[0]), XL=normed(netC.P[1]))


class Unit:                                                                                                                        # the writer's own unit for the agreement and the reader's naming layer for it
    def __init__(self, R, U, rng, eta):
        N = len(R.PB); self.R, self.U, self.eta = R, U, eta; self.lR, self.lL = np.log(R.PB.sum(0)), np.log(R.PB.sum(1)); self.b, self.ebar = 0.0, None
        self.D1, self.D2, self.PI = INIT * rng.standard_normal((U, N)), INIT * rng.standard_normal((U, N)), np.zeros(U); self.W2, self.W3, self.c = INIT * rng.standard_normal((U, D)), INIT * rng.standard_normal((U, D)), np.zeros(U)

    def tables(self): return softmax(self.PI), softmax(self.lR + self.D1, 1), softmax(self.lL + self.D2, 1)

    def pair(self, tab=None): pi, p1, p2 = tab or self.tables(); return np.einsum('u,ui,uj->ij', pi, p1, p2)

    def sample(self, rng, tab): pi, p1, p2 = tab; u = rng.choice(self.U, BATCH, p=pi); return u, draw(p1[u], rng), draw(p2[u], rng)

    def name(self, k2, k3): return softmax(GAIN * (self.R.XR[k2] @ self.W2.T + self.R.XL[k3] @ self.W3.T) + self.c, 1)

    def posterior(self, k2, k3, tab): pi, p1, p2 = tab; q = pi * p1[:, k2].T * p2[:, k3].T; return q / q.sum(1, keepdims=True)

    def step(self, u, k2, k3, d, tab, eta):                                                                                        # three-factor step at the unit u: d x (what was written - what the unit writes on average)
        pi, p1, p2 = tab; n = np.bincount(u, d, self.U); np.add.at(self.D1, (u, k2), eta * d); self.D1 -= eta * n[:, None] * p1; np.add.at(self.D2, (u, k3), eta * d); self.D2 -= eta * n[:, None] * p2; self.PI += self.eta[3] * eta * (n - d.sum() * pi)

    def wake(self, k2, k3, rng, tied=False): tab = self.tables(); self.step(draw(self.posterior(k2, k3, tab) if tied else self.name(k2, k3), rng), k2, k3, np.ones(len(k2)), tab, self.eta[0])

    def sleep(self, u, k2, k3): g = -self.name(k2, k3); g[np.arange(len(u)), u] += 1; self.W2 += self.eta[1] * g.T @ self.R.XR[k2]; self.W3 += self.eta[1] * g.T @ self.R.XL[k3]; self.c += self.eta[1] * g.sum(0)

    def practice(self, u, k2, k3, tab):
        e = self.R.pE[k2, k3]; pos = e > 0; self.ebar = (float(e[pos].mean()) if self.ebar is None else 0.95 * self.ebar + 0.05 * float(e[pos].mean())) if pos.any() else (self.ebar or 1 / e.size)
        delta = np.log(self.pair(tab)[k2, k3]) - np.log(np.where(pos, e, FLOOR * self.ebar)); delta[~pos] = np.maximum(delta[~pos], 0); delta = np.clip(delta, -CLIP, CLIP)
        self.step(u, k2, k3, -(delta - self.b), tab, self.eta[2]); self.b += 0.05 * (float(delta.mean()) - self.b)

    def score(self, c, tied=False):
        tab = self.tables(); P = self.pair(tab); ps = 1 / c.agree.sum(); k2, k3 = np.nonzero(c.agree); q = self.posterior(k2, k3, tab) if tied else self.name(k2, k3)
        J = np.stack([q[c.fR[k2] == f].sum(0) for f in range(RHO)]) / len(k2); ind = J.sum(1, keepdims=True) * J.sum(0)                 # J[feature, read u]
        return float((ps * (np.log(ps) - np.log(P[c.agree] + 1e-300))).sum()), float(P[c.agree].sum()), float((J * (np.log(J + 1e-300) - np.log(ind + 1e-300))).sum() / np.log(RHO))


def run(c, R, RT, S, arm, seed, eta, U, rounds, fresh):
    rng, rT = np.random.default_rng(seed), np.random.default_rng(seed + 1); batch = lambda r: c.teacher(r, BATCH) if fresh else S[r.integers(len(S), size=BATCH)]; L = Unit(R, U, rng, eta); T = Unit(RT, U, rT, eta) if arm == 'WSPtwin' else None; cs = set(checks(rounds)); out, outT = [], []
    arm, nw, no = SPLIT.get(arm, (arm, 1, 1))
    for t in range(rounds + 1):
        if t in cs: out.append(L.score(c, arm == 'Wtied')); outT.append(T.score(c) if T else out[-1])
        for _ in range(nw if arm != 'SP' else 0): x = batch(rng); L.wake(x[:, 1], x[:, 2], rng, arm == 'Wtied')
        for _ in range(no if arm not in ('W', 'Wtied') else 0):
            tab = L.tables(); u, k2, k3 = L.sample(rng, tab)
            if arm != 'WP': L.sleep(u, k2, k3)
            if arm in ('WSP', 'WP', 'SP'): L.practice(u, k2, k3, tab)
        if arm == 'WSPread': tab = L.tables(); _, k2, k3 = L.sample(rng, tab); L.practice(draw(L.name(k2, k3), rng), k2, k3, tab)
        if arm == 'WSPtwin':
            x = batch(rT); T.wake(x[:, 1], x[:, 2], rT); tabT = T.tables(); uT, a, b = T.sample(rT, tabT); T.sleep(uT, a, b); T.practice(uT, a, b, tabT)
            _, k2, k3 = T.sample(rT, T.tables()); tab = L.tables(); L.practice(draw(L.name(k2, k3), rng), k2, k3, tab)
    return np.array(out), np.array(outT)


def one(job):
    n, seed, U, rounds, confs, fresh = job; c = Cut(seed); rng = np.random.default_rng(110_000 + seed); S = c.teacher(rng, n); val = rng.random(n) < 0.15; R = reader(c, S, val, 120_000 + seed)
    RT = reader(c, S, val, 140_000 + seed) if any('WSPtwin' in arms for _, arms in confs) else None; res = np.full((len(confs), len(ARMS), len(checks(rounds)), len(KEYS)), np.nan)
    for i, (eta, arms) in enumerate(confs):
        for arm in arms:
            res[i, ARMS.index(arm)], t = run(c, R, RT, S, arm, 150_000 + seed, eta, U, rounds, fresh)
            if arm == 'WSPtwin': res[i, ARMS.index('twin')] = t
    P0 = np.outer(R.PB.sum(0), R.PB.sum(1)); ps = 1 / c.agree.sum()
    return job, res, [float(R.pE[c.agree].sum()), float((R.pE > 0)[~c.agree].mean()), float(P0[c.agree].sum()), float((ps * (np.log(ps) - np.log(0.999 * R.pE[c.agree] + 0.001 / R.pE.size))).sum())]   # judge: mass on agreeing pairs, share of the other pairs it accepts; block part alone: share that agrees; the judge's own loss on the pair


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='grid'); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--ns', default='1024'); ap.add_argument('--U', type=int, default=2)
    ap.add_argument('--etas', default='0.2,0.05,0.2,0'); ap.add_argument('--etas2', default='0.1,0.05,0.2,0'); ap.add_argument('--fresh', action='store_true'); ap.add_argument('--rounds', type=int, default=2048); ap.add_argument('--gw', default='0.0125,0.05,0.2'); ap.add_argument('--gs', default='0.003,0.0125,0.05,0.2'); ap.add_argument('--gp', default='0.0125,0.05,0.2'); ap.add_argument('--priors', default='1,0'); ap.add_argument('--tag', default=''); args = ap.parse_args(); ns = [int(x) for x in args.ns.split(',')]; eta0, eta1 = (tuple(float(x) for x in e.split(',')) for e in (args.etas, args.etas2))
    GW, GS, GP, FX = (tuple(float(x) for x in getattr(args, k).split(',')) for k in ('gw', 'gs', 'gp', 'priors'))
    confs = [(eta0, ARMS[:-1]), (eta1, ('W', 'WS', 'WSP', 'Wtied'))] if args.run == 'grid' else [((a, b, pp, f), ('WS', 'WSP', 'Wtied') if (b, pp) == (GS[0], GP[0]) else ('WS', 'WSP') if pp == GP[0] else ('WSP',)) for f in FX for a in GW for b in GS for pp in GP]
    jobs = [(n, s, args.U, args.rounds, confs, args.fresh) for n in ns for s in range(args.worlds)]; res, aux = np.zeros((len(ns), args.worlds, len(confs), len(ARMS), len(checks(args.rounds)), len(KEYS))), np.zeros((len(ns), args.worlds, 4))
    with Pool(args.procs) as pool:
        for (n, s, *_), r, a in tqdm(pool.imap_unordered(one, jobs), total=len(jobs)): res[ns.index(n), s], aux[ns.index(n), s] = r, a
    name = f"{OUT}/cut_practice{'_scan' if args.run == 'scan' else ''}{'_' + args.tag if args.tag else ''}.npz"
    np.savez(name, res=res, aux=aux, ns=np.array(ns), etas=np.array([e for e, _ in confs]), checks=checks(args.rounds), arms=ARMS, keys=KEYS, uses=np.array([USES[a] for a in ARMS]), U=args.U, fresh=args.fresh); print(name)
