"""Model 5, two levels of the writer's own units: does binding go bottom-up without being told to, and is practice's lead over wake and sleep larger at the upper level?
World: four blocks of model5_lateral_verdict.py side by side, codes k1 k2 | k3 k4 | k5 k6 | k7 k8, all from one grammar.
  level 1   at the cut (k2, k3) a four-valued feature of the two classes has to agree; its value is a. The same at the cut (k6, k7), value b.
  level 2   a rule on the pair of values (a, b). --top table (default): b = a or b = a + 1 (mod 4), a table with two partners per value and no feature of a single value behind it, so the rule
            cannot be read off single codes. --top feature: a two-valued feature g of the two values has to agree, g(a) = g(b); here every code carries g.     Teacher: uniform over the valid strings.
Learner (separate weights, small random starts):
  from the stored teacher strings only: block net and cut net (local rule, model5_network_practice.local_net; sibling pairs and cut pairs pooled over their places); the judge of a cut
     p_E(k2, k3) (teacher pairs spread over the cut net's hidden-layer neighbours); the reader's class of a cut pair (units recruited from the cut net's hidden vectors of the right-hand
     code); the judge of level 2, r_E[class, class] = how much more often two classes are met at the two cuts than by chance (a count table); the writer's block part. No arm changes these.
  the writer's own units:  z (4 values for the table, 2 for the feature; uniform)  ->  u_A ~ T_A[z], u_B ~ T_B[z] (U values each, T = softmax(TH[z]))  ->  the codes at their cut,
     p(k2 | u) = softmax(log m_R + D1[u]),  p(k3 | u) = softmax(log m_L + D2[u])     (the same weights at both cuts);  the other four codes follow from the block part.
  the reader's naming layers: level 1, q(u | k2, k3) on the cut net's hidden vectors; level 2, q(z | u_A, u_B) = softmax(N_A[u_A] + N_B[u_B] + c) on the units named at level 1.
Rules (delta rules, one step per sample):
  wake      a teacher string, u_A, u_B drawn from the naming layer of level 1 and z from that of level 2:  D[u] += eta_w (e_k - p(. | u)) at both cuts,  TH_A[z] += eta_w (e_uA - T_A[z]), TH_B likewise
  sleep     an own sample with the units that wrote it: both naming layers move towards them (level 2 from the true u_A, u_B)
  practice  an own sample, one verdict per level, each credited to the unit directly above what it judges:
     level 1, at each cut:  delta = log p(k2, k3) - log p_E(k2, k3)                                                    -> the wake step of D at u with factor -(delta - mean)
     level 2, only if the judge accepts both cuts:  delta = log p(x_A, x_B) - log [p_E(x_A) p_E(x_B) r_E]               -> the wake step of TH at z with factor -(delta - mean)
     (cut at +-CLIP; what the judge refuses counts as FLOOR x its mean positive answer and is only ever moved down)
Arms, steps and counting of samples as in model5_cut_practice.py (a sample is one string, teacher or own); credit 'read': the units are those the naming layers read, level by level.
Scores (exact): loss = KL(p*(x_A, x_B) || p(x_A, x_B)), the part of the string's population loss the arms can change;  valid1 = share of written strings whose two cuts both agree;
  valid2 = of those, the share that also obeys level 2 (0.5: not at all);  read1 = information of the named u about a, in units of log 4;  read2 = information of the named z about the pair (a, b), in units of log(number of values of z).
Writes OUT/cut_depth_<top>[_scan][_tag].npz.   Usage: python model5_cut_depth.py [--run grid|scan] [--worlds 16] [--procs 16] [--ns 4096] [--U 4] [--etas 0.1,0.05,0.1] [--top table|feature] [--fresh] [--rounds 4096] [--tag x]"""

import argparse
import os
from multiprocessing import Pool
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm

from model5_cut_check import units
from model5_cut_practice import checks
from model5_lateral_verdict import OUT, World, draw
from model5_network_practice import D, bandwidth, evidence, local_net
from model5_network_units import normed
from model5_own_benefit import ALPHAS

R1, R2, ZS, INIT, BATCH, GAIN, FLOOR, CLIP = 4, 2, dict(table=4, feature=2), 0.1, 64, 4.0, 1e-3, 5.0
ARMS = ('W', 'WS', 'WSP', 'WP', 'Wtied', 'WSPread', 'WSPtwin', 'WS31', 'WS13', 'WSP31', 'WSP13', 'twin'); KEYS = ('loss', 'valid1', 'valid2', 'read1', 'read2')
SPLIT = dict(WS31=('WS', 3, 1), WS13=('WS', 1, 3), WSP31=('WSP', 3, 1), WSP13=('WSP', 1, 3)); USES = dict(W=1, WS=2, WSP=2, WP=2, Wtied=1, WSPread=3, WSPtwin=3, WS31=4, WS13=4, WSP31=4, WSP13=4, twin=2)


class Deep:
    def __init__(self, seed, top='table'):
        self.w = w = World(seed=seed); rng = np.random.default_rng(2000 + seed); f = np.repeat(np.arange(R1), w.v // R1); self.fR, self.fL = rng.permutation(f)[w.cls], rng.permutation(f)[w.cls]   # the value of a code as k2, as k3
        self.g = rng.permutation(np.repeat(np.arange(R2), R1 // R2)); self.agree = self.fR[:, None] == self.fL[None, :]; self.pairs = np.argwhere(self.agree); self.a = self.fR[self.pairs[:, 0]]   # valid cut pairs and their values
        V = np.arange(R1); self.ok = np.isin((V[None, :] - V[:, None]) % R1, (0, 1)) if top == 'table' else self.g[:, None] == self.g[None, :]; self.ab = np.argwhere(self.ok); self.Z = ZS[top]            # ok[a, b]: the valid pairs of values
        self.ok2 = self.ok[np.ix_(self.a, self.a)]; self.bya = np.stack([np.flatnonzero(self.a == v) for v in range(R1)])
        self.Lp, self.Rp = np.stack([np.flatnonzero(w.valid[:, k]) for k in range(w.N)]), np.stack([np.flatnonzero(w.valid[k]) for k in range(w.N)])                                               # the valid partners of a code

    def cuts(self, rng, n):                                                                                                        # (k2, k3, k6, k7) of n teacher strings
        ab = self.ab[rng.integers(len(self.ab), size=n)]; ia, ib = (self.bya[v, rng.integers(self.bya.shape[1], size=n)] for v in ab.T); return np.concatenate([self.pairs[ia], self.pairs[ib]], 1)

    def teacher(self, rng, n):                                                                                                     # whole strings: every code has as many valid partners, so this is uniform over the valid strings
        x = self.cuts(rng, n); pick = lambda P, k: P[k, rng.integers(P.shape[1], size=n)]; return np.stack([pick(self.Lp, x[:, 0]), x[:, 0], x[:, 1], pick(self.Rp, x[:, 1]), pick(self.Lp, x[:, 2]), x[:, 2], x[:, 3], pick(self.Rp, x[:, 3])], 1)


def reader(c, S, val, seed):                                                                                                       # everything that comes from the teacher strings alone
    w = c.w; cw = SimpleNamespace(N=w.N, v=R1, valid=c.agree, pstar=c.agree / c.agree.sum()); left, right, k2, k3 = S[:, 0::2].T.ravel(), S[:, 1::2].T.ravel(), S[:, [1, 5]].T.ravel(), S[:, [2, 6]].T.ravel()
    netB, _, _, QB, QvB, _ = local_net(w, left, right, np.tile(val, 4), np.random.default_rng(seed)); netC, _, _, QC, QvC, _ = local_net(cw, k2, k3, np.tile(val, 2), np.random.default_rng(seed + 1)); tauC = bandwidth(netC, QC, QvC)
    uL, uR = (units(normed(X), shown, bandwidth(netB, QB, QvB)) for X, shown in ((netB.P[0], left), (netB.P[1], right))); size = np.outer(np.bincount(uL)[uL], np.bincount(uR)[uR])
    def table(M, al): C = np.zeros((uL.max() + 1, uR.max() + 1)); np.add.at(C, (uL[:, None], uR[None, :]), M); return ((C + al) / (C + al).sum())[np.ix_(uL, uR)] / size
    al = min(ALPHAS, key=lambda a: -(QvB * np.log(table(QB, a))).sum()); cls1 = units(normed(netC.P[0]), k2, tauC); M2 = np.zeros((cls1.max() + 1,) * 2); np.add.at(M2, (cls1[S[:, 1]], cls1[S[:, 5]]), 1 / len(S))
    den = np.outer(M2.sum(1), M2.sum(0)); return SimpleNamespace(pE=evidence(netC, QC + QvC, tauC), PB=table(QB + QvB, al), XR=normed(netC.P[0]), XL=normed(netC.P[1]), cls1=cls1, r2=np.divide(M2, den, out=np.zeros_like(M2), where=den > 0))


class Stack:                                                                                                                       # the writer's own units z -> u_A, u_B -> codes at the cuts, and the reader's two naming layers
    def __init__(self, R, U, Z, rng, eta):
        N = len(R.PB); self.R, self.U, self.Z, self.eta = R, U, Z, eta; self.lR, self.lL = np.log(R.PB.sum(0)), np.log(R.PB.sum(1)); self.b, self.ebar, self.rbar, self.only = [0.0, 0.0], None, None, None; D = R.XR.shape[1]   # only: 'refused' or 'accepted' keeps the verdicts of one kind (model5_cut_mechanism.py)
        self.D1, self.D2, self.THA, self.THB = INIT * rng.standard_normal((U, N)), INIT * rng.standard_normal((U, N)), INIT * rng.standard_normal((Z, U)), INIT * rng.standard_normal((Z, U))
        self.W2, self.W3, self.c, self.NA, self.NB, self.c2 = INIT * rng.standard_normal((U, D)), INIT * rng.standard_normal((U, D)), np.zeros(U), INIT * rng.standard_normal((U, Z)), INIT * rng.standard_normal((U, Z)), np.zeros(Z)

    def tables(self): return softmax(self.THA, 1), softmax(self.THB, 1), softmax(self.lR + self.D1, 1), softmax(self.lL + self.D2, 1)

    def pairz(self, tab): TA, TB, p1, p2 = tab; return np.einsum('zu,ui,uj->zij', TA, p1, p2), np.einsum('zu,ui,uj->zij', TB, p1, p2)   # p(k2, k3 | z), p(k6, k7 | z)

    def sample(self, rng, tab): TA, TB, p1, p2 = tab; z = rng.integers(self.Z, size=BATCH); uA, uB = draw(TA[z], rng), draw(TB[z], rng); return z, uA, uB, np.stack([draw(p1[uA], rng), draw(p2[uA], rng), draw(p1[uB], rng), draw(p2[uB], rng)], 1)

    def name1(self, k2, k3): return softmax(GAIN * (self.R.XR[k2] @ self.W2.T + self.R.XL[k3] @ self.W3.T) + self.c, 1)

    def name2(self, uA, uB): return softmax(self.NA[uA] + self.NB[uB] + self.c2, 1)

    def read(self, x, rng): uA, uB = draw(self.name1(x[:, 0], x[:, 1]), rng), draw(self.name1(x[:, 2], x[:, 3]), rng); return draw(self.name2(uA, uB), rng), uA, uB

    def posterior(self, x, tab, rng):                                                                                              # the writer's own posterior over (z, u_A, u_B), drawn jointly
        TA, TB, p1, p2 = tab; la, lb = p1[:, x[:, 0]].T * p2[:, x[:, 1]].T, p1[:, x[:, 2]].T * p2[:, x[:, 3]].T; z = draw((la @ TA.T) * (lb @ TB.T), rng); return z, draw(TA[z] * la, rng), draw(TB[z] * lb, rng)

    def step1(self, u, k2, k3, d, tab, eta): p1, p2 = tab[2:]; n = np.bincount(u, d, self.U); np.add.at(self.D1, (u, k2), eta * d); self.D1 -= eta * n[:, None] * p1; np.add.at(self.D2, (u, k3), eta * d); self.D2 -= eta * n[:, None] * p2

    def step2(self, z, uA, uB, d, tab, eta): n = np.bincount(z, d, self.Z)[:, None]; np.add.at(self.THA, (z, uA), eta * d); self.THA -= eta * n * tab[0]; np.add.at(self.THB, (z, uB), eta * d); self.THB -= eta * n * tab[1]

    def wake(self, x, rng, tied=False):
        tab = self.tables(); z, uA, uB = self.posterior(x, tab, rng) if tied else self.read(x, rng); one = np.ones(len(x)); self.step1(uA, x[:, 0], x[:, 1], one, tab, self.eta[0]); self.step1(uB, x[:, 2], x[:, 3], one, tab, self.eta[0]); self.step2(z, uA, uB, one, tab, self.eta[0])

    def sleep(self, z, uA, uB, x):
        for u, k2, k3 in ((uA, x[:, 0], x[:, 1]), (uB, x[:, 2], x[:, 3])): g = -self.name1(k2, k3); g[np.arange(len(u)), u] += 1; self.W2 += self.eta[1] * g.T @ self.R.XR[k2]; self.W3 += self.eta[1] * g.T @ self.R.XL[k3]; self.c += self.eta[1] * g.sum(0)
        g = -self.name2(uA, uB); g[np.arange(len(z)), z] += 1; np.add.at(self.NA, uA, self.eta[1] * g); np.add.at(self.NB, uB, self.eta[1] * g); self.c2 += self.eta[1] * g.sum(0)

    def practice(self, z, uA, uB, x, tab):
        pA, pB = self.pairz(tab); R = self.R; eA, eB = R.pE[x[:, 0], x[:, 1]], R.pE[x[:, 2], x[:, 3]]; e = np.concatenate([eA, eB]); pos = e > 0; n = len(x)
        self.ebar = (float(e[pos].mean()) if self.ebar is None else 0.95 * self.ebar + 0.05 * float(e[pos].mean())) if pos.any() else (self.ebar or 1 / R.pE.size); logE = np.log(np.where(pos, e, FLOOR * self.ebar))
        d1 = np.log(np.concatenate([pA.mean(0)[x[:, 0], x[:, 1]], pB.mean(0)[x[:, 2], x[:, 3]]])) - logE; d1[~pos] = np.maximum(d1[~pos], 0); d1 = np.clip(d1, -CLIP, CLIP); f = -(d1 - self.b[0])
        ok = (eA > 0) & (eB > 0); r = R.r2[R.cls1[x[:, 0]], R.cls1[x[:, 2]]]; rp = r > 0; self.rbar = (float(r[ok & rp].mean()) if self.rbar is None else 0.95 * self.rbar + 0.05 * float(r[ok & rp].mean())) if (ok & rp).any() else (self.rbar or 1.0)
        d2 = np.log((pA[:, x[:, 0], x[:, 1]] * pB[:, x[:, 2], x[:, 3]]).mean(0)) - logE[:n] - logE[n:] - np.log(np.where(rp, r, FLOOR * self.rbar)); d2[~rp] = np.maximum(d2[~rp], 0); d2 = np.clip(d2, -CLIP, CLIP)
        f2 = np.where(ok, -(d2 - self.b[1]), 0.0)
        if self.only: f, f2 = f * (pos if self.only == 'accepted' else ~pos), f2 * (rp if self.only == 'accepted' else ~rp)
        self.step1(uA, x[:, 0], x[:, 1], f[:n], tab, self.eta[2]); self.step1(uB, x[:, 2], x[:, 3], f[n:], tab, self.eta[2]); self.step2(z, uA, uB, f2, tab, self.eta[2])
        self.b[0] += 0.05 * (float(d1.mean()) - self.b[0]); self.b[1] += 0.05 * (float(d2[ok].mean()) - self.b[1]) if ok.any() else 0.0

    def score(self, c):
        tab = self.tables(); pA, pB = (P[:, c.pairs[:, 0], c.pairs[:, 1]] for P in self.pairz(tab)); J = np.einsum('zi,zj->ij', pA, pB) / self.Z; v1 = float(J.sum()); loss = float(-np.log(c.ok2.sum()) - np.log(J[c.ok2] + 1e-300).mean())
        info = lambda J: float((J * (np.log(J + 1e-300) - np.log(J.sum(1, keepdims=True) * J.sum(0) + 1e-300))).sum()); q = np.stack([self.name1(*c.pairs[c.a == v].T).mean(0) for v in range(R1)])                  # q[value a, named u]
        n2 = softmax(self.NA[:, None, :] + self.NB[None, :, :] + self.c2, 2); J2 = np.stack([np.einsum('u,v,uvz->z', q[a], q[b], n2) for a, b in c.ab]) / len(c.ab)                                             # J2[valid pair of values, named z]
        return loss, v1, float(J[c.ok2].sum() / max(v1, 1e-300)), info(q / R1) / np.log(R1), info(J2) / np.log(self.Z)


def run(c, R, RT, S, arm, seed, eta, U, rounds, fresh):
    rng, rT = np.random.default_rng(seed), np.random.default_rng(seed + 1); batch = lambda r: c.cuts(r, BATCH) if fresh else S[r.integers(len(S), size=BATCH)][:, [1, 2, 5, 6]]; L = Stack(R, U, c.Z, rng, eta); T = Stack(RT, U, c.Z, rT, eta) if arm == 'WSPtwin' else None
    cs = set(checks(rounds)); out, outT = [], []; arm, nw, no = SPLIT.get(arm, (arm, 1, 1))
    for t in range(rounds + 1):
        if t in cs: out.append(L.score(c)); outT.append(T.score(c) if T else out[-1])
        for _ in range(nw): L.wake(batch(rng), rng, arm == 'Wtied')
        for _ in range(no if arm not in ('W', 'Wtied') else 0):
            tab = L.tables(); z, uA, uB, x = L.sample(rng, tab)
            if arm != 'WP': L.sleep(z, uA, uB, x)
            if arm in ('WSP', 'WP'): L.practice(z, uA, uB, x, tab)
        if arm == 'WSPread': tab = L.tables(); x = L.sample(rng, tab)[3]; L.practice(*L.read(x, rng), x, tab)
        if arm == 'WSPtwin':
            T.wake(batch(rT), rT); tabT = T.tables(); zT, a, b, xT = T.sample(rT, tabT); T.sleep(zT, a, b, xT); T.practice(zT, a, b, xT, tabT)
            x = T.sample(rT, T.tables())[3]; tab = L.tables(); L.practice(*L.read(x, rng), x, tab)
    return np.array(out), np.array(outT)


def one(job):
    n, seed, U, rounds, confs, fresh, top = job; c = Deep(seed, top); rng = np.random.default_rng(210_000 + seed); S = c.teacher(rng, n); val = rng.random(n) < 0.15; R = reader(c, S, val, 220_000 + seed)
    RT = reader(c, S, val, 240_000 + seed) if any('WSPtwin' in arms for _, arms in confs) else None; res = np.full((len(confs), len(ARMS), len(checks(rounds)), len(KEYS)), np.nan)
    for i, (eta, arms) in enumerate(confs):
        for arm in arms:
            res[i, ARMS.index(arm)], t = run(c, R, RT, S, arm, 250_000 + seed, eta, U, rounds, fresh)
            if arm == 'WSPtwin': res[i, ARMS.index('twin')] = t
    cA = R.cls1[c.pairs[:, 0]]; r = R.r2[cA[:, None], cA[None, :]]; ps = 1 / c.agree.sum()                                           # the two judges on the valid cut pairs
    return job, res, [float((R.pE > 0)[c.agree].mean()), float((R.pE > 0)[~c.agree].mean()), float((r > 0)[c.ok2].mean()), float((r > 0)[~c.ok2].mean()), float((ps * (np.log(ps) - np.log(0.999 * R.pE[c.agree] + 0.001 / R.pE.size))).sum()), R.cls1.max() + 1]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='grid'); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--ns', default='4096'); ap.add_argument('--U', type=int, default=4)
    ap.add_argument('--etas', default='0.1,0.05,0.1'); ap.add_argument('--fresh', action='store_true'); ap.add_argument('--top', default='table'); ap.add_argument('--rounds', type=int, default=4096); ap.add_argument('--gw', default='0.05,0.1,0.2'); ap.add_argument('--gs', default='0.0125,0.05,0.2'); ap.add_argument('--gp', default='0.05,0.1,0.2')
    ap.add_argument('--tag', default=''); args = ap.parse_args(); ns = [int(x) for x in args.ns.split(',')]; eta0 = tuple(float(x) for x in args.etas.split(',')); GW, GS, GP = (tuple(float(x) for x in getattr(args, k).split(',')) for k in ('gw', 'gs', 'gp'))
    confs = [(eta0, ARMS[:-1])] if args.run == 'grid' else [((a, b, pp), ('WS', 'WSP', 'Wtied') if (b, pp) == (GS[0], GP[0]) else ('WS', 'WSP') if pp == GP[0] else ('WSP',)) for a in GW for b in GS for pp in GP]
    jobs = [(n, s, args.U, args.rounds, confs, args.fresh, args.top) for n in ns for s in range(args.worlds)]; res, aux = np.zeros((len(ns), args.worlds, len(confs), len(ARMS), len(checks(args.rounds)), len(KEYS))), np.zeros((len(ns), args.worlds, 6))
    with Pool(args.procs) as pool:
        for (n, s, *_), r, a in tqdm(pool.imap_unordered(one, jobs), total=len(jobs)): res[ns.index(n), s], aux[ns.index(n), s] = r, a
    name = f"{OUT}/cut_depth_{args.top}{'_scan' if args.run == 'scan' else ''}{'_' + args.tag if args.tag else ''}.npz"
    np.savez(name, res=res, aux=aux, ns=np.array(ns), etas=np.array([e for e, _ in confs]), checks=checks(args.rounds), arms=ARMS, keys=KEYS, uses=np.array([USES[a] for a in ARMS]), U=args.U, fresh=args.fresh, top=args.top); print(name)
