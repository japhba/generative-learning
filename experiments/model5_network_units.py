"""Model 5, a network learner with latent units in the writer. One block of model5_lateral_verdict.py (64 codes per side, 16 classes, 1,024 valid pairs of 4,096).
Three parts with separate weights, all from small random starts:
  trunk    the reader's hidden layer: the local-rule network of model5_network_practice.local_net (hidden vectors u_k, v_k' of the codes, 16 units), trained on teacher pairs only
           (a teacher pair raises u_k . v_k', the same left code with the right code of another teacher pair lowers it). X_L, X_R: the hidden vectors, centred and scaled to unit length.
  judge    p_E = K_L M K_R^T / n, the stored teacher pairs spread over hidden-layer neighbours (as in section 8 of the note); a refusal counts as FLOOR times the mean positive answer.
  writer   latent units a, a' (K per side):  p(k, k') = sum_{a,a'} T[a,a'] G_L[k|a] G_R[k'|a'],  G_L = softmax_k(LL), G_R = softmax_k'(LR), T = softmax_{aa'}(TH).
  head     the reader's units for a pair of codes, a and a' drawn independently from q_L(a|k,k') and q_R(a'|k,k'). Four versions:
             tied   the writer's own posterior q(a,a'|k,k') ~ T G_L G_R (units drawn jointly; what sleep run to the end would give)
             raw    q_L(a|k,k') = softmax_a(H[k, a] + H'[k', a]): one layer from the two codes, trained by sleep (classic wake-sleep)
             trunk  q_L(a|k,k') = softmax_a(GAIN (W_a . X_L[k] + W'_a . X_R[k']) + c_a): one layer on the trunk's hidden vectors of the two codes, trained by sleep
             comp   no own samples: units are recruited from the trunk's hidden vectors of the teacher codes (a code joins the unit with the most similar mean vector if the cosine is >= tau, otherwise it founds a unit)
Rules (delta rules on the active units; d is the third factor):
  wake      teacher pair (k, k'), units (a, a') read by the head:  LL[a] += eta (e_k - G_L[.|a]), LR[a'] likewise, TH += eta (e_aa' - T)
  sleep     own sample with its units: the head's layer moves towards the units that wrote it (delta rule; raw and trunk only)
  practice  own sample x = (k, k') written by units z = (a, a'):  delta = log p(x) - log p_E(x), cut at +-CLIP;  the wake step at z with factor d = -(delta - running mean of delta).
            Averaged over own samples this is the gradient of KL(p_writer || p_E). credit 'gen': the units that wrote x; 'read': the units the head reads x into.
            A foreign sample (twin, teacher) is read into the learner's units by its head. A refused sample is only ever moved down.
Protocol: n stored teacher pairs (15% held out for the trunk and the judge's bandwidth). Watching: WATCH batches, each a wake batch from the stored pairs and (raw, trunk) a sleep batch.
  Then 512 batches of 64 by arm:  replay (more watching)   own (practice only, credit to the writing units)   own_read (credit by reading)   twin (samples of a twin: same teacher pairs, other random
  starts, practising on its own samples at the same pace)   own_replay (a practice batch and a wake batch in turn)   blind (the judge accepts everything: p_E uniform)
  fresh: the same learner on twice the teacher pairs is the next size of the grid.
Scores (exact, of a running mean of the writer's weights over about 50 batches): population loss KL(p* || p_writer), valid share, purity (mass a unit puts on its main class, weighted by use), reading (probability that the head reads a teacher code into a unit whose main class is the code's).
  --run scan: after 4,096 teacher pairs with 16 own units, the course of watching and 524,288 further samples at other step sizes, and with a judge that only accepts or refuses (target: uniform over what it accepts).
Writes /var/tmp/jbauer/model5_lateral_verdict/network_units[_scan].npz.   Usage: python model5_network_units.py [--run scan] [--worlds 16] [--procs 16] [--ns 1024,...] [--Ks 16,32] [--tag x]"""

import argparse
import copy
import os
from multiprocessing import Pool

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm

from model5_lateral_verdict import OUT, World, draw, table_kl
from model5_network_baseline import groups
from model5_network_practice import D, bandwidth, evidence, local_net
from model5_own_benefit import class_table, kl

INIT, BATCH, ETA_W, ETA_S, ETA_P, GAIN, WATCH, FLOOR, CLIP, RHO = 0.1, 64, 0.05, 0.05, 0.05, 4.0, 40000, 1e-3, 5.0, 0.98
SCAN = (('replay', 0.05, 'replay'), ('own_replay', 0.05, 'mix'), ('own_replay_slow', 0.01, 'mix'), ('own_slow', 0.01, 'own'), ('binary', 0.02, 'mix')); SCHECK, WCHECK = (0, 512, 2048, 8192), 2000   # scan: arm, practice step, schedule
CHECK = np.array([0, 16, 32, 64, 128, 256, 512]); HEADS = ('tied', 'raw', 'trunk', 'comp'); ARMS = ('replay', 'own', 'own_read', 'twin', 'own_replay', 'blind'); KEYS = ('kl', 'valid', 'purity', 'reading')


def normed(X): X = X - X.mean(0); return X / np.linalg.norm(X, axis=1, keepdims=True)


def recruit(X, order, tau):                                                                                                        # units recruited from the hidden vectors: code -> unit
    sums, unit = [], -np.ones(len(X), int)
    for k in order:
        s = np.array([p @ X[k] / np.linalg.norm(p) for p in sums]) if sums else np.array([-1.0]); j = int(s.argmax())
        if s[j] >= tau: sums[j] += X[k]; unit[k] = j
        else: sums.append(X[k].copy()); unit[k] = len(sums) - 1
    return unit


class Units:
    def __init__(self, w, X, K, head, rng, unit=None):
        N = w.N; self.w, self.X, self.N, self.head = w, X, N, head; self.K = K = K if unit is None else max(int(u.max()) + 1 for u in unit)
        self.LL, self.LR, self.TH = (INIT * rng.standard_normal(s) for s in ((K, N), (K, N), (K, K))); self.b, self.ebar = 0.0, None; self.avg = [x.copy() for x in (self.LL, self.LR, self.TH)]   # avg: running mean of the writer's weights, for scoring
        if head == 'raw': self.H = INIT * rng.standard_normal((2, 2, N, K))                                                        # [side of the unit, own code or partner's code]
        elif head == 'trunk': self.W, self.c = INIT * rng.standard_normal((2, 2, K, D)), np.zeros((2, K))
        elif head == 'comp': self.unit = unit

    def tables(self, avg=False): LL, LR, TH = self.avg if avg else (self.LL, self.LR, self.TH); return softmax(LL, 1), softmax(LR, 1), softmax(TH.ravel()).reshape(self.K, self.K)

    def readprob(self, k, kp, tab=None):                                                                                           # the reader's units for the pairs (k, k'): probabilities of a and of a' (each len(k) x K)
        if self.head == 'comp': return np.eye(self.K)[self.unit[0][k]], np.eye(self.K)[self.unit[1][kp]]
        if self.head == 'tied': GL, GR, T = tab or self.tables(); qa, qb = GL[:, k].T * (T @ GR)[:, kp].T, GR[:, kp].T * (GL.T @ T)[k]; return qa / qa.sum(1, keepdims=True), qb / qb.sum(1, keepdims=True)
        if self.head == 'raw': return softmax(self.H[0, 0][k] + self.H[0, 1][kp], 1), softmax(self.H[1, 0][kp] + self.H[1, 1][k], 1)
        XL, XR = self.X[0][k], self.X[1][kp]; return softmax(GAIN * (XL @ self.W[0, 0].T + XR @ self.W[0, 1].T) + self.c[0], 1), softmax(GAIN * (XR @ self.W[1, 0].T + XL @ self.W[1, 1].T) + self.c[1], 1)

    def read(self, k, kp, rng, tab=None):
        if self.head == 'tied': GL, GR, T = tab or self.tables(); a = draw((GL[:, k] * (T @ GR)[:, kp]).T, rng); return a, draw(T[a] * GR[:, kp].T, rng)   # the writer's own posterior, units drawn jointly
        qa, qb = self.readprob(k, kp); return draw(qa, rng), draw(qb, rng)

    def step(self, a, ap, k, kp, d, tab, eta):                                                                                     # three-factor step at the units (a, a'): d x (what was written - what the unit writes on average)
        GL, GR, T = tab; np.add.at(self.LL, (a, k), eta * d); self.LL -= eta * np.bincount(a, d, self.K)[:, None] * GL
        np.add.at(self.LR, (ap, kp), eta * d); self.LR -= eta * np.bincount(ap, d, self.K)[:, None] * GR; np.add.at(self.TH, (a, ap), eta * d); self.TH -= eta * d.sum() * T
        for A, L in zip(self.avg, (self.LL, self.LR, self.TH)): A += (1 - RHO) * (L - A)

    def sample(self, n, rng, tab=None):
        GL, GR, T = tab or self.tables(); c = rng.choice(self.K ** 2, n, p=T.ravel()); a, ap = c // self.K, c % self.K; return draw(GL[a], rng), draw(GR[ap], rng), a, ap

    def wake(self, k, kp, rng): tab = self.tables(); self.step(*self.read(k, kp, rng, tab), k, kp, np.ones(len(k)), tab, ETA_W)

    def sleep(self, a, ap, k, kp):
        if self.head in ('comp', 'tied'): return
        for s, (u, q, x, y) in enumerate(zip((a, ap), self.readprob(k, kp), (k, kp), (kp, k))):                                    # x: the unit's own code, y: the partner's code
            g = -q; g[np.arange(len(u)), u] += 1
            if self.head == 'raw': np.add.at(self.H[s, 0], x, ETA_S * g); np.add.at(self.H[s, 1], y, ETA_S * g)
            else: self.W[s, 0] += ETA_S * g.T @ self.X[s][x]; self.W[s, 1] += ETA_S * g.T @ self.X[1 - s][y]; self.c[s] += ETA_S * g.sum(0)

    def dream(self, rng):
        if self.head not in ('comp', 'tied'): k, kp, a, ap = self.sample(BATCH, rng); self.sleep(a, ap, k, kp)

    def practice(self, pE, rng, source=None, credit='gen', eta=None):
        tab = self.tables(); k, kp, a, ap = self.sample(BATCH, rng, tab) if source is None else (*source, None, None)
        if source is None: self.sleep(a, ap, k, kp)
        if source is not None or credit == 'read': a, ap = self.read(k, kp, rng, tab)
        e = pE[k, kp]; pos = e > 0; self.ebar = (float(e[pos].mean()) if self.ebar is None else 0.95 * self.ebar + 0.05 * float(e[pos].mean())) if pos.any() else (self.ebar or 1 / pE.size)
        delta = np.log((tab[0].T @ tab[2] @ tab[1])[k, kp]) - np.log(np.where(pos, e, FLOOR * self.ebar)); delta[~pos] = np.maximum(delta[~pos], 0); delta = np.clip(delta, -CLIP, CLIP)
        self.step(a, ap, k, kp, -(delta - self.b), tab, eta or ETA_P); self.b += 0.05 * (float(delta.mean()) - self.b)

    def score(self):                                                                                                               # of the averaged weights
        w = self.w; GL, GR, T = tab = self.tables(avg=True); P = GL.T @ T @ GR; out = [kl(w, P), float(P[w.valid].sum())]; vk, vkp = np.nonzero(w.valid); pur, rd = [], []
        for G, pi, q, x in zip((GL, GR), (T.sum(1), T.sum(0)), self.readprob(vk, vkp, tab), (vk, vkp)):
            mass = np.stack([np.bincount(w.cls, G[a], w.v) for a in range(self.K)]); main = mass.argmax(1); pur.append(float(pi @ mass.max(1))); rd.append(float((q * (main[None, :] == w.cls[x][:, None])).sum(1).mean()))
        return out + [float(np.mean(pur)), float(np.mean(rd))]


def build(w, k, kp, val, K, head, seed):                                                                                           # trunk, judge and a learner after watching
    rng = np.random.default_rng(seed); net, _, _, Q, Qv, _ = local_net(w, k, kp, val, rng); X = [normed(net.P[0]), normed(net.P[1])]; tau = bandwidth(net, Q, Qv); pE = evidence(net, Q + Qv, tau); n = len(k)
    unit = [recruit(X[s], kk[np.sort(np.unique(kk, return_index=True)[1])], tau) for s, kk in enumerate((k, kp))] if head == 'comp' else None; L = Units(w, X, K, head, rng, unit)
    for _ in range(WATCH):
        i = rng.integers(n, size=BATCH); L.wake(k[i], kp[i], rng)
        L.dream(rng)
    return L, pE, rng, net, Q + Qv


def one(job):
    n, K, head, seed = job; w = World(seed=seed); rng0 = np.random.default_rng(50_000 + seed); k, kp = w.teacher(rng0, n); val = rng0.random(n) < 0.15
    L0, pE, rng, net, M = build(w, k, kp, val, K, head, 60_000 + seed); T0, pT, rngT, _, _ = build(w, k, kp, val, K, head, 70_000 + seed); res = np.zeros((len(ARMS), len(CHECK), len(KEYS)))
    ref = [table_kl(w, M), class_table(w, M, w.cls, w.cls), class_table(w, M, groups(net.P[0], w.v), groups(net.P[1], w.v)), kl(w, 0.999 * pE + 0.001 / pE.size), int((pE > 0)[~w.valid].sum()), int((pE == 0)[w.valid].sum()), L0.K, T0.score()[0], class_table(w, M, *L0.unit) if head == 'comp' else np.nan]
    for i, arm in enumerate(ARMS):
        L, T, r, rT = copy.deepcopy(L0), copy.deepcopy(T0), np.random.default_rng(80_000 + seed), np.random.default_rng(90_000 + seed); judge = np.full_like(pE, 1 / pE.size) if arm == 'blind' else pE
        for t in range(CHECK[-1] + 1):
            if t in CHECK: res[i, list(CHECK).index(t)] = L.score()
            if arm == 'replay' or (arm == 'own_replay' and t % 2):
                j = r.integers(n, size=BATCH); L.wake(k[j], kp[j], r)
                L.dream(r)
            elif arm == 'twin': T.practice(pT, rT); L.practice(judge, r, source=T.sample(BATCH, rT)[:2])
            else: L.practice(judge, r, credit='read' if arm == 'own_read' else 'gen')
    return job, res, ref


def one_scan(job):                                                                                                                 # 4,096 teacher pairs, 16 own units: the course of watching, then long practice at other step sizes and with a verdict that only accepts or refuses
    head, seed = job; n, K = 4096, 16; w = World(seed=seed); rng0 = np.random.default_rng(50_000 + seed); k, kp = w.teacher(rng0, n); val = rng0.random(n) < 0.15; rng = np.random.default_rng(60_000 + seed)
    net, _, _, Q, Qv, _ = local_net(w, k, kp, val, rng); X = [normed(net.P[0]), normed(net.P[1])]; pE = evidence(net, Q + Qv, bandwidth(net, Q, Qv)); L0 = Units(w, X, K, head, rng); curve = []; res = np.zeros((len(SCAN), len(SCHECK), len(KEYS)))
    for t in range(WATCH + 1):
        if t % WCHECK == 0: curve.append(L0.score())
        if t < WATCH: i = rng.integers(n, size=BATCH); L0.wake(k[i], kp[i], rng); L0.dream(rng)
    for i, (arm, eta, mode) in enumerate(SCAN):
        L, r = copy.deepcopy(L0), np.random.default_rng(80_000 + seed); judge = np.where(pE > 0, 1 / (pE > 0).sum(), 0.0) if arm == 'binary' else pE                               # binary: uniform over what the judge accepts (the most even writer among those it accepts)
        for t in range(SCHECK[-1] + 1):
            if t in SCHECK: res[i, SCHECK.index(t)] = L.score()
            if mode == 'replay' or (mode == 'mix' and t % 2): j = r.integers(n, size=BATCH); L.wake(k[j], kp[j], r); L.dream(r)
            else: L.practice(judge, r, eta=eta)
    return job, np.array(curve), res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--ns', default='1024,2048,4096,8192'); ap.add_argument('--Ks', default='16,32')
    ap.add_argument('--heads', default=','.join(HEADS)); ap.add_argument('--tag', default=''); ap.add_argument('--run', default='grid'); args = ap.parse_args(); ns, Ks, heads = [int(x) for x in args.ns.split(',')], [int(x) for x in args.Ks.split(',')], args.heads.split(',')
    if args.run == 'scan':
        sh = [h for h in heads if h != 'comp']; out = {}
        with Pool(args.procs) as p:
            for job, curve, res in tqdm(p.imap_unordered(one_scan, [(h, s) for h in sh for s in range(args.worlds)]), total=len(sh) * args.worlds): out[job] = (curve, res)
        C, R = (np.array([[out[h, s][i] for s in range(args.worlds)] for h in sh]) for i in (0, 1)); np.savez(OUT / f'network_units_scan{args.tag}.npz', curve=C, res=R, heads=sh, arms=[a for a, _, _ in SCAN], check=np.array(SCHECK) * BATCH, wcheck=np.arange(C.shape[2]) * WCHECK, keys=KEYS)
        for i, h in enumerate(sh):
            print(h, 'watching, loss by batch', np.round(np.median(C[i, :, :, 0], 0), 3).tolist())
            for j, (a, _, _) in enumerate(SCAN): print(f'   {a:16s} loss', np.round(np.median(R[i, :, j, :, 0], 0), 3).tolist(), 'valid', np.round(np.median(R[i, :, j, :, 1], 0), 3).tolist(), 'purity', np.round(np.median(R[i, :, j, :, 2], 0), 3).tolist(), 'worlds below more watching', int((R[i, :, j, -1, 0] < R[i, :, 0, -1, 0]).sum()))
        raise SystemExit
    jobs = [(n, K, h, s) for n in ns[::-1] for K in Ks for h in heads for s in range(args.worlds) if not (h == 'comp' and K != Ks[0])]; out = {}; OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.procs) as p:
        for job, res, ref in tqdm(p.imap_unordered(one, jobs), total=len(jobs)): out[job] = (res, ref)
    cfg = [(K, h) for K in Ks for h in heads if not (h == 'comp' and K != Ks[0])]; R = np.array([[[out[n, K, h, s][0] for s in range(args.worlds)] for K, h in cfg] for n in ns]); F = np.array([[[out[n, K, h, s][1] for s in range(args.worlds)] for K, h in cfg] for n in ns])
    np.savez(OUT / f'network_units{args.tag}.npz', res=R, ref=F, ns=ns, cfg=[f'{h}{K}' for K, h in cfg], arms=ARMS, check=CHECK * BATCH, keys=KEYS, refkeys=['table', 'classes', 'hidden', 'judge', 'invalid', 'refused', 'units', 'twin', 'limit'])
    med = np.median(R, 2); rf = np.median(F, 2)
    for a, n in enumerate(ns):
        print(f'n = {n}: table {rf[a, 0, 0]:.3f}  true classes {rf[a, 0, 1]:.3f}  hidden layer read out {rf[a, 0, 2]:.3f}  judge {rf[a, 0, 3]:.3f} (invalid accepted {rf[a, 0, 4]:.0f}, valid refused {rf[a, 0, 5]:.0f})')
        for c, name in enumerate(f'{h}{K}' for K, h in cfg):
            print(f'  {name:8s} units {rf[a, c, 6]:.0f}  limit of wake {rf[a, c, 8]:.3f}  twin {rf[a, c, 7]:.3f}  after watching: loss {med[a, c, 0, 0, 0]:.3f} valid {med[a, c, 0, 0, 1]:.3f} purity {med[a, c, 0, 0, 2]:.3f} reading {med[a, c, 0, 0, 3]:.3f}')
            for i, arm in enumerate(ARMS): print(f'      {arm:10s} loss by samples ' + ' '.join(f'{x:.3f}' for x in med[a, c, i, :, 0]) + f' | end: valid {med[a, c, i, -1, 1]:.3f} purity {med[a, c, i, -1, 2]:.3f} reading {med[a, c, i, -1, 3]:.3f}')
