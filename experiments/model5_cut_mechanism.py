"""Model 5, inside the two-level learner of model5_cut_depth.py (--top table): what the units come to stand for, which samples make them selective and when, and what the reader recovers.
Two arms, classic wake-sleep (WS) and the same with the judged step on the own samples (WSP), 4,096 stored teacher strings, fresh strings for wake, 16 worlds; a twin with other random starts runs the same arm.
At every check of the first learner:
  scores        as in model5_cut_depth.py (loss, valid1, valid2, read1, read2)
  selectivity   sel1 = information of a level-1 unit about the value it writes at its cut; sel2w = information of the top unit about the pair of level-1 units it chooses (its weights);
                sel2x = information of the top unit about the pair of values in the written string (its effect on the string). All in units of log 4.
  reach         how much of the unit that wrote a sample the reader's naming layers recover, for own samples and for the twin's, level 1 and level 2 (model5_cut_twin.reach)
  tuning        for one look at the units: A[u, a] the value a unit writes, T_A[z, u] and T_B[z, u] the units the top unit chooses, q[a, u] the unit named for a teacher pair of value a
  agreement     do reader and writer agree on the units of a string: the probability that the named unit is the one the writer's own posterior would draw (0: chance, 1: always), level 1 and level 2,
                for own samples, fresh teacher strings and the twin's samples
  probes        the learner is copied and one rule alone is applied for 256 batches at a step 500 times smaller; the change of what is written (loss, valid1, valid2) and of the selectivities per
                sample at the run's step is recorded:  wake (fresh teacher strings, units named by the reader)   wake_tied (the same strings, units drawn from the writer's own posterior: what a
                teacher string gives when it is read into the writer's units as well as possible)   practice (own samples, credited to the units that wrote them)   refused / accepted (the same,
                keeping only the verdicts of one kind)   own_read (own samples, credited by reading)   twin_read (the twin's samples, credited by reading)
Also: the same two arms with a naming layer that reads the codes instead of the cut net's hidden vectors (four steps of sleep), and, for four sizes of the teacher set, the share of the valid
cut pairs that the stored strings contain against the share the judge accepts.
Writes OUT/cut_mechanism.npz.   Usage: python model5_cut_mechanism.py [--worlds 16] [--procs 16]"""

import argparse
import copy
import os
from multiprocessing import Pool

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.special import softmax
from tqdm.auto import tqdm

from model5_cut_depth import BATCH, R1, Deep, Stack, reader
from model5_cut_twin import reach
from model5_lateral_verdict import OUT

N0, ETA, U, ROUNDS, NB, EPS = 4096, (0.1, 0.003, 0.1), 4, 2048, 256, 2e-4; CK = (0, 4, 8, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 2048); ARMS = ('WS', 'WSP')
RULES = ('wake', 'wake_tied', 'practice', 'refused', 'accepted', 'own_read', 'twin_read'); PK = ('loss', 'valid1', 'valid2', 'sel1', 'sel2w', 'sel2x')
KEYS = ('loss', 'valid1', 'valid2', 'read1', 'read2', 'sel1', 'sel2w', 'sel2x', 'reach1_own', 'reach2_own', 'reach1_twin', 'reach2_twin', 'agree1_own', 'agree2_own', 'agree1_teacher', 'agree2_teacher', 'agree1_twin', 'agree2_twin'); SLEEPS = (0.003, 0.0125, 0.05, 0.2); SIZES = (256, 512, 1024, 4096)


def mi(J): J = J / J.sum(); return float((J * (np.log(J + 1e-300) - np.log(J.sum(1, keepdims=True) * J.sum(0) + 1e-300))).sum())


def tuning(L, c):
    TA, TB, p1, p2 = L.tables(); return np.stack([np.stack([p1[:, c.fR == v].sum(1) for v in range(R1)], 1), TA, TB, np.stack([L.name1(*c.pairs[c.a == v].T).mean(0) for v in range(R1)])])


def sel(L, c):
    A, TA, TB, _ = tuning(L, c); return np.array([(mi(TA.mean(0)[:, None] * A) + mi(TB.mean(0)[:, None] * A)) / 2 / np.log(R1), mi((TA[:, :, None] * TB[:, None, :]).reshape(L.Z, -1)) / np.log(L.Z), mi(((TA @ A)[:, :, None] * (TB @ A)[:, None, :]).reshape(L.Z, -1)) / np.log(L.Z)])


def state(L, c): return np.array([*L.score(c)[:3], *sel(L, c)])                                                                      # what is written (loss, valid1, valid2) and the three selectivities


def agree(L, x):                                                                                                                   # do reader and writer agree on the units of the strings x? Probability that the named unit is the one the writer's own
    TA, TB, p1, p2 = L.tables(); la, lb = p1[:, x[:, 0]].T * p2[:, x[:, 1]].T, p1[:, x[:, 2]].T * p2[:, x[:, 3]].T; qa, qb = L.name1(x[:, 0], x[:, 1]), L.name1(x[:, 2], x[:, 3])   # posterior would draw, level 1 and level 2; 0: chance, 1: always
    pa, pb = la * TA.mean(0), lb * TB.mean(0); a1 = ((qa * pa).sum(1) / pa.sum(1) + (qb * pb).sum(1) / pb.sum(1)).mean() / 2; pz = (la @ TA.T) * (lb @ TB.T); n2 = softmax(L.NA[:, None, :] + L.NB[None, :, :] + L.c2, 2)
    a2 = ((np.einsum('nu,nv,uvz->nz', qa, qb, n2) * pz).sum(1) / pz.sum(1)).mean(); return [(a1 - 1 / L.U) / (1 - 1 / L.U), (a2 - 1 / L.Z) / (1 - 1 / L.Z)]


def many(L, rng): return np.concatenate([L.sample(rng, L.tables())[3] for _ in range(64)])                                           # 4,096 samples of a learner


def probe(L, c, rule, rng, twin):                                                                                                  # one rule alone on a copy, at a small step: change of what is written and of the selectivities per sample at the run's step
    P = copy.deepcopy(L); P.eta = (EPS, 0.0, EPS); P.only = rule if rule in ('refused', 'accepted') else None; s0 = state(P, c)
    for _ in range(NB):
        if rule in ('wake', 'wake_tied'): P.wake(c.cuts(rng, BATCH), rng, rule == 'wake_tied'); continue
        tab = P.tables(); z, uA, uB, x = P.sample(rng, tab)
        if rule == 'twin_read': x = twin.sample(rng, twin.tables())[3]
        if rule in ('own_read', 'twin_read'): z, uA, uB = P.read(x, rng)
        P.practice(z, uA, uB, x, tab)
    return (state(P, c) - s0) / (NB * BATCH * EPS) * ETA[0]


def run(c, R, RT, arm, seed, eta, probes=True):
    rng, rT, rE = np.random.default_rng(seed), np.random.default_rng(seed + 1), np.random.default_rng(seed + 2); L, T = Stack(R, U, c.Z, rng, eta), Stack(RT, U, c.Z, rT, eta); res, prb, tun = [], [], []
    for t in range(ROUNDS + 1):
        if t in CK:
            res.append([*L.score(c), *sel(L, c), *reach(L, L, rE), *reach(L, T, rE), *agree(L, many(L, rE)), *agree(L, c.cuts(rE, 4096)), *agree(L, many(T, rE))]); tun.append(tuning(L, c))
            if probes: prb.append([probe(L, c, rule, rE, T) for rule in RULES])
        for X, r in ((L, rng), (T, rT)):
            X.wake(c.cuts(r, BATCH), r); tab = X.tables(); z, uA, uB, x = X.sample(r, tab); X.sleep(z, uA, uB, x)
            if arm == 'WSP': X.practice(z, uA, uB, x, tab)
    return np.array(res), np.array(prb), np.array(tun)


def one(seed):
    c = Deep(seed); rng = np.random.default_rng(210_000 + seed); S = c.teacher(rng, N0); val = rng.random(N0) < 0.15; R, RT = reader(c, S, val, 220_000 + seed), reader(c, S, val, 240_000 + seed)
    out = [run(c, R, RT, arm, 250_000 + seed, ETA) for arm in ARMS]; eye = np.eye(c.w.N); Rc, RTc = (copy.copy(X) for X in (R, RT)); Rc.XR = Rc.XL = RTc.XR = RTc.XL = eye                 # naming layer on the codes
    abl = np.array([[run(c, Rc, RTc, arm, 250_000 + seed, (ETA[0], es, ETA[2]), probes=False)[0][:, :5] for arm in ARMS] for es in SLEEPS]); jud = []
    for n in SIZES:                                                                                                                # the judge of a cut against the stored strings themselves
        Sn = c.teacher(np.random.default_rng(310_000 + seed), n); vn = np.random.default_rng(320_000 + seed).random(n) < 0.15; Rn = reader(c, Sn, vn, 330_000 + seed); seen = np.zeros_like(c.agree); seen[Sn[:, [1, 5]].ravel(), Sn[:, [2, 6]].ravel()] = True
        jud.append([seen[c.agree].mean(), (Rn.pE > 0)[c.agree].mean(), (Rn.pE > 0)[~c.agree].mean(), ((Rn.pE > 0) & ~seen)[c.agree].sum() / max((~seen)[c.agree].sum(), 1)])
    return seed, [o[0] for o in out], [o[1] for o in out], [o[2] for o in out], abl, np.array(jud)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=16); args = ap.parse_args(); W = args.worlds
    res, prb, tun, abl, jud = np.zeros((W, 2, len(CK), len(KEYS))), np.zeros((W, 2, len(CK), len(RULES), len(PK))), np.zeros((W, 2, len(CK), 4, 4, 4)), np.zeros((W, len(SLEEPS), 2, len(CK), 5)), np.zeros((W, len(SIZES), 4))
    with Pool(args.procs) as pool:
        for s, r, p, t, a, j in tqdm(pool.imap_unordered(one, range(W)), total=W): res[s], prb[s], tun[s], abl[s], jud[s] = r, p, t, a, j
    np.savez(f'{OUT}/cut_mechanism.npz', res=res, prb=prb, tun=tun, abl=abl, jud=jud, checks=np.array(CK), arms=ARMS, rules=RULES, keys=KEYS, pkeys=PK, sleeps=np.array(SLEEPS), sizes=np.array(SIZES), etas=np.array(ETA)); print(f'{OUT}/cut_mechanism.npz')
