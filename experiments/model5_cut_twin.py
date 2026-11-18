"""Model 5, whose samples at two levels: are a twin's samples alike at level 1 and different at level 2, and do they then do worse?
World, learner, rules and steps of model5_cut_depth.py (--top table). Two learners with the same stored teacher strings and other random starts of every part (reader nets, own units, naming layers).
Every round, each learner reads one batch of fresh teacher strings (wake), writes one batch of own samples (sleep on them), and takes one judged batch:
  own_cause    its own batch, each verdict credited to the units that wrote the sample
  own_read     its own batch, credited to the units its naming layers read
  cross_read   the other learner's batch, credited by reading; the other does the same with this learner's batch, so the two are at the same stage throughout
  ahead_read   the batch of a twin that runs own_cause (and is therefore ahead), credited by reading
Scores of the first learner as in model5_cut_depth.py, and how alike the two learners' samples are (total variation distance, 0: the same, 1: disjoint):
  tv1   between what the two write at a cut (the pair of codes, mean of the two cuts)
  tv2   between their tables of the two values (a, b), among the strings whose two cuts agree
and how much of a sample's causes the first learner's reader recovers (information of the named unit about the unit that wrote the sample, in units of log 4; from 4,096 samples):
  reach1_own, reach2_own     its own samples, at level 1 (u) and level 2 (z)          reach1_twin, reach2_twin   the other learner's samples
Writes OUT/cut_twin.npz.   Usage: python model5_cut_twin.py [--worlds 16] [--procs 16] [--ns 4096] [--etas 0.1,0.003,0.1] [--rounds 2048]"""

import argparse
import os
from multiprocessing import Pool

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from tqdm.auto import tqdm

from model5_cut_depth import BATCH, R1, Deep, Stack, reader
from model5_cut_practice import checks
from model5_lateral_verdict import OUT, draw

ARMS = ('own_cause', 'own_read', 'cross_read', 'ahead_read'); KEYS = ('loss', 'valid1', 'valid2', 'read1', 'read2', 'tv1', 'tv2', 'twin_valid1', 'twin_valid2', 'reach1_own', 'reach2_own', 'reach1_twin', 'reach2_twin'); M = 4096


def written(L, c):                                                                                                                 # what a learner writes: the pair at a cut, and the table of the two values among the strings whose cuts agree
    pA, pB = L.pairz(L.tables()); J = np.einsum('zi,zj->ij', pA[:, c.pairs[:, 0], c.pairs[:, 1]], pB[:, c.pairs[:, 0], c.pairs[:, 1]]) / L.Z; V = np.zeros((R1, R1)); np.add.at(V, (c.a[:, None], c.a[None, :]), J)
    return (pA.mean(0) + pB.mean(0)) / 2, V / V.sum()


def reach(L, src, rng):                                                                                                            # how much of the causes of src's samples L's naming layers recover, level 1 and level 2
    TA, TB, p1, p2 = src.tables(); z = rng.integers(src.Z, size=M); uA, uB = draw(TA[z], rng), draw(TB[z], rng); zh, ah, bh = L.read(np.stack([draw(p1[uA], rng), draw(p2[uA], rng), draw(p1[uB], rng), draw(p2[uB], rng)], 1), rng)
    def info(a, b): J = np.zeros((4, 4)); np.add.at(J, (a, b), 1 / len(a)); return float((J * (np.log(J + 1e-300) - np.log(J.sum(1, keepdims=True) * J.sum(0) + 1e-300))).sum() / np.log(4))
    return [(info(uA, ah) + info(uB, bh)) / 2, info(z, zh)]


def run(c, R, RT, arm, seed, eta, U, rounds):
    rng, rT, rE = np.random.default_rng(seed), np.random.default_rng(seed + 1), np.random.default_rng(seed + 2); L, T = Stack(R, U, c.Z, rng, eta), Stack(RT, U, c.Z, rT, eta); cs = set(checks(rounds)); out = []
    for t in range(rounds + 1):
        if t in cs: (P, V), (PT, VT), sT = written(L, c), written(T, c), T.score(c); out.append([*L.score(c), 0.5 * np.abs(P - PT).sum(), 0.5 * np.abs(V - VT).sum(), sT[1], sT[2], *reach(L, L, rE), *reach(L, T, rE)])
        L.wake(c.cuts(rng, BATCH), rng); T.wake(c.cuts(rT, BATCH), rT); tabL, tabT = L.tables(), T.tables(); zL, aL, bL, xL = L.sample(rng, tabL); zT, aT, bT, xT = T.sample(rT, tabT); L.sleep(zL, aL, bL, xL); T.sleep(zT, aT, bT, xT)
        if arm == 'own_cause': L.practice(zL, aL, bL, xL, tabL); T.practice(zT, aT, bT, xT, tabT)
        elif arm == 'own_read': L.practice(*L.read(xL, rng), xL, tabL); T.practice(*T.read(xT, rT), xT, tabT)
        elif arm == 'cross_read': L.practice(*L.read(xT, rng), xT, tabL); T.practice(*T.read(xL, rT), xL, tabT)
        else: T.practice(zT, aT, bT, xT, tabT); L.practice(*L.read(xT, rng), xT, tabL)
    return np.array(out)


def one(job):
    n, seed, eta, U, rounds = job; c = Deep(seed); rng = np.random.default_rng(210_000 + seed); S = c.teacher(rng, n); val = rng.random(n) < 0.15; R, RT = reader(c, S, val, 220_000 + seed), reader(c, S, val, 240_000 + seed)
    return job, np.stack([run(c, R, RT, arm, 250_000 + seed, eta, U, rounds) for arm in ARMS])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--ns', default='4096'); ap.add_argument('--U', type=int, default=4); ap.add_argument('--etas', default='0.1,0.003,0.1'); ap.add_argument('--rounds', type=int, default=2048)
    args = ap.parse_args(); ns = [int(x) for x in args.ns.split(',')]; eta = tuple(float(x) for x in args.etas.split(',')); jobs = [(n, s, eta, args.U, args.rounds) for n in ns for s in range(args.worlds)]; res = np.zeros((len(ns), args.worlds, len(ARMS), len(checks(args.rounds)), len(KEYS)))
    with Pool(args.procs) as pool:
        for (n, s, *_), r in tqdm(pool.imap_unordered(one, jobs), total=len(jobs)): res[ns.index(n), s] = r
    np.savez(f'{OUT}/cut_twin.npz', res=res, ns=np.array(ns), checks=checks(args.rounds), arms=ARMS, keys=KEYS, etas=np.array(eta)); print(f'{OUT}/cut_twin.npz')
