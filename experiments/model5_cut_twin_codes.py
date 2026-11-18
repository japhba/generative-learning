"""Model 5, check for model5_cut_twin.py: which deep code two finished learners hold. Reruns the cross_read arm (4,096 teacher strings, 16 worlds) and prints, for both learners, how much the top unit says
about the value at the first and at the second cut in the learner's own samples, and how much of the top unit of its own and of the twin's samples the first learner's reader recovers.
Writes OUT/cut_twin_codes.json.   Usage: python model5_cut_twin_codes.py"""

import os

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool
import numpy as np
from model5_cut_depth import BATCH, Deep, Stack, reader
from model5_cut_twin import reach
from model5_lateral_verdict import draw

def info(a, b):
    J = np.zeros((4, 4)); np.add.at(J, (a, b), 1 / len(a)); return float((J * (np.log(J + 1e-300) - np.log(J.sum(1, keepdims=True) * J.sum(0) + 1e-300))).sum() / np.log(4))

def code(L, c, rng, n=8192):                                # what the top unit says about the first and about the second cut's value in the learner's own samples
    TA, TB, p1, p2 = L.tables(); z = rng.integers(L.Z, size=n); uA, uB = draw(TA[z], rng), draw(TB[z], rng); return info(z, c.fR[draw(p1[uA], rng)]), info(z, c.fR[draw(p1[uB], rng)])

def one(seed):
    n, eta, rounds = 4096, (0.1, 0.003, 0.1), 2048; c = Deep(seed); rng0 = np.random.default_rng(210_000 + seed); S = c.teacher(rng0, n); val = rng0.random(n) < 0.15; R, RT = reader(c, S, val, 220_000 + seed), reader(c, S, val, 240_000 + seed)
    rng, rT, rE = np.random.default_rng(250_000 + seed), np.random.default_rng(250_001 + seed), np.random.default_rng(7); L, T = Stack(R, 4, c.Z, rng, eta), Stack(RT, 4, c.Z, rT, eta)
    for t in range(rounds):                                 # the cross_read arm of model5_cut_twin.py
        L.wake(c.cuts(rng, BATCH), rng); T.wake(c.cuts(rT, BATCH), rT); tabL, tabT = L.tables(), T.tables(); zL, aL, bL, xL = L.sample(rng, tabL); zT, aT, bT, xT = T.sample(rT, tabT); L.sleep(zL, aL, bL, xL); T.sleep(zT, aT, bT, xT)
        L.practice(*L.read(xT, rng), xT, tabL); T.practice(*T.read(xL, rT), xL, tabT)
    return seed, code(L, c, rE), code(T, c, rE), reach(L, L, rE)[1], reach(L, T, rE)[1], L.score(c)[1:3], T.score(c)[1:3]

if __name__ == '__main__':
    with Pool(16) as pool: out = pool.map(one, range(16))
    print('world | first learner: top unit about value 1, value 2 | twin: same | reader recovers level 2 of own, of twin | written right (level 1, 2): first, twin')
    import json; from model5_lateral_verdict import OUT
    done = [(cl, ct, ro, rt) for s, cl, ct, ro, rt, sl, st in out if min(*sl, *st) > 0.9]; diff = [(cl[0] > cl[1]) != (ct[0] > ct[1]) for cl, ct, _, _ in done]                                      # pairs in which both learners finished; do they hold different codes
    json.dump(dict(worlds=len(out), finished=len(done), different=int(sum(diff)), own_when_different=float(np.median([r[2] for r, x in zip(done, diff) if x])), twin_when_different=float(np.median([r[3] for r, x in zip(done, diff) if x])),
                   own_when_same=float(np.median([r[2] for r, x in zip(done, diff) if not x])), twin_when_same=float(np.median([r[3] for r, x in zip(done, diff) if not x])), top_about_own_value=float(np.median([max(cl) for cl, _, _, _ in done])), top_about_other_value=float(np.median([min(cl) for cl, _, _, _ in done]))), open(OUT / 'cut_twin_codes.json', 'w'), indent=1)
    for s, cl, ct, ro, rt, sl, st in out: print(f'{s:3d} | {cl[0]:.2f} {cl[1]:.2f} | {ct[0]:.2f} {ct[1]:.2f} | {ro:.2f} {rt:.2f} | {sl[0]:.2f} {sl[1]:.2f}  {st[0]:.2f} {st[1]:.2f}', '  same code' if (cl[0] > cl[1]) == (ct[0] > ct[1]) else '  different code')
