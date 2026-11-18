"""Model 5, reader and writer in sync: what is left of practice's lead over wake and sleep, and of own samples' lead over a twin's, when the reader names the writer's units as well as can be?
World, learner, rules and steps of model5_cut_depth.py (--top table). Two learners with the same stored teacher strings and other random starts of every part (model5_cut_twin.py).
Every round each learner reads one batch of fresh teacher strings (wake), writes one batch of own samples and, in the arms with practice, takes one judged batch.
How the reader is kept in sync with the writer:
  exact     no naming layers: wake and credit draw the units from the writer's own posterior. This is what sleep converges to when it is run to the end after every step of the writer
            (the naming layers can represent the posterior), and it can be computed here because the writer has 4 x 4 x 4 states.
  sleep s   the naming layers of the note, trained on the own batch at step s (0.003 is the note's step; larger steps follow the writer more closely).
Arms:
  wake        no judged batch. With naming layers this is classic wake-sleep (2 samples a round); with the exact posterior it uses teacher strings only (1 sample a round).
  own_cause   the own batch is judged, each verdict credited to the units that wrote the sample
  own_read    the own batch is judged, credited to the units the reader names (exact: drawn from the posterior; in distribution this is own_cause)
  twin_read   the other learner's batch is judged, credited by reading; the other learner does the same with this learner's batch, so the two are at the same stage throughout
Scores of the first learner: loss, valid1, valid2, read1, read2 as in model5_cut_depth.py; valid1, valid2 of the second; and, with naming layers, whether reader and writer agree on the
units of a string (model5_cut_mechanism.agree; 0: chance, 1: always) at level 1 and level 2, for own samples, fresh teacher strings and the other learner's samples.
--decay T: the wake step is eta until round T and eta x T / round after it (does a wake step that shrinks close the gap that the noise of a constant step leaves?).
Writes OUT/cut_sync[_tag].npz.   Usage: python model5_cut_sync.py [--worlds 16] [--procs 16] [--n 4096] [--rounds 2048] [--sleeps 0.003,0.0125,0.05,0.2] [--etas 0.1,0.1] [--decay 0] [--tag x]"""

import argparse
import os
from multiprocessing import Pool

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from tqdm.auto import tqdm

from model5_cut_depth import BATCH, Deep, Stack, reader
from model5_cut_mechanism import agree, many
from model5_cut_practice import checks
from model5_lateral_verdict import OUT

ARMS = ('wake', 'own_cause', 'own_read', 'twin_read'); U = 4
KEYS = ('loss', 'valid1', 'valid2', 'read1', 'read2', 'twin_valid1', 'twin_valid2', 'agree1_own', 'agree2_own', 'agree1_teacher', 'agree2_teacher', 'agree1_twin', 'agree2_twin')


def run(c, R, RT, sync, arm, seed, eta, rounds, decay=0):
    rng, rT, rE = np.random.default_rng(seed), np.random.default_rng(seed + 1), np.random.default_rng(seed + 2); ex = sync == 'exact'; e = (eta[0], 0.0 if ex else sync, eta[1]); L, T = Stack(R, U, c.Z, rng, e), Stack(RT, U, c.Z, rT, e)
    name = lambda X, x, tab, r: X.posterior(x, tab, r) if ex else X.read(x, r); cs = set(checks(rounds)); out = []
    for t in range(rounds + 1):
        if t in cs: out.append([*L.score(c), *T.score(c)[1:3], *([np.nan] * 6 if ex else [*agree(L, many(L, rE)), *agree(L, c.cuts(rE, 4096)), *agree(L, many(T, rE))])])
        if decay: L.eta = T.eta = (eta[0] * min(1.0, decay / max(t, 1)), e[1], e[2])                                                 # wake step eta, then eta x decay / round
        L.wake(c.cuts(rng, BATCH), rng, ex); T.wake(c.cuts(rT, BATCH), rT, ex)
        if ex and arm == 'wake': continue
        tabL, tabT = L.tables(), T.tables(); zL, aL, bL, xL = L.sample(rng, tabL); zT, aT, bT, xT = T.sample(rT, tabT)
        if not ex: L.sleep(zL, aL, bL, xL); T.sleep(zT, aT, bT, xT)
        if arm == 'own_cause': L.practice(zL, aL, bL, xL, tabL); T.practice(zT, aT, bT, xT, tabT)
        elif arm == 'own_read': L.practice(*name(L, xL, tabL, rng), xL, tabL); T.practice(*name(T, xT, tabT, rT), xT, tabT)
        elif arm == 'twin_read': L.practice(*name(L, xT, tabL, rng), xT, tabL); T.practice(*name(T, xL, tabT, rT), xL, tabT)
    return np.array(out)


def one(job):
    n, seed, syncs, eta, rounds, decay = job; c = Deep(seed); rng = np.random.default_rng(210_000 + seed); S = c.teacher(rng, n); val = rng.random(n) < 0.15; R, RT = reader(c, S, val, 220_000 + seed), reader(c, S, val, 240_000 + seed)
    return seed, np.stack([[run(c, R, RT, s, arm, 250_000 + seed, eta, rounds, decay) for arm in ARMS] for s in syncs])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--n', type=int, default=4096); ap.add_argument('--rounds', type=int, default=2048)
    ap.add_argument('--sleeps', default='0.003,0.0125,0.05,0.2'); ap.add_argument('--etas', default='0.1,0.1'); ap.add_argument('--decay', type=int, default=0); ap.add_argument('--tag', default=''); args = ap.parse_args()
    syncs = ('exact', *(float(s) for s in args.sleeps.split(',') if s)); eta = tuple(float(x) for x in args.etas.split(',')); res = np.zeros((args.worlds, len(syncs), len(ARMS), len(checks(args.rounds)), len(KEYS)))
    with Pool(args.procs) as pool:
        for s, r in tqdm(pool.imap_unordered(one, [(args.n, s, syncs, eta, args.rounds, args.decay) for s in range(args.worlds)]), total=args.worlds): res[s] = r
    name = f"{OUT}/cut_sync{'_' + args.tag if args.tag else ''}.npz"; uses = np.array([[1 if s == 'exact' and a == 'wake' else 2 for a in ARMS] for s in syncs])
    np.savez(name, res=res, checks=checks(args.rounds), syncs=np.array([str(s) for s in syncs]), arms=ARMS, keys=KEYS, uses=uses, etas=np.array(eta), n=args.n, decay=args.decay); print(name)
