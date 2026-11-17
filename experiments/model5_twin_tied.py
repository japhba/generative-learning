"""Model 5: does a reader that is aligned to its own writer give own samples an advantage over a twin's? One block of model5_lateral_verdict.py.
Both learners here read through the transpose of their writer, which is what a shared start followed by a completed sleep stage gives; credit is by that reading, never by the code that wrote the sample.
  gradient   GradientLearner, 16 units from a random start (weak: noise 0.1; sharp: one random code per unit at 6), optional threshold on the reader's drive.
             A stored teacher set (shown to learner and twin alike, replayed in wake batches) alternates with practice batches: none, own samples, or the samples of a twin
             (same teacher pairs and wake batches, another random start, practising on its own samples). Reach: share of samples whose reading has drive > 1 under the learner's reader.
  grower     Grower (units start as copies of the codes, no random start): practice on own samples, on those of a twin with the same teacher pairs, and of a twin with its own.
Writes /var/tmp/jbauer/model5_lateral_verdict/twin_tied.npz.   Usage: python model5_twin_tied.py [--worlds 16] [--procs 28]"""

import argparse
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm

from model5_lateral_verdict import BATCH, CHECK, OUT, GradientLearner, Grower, World, practise

N_TEACHER, STEPS, MARKS = 8192, 10000, (0, 100, 300, 1000, 3000, 10000)
STARTS = dict(weak=('noise', 0.1), sharp=('random', 6.0)); ARMS = ('none', 'own', 'twin')


def reach(L, k, kp, rng): return float((L.read(k, kp, L.tables(), rng)[2] > 1).mean())                                             # share of the pairs that the learner's reader explains above chance


def one_gradient(job):
    start, theta, arm, seed = job; w = World(seed=seed); rng = np.random.default_rng(40_000 + seed); k, kp = w.teacher(rng, N_TEACHER); kind, s = STARTS[start]
    L, T = (GradientLearner(w, K=w.v, start=kind, s=s, theta=theta, lam=0, even=True, seed=sd) for sd in (seed, 1000 + seed))
    for G in (L, T): G.watch(k, kp, rng, wake=False)                                                                                # the pair counts M of the stored teacher set, the same for both
    out = []
    for t in range(STEPS + 1):
        if t in MARKS: sc = L.score(); out.append([sc['kl'], sc['valid'], sc['purity'], reach(L, *L.sample(1024, rng)[:2], rng), reach(L, *T.sample(1024, rng)[:2], rng), reach(L, *w.teacher(rng, 1024), rng), T.score()['kl']])
        i = rng.integers(N_TEACHER, size=BATCH); L.watch(k[i], kp[i], rng); T.watch(k[i], kp[i], rng); T.practice(BATCH, rng)       # the same wake batch for both; the twin always practises on its own samples
        if arm == 'own': L.practice(BATCH, rng)
        elif arm == 'twin': L.practice(BATCH, rng, source=T.sample(BATCH, rng)[:2])
    return job, np.array(out), [L.n['el_own'] / max(L.n['own'], 1), L.n['el_teacher'] / max(L.n['teacher'] - N_TEACHER, 1)]


def one_grower(job):
    src, seed = job; w = World(seed=seed); rng = np.random.default_rng(41_000 + seed); k, kp = w.teacher(rng, 4096); L, T = Grower(w), Grower(w); k2, kp2 = (k, kp) if src != 'twin_other' else w.teacher(rng, 4096)
    for i in range(0, 4096, 256): L.watch(k[i:i + 256], kp[i:i + 256], rng); T.watch(k2[i:i + 256], kp2[i:i + 256], rng)
    def foreign(): T.practice(BATCH, rng); return T.sample(BATCH, rng)
    sc = practise(L, rng, 32768, None if src == 'own' else foreign); return job, [[x['kl'], x['valid']] for x in sc], T.score()['kl']


def run(job): return one_gradient(job[1:]) if job[0] == 'gradient' else one_grower(job[1:])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=28); args = ap.parse_args(); S = range(args.worlds)
    gj = [('gradient', st, th, arm, s) for st in STARTS for th in (0, 1) for arm in ARMS for s in S]; wj = [('grower', src, s) for src in ('own', 'twin_same', 'twin_other') for s in S]; res = {}
    with Pool(args.procs) as p:
        for job, *out in tqdm(p.imap_unordered(run, wj + gj), total=len(gj) + len(wj)): res[job] = out
    g = np.array([[[[res[st, th, arm, s][0] for s in S] for arm in ARMS] for th in (0, 1)] for st in STARTS]); el = np.array([[[[res[st, th, arm, s][1] for s in S] for arm in ARMS] for th in (0, 1)] for st in STARTS])
    gr = np.array([[res[src, s][0] for s in S] for src in ('own', 'twin_same', 'twin_other')]); tk = np.array([[res[src, s][1] for s in S] for src in ('own', 'twin_same', 'twin_other')])
    np.savez(OUT / 'twin_tied.npz', gradient=g, eligible=el, marks=MARKS, starts=list(STARTS), arms=ARMS, keys=['kl', 'valid', 'purity', 'reach_own', 'reach_twin', 'reach_teacher', 'twin_kl'], grower=gr, grower_twin_kl=tk, check=CHECK[CHECK <= 32768] * BATCH)
    print('Grower, loss after practice (median, worlds own < twin):', {src: round(float(np.median(gr[i, :, -1, 0])), 3) for i, src in enumerate(('own', 'twin_same', 'twin_other'))}, [int((gr[0, :, -1, 0] < gr[i, :, -1, 0]).sum()) for i in (1, 2)], 'twins themselves', np.round(np.median(tk, 1), 3))
    for a, st in enumerate(STARTS):
        for b, th in enumerate((0, 1)):
            m = np.median(g[a, b], 1); print(f'{st} start, threshold {th}: reach before practice own / twin / teacher', np.round(m[1, 0, 3:6], 3), '| at the end (own arm)', np.round(m[1, -1, 3:6], 3))
            print('   loss by mark', {arm: np.round(m[i, :, 0], 3).tolist() for i, arm in enumerate(ARMS)}, '| valid share at the end', {arm: round(float(m[i, -1, 1]), 3) for i, arm in enumerate(ARMS)}, '| own < twin in', int((g[a, b, 1, :, -1, 0] < g[a, b, 2, :, -1, 0]).sum()), 'of', args.worlds,
                  '| practised samples that pass the threshold (own, twin)', np.round(np.median(el[a, b, 1:, :, 0], 1), 3))
