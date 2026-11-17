"""Model 5, the smallest world with an agreement between neighbours that are not siblings, with networks. Two blocks of model5_lateral_verdict.py side by side (codes k1 k2 | k3 k4, both
blocks from the same grammar) and one agreement: a two-valued feature of the class of k2 has to equal a two-valued feature of the class of k3. Teacher: uniform over the valid strings.
Question (the count of 17 Nov): how many teacher strings until the reader can CHECK the agreement, and how many until teacher strings can TEACH it to a writer that takes its units from the reader.
  block net   the local-rule network of model5_network_practice.local_net on the sibling pairs (k1, k2) and (k3, k4), pooled
  cut net     the same network on the pairs (k2, k3)
  judges      of a pair of codes. own: the network gives the pair more probability than the product of its marginals. pooled: teacher pairs spread over hidden-layer neighbours (section 8) > 0
  units       recruited from the hidden vectors as in section 11 (a code joins the unit with the most similar mean vector if the cosine is >= the judge's bandwidth, otherwise it founds a unit):
              block units per side from the block net, the cut unit from the cut net's vectors of k2. A code never shown founds a unit of its own.
  writer      p(x) = sum_u pi_u  T1_u[a, a'] T2_u[b, b'] / sizes: the cut unit u feeds both blocks; a, a', b, b' are the units of the four codes; uniform within a unit. The unit of k1 and k4 is
              the block unit; the unit of k2 and k3, which are in both networks, is the pair (block unit, unit in the cut net). The tables are the counts of the teacher strings read by the
              reader into these units (what wake steps converge to as the step shrinks), with the best uniform smoothing as in class_table.
Cut unit: recruited as above (cut, all), or one unit with two values, the cut net's hidden vectors in two groups by Ward linkage (cut2, all2).
Writers: none (true classes, no cut unit)   cut2, cut (true classes, cut unit from the cut net)   true (true classes, true feature)   all2, all (block units and cut unit from the reader)   blocks (block units from the reader, no cut unit)
Scores: population loss KL(p* || p), share of written strings whose two blocks agree, share of written strings that are valid.
Writes OUT/cut_check.npz and figures/model5-cut-check.{svg,png}.   Usage: python model5_cut_check.py [--worlds 16] [--procs 16] [--plot]"""

import argparse
import os
from multiprocessing import Pool
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from tqdm.auto import tqdm

from model5_lateral_verdict import OUT, World
from model5_network_baseline import groups
from model5_network_practice import bandwidth, evidence, local_net
from model5_network_units import normed, recruit
from model5_own_benefit import ALPHAS

RHO = 2; NS = (128, 256, 512, 1024, 2048, 4096, 8192); WRITERS = ('none', 'cut2', 'cut', 'true', 'all2', 'all', 'blocks'); SCORES = ('loss', 'agree', 'valid')
AUX = ('block_own_acc', 'block_own_ref', 'block_pool_acc', 'block_pool_ref', 'cut_own_acc', 'cut_own_ref', 'cut_pool_acc', 'cut_pool_ref', 'cut_purity', 'cut_units', 'left_purity', 'right_purity', 'left_units', 'right_units', 'cut2_purity')


class Cut:
    def __init__(self, seed):
        self.w = w = World(seed=seed); rng = np.random.default_rng(1000 + seed); half = np.repeat(np.arange(RHO), w.v // RHO)
        self.fR, self.fL = rng.permutation(half)[w.cls], rng.permutation(half)[w.cls]                                              # the feature of a code as k2, as k3
        self.agree = self.fR[:, None] == self.fL[None, :]; self.pairs = np.argwhere(w.valid); self.ok = self.agree[np.ix_(self.pairs[:, 1], self.pairs[:, 0])]   # ok[i, j]: valid pair i as block 1 fits valid pair j as block 2

    def teacher(self, rng, n):                                                                                                     # two valid blocks, kept if they agree: uniform over the valid strings
        S = np.zeros((0, 4), int)
        while len(S) < n: a, b = np.stack(self.w.teacher(rng, 2 * n), 1), np.stack(self.w.teacher(rng, 2 * n), 1); S = np.concatenate([S, np.concatenate([a, b], 1)[self.agree[a[:, 1], b[:, 0]]]])
        return S[:n]


def units(X, shown, tau):                                                                                                          # code -> unit, codes in the order of their first appearance
    u = recruit(X, shown[np.sort(np.unique(shown, return_index=True)[1])], tau); miss = u < 0; u[miss] = u.max() + 1 + np.arange(miss.sum()); return u


def purity(u, f): return sum(np.bincount(f[u == j]).max() for j in range(u.max() + 1)) / len(u)                                     # share of the codes that hold their unit's most frequent label


def judged(valid, net, Q, Qv):                                                                                                     # shares of valid pairs accepted and of invalid pairs refused: the network's own answer, the pooled counts
    tau = bandwidth(net, Q, Qv); pE = evidence(net, Q + Qv, tau); own = net.logp() > net.P[2] + net.P[3]
    return [own[valid].mean(), (~own)[~valid].mean(), (pE > 0)[valid].mean(), (pE == 0)[~valid].mean()], tau


def join(a, b): return np.unique(a * (b.max() + 1) + b, return_inverse=True)[1]                                                       # a code's unit in both networks: the pair of its two units


def score(c, S, U, uc):                                                                                                            # the writer whose tables are the teacher strings counted in the units (U: code -> block unit at the four places; uc: k2 -> value of the cut unit)
    u = uc[S[:, 1]]; nu = uc.max() + 1; n_u = np.bincount(u, minlength=nu); best = (np.inf,); C = [np.zeros((nu, U[2 * b].max() + 1, U[2 * b + 1].max() + 1)) for b in (0, 1)]
    size = [np.outer(np.bincount(U[2 * b])[U[2 * b]], np.bincount(U[2 * b + 1])[U[2 * b + 1]]) for b in (0, 1)]
    for b in (0, 1): np.add.at(C[b], (u, U[2 * b][S[:, 2 * b]], U[2 * b + 1][S[:, 2 * b + 1]]), 1)
    for al in ALPHAS:
        pi = (n_u + al) / (n_u + al).sum(); P1, P2 = (((C[b] + al) / (C[b] + al).sum((1, 2), keepdims=True))[:, U[2 * b][:, None], U[2 * b + 1][None, :]] / size[b] for b in (0, 1))   # (cut unit, left code, right code)
        V = np.einsum('u,ui,uj->ij', pi, P1[:, c.pairs[:, 0], c.pairs[:, 1]], P2[:, c.pairs[:, 0], c.pairs[:, 1]]); r = (float(-np.log(c.ok.sum()) - np.log(V[c.ok] + 1e-300).mean()), float(np.einsum('u,ui,ij,uj->', pi, P1.sum(1), c.agree, P2.sum(2))), float(V[c.ok].sum()))
        if r[0] < best[0]: best = r
    return best


def one(job):
    n, seed = job; c = Cut(seed); w = c.w; rng = np.random.default_rng(110_000 + seed); S = c.teacher(rng, n); val = rng.random(n) < 0.15; cw = SimpleNamespace(N=w.N, v=RHO, valid=c.agree, pstar=c.agree / c.agree.sum())
    left, right = np.concatenate([S[:, 0], S[:, 2]]), np.concatenate([S[:, 1], S[:, 3]]); netB, _, _, QB, QvB, _ = local_net(w, left, right, np.concatenate([val, val]), np.random.default_rng(120_000 + seed))
    netC, _, _, QC, QvC, _ = local_net(cw, S[:, 1], S[:, 2], val, np.random.default_rng(130_000 + seed)); jB, tauB = judged(w.valid, netB, QB, QvB); jC, tauC = judged(c.agree, netC, QC, QvC)
    uL, uR, zero = units(normed(netB.P[0]), left, tauB), units(normed(netB.P[1]), right, tauB), np.zeros(w.N, int); ucR, ucL = units(normed(netC.P[0]), S[:, 1], tauC), units(normed(netC.P[1]), S[:, 2], tauC)
    c2R, c2L = groups(netC.P[0], RHO), groups(netC.P[1], RHO)                                                                    # the cut unit as one unit with RHO values: the hidden vectors of the cut net in RHO groups
    W = dict(none=((w.cls,) * 4, zero), cut2=((w.cls, join(w.cls, c2R), join(w.cls, c2L), w.cls), c2R), cut=((w.cls, join(w.cls, ucR), join(w.cls, ucL), w.cls), ucR), true=((w.cls,) * 4, c.fR),
             all2=((uL, join(uR, c2R), join(uL, c2L), uR), c2R), all=((uL, join(uR, ucR), join(uL, ucL), uR), ucR), blocks=((uL, uR, uL, uR), zero))
    return job, np.array([score(c, S, *W[k]) for k in WRITERS]), np.array(jB + jC + [purity(ucR, c.fR), ucR.max() + 1, purity(uL, w.cls), purity(uR, w.cls), uL.max() + 1, uR.max() + 1, purity(c2R, c.fR)])


def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = np.load(f'{OUT}/cut_check.npz'); res, aux, ns = np.median(d['res'], 1), np.median(d['aux'], 1), d['ns']; A = dict(zip(AUX, aux.T)); R = {k: res[:, i] for i, k in enumerate(WRITERS)}
    fig, ax = plt.subplots(1, 3, figsize=plt.figaspect(0.36), constrained_layout=True)
    for key, lab, ls in (('cut_pool_ref', 'refuses pairs that do not agree', '-'), ('cut_pool_acc', 'accepts pairs that agree', '--')): ax[0].plot(ns, A[key], ls, color='C0', marker='o', label=f'judge of the agreement {lab}')
    for key, lab, ls in (('block_pool_ref', 'refuses invalid pairs', '-'), ('block_pool_acc', 'accepts valid pairs', '--')): ax[0].plot(ns, A[key], ls, color='C1', marker='s', label=f'judge of a block {lab}')
    ax[0].set_ylabel('share of the pairs of codes'); ax[0].set_title('What the reader can check')
    for k, lab in (('none', 'no unit for the agreement'), ('cut2', 'unit taken from the judge of the agreement'), ('all2', 'the same, block units taken from the reader too')): ax[1].plot(ns, R[k][:, 1], marker='o', label=lab)
    ax[1].set_ylabel('share of written strings whose blocks agree'); ax[1].set_title('What teacher strings teach the writer')
    for k, lab in (('none', 'true classes, no unit for the agreement'), ('cut2', 'true classes, unit from the judge'), ('true', 'true classes, true feature'), ('all2', 'all units from the reader'), ('blocks', 'block units from the reader, no unit for the agreement')): ax[2].plot(ns, R[k][:, 0], marker='o', label=lab)
    ax[2].set_yscale('log'); ax[2].set_ylabel('population loss (nats)'); ax[2].set_title('Loss of the writer')
    for a in ax: a.set_xscale('log', base=2); a.set_xlabel('teacher strings'); a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False)
    os.makedirs(f'{OUT}/figures', exist_ok=True); [fig.savefig(f'{OUT}/figures/model5-cut-check.{e}') for e in ('svg', 'png')]; print(f'{OUT}/figures/model5-cut-check.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--plot', action='store_true'); args = ap.parse_args()
    if not args.plot:
        jobs = [(n, s) for n in NS for s in range(args.worlds)]; res, aux = np.zeros((len(NS), args.worlds, len(WRITERS), len(SCORES))), np.zeros((len(NS), args.worlds, len(AUX)))
        with Pool(args.procs) as pool:
            for (n, s), r, a in tqdm(pool.imap_unordered(one, jobs), total=len(jobs)): res[NS.index(n), s], aux[NS.index(n), s] = r, a
        np.savez(f'{OUT}/cut_check.npz', res=res, aux=aux, ns=np.array(NS), writers=WRITERS, scores=SCORES, keys=AUX)
    d = np.load(f'{OUT}/cut_check.npz'); res, aux = np.median(d['res'], 1), np.median(d['aux'], 1); np.set_printoptions(precision=3, suppress=True, linewidth=250)
    print('teacher strings', list(d['ns']))
    for i, k in enumerate(AUX): print(f'{k:16s}', aux[:, i])
    for j, s in enumerate(SCORES):
        for i, k in enumerate(WRITERS): print(f'{s:6s} {k:7s}', res[:, i, j])
    plot()
