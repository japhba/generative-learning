"""Model 5 with networks, three levels: strings of four codes (k1 k2 | k3 k4) of model5_depth.World3, 262,144 valid strings of 16.8 million.
A string is a pair of blocks (L, R), L = 64 k1 + k2, R = 64 k3 + k4. Network (two levels of the network of model5_network_practice, separate weights per half h):
  level 1   hidden vectors u_k, v_k' (16 units) of the two codes of a block, block score s_h(k, k') = u_k . v_k' + a_k + b_k'
  level 2   the hidden vector of a block, e_h(k, k') = A_h^T (u_k (x) v_k') (16 units; the outer product of the two level-1 hidden vectors, each scaled to unit length, through a weight tensor A_h)
  writes    p(L, R) = softmax_{LR}( s_0(L) + s_1(R) + e_0(L) . e_1(R) )
Local rule (watching): at each level a teacher pair raises the product of the two hidden vectors and the same left item with a right item from another teacher string lowers it
  (level 1: the two codes of a block; level 2: the two blocks of a string); only the weights of the active items change, and level 2 does not change level 1.
Backpropagation (reference): full-batch Adam on the likelihood of the teacher strings, all weights, level-2 input not scaled.
Read out: codes grouped by their level-1 hidden vectors (16 classes per position), the class pairs shown by the teacher grouped by their mean level-2 vector (16 symbols per side), pairs of symbols counted.
Judge: reader network (own random start) and the stored teacher strings. A block is accepted if a neighbouring block (level-1 hidden vectors, cosine >= tau1) was shown; accepted blocks are neighbours
  at level 2 if their level-2 vectors have cosine >= tau2; p_E = K_0 M K_1^T / n, the teacher strings spread over neighbouring strings (bandwidths by the likelihood of held-out teacher strings).
Practice: the writer network (own random start) proposes strings; for each the judge answers; one local delta step per level, nothing is kept:
  level 1   log-probability of the block among all blocks, log softmax_L s_h(L), towards the log of the judge's answer for the block, sum_R p_E(L, R) (only the hidden vectors and biases of its two codes move)
  level 2   e_0(L) . e_1(R) + b2 towards the judge's log p_E(L, R) - log p_E(L) - log p_E(R) (only A_0, A_1 and the offset b2 move; A_h by error x the block's input x the other block's hidden vector)
  a refused block or string is moved down to FLOOR times the mean positive answer, never up.
Proposal sources: own; twin (same teacher strings, other random starts, practising at the same pace); shallow (the writer's two block factors, combined at random); uniform;
  judge (strings drawn from p_E itself: possible here because the blocks can be listed; no writer is needed to propose); mix, mixtwin (every other batch from p_E, the rest own or the twin's).
Runs: watching, saddle, curves. Practice is run from networks trained by the local rule only: the step per level assumes the form that rule gives (block scores that are block marginals,
  a level-2 product that is the log-ratio up to an offset) and breaks a network trained by backpropagation (pilot on two worlds: loss 0.33 to above 2 nats).   Writes /var/tmp/jbauer/model5_lateral_verdict/network_depth_<run>.npz.   Usage: python model5_network_depth.py <run> [--worlds 16] [--procs 28]"""

import argparse
import os
from multiprocessing import Pool

os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'                                                            # one thread per worker: the 4,096 x 4,096 products would otherwise oversubscribe the cores

import numpy as np
from scipy.special import expit, log_softmax, softmax
from tqdm.auto import tqdm

from model5_depth import World3
from model5_lateral_verdict import OUT
from model5_network_baseline import groups
from model5_network_practice import neighbours

D, INIT, BATCH, ETA, FLOOR, EPOCHS, NEGS, LR, STEPS, DECAYS = 16, 0.1, 64, 0.5, 1e-3, 60, (4,), 0.02, 300, (1e-4, 1e-3)
TAUS = np.arange(0.3, 0.96, 0.05); CHECK = BATCH * 2 ** np.arange(4, 9); SKIP = (0, 11, 29, 46, 57)                                # own strings at which practice is scored: 1,024 .. 16,384; worlds with two classes of the same partners are left out
EPS = 10.0 ** np.arange(-8, -0.4, 0.5)


class Strings:                                                                                                                     # the three-level world in block coordinates
    def __init__(self, seed):
        self.W = W = World3(seed); self.w = w = W.w; N = self.N = w.N; rule = -np.ones((w.v, w.v), int); rule[w.rules[:, 0], w.rules[:, 1]] = np.arange(len(w.rules)); symr = np.empty(len(w.rules), int); symr[W.prods.ravel()] = np.repeat(np.arange(w.v), w.m)
        r = rule[w.cls[:, None], w.cls[None, :]].ravel(); self.sym = np.where(r >= 0, symr[r], -1); self.ok = self.sym >= 0                                    # symbol of every block (-1: not a valid block)
        B = np.zeros((w.v, w.v), bool); B[W.top[:, 0], W.top[:, 1]] = True; self.valid = np.zeros((N * N, N * N), bool); self.valid[np.ix_(self.ok, self.ok)] = B[np.ix_(self.sym[self.ok], self.sym[self.ok])]
        self.idx = np.flatnonzero(self.valid.ravel()); self.vL, self.vR = self.idx // N ** 2, self.idx % N ** 2

    def teacher(self, rng, n): x = self.W.strings(rng, n); return x[:, 0] * self.N + x[:, 1], x[:, 2] * self.N + x[:, 3]

    def kl(self, lp): return float(-np.log(len(self.idx)) - lp.ravel()[self.idx].mean(dtype=np.float64))                             # lp: log-probabilities of all strings (4,096 x 4,096)

    def mixed(self, pv):                                                                                                            # loss of a distribution given by its values pv at the valid strings, mixed with the uniform one (best mixing weight)
        return min(float(-np.log(len(self.idx)) - np.log((1 - e) * pv + e / self.N ** 4).mean()) for e in EPS)

    def scores(self, lp, P):                                                                                                        # loss, valid share, valid strings hardly written, share of valid blocks (two halves), valid share among strings of valid blocks
        pv = P.ravel()[self.idx]; inner = float(P[np.ix_(self.ok, self.ok)].sum(dtype=np.float64)) + 1e-300; va = float(pv.sum(dtype=np.float64))
        return [self.kl(lp), va, float((pv < 0.25 / len(self.idx)).mean()), float(P.sum(1, dtype=np.float64)[self.ok].sum()), float(P.sum(0, dtype=np.float64)[self.ok].sum()), va / inner]


class Stack:
    def __init__(self, N, rng, init=INIT, scaled=True): self.N, self.scaled = N, scaled; self.U, self.V = init * rng.standard_normal((2, N, D)), init * rng.standard_normal((2, N, D)); self.A = init * rng.standard_normal((2, D * D, D)); self.a, self.b = np.zeros((2, N)), np.zeros((2, N)); self.b2 = 0.0   # b2: offset between e_0 . e_1 and log p(L, R) - log p(L) - log p(R)

    def copy(self): c = Stack.__new__(Stack); c.N, c.scaled = self.N, self.scaled; c.U, c.V, c.A, c.a, c.b = (x.copy() for x in (self.U, self.V, self.A, self.a, self.b)); c.b2 = self.b2; return c

    def feat(self, h, k=None, kp=None):                                                                                             # level-2 input of blocks (all, or those of the codes k, kp): outer product of the two level-1 hidden vectors
        U, V = self.U[h], self.V[h]
        if self.scaled: U, V = U / np.linalg.norm(U, axis=1, keepdims=True), V / np.linalg.norm(V, axis=1, keepdims=True)
        return (U[:, None, :, None] * V[None, :, None, :]).reshape(self.N ** 2, D * D) if k is None else (U[k][:, :, None] * V[kp][:, None, :]).reshape(len(k), D * D)

    def emb(self, h): return self.feat(h) @ self.A[h]                                                                               # level-2 hidden vector of every block

    def block(self, h): return (self.U[h] @ self.V[h].T + self.a[h][:, None] + self.b[h][None, :]).ravel()                          # level-1 score of every block

    def table(self):                                                                                                                # log-probabilities and probabilities of all strings
        S = self.emb(0).astype(np.float32) @ self.emb(1).astype(np.float32).T; S += self.block(0).astype(np.float32)[:, None]; S += self.block(1).astype(np.float32)[None, :]
        m = float(S.max()); P = np.exp(S - np.float32(m)); Z = float(P.sum(dtype=np.float64)); S -= np.float32(m + np.log(Z)); P /= np.float32(Z); return S, P


def split(S, kL, kR, val):                                                                                                          # training strings as four code columns, and the flat index of the held-out strings
    N = S.N; tr = ~val; return [kL[tr] // N, kL[tr] % N, kR[tr] // N, kR[tr] % N], kL[val] * N ** 2 + kR[val]


def watch_local(S, kL, kR, val, rng, init=INIT, neg=4, epochs=EPOCHS, eta=0.05, decay=1e-4, curve=False):                           # the local rule at both levels; the network is kept at the pass with the lowest loss on the held-out strings
    N = S.N; net = Stack(N, rng, init); k, vi = split(S, kL, kR, val); n = len(k[0]); c = np.zeros(3); best, read = (np.inf,), (np.inf,); hist = []; kv = [kL[val] // N, kL[val] % N, kR[val] // N, kR[val] % N]; sh = np.random.default_rng(0).integers(len(kv[0]), size=(neg, len(kv[0])))
    for h in range(2): pl, pr = np.bincount(k[2 * h], minlength=N) + 1, np.bincount(k[2 * h + 1], minlength=N) + 1; net.a[h], net.b[h] = np.log(pl / pl.sum()), np.log(pr / pr.sum())   # biases: how often each code was shown
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            j = order[i:i + BATCH]; m = len(j); y = np.concatenate([np.ones(m), np.zeros(m * neg)])
            for h in range(2):                                                                                                      # level 1: the two codes of a block against the same left code with the right code of another string
                x, xp = np.concatenate([k[2 * h][j], np.repeat(k[2 * h][j], neg)]), np.concatenate([k[2 * h + 1][j], k[2 * h + 1][rng.integers(n, size=m * neg)]]); U, V = net.U[h], net.V[h]
                g = y - expit((U[x] * V[xp]).sum(1) + c[h]); dU, dV = np.zeros_like(U), np.zeros_like(V); np.add.at(dU, x, g[:, None] * V[xp]); np.add.at(dV, xp, g[:, None] * U[x]); U += eta * (dU - decay * U); V += eta * (dV - decay * V); c[h] += eta * g.mean()
            jl, jr = np.concatenate([j, np.repeat(j, neg)]), np.concatenate([j, rng.integers(n, size=m * neg)])                     # level 2: the two blocks of a string against the same left block with the right block of another string
            fl, fr = net.feat(0, k[0][jl], k[1][jl]), net.feat(1, k[2][jr], k[3][jr]); el, er = fl @ net.A[0], fr @ net.A[1]; g = y - expit((el * er).sum(1) + c[2])
            net.A[0] += eta * (fl.T @ (g[:, None] * er) - decay * net.A[0]); net.A[1] += eta * (fr.T @ (g[:, None] * el) - decay * net.A[1]); c[2] += eta * g.mean()
        net.b2 = float(c[2] + np.log(neg))                                                                                          # at the optimum of the contrast, e_0 . e_1 + c + log(neg) = log p(L, R) - log p(L) - log p(R)
        lp, P = net.table(); vl = float(-lp.ravel()[vi].mean(dtype=np.float64)); E = [net.emb(h) for h in range(2)]; el = E[0][kv[0] * N + kv[1]]                                # held-out strings: likelihood of what the network writes, and the level-2 contrast loss
        cl = float(-np.log(expit((el * E[1][kv[2] * N + kv[3]]).sum(1) + c[2])).mean() - sum(np.log(expit(-(el * E[1][kv[2][j] * N + kv[3][j]]).sum(1) - c[2])).mean() for j in sh))   # (held-out strings against the same strings with the right blocks shuffled)
        if curve: r = Readout(S, net, kL, kR); hist.append([r.right1, r.right2, r.loss, S.kl(lp), cl])
        if vl < best[0]: best = (vl, net.copy(), ep)
        if cl < read[0]: read = (cl, net.copy(), ep)
    return best[1], best[0], best[2], np.array(hist), read[1], read[2]                                                               # the writer: kept at the pass at which it writes the held-out strings best; the reader: at the pass at which it tells them best from shuffled ones


def best_local(S, data, rng, reader=False):                                                                                         # the number of contrast pairs by the held-out likelihood of what the network writes
    seed = rng.integers(1 << 30); return min((watch_local(S, *data, np.random.default_rng(seed), neg=neg) for neg in NEGS), key=lambda r: r[1])[4 if reader else 0]


def grads(net, G, decay):                                                                                                           # gradient with respect to U, V, A, a, b, from the gradient G with respect to the scores of all strings (level-2 input not scaled)
    N = net.N; E = [net.emb(h).astype(np.float32) for h in range(2)]; dE = [(G @ E[1]).astype(float), (G.T @ E[0]).astype(float)]; H = [G.sum(1, dtype=np.float64).reshape(N, N), G.sum(0, dtype=np.float64).reshape(N, N)]; g = [np.zeros_like(x) for x in (net.U, net.V, net.A, net.a, net.b)]
    for h in range(2):
        T = (dE[h] @ net.A[h].T).reshape(N, N, D, D); g[0][h] = H[h] @ net.V[h] + np.einsum('klab,lb->ka', T, net.V[h]) + decay * net.U[h]; g[1][h] = H[h].T @ net.U[h] + np.einsum('klab,ka->lb', T, net.U[h]) + decay * net.V[h]
        g[2][h] = net.feat(h).T @ dE[h] + decay * net.A[h]; g[3][h] = H[h].sum(1); g[4][h] = H[h].sum(0)
    return g


def watch_backprop(S, kL, kR, val, rng):                                                                                            # reference: the same network trained by backpropagation on the likelihood of the teacher strings (weight decay and stopping step by the held-out strings)
    N = S.N; k, vi = split(S, kL, kR, val); Q = np.zeros((N * N, N * N), np.float32); np.add.at(Q, (k[0] * N + k[1], k[2] * N + k[3]), 1); Q /= Q.sum(); best = (np.inf,); seed = rng.integers(1 << 30)
    for decay in DECAYS:
        net = Stack(N, np.random.default_rng(seed), scaled=False); par = [net.U, net.V, net.A, net.a, net.b]; mo = [np.zeros_like(x) for x in par]; ve = [np.zeros_like(x) for x in par]
        for t in range(1, STEPS + 1):
            lp, P = net.table()
            if t % 10 == 0:
                vl = float(-lp.ravel()[vi].mean(dtype=np.float64))
                if vl < best[0]: best = (vl, net.copy(), decay, t)
            P -= Q; g = grads(net, P, decay)
            for i in range(5): mo[i] = 0.9 * mo[i] + 0.1 * g[i]; ve[i] = 0.999 * ve[i] + 0.001 * g[i] ** 2; par[i] -= LR * (mo[i] / (1 - 0.9 ** t)) / (np.sqrt(ve[i] / (1 - 0.999 ** t)) + 1e-8)
    return best[1], best[2], best[3]


class Readout:                                                                                                                      # what the hidden layers know, read out without the writer: classes, symbols, counted pairs of symbols
    def __init__(self, S, net, kL, kR, true=False):
        N, v = S.N, S.w.v; self.g = g = [S.w.cls] * 4 if true else [groups(X, v) for X in (net.U[0], net.V[0], net.U[1], net.V[1])]; cp = [(g[0][:, None] * v + g[1][None, :]).ravel(), (g[2][:, None] * v + g[3][None, :]).ravel()]; self.sy = []
        for h, kb in enumerate((kL, kR)):                                                                                           # the class pairs that were shown, grouped by the mean level-2 vector of their shown blocks
            if true: self.sy.append(S.sym); continue
            E = net.emb(h); ub = np.unique(kb); seen = np.unique(cp[h][ub]); s = -np.ones(v * v, int); s[seen] = groups(np.stack([E[ub[cp[h][ub] == q]].mean(0) for q in seen]), min(v, len(seen))); self.sy.append(s[cp[h]])
        C = np.zeros((v, v)); np.add.at(C, (self.sy[0][kL], self.sy[1][kR]), 1); size = [np.bincount(s[s >= 0], minlength=v) for s in self.sy]; sl, sr = self.sy[0][S.vL], self.sy[1][S.vR]
        self.loss = S.mixed(np.where((sl >= 0) & (sr >= 0), C[sl, sr] / C.sum() / np.maximum(size[0][sl] * size[1][sr], 1), 0.0))  # a pair of symbols writes each of its strings equally often
        self.right1 = float(np.mean([len(set(c[S.w.members[a]])) == 1 for c in g for a in range(v)])); self.right2 = float(np.mean([len(set(s[S.sym == a])) == 1 and s[S.sym == a][0] >= 0 for s in self.sy for a in range(v)]))   # classes, symbols held in one group


class Judge:
    def __init__(self, S, net, kL, kR, val):
        N = S.N; tr = ~val; self.S = S; self.acc, self.tau1, self.pos = [], [], []
        for h, kb in enumerate((kL, kR)):                                                                                           # level 1: a block is accepted if a neighbouring block was shown
            Q, Qv = np.zeros((N, N)), np.zeros((N, N)); np.add.at(Q, (kb[tr] // N, kb[tr] % N), 1); np.add.at(Qv, (kb[val] // N, kb[val] % N), 1)
            est = lambda tau: neighbours(net.U[h], tau) @ (Q / Q.sum()) @ neighbours(net.V[h], tau).T
            tau = float(TAUS[np.argmin([-(Qv * np.log(0.999 * est(t) + 0.001 / Q.size)).sum() for t in TAUS])]); acc = np.flatnonzero(est(tau).ravel() > 0); pos = -np.ones(N * N, int); pos[acc] = np.arange(len(acc)); self.tau1.append(tau); self.acc.append(acc); self.pos.append(pos)
        M = np.zeros((len(self.acc[0]), len(self.acc[1]))); np.add.at(M, (self.pos[0][kL[tr]], self.pos[1][kR[tr]]), 1); X = [net.emb(h)[self.acc[h]] for h in range(2)]; vl, vr = self.pos[0][kL[val]], self.pos[1][kR[val]]; ok = (vl >= 0) & (vr >= 0)
        est = lambda tau: neighbours(X[0], tau) @ (M / M.sum()) @ neighbours(X[1], tau).T                                           # level 2: the teacher strings spread over the strings of neighbouring accepted blocks
        self.tau2 = float(TAUS[np.argmin([-np.log(0.999 * est(t)[vl[ok], vr[ok]] + 0.001 / N ** 4).sum() for t in TAUS])]); self.E = est(self.tau2); self.marg = [self.E.sum(1), self.E.sum(0)]

    def __call__(self, L, R): i, j = self.pos[0][L], self.pos[1][R]; return np.where((i >= 0) & (j >= 0), self.E[i, j], 0.0)         # the answer for strings (L, R)

    def blocks(self, h, B): i = self.pos[h][B]; return np.where(i >= 0, self.marg[h][i], 0.0)                                       # the answer for blocks: the judge's block marginal (zero: refused)

    def scores(self):                                                                                                               # loss of the estimate itself, invalid strings accepted, valid strings refused, bandwidths
        S = self.S; F = np.zeros((S.N ** 2, S.N ** 2)); F[np.ix_(self.acc[0], self.acc[1])] = self.E; a = F > 0
        return [S.mixed(F.ravel()[S.idx]), int((a & ~S.valid).sum()), int((~a & S.valid).sum()), float(np.mean(self.tau1)), self.tau2, int((~S.ok[self.acc[0]]).sum() + (~S.ok[self.acc[1]]).sum())]


def draw(P, rng, m):                                                                                                                # m strings from a table of probabilities
    r = P.sum(1, dtype=np.float64); L = rng.choice(len(r), m, p=r / r.sum()); cs = P[L].astype(np.float64).cumsum(1); return L, (rng.random(m)[:, None] * cs[:, -1:] < cs).argmax(1)


class Learner:
    def __init__(self, S, data, rng):
        self.S, self.data = S, data; self.reader, self.writer = best_local(S, data, rng, reader=True), best_local(S, data, rng); self.judge = Judge(S, self.reader, *data); self.ebar = [None, None, None]; self.batches = 0   # separate weights: two random starts

    def target(self, i, e):                                                                                                         # log of the judge's answer; a refusal counts as FLOOR times the mean positive answer so far
        if (e > 0).any(): self.ebar[i] = e[e > 0].mean() if self.ebar[i] is None else 0.9 * self.ebar[i] + 0.1 * e[e > 0].mean()
        return np.log(np.maximum(e, FLOOR * (self.ebar[i] or 1 / self.S.N ** (2 if i < 2 else 4))))

    def practice(self, rng, m, source='own', twin=None, levels=(1, 2)):
        net, S, N = self.writer, self.S, self.S.N
        for it in range(m // BATCH):
            src = source if not source.startswith('mix') else 'judge' if (self.batches + it) % 2 == 0 else source[3:] or 'own'       # mix, mixtwin: every other batch from the judge's estimate, the rest own or the twin's
            if twin is not None and src == 'judge': twin.practice(rng, BATCH, source='judge', levels=levels)                        # the twin practises by the same protocol at the same pace
            if src == 'own': L, R = draw(net.table()[1], rng, BATCH)
            elif src == 'twin': L, R = draw(twin.writer.table()[1], rng, BATCH); twin.practice(rng, BATCH, levels=levels)
            elif src == 'shallow': L, R = (rng.choice(N * N, BATCH, p=softmax(net.block(h))) for h in range(2))                     # the writer's blocks, combined at random
            elif src == 'judge': i, j = draw(self.judge.E, rng, BATCH); L, R = self.judge.acc[0][i], self.judge.acc[1][j]           # no writer needed: strings drawn from the judge's own estimate (teacher strings with blocks swapped for neighbours)
            else: L, R = rng.integers(N * N, size=BATCH), rng.integers(N * N, size=BATCH)
            e, eb = self.judge(L, R), [self.judge.blocks(0, L), self.judge.blocks(1, R)]; tb = [self.target(h, eb[h]) for h in range(2)]; ts = self.target(2, e); B = [L, R]; k = [L // N, L % N, R // N, R % N]
            for i in range(BATCH):
                for h in range(2) if 1 in levels else ():                                                                           # level 1, each half: log-probability of the block among all blocks against the log of the judge's answer for the block
                    x, xp = k[2 * h][i], k[2 * h + 1][i]; d = log_softmax(net.block(h))[B[h][i]] - tb[h][i]; d = float(np.clip(d if eb[h][i] > 0 else max(d, 0.0), -5, 5))
                    st = ETA * d / ((net.U[h][x] ** 2).sum() + (net.V[h][xp] ** 2).sum() + 2); du = st * net.V[h][xp]; net.V[h][xp] -= st * net.U[h][x]; net.U[h][x] -= du; net.a[h][x] -= st; net.b[h][xp] -= st
                if 2 not in levels or eb[0][i] == 0 or eb[1][i] == 0: continue                                                      # level 2, only for strings of two accepted blocks: the product of the two block vectors against the judge's
                fl, fr = net.feat(0, k[0][i:i + 1], k[1][i:i + 1])[0], net.feat(1, k[2][i:i + 1], k[3][i:i + 1])[0]; el, er = fl @ net.A[0], fr @ net.A[1]; d = el @ er + net.b2 - (ts[i] - tb[0][i] - tb[1][i])   # log p_E(L, R) - log p_E(L) - log p_E(R)
                d = float(np.clip(d if e[i] > 0 else max(d, 0.0), -5, 5)); st = ETA * d / ((fl ** 2).sum() * (er ** 2).sum() + (fr ** 2).sum() * (el ** 2).sum() + 1); net.A[0] -= st * np.outer(fl, er); net.A[1] -= st * np.outer(fr, el); net.b2 -= st
        self.batches += m // BATCH

    def score(self): return self.S.scores(*self.writer.table())


def overlap(S, P, Pt):                                                                                                              # before practice: share of the learner's invalid mass that the twin writes as well; correlation of their log-probabilities of the valid strings
    bad = ~S.valid; return [float(np.minimum(P, Pt)[bad].sum(dtype=np.float64) / P[bad].sum(dtype=np.float64)), float(np.corrcoef(np.log(P.ravel()[S.idx]), np.log(Pt.ravel()[S.idx]))[0, 1])]


def table_loss(S, kL, kR):                                                                                                          # the counted table of whole strings with the best uniform smoothing
    c = np.unique(kL * S.N ** 2 + kR, return_counts=True)[1]; n, nv, tot = len(kL), len(S.idx), S.N ** 4
    return min(float(-np.log(nv) - (np.log((c + a) / (n + a * tot)).sum() + (nv - len(c)) * np.log(a / (n + a * tot))) / nv) for a in 10.0 ** np.arange(-10, 0.1, 0.5))


def data_of(S, n, seed): rng = np.random.default_rng(30_000 + seed); kL, kR = S.teacher(rng, n); return (kL, kR, rng.random(n) < 0.15)


def one_watching(job):                                                                                                              # what watching gives at both levels, local rule and backpropagation: output, hidden layers read out, judge
    rule, n, seed = job; S = Strings(seed); data = data_of(S, n, seed); rng = np.random.default_rng(31_000 + seed); kL, kR, val = data
    if rule == 'local': net, _, stop, _, rd, rstop = watch_local(S, *data, rng)                                                       # one run, two passes kept: what it writes is scored at the writer's pass, the hidden layers and the judge at the reader's
    else: net, _, stop = watch_backprop(S, *data, rng); rd, rstop = net, stop
    r = Readout(S, rd, kL, kR); J = Judge(S, rd, *data)
    return job, S.scores(*net.table()) + [r.loss, r.right1, r.right2, stop] + J.scores() + [table_loss(S, kL, kR), Readout(S, net, kL, kR, true=True).loss, rstop]


WKEYS = ['writes', 'valid', 'holes', 'blocks0', 'blocks1', 'top', 'hidden', 'right1', 'right2', 'stop', 'judge', 'invalid', 'refused', 'tau1', 'tau2', 'badblocks', 'table', 'oracle', 'rstop']


def watching():
    ns = [int(x) for x in args.ns.split(',')]; seeds = worlds(); rules = args.rules.split(','); jobs = [(r, n, s) for r in rules for n in ns for s in seeds]; res = {}
    with Pool(args.procs) as p:
        for job, out in tqdm(p.imap_unordered(one_watching, jobs), total=len(jobs)): res[job] = out
    np.savez(OUT / f'network_depth_watching{args.tag}.npz', ns=ns, keys=WKEYS, rules=rules, **{r: np.array([[res[r, n, s] for s in seeds] for n in ns]) for r in rules})
    for r in rules:
        for n in ns: print(r, n, dict(zip(WKEYS, np.round(np.median([res[r, n, s] for s in seeds], 0), 4))))


def one_saddle(job):                                                                                                                # learning curve of the local rule by pass, for random starts of different size
    n, seed, init = job; S = Strings(seed); data = data_of(S, n, seed); return job, watch_local(S, *data, np.random.default_rng(32_000 + seed), init=init, curve=True, epochs=60)[3]


def saddle():
    inits = [0.1, 0.01, 0.001, 0.0001]; seeds = worlds(); jobs = [(args.n, s, i) for i in inits for s in seeds]; res = {}
    with Pool(args.procs) as p:
        for job, out in tqdm(p.imap_unordered(one_saddle, jobs), total=len(jobs)): res[job] = out
    np.savez(OUT / f'network_depth_saddle{args.tag}.npz', inits=inits, n=args.n, keys=['right1', 'right2', 'hidden', 'writes', 'contrast'], curves=np.array([[res[args.n, s, i] for s in seeds] for i in inits]))
    for i, c in zip(inits, np.median([[res[args.n, s, i] for s in seeds] for i in inits], 1)): print('start', i, 'classes held from pass', int(np.argmax(c[:, 0] >= 0.99)) + 1, 'symbols from pass', int(np.argmax(c[:, 1] >= 0.99)) + 1, 'final', np.round(c[-1], 3))


def one_curve(job):
    kind, n, seed, arm = job; S = Strings(seed)
    if kind == 'fresh': data = data_of(S, n, 1000 + seed); net = best_local(S, data, np.random.default_rng(33_000 + seed)); return job, [S.kl(net.table()[0]), Readout(S, net, data[0], data[1]).loss, table_loss(S, data[0], data[1])]
    data = data_of(S, n, seed); rng = np.random.default_rng(34_000 + seed); L = Learner(S, data, rng); T = Learner(S, data, rng) if 'twin' in arm else None; out = [L.score()]; done = 0; source, lev = arm.split('_') if '_' in arm else (arm, '12'); levels = tuple(int(c) for c in lev)
    start = overlap(S, L.writer.table()[1], T.writer.table()[1]) if T is not None else [1.0, 1.0]; extra = L.judge.scores() + [Readout(S, L.reader, data[0], data[1]).loss] + start + ([T.score()[0]] if T is not None else [np.nan])
    for c in CHECK: L.practice(rng, int(c) - done, source=source, twin=T, levels=levels); done = int(c); out.append(L.score())
    return job, (np.array(out), extra + ([T.score()[0]] if T is not None else [np.nan]))


def curves():                                                                                                                       # practice by proposal source against as many fresh teacher strings
    ns = [int(x) for x in args.ns.split(',')]; seeds = worlds(); arms = args.arms.split(','); totals = sorted({n + int(c) for n in ns for c in np.concatenate([[0], CHECK])})
    jobs = [('arm', n, s, arm) for n in ns for arm in arms for s in seeds] + [('fresh', t, s, '') for t in totals for s in seeds]; res = {}
    with Pool(args.procs) as p:
        for job, out in tqdm(p.imap_unordered(one_curve, jobs), total=len(jobs)): res[job] = out
    np.savez(OUT / f'network_depth_curves{args.tag}.npz', ns=ns, P=np.concatenate([[0], CHECK]), totals=totals, arms=arms, fresh=np.array([[res['fresh', t, s, ''] for s in seeds] for t in totals]),
             **{arm: np.array([[res['arm', n, s, arm][0] for s in seeds] for n in ns]) for arm in arms}, **{f'extra_{arm}': np.array([[res['arm', n, s, arm][1] for s in seeds] for n in ns]) for arm in arms})
    f = {t: np.median([res['fresh', t, s, ''] for s in seeds], 0) for t in totals}
    for n in ns:
        print(f'n = {n}: judge (loss, invalid accepted, valid refused, tau1, tau2, invalid blocks accepted), reader read out', np.round(np.median([res['arm', n, s, arms[0]][1][:7] for s in seeds], 0), 4))
        for i, c in enumerate(np.concatenate([[0], CHECK])): print(f'  P = {int(c):6d}  fresh net {f[n + int(c)][0]:.3f}  read out {f[n + int(c)][1]:.3f}  table {f[n + int(c)][2]:.2f}  ' + '  '.join(f"{arm} {np.median([res['arm', n, s, arm][0][i, 0] for s in seeds]):.3f}" for arm in arms))


def worlds(): return [s for s in range(64) if s not in SKIP][:args.worlds]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=28); ap.add_argument('--ns', default='4096,8192,16384'); ap.add_argument('--n', type=int, default=4096)
    ap.add_argument('--arms', default='own,twin,shallow,uniform'); ap.add_argument('--rules', default='local,backprop'); ap.add_argument('--tag', default=''); args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); globals()[args.run]()
