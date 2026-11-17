"""Model 5 with networks: does practice make a network write what its hidden layer already knows, and is an own sample worth a fresh teacher pair?
One block of model5_lateral_verdict.py (64 codes per side, 16 classes, 1,024 valid pairs). Two networks of the same form with separate weights,
  p(k, k') = softmax_{kk'}(u_k . v_k' + a_k + b_k')   (code -> 16 hidden units -> partner, two linear layers),
a READER and a WRITER, both trained by backpropagation on the teacher pairs (Adam, weight decay and stopping step chosen on a held-out 15%).
Judge: the reader's hidden layer and the lateral memory M of the teacher pairs. Codes are neighbours if their (centred) hidden vectors have cosine >= tau;
  p_E = K_L M K_R^T / n,  K column-normalised neighbourhood matrices: every stored pair is spread over the pairs of its neighbours.
  tau is the bandwidth of this kernel estimate and is chosen by the likelihood of the held-out teacher pairs.
Practice: the writer proposes pairs by drawing them from its own output; the judge returns the evidence p_E(x) for each; the learner keeps the judged value:
  in the memory that the writer is trained on, the raw teacher count of a proposed pair is replaced by its judged count n p_E(x) (zero if no neighbouring
  pair was shown). After each round the writer is fitted again to that memory (same procedure as in watching) and proposes the next round.
Arms after n teacher pairs, P further samples:  fresh    a network trained on n + P teacher pairs (and the counted table of them)
  own      practice on the writer's own proposals          twin     the judged proposals of a twin (own teacher set, practising at the same time) instead of own ones
  draws    P pairs drawn from p_E and added to the teacher pairs as if the teacher had shown them (a verified sample used as a sample, not as a question)
  steps    no memory of judged pairs: one likelihood step per batch of own samples, each weighted by p_E(x) / p_writer(x)
  written  the judge without the hidden layer (tau > 1: was exactly this pair shown?)
  online   no memory of judged pairs: for every own proposal one normalised delta step of the writer's weights towards the judge's answer (Learner.online)
  online_twin  the same on a twin's proposals        online_explore  proposals drawn from the writer's output at temperature 2        online_top  only the top of the writer learns
  online_local  the same step without the normaliser's part: only the hidden vectors and biases of the two written codes move, by error x the partner's hidden vector
  own_reader  as own, and after every round the reader is fitted to the same memory, so that the judge is rebuilt from a hidden layer that has learned from the judged pairs
Closed forms: p_E itself; the table over the true classes.
Runs: curves (arms against P at fixed n), budget (total by own share), local (the network trained by a local rule: hidden layer, output, plateau).   Writes /var/tmp/jbauer/model5_lateral_verdict/network_practice_<run>.npz.
Usage: python model5_network_practice.py <run> [--worlds 16] [--procs 28]"""

import argparse
from multiprocessing import Pool

import numpy as np
from scipy.special import expit, log_softmax
from tqdm.auto import tqdm

from model5_lateral_verdict import OUT, World, table_kl
from model5_network_baseline import groups
from model5_own_benefit import class_table, kl

D, INIT, LR, STEPS, DECAYS, BATCH, LR_P, ETA, FLOOR = 16, 0.1, 0.02, 600, (1e-4, 1e-3, 3e-3), 64, 0.01, 0.5, 1e-3; RULE = 'backprop'   # --rule local: reader and writer trained by the local rule of local_net
TAUS = np.arange(0.3, 0.96, 0.05); CHECK = BATCH * 2 ** np.arange(2, 10)                                                             # own samples at which practice is scored: 256 .. 32,768


class Net:
    def __init__(self, N, rng): self.P = [INIT * rng.standard_normal((N, D)), INIT * rng.standard_normal((N, D)), np.zeros((N, 1)), np.zeros((1, N))]; self.N = N; self.reset()

    def reset(self): self.m = [np.zeros_like(x) for x in self.P]; self.v = [np.zeros_like(x) for x in self.P]; self.t = 0

    def logp(self): U, V, a, b = self.P; return log_softmax((U @ V.T + a + b).ravel()).reshape(self.N, self.N)

    def step(self, G, lr, decay):                                                                                                  # G: gradient of the loss with respect to the logits
        U, V, a, b = self.P; g = [G @ V + decay * U, G.T @ U + decay * V, G.sum(1, keepdims=True), G.sum(0, keepdims=True)]; self.t += 1
        for i in range(4): self.m[i] = 0.9 * self.m[i] + 0.1 * g[i]; self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] ** 2; self.P[i] = self.P[i] - lr * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)

    def copy(self): c = Net.__new__(Net); c.N = self.N; c.P = [x.copy() for x in self.P]; c.reset(); return c


def watch(w, Q, Qv, rng):                                                                                                          # a network fitted to the training pairs Q; decay and stopping step by the held-out pairs Qv
    best = (np.inf,); Qn, Qvn = Q / Q.sum(), Qv / Qv.sum(); seed = rng.integers(1 << 30)
    for decay in DECAYS:
        net = Net(w.N, np.random.default_rng(seed))
        for t in range(1, STEPS + 1):
            lp = net.logp()
            if t % 5 == 0 and -(Qvn * lp).sum() < best[0]: best = (float(-(Qvn * lp).sum()), net.copy(), decay)
            net.step(np.exp(lp) - Qn, LR, decay)
    return best[1], best[2]


def neighbours(X, tau):                                                                                                            # column-normalised neighbourhood matrix of the rows of X
    X = X - X.mean(0); X = X / np.linalg.norm(X, axis=1, keepdims=True); K = (X @ X.T >= tau) | np.eye(len(X), dtype=bool); return K / K.sum(0, keepdims=True)


def evidence(reader, M, tau): return neighbours(reader.P[0], tau) @ (M / M.sum()) @ neighbours(reader.P[1], tau).T


def bandwidth(reader, Q, Qv):                                                                                                      # tau with the highest likelihood of the held-out pairs under the kernel estimate from the training pairs
    return float(TAUS[np.argmin([-(Qv * np.log(0.999 * evidence(reader, Q, tau) + 0.001 / Q.size)).sum() for tau in TAUS])])


class Learner:
    def __init__(self, w, n, rng, tau=None):
        self.w = w; k, kp = w.teacher(rng, n); val = rng.random(n) < 0.15; N = w.N; self.Q, self.Qv = np.zeros((N, N)), np.zeros((N, N)); np.add.at(self.Q, (k[~val], kp[~val]), 1); np.add.at(self.Qv, (k[val], kp[val]), 1)
        if RULE == 'local': self.reader, self.writer = (best_local(w, k, kp, val, rng)[0] for _ in range(2)); self.decay = 0.0                                                        # both networks trained by the local rule
        else: self.reader, _ = watch(w, self.Q, self.Qv, rng); self.writer, self.decay = watch(w, self.Q, self.Qv, rng)                                                                # separate weights: two random starts
        self.tau = bandwidth(self.reader, self.Q, self.Qv) if tau is None else tau
        self.pE = evidence(self.reader, self.Q, self.tau) if self.tau <= 1 else self.Q / self.Q.sum(); self.asked = np.zeros((N, N), bool); self.extra = np.zeros((N, N)); self.ebar = None

    def propose(self, rng, m): return rng.choice(self.Q.size, m, p=np.exp(self.writer.logp()).ravel())

    def keep(self, x): self.asked.ravel()[x] = True                                                                                 # the judged count replaces the raw count at the pairs that were proposed

    def memory(self): return np.where(self.asked, self.Q.sum() * self.pE, self.Q) + self.extra

    def refit(self, rng, reader=False):
        if reader: self.reader, _ = watch(self.w, self.memory(), self.Qv, rng); self.tau = bandwidth(self.reader, self.Q, self.Qv); self.pE = evidence(self.reader, self.Q, self.tau)   # the reader learns from the judged pairs as well; the judge is
        self.writer, self.decay = watch(self.w, self.memory(), self.Qv, rng)                                                                                                         # rebuilt from its new hidden layer and the teacher pairs

    def steps(self, rng, m):                                                                                                       # weighted likelihood steps on own samples, nothing kept
        for _ in range(m // BATCH):
            p = np.exp(self.writer.logp()); x = rng.choice(p.size, BATCH, p=p.ravel()); wt = self.pE.ravel()[x] / p.ravel()[x]; self.writer.step(wt.mean() * p - np.bincount(x, wt, p.size).reshape(p.shape) / BATCH, LR_P, self.decay)

    def online(self, rng, m, source=None, temp=1.0, top=False, local=False):                                                                    # nothing is kept: for every proposal one step of the writer's weights that moves its log-probability of the
        U, V, a, b = [x.copy() for x in self.writer.P]; N = self.w.N; B = getattr(self, 'B', np.eye(D))                                                  # proposed pair towards the log of the judge's answer (an answer of zero counts as FLOOR times the mean positive answer);
        for _ in range(m):                                                                                                         # the step is normalised by the squared gradient, so ETA is the share of the error removed at that pair
            lp = log_softmax((U @ B @ V.T + a + b).ravel()).reshape(N, N); p = np.exp(lp); q = p if source is None else np.exp(source.writer.logp())
            if temp != 1: q = q ** (1 / temp); q = q / q.sum()
            k, kp = divmod(int(rng.choice(N * N, p=q.ravel())), N); e = self.pE[k, kp]
            if e > 0: self.ebar = e if self.ebar is None else 0.99 * self.ebar + 0.01 * e
            err = float(np.clip(lp[k, kp] - np.log(max(e, FLOOR * (self.ebar or 1 / N ** 2))), -5, 5)); ga = -p.sum(1, keepdims=True); ga[k] += 1; gb = -p.sum(0, keepdims=True); gb[0, kp] += 1
            if local: st = ETA * err / ((U[k] ** 2).sum() + (V[kp] ** 2).sum() + 2); du = st * V[kp]; V[kp] = V[kp] - st * U[k]; U[k] = U[k] - du; a[k] -= st; b[0, kp] -= st; continue   # only the weights of the two written codes move: error x partner's hidden vector
            if top: gB = np.outer(U[k], V[kp]) - U.T @ p @ V; st = ETA * err / ((gB ** 2).sum() + (ga ** 2).sum() + (gb ** 2).sum()); B = B - st * gB
            else: gU = -(p @ V); gU[k] += V[kp]; gV = -(p.T @ U); gV[kp] += U[k]; st = ETA * err / ((gU ** 2).sum() + (gV ** 2).sum() + (ga ** 2).sum() + (gb ** 2).sum()); U = U - st * gU; V = V - st * gV
            a = a - st * ga; b = b - st * gb
        self.writer.P = [U @ B, V, a, b] if top else [U, V, a, b]
        if top: self.B = np.eye(D)                                                                                                 # the top matrix is folded into the stored weights at every checkpoint; the hidden vectors themselves stay as they were up to it

    def apart(self):                                                                                                               # how far the writer's hidden layer keeps the classes apart: cosine of hidden vectors, 5% quantile within a class minus
        out = []                                                                                                                   # 95% quantile between classes, mean of the two sides
        for X in self.writer.P[:2]:
            X = X - X.mean(0); X = X / np.linalg.norm(X, axis=1, keepdims=True); C = X @ X.T; same = self.w.cls[:, None] == self.w.cls[None, :]; out.append(np.quantile(C[same & ~np.eye(len(C), dtype=bool)], 0.05) - np.quantile(C[~same], 0.95))
        return float(np.mean(out))

    def score(self):                                                                                                               # loss, valid share of what it writes, valid pairs it hardly writes, valid pairs asked about
        w = self.w; p = np.exp(self.writer.logp()); return [kl(w, p), float(p[w.valid].sum()), float((p[w.valid] < 0.25 / w.valid.sum()).mean()), float(self.asked[w.valid].mean())] + self.judge() + [self.apart()]

    def judge(self):                                                                                                               # the kernel estimate itself, and how the judge errs
        w = self.w; a = self.pE > 0; return [kl(w, 0.999 * self.pE + 0.001 / self.pE.size), int(a[~w.valid].sum()), int((~a[w.valid]).sum()), self.tau]


def run_arm(w, n, seed, arm):
    rng = np.random.default_rng(5000 + seed); L = Learner(w, n, rng, tau=1.5 if arm == 'written' else None); T = Learner(w, n, rng) if arm in ('twin', 'online_twin') else None; out = [L.score()]; done = 0
    if arm == 'steps': L.writer.reset()
    for c in CHECK:
        m = int(c) - done; done = int(c)
        if arm == 'steps': L.steps(rng, m)
        elif arm == 'online_twin':                                                                                                  # the twin practises on its own proposals at the same pace; the learner takes the twin's proposals to its own judge
            for _ in range(m // BATCH): L.online(rng, BATCH, source=T); T.online(rng, BATCH)
        elif arm.startswith('online'): L.online(rng, m, temp=2.0 if arm == 'online_explore' else 1.0, top=arm == 'online_top', local=arm == 'online_local')
        elif arm == 'draws': L.extra += np.bincount(rng.choice(L.Q.size, m, p=L.pE.ravel()), minlength=L.Q.size).reshape(L.Q.shape); L.refit(rng)
        elif arm == 'twin': x = T.propose(rng, m); T.keep(x); T.refit(rng); L.keep(x); L.refit(rng)
        else: L.keep(L.propose(rng, m)); L.refit(rng, reader=arm == 'own_reader')
        out.append(L.score())
    return np.array(out), L.judge(), (T.score()[0] if T is not None else np.nan)


def fresh(w, total, seed):                                                                                                         # a network and a table given `total` teacher pairs
    rng = np.random.default_rng(6000 + seed); k, kp = w.teacher(rng, total); val = rng.random(total) < 0.15; N = w.N; Q, Qv = np.zeros((N, N)), np.zeros((N, N)); np.add.at(Q, (k[~val], kp[~val]), 1); np.add.at(Qv, (k[val], kp[val]), 1)
    net = best_local(w, k, kp, val, rng)[0] if RULE == 'local' else watch(w, Q, Qv, rng)[0]; return kl(w, np.exp(net.logp())), table_kl(w, Q + Qv), class_table(w, Q + Qv, w.cls, w.cls)


def one_budget(job):                                                                                                               # a budget of B samples of which a share f are own proposals, in four rounds
    B, f, seed, arm = job; w = World(seed=seed); P = int(round(f * B))
    if P == 0: return job, fresh(w, B, seed)[:2]
    rng = np.random.default_rng(7000 + seed); L = Learner(w, B - P, rng); done = 0
    for c in (P // 8, P // 4, P // 2, P): L.keep(L.propose(rng, c - done)); done = c; L.refit(rng, reader=arm == 'own_reader')
    return job, (L.score()[0], L.judge()[0])


def budget():
    Bs = [int(2 ** e) for e in range(11, 17)]; fs = [0, 0.125, 0.25, 0.5, 0.75, 0.875]; S = range(args.worlds); arms = args.arms.split(','); jobs = [(B, f, s, arm) for arm in arms for B in Bs for f in fs for s in S if f > 0 or arm == arms[0]]; res = {}
    with Pool(args.procs) as p:
        for job, out in tqdm(p.imap_unordered(one_budget, jobs), total=len(jobs)): res[job] = out
    np.savez(OUT / f'network_practice_budget{args.tag}.npz', total=Bs, shares=fs, arms=arms, fresh=np.array([[res[B, 0, s, arms[0]] for s in S] for B in Bs]), **{arm: np.array([[[res[B, f, s, arm] for s in S] for B in Bs] for f in fs[1:]]) for arm in arms})
    for B in Bs: print(B, 'fresh net', round(float(np.median([res[B, 0, s, arms[0]][0] for s in S])), 4), 'table', round(float(np.median([res[B, 0, s, arms[0]][1] for s in S])), 4), {arm: {f: round(float(np.median([res[B, f, s, arm][0] for s in S])), 4) for f in fs[1:]} for arm in arms})


def local_net(w, k, kp, val, rng, init=0.1, neg=4, epochs=120, eta=0.05, decay=1e-4):                                              # the same network trained by a local rule, no normaliser: a teacher pair raises the product of the two
    N = w.N; U, V, c = init * rng.standard_normal((N, D)), init * rng.standard_normal((N, D)), 0.0; kt, kpt = k[~val], kp[~val]      # hidden vectors (each moves by error x the other), the same left code with a partner drawn from another teacher pair lowers it
    Q, Qv = np.zeros((N, N)), np.zeros((N, N)); np.add.at(Q, (kt, kpt), 1); np.add.at(Qv, (k[val], kp[val]), 1); pl, pr = Q.sum(1, keepdims=True) + 1, Q.sum(0, keepdims=True) + 1; a, b = np.log(pl / pl.sum()), np.log(pr / pr.sum()); best = (np.inf,); hist = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(kt))
        for i in range(0, len(order), BATCH):
            x, y = kt[order[i:i + BATCH]], kpt[order[i:i + BATCH]]; A = np.concatenate([x, np.repeat(x, neg)]); Bq = np.concatenate([y, rng.choice(kpt, len(x) * neg)]); g = np.concatenate([np.ones(len(x)), np.zeros(len(x) * neg)]) - expit((U[A] * V[Bq]).sum(1) + c)
            dU, dV = np.zeros_like(U), np.zeros_like(V); np.add.at(dU, A, g[:, None] * V[Bq]); np.add.at(dV, Bq, g[:, None] * U[A]); U += eta * (dU - decay * U); V += eta * (dV - decay * V); c += eta * g.mean()
        lp = log_softmax((U @ V.T + a + b).ravel()).reshape(N, N); vl = float(-(Qv / Qv.sum() * lp).sum()); hist.append([class_table(w, Q + Qv, groups(U, w.v), groups(V, w.v)), kl(w, np.exp(lp))])   # what it writes: p(k) p(k') exp(u . v), normalised
        if vl < best[0]: best = (vl, U.copy(), V.copy(), ep)
    net = Net(N, rng); net.P = [best[1], best[2], a, b]; return net, best[0], best[3], Q, Qv, np.array(hist)


def best_local(w, k, kp, val, rng):                                                                                               # the number of contrast pairs by the held-out likelihood of what the network writes
    seed = rng.integers(1 << 30); return min((local_net(w, k, kp, val, np.random.default_rng(seed), neg=neg) for neg in (1, 4, 16)), key=lambda r: r[1])


def one_local(job):
    n, seed, init = job; w = World(seed=seed); rng = np.random.default_rng(8000 + seed); k, kp = w.teacher(rng, n); val = rng.random(n) < 0.15; runs = [local_net(w, k, kp, val, np.random.default_rng(9000 + seed), init, neg) for neg in (1, 4, 16)]
    net, _, stop, Q, Qv, hist = min(runs, key=lambda r: r[1]); tau = bandwidth(net, Q, Qv); pE = evidence(net, Q, tau); gl, gr = groups(net.P[0], w.v), groups(net.P[1], w.v)                       # the number of contrast pairs by the held-out likelihood of what it writes
    return job, [kl(w, np.exp(net.logp())), class_table(w, Q + Qv, gl, gr), float(np.mean([len(set(g[w.members[c]])) == 1 for g in (gl, gr) for c in range(w.v)])), kl(w, 0.999 * pE + 0.001 / pE.size), int((pE > 0)[~w.valid].sum()), int((pE == 0)[w.valid].sum()), stop,
                 table_kl(w, Q + Qv)], runs[1][5]


def local():                                                                                                                       # section 9: what a local rule gives the hidden layer and the output; the plateau at small weights by the size of the random start
    ns = [1024, 2048, 4096, 8192, 16384]; inits = [0.1, 0.01, 0.001, 0.0001]; S = range(args.worlds); jobs = [(n, s, 0.1) for n in ns for s in S] + [(4096, s, i) for i in inits[1:] for s in S]; res = {}
    with Pool(args.procs) as p:
        for job, out, hist in tqdm(p.imap_unordered(one_local, jobs), total=len(jobs)): res[job] = (out, hist)
    np.savez(OUT / 'network_practice_local.npz', ns=ns, inits=inits, stats=np.array([[res[n, s, 0.1][0] for s in S] for n in ns]), curves=np.array([[res[4096, s, i][1] for s in S] for i in inits]), keys=['writes', 'hidden', 'right', 'judge', 'invalid', 'refused', 'stop', 'table'])
    for n in ns: print(n, dict(zip(['writes', 'hidden', 'right', 'judge', 'invalid', 'refused', 'stop', 'table'], np.round(np.median([res[n, s, 0.1][0] for s in S], 0), 4))))
    for i in inits: print('start', i, 'hidden layer read out below 0.05 nats from pass', int(np.argmax(np.median([res[4096, s, i][1][:, 0] for s in S], 0) < 0.05)) + 1)


def one_curve(job):
    kind, n, seed, arm = job; w = World(seed=seed)
    return (job, fresh(w, n, seed)) if kind == 'fresh' else (job, run_arm(w, n, seed, arm))


ARMS = ('own', 'own_reader', 'twin', 'draws', 'steps', 'written')                                                                # run with --arms own,steps,online,online_local,online_twin,online_explore,online_top --tag _online for section 9


def curves():
    ns = [int(x) for x in args.ns.split(',')]; S = range(args.worlds); totals = sorted({n + int(c) for n in ns for c in np.concatenate([[0], CHECK])}); arms = args.arms.split(',')
    jobs = [('arm', n, s, arm) for n in ns for arm in arms for s in S] + [('fresh', t, s, '') for t in totals for s in S]; res = {}
    with Pool(args.procs) as p:
        for job, out in tqdm(p.imap_unordered(one_curve, jobs), total=len(jobs)): res[job] = out
    np.savez(OUT / f'network_practice_curves{args.tag}.npz', ns=ns, P=np.concatenate([[0], CHECK]), totals=totals, arms=arms, fresh=np.array([[res['fresh', t, s, ''] for s in S] for t in totals]),
             judge=np.array([[res['arm', n, s, arms[0]][1] for s in S] for n in ns]), twin_kl=np.array([[res['arm', n, s, 'twin'][2] for s in S] for n in ns]) if 'twin' in arms else 0,
             **{arm: np.array([[res['arm', n, s, arm][0] for s in S] for n in ns]) for arm in arms})
    f = {t: np.median([res['fresh', t, s, ''] for s in S], 0) for t in totals}
    for n in ns:
        print(f'n = {n}: judge (loss of p_E, invalid pairs accepted, valid pairs refused, tau)', np.round(np.median([res['arm', n, s, arms[0]][1] for s in S], 0), 3))
        for i, c in enumerate(np.concatenate([[0], CHECK])): print(f'  P = {int(c):6d}  fresh net {f[n + int(c)][0]:.3f}  table {f[n + int(c)][1]:.3f}  ' + '  '.join(f"{arm} {np.median([res['arm', n, s, arm][0][i, 0] for s in S]):.3f}" for arm in arms))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=28); ap.add_argument('--ns', default='1024,2048,4096')
    ap.add_argument('--arms', default=','.join(ARMS)); ap.add_argument('--tag', default=''); ap.add_argument('--rule', default='backprop'); args = ap.parse_args(); RULE = args.rule; OUT.mkdir(parents=True, exist_ok=True); globals()[args.run]()
