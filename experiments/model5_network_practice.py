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
  own_reader  as own, and after every round the reader is fitted to the same memory, so that the judge is rebuilt from a hidden layer that has learned from the judged pairs
Closed forms: p_E itself; the table over the true classes.
Runs: curves (arms against P at fixed n), budget (total by own share).   Writes /var/tmp/jbauer/model5_lateral_verdict/network_practice_<run>.npz.
Usage: python model5_network_practice.py <run> [--worlds 16] [--procs 28]"""

import argparse
from multiprocessing import Pool

import numpy as np
from scipy.special import log_softmax
from tqdm.auto import tqdm

from model5_lateral_verdict import OUT, World, table_kl
from model5_own_benefit import class_table, kl

D, INIT, LR, STEPS, DECAYS, BATCH, LR_P = 16, 0.1, 0.02, 600, (1e-4, 1e-3, 3e-3), 64, 0.01
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
        self.reader, _ = watch(w, self.Q, self.Qv, rng); self.writer, self.decay = watch(w, self.Q, self.Qv, rng); self.tau = bandwidth(self.reader, self.Q, self.Qv) if tau is None else tau   # separate weights: two random starts
        self.pE = evidence(self.reader, self.Q, self.tau) if self.tau <= 1 else self.Q / self.Q.sum(); self.asked = np.zeros((N, N), bool); self.extra = np.zeros((N, N))

    def propose(self, rng, m): return rng.choice(self.Q.size, m, p=np.exp(self.writer.logp()).ravel())

    def keep(self, x): self.asked.ravel()[x] = True                                                                                 # the judged count replaces the raw count at the pairs that were proposed

    def memory(self): return np.where(self.asked, self.Q.sum() * self.pE, self.Q) + self.extra

    def refit(self, rng, reader=False):
        if reader: self.reader, _ = watch(self.w, self.memory(), self.Qv, rng); self.tau = bandwidth(self.reader, self.Q, self.Qv); self.pE = evidence(self.reader, self.Q, self.tau)   # the reader learns from the judged pairs as well; the judge is
        self.writer, self.decay = watch(self.w, self.memory(), self.Qv, rng)                                                                                                         # rebuilt from its new hidden layer and the teacher pairs

    def steps(self, rng, m):                                                                                                       # weighted likelihood steps on own samples, nothing kept
        for _ in range(m // BATCH):
            p = np.exp(self.writer.logp()); x = rng.choice(p.size, BATCH, p=p.ravel()); wt = self.pE.ravel()[x] / p.ravel()[x]; self.writer.step(wt.mean() * p - np.bincount(x, wt, p.size).reshape(p.shape) / BATCH, LR_P, self.decay)

    def score(self):                                                                                                               # loss, valid share of what it writes, valid pairs it hardly writes, valid pairs asked about
        w = self.w; p = np.exp(self.writer.logp()); return [kl(w, p), float(p[w.valid].sum()), float((p[w.valid] < 0.25 / w.valid.sum()).mean()), float(self.asked[w.valid].mean())] + self.judge()

    def judge(self):                                                                                                               # the kernel estimate itself, and how the judge errs
        w = self.w; a = self.pE > 0; return [kl(w, 0.999 * self.pE + 0.001 / self.pE.size), int(a[~w.valid].sum()), int((~a[w.valid]).sum()), self.tau]


def run_arm(w, n, seed, arm):
    rng = np.random.default_rng(5000 + seed); L = Learner(w, n, rng, tau=1.5 if arm == 'written' else None); T = Learner(w, n, rng) if arm == 'twin' else None; out = [L.score()]; done = 0
    if arm == 'steps': L.writer.reset()
    for c in CHECK:
        m = int(c) - done; done = int(c)
        if arm == 'steps': L.steps(rng, m)
        elif arm == 'draws': L.extra += np.bincount(rng.choice(L.Q.size, m, p=L.pE.ravel()), minlength=L.Q.size).reshape(L.Q.shape); L.refit(rng)
        elif arm == 'twin': x = T.propose(rng, m); T.keep(x); T.refit(rng); L.keep(x); L.refit(rng)
        else: L.keep(L.propose(rng, m)); L.refit(rng, reader=arm == 'own_reader')
        out.append(L.score())
    return np.array(out), L.judge(), (T.score()[0] if T is not None else np.nan)


def fresh(w, total, seed):                                                                                                         # a network and a table given `total` teacher pairs
    rng = np.random.default_rng(6000 + seed); k, kp = w.teacher(rng, total); val = rng.random(total) < 0.15; N = w.N; Q, Qv = np.zeros((N, N)), np.zeros((N, N)); np.add.at(Q, (k[~val], kp[~val]), 1); np.add.at(Qv, (k[val], kp[val]), 1)
    net, _ = watch(w, Q, Qv, rng); return kl(w, np.exp(net.logp())), table_kl(w, Q + Qv), class_table(w, Q + Qv, w.cls, w.cls)


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


def one_curve(job):
    kind, n, seed, arm = job; w = World(seed=seed)
    return (job, fresh(w, n, seed)) if kind == 'fresh' else (job, run_arm(w, n, seed, arm))


ARMS = ('own', 'own_reader', 'twin', 'draws', 'steps', 'written')


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
    ap.add_argument('--arms', default=','.join(ARMS)); ap.add_argument('--tag', default=''); args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); globals()[args.run]()
