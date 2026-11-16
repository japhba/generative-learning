"""Model 5 (lateral verdict + compositional prior), one frontier block of an RHM. Note: tex/model_5_lateral_verdict.html.

World: codes k (left) and k' (right), N = v*m each (the productions of the level below, read by the shallow layer); v classes of m codes;
m*v valid class pairs (the productions of the level above), every class m times on each side. Population: uniform over valid (k, k').
Both learners: class units per side with pools G_L[k|a], G_R[k'|a'] and a table T[a, a'] over pairs of units (logits LL, LR, TH),
  p(k, k') = sum_{a,a'} T[a,a'] G_L[k|a] G_R[k'|a'];   reader = transpose, q(a,a'|k,k') ~ T G_L G_R, one winner;
  M[k, k']: lateral Hebbian count of teacher pairs between the shallow codes (needs no class unit).
Grower (the learner of the note): N units per side that start as copies of the codes, additive bounded weights, practice step eta (v - c vbar).
GradientLearner (first attempt, section 2 of the note): K units from a random start, softmax-gradient steps: wake (+1) on teacher pairs,
  -1 on own pairs that M has never seen; optional threshold theta on the drive T G_L G_R N^2 of the read explanation; even use of the units.
Runs (ablation, efficiency, heatmap, source, gradient) write /var/tmp/jbauer/model5_lateral_verdict/; model5_depth.py adds the three-level run,
model5_figures.py plots, model5_page.py builds the note. Results are kept in /nfs/gatsbystor/jbauer/generative-learning/model5_lateral_verdict/.
Usage: python model5_lateral_verdict.py <run> [--worlds 16] [--procs 28]"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from tqdm.auto import tqdm

OUT = Path('/var/tmp/jbauer/model5_lateral_verdict'); V, MM, BATCH = 16, 4, 64


class World:
    def __init__(self, v=V, m=MM, seed=0):
        rng = np.random.default_rng(seed); self.v, self.m, self.N = v, m, v * m; self.cls = rng.permutation(np.repeat(np.arange(v), m))  # code -> class
        while True:                                                                                                                # m permutations: every class m times left, m times right
            self.B = np.zeros((v, v), bool)
            for _ in range(m): self.B[np.arange(v), rng.permutation(v)] = True
            if self.B.sum() == v * m: break
        self.members = np.stack([np.flatnonzero(self.cls == a) for a in range(v)]); self.rules = np.argwhere(self.B)
        self.valid = self.B[np.ix_(self.cls, self.cls)]; self.pstar = self.valid / self.valid.sum()

    def teacher(self, rng, n):
        r = self.rules[rng.integers(len(self.rules), size=n)]
        return self.members[r[:, 0], rng.integers(self.m, size=n)], self.members[r[:, 1], rng.integers(self.m, size=n)]


def draw(w, rng, hard=False):                                                                                                      # one index per row of unnormalised weights
    return w.argmax(1) if hard else (rng.random(len(w))[:, None] * w.sum(1, keepdims=True) < w.cumsum(1)).argmax(1)


class GradientLearner:
    def __init__(self, world, K=None, start='identity', s=6.0, seed=0, eta=0.05, lam=0.002, theta=0.0, rule='punish', hard=False, credit='read', even=False, gain=1.0, clip=None, wrong=None):
        rng = np.random.default_rng(seed); self.w, self.N = world, world.N; self.K = K = K or world.N; self.eta, self.lam, self.theta, self.rule, self.hard, self.cred, self.even, self.gain, self.clip = eta, lam, theta, rule, hard, credit, even, gain, clip
        def pool():
            if start == 'classes':                                                                                                  # every code in the unit of its class, a fraction `wrong` of the codes in a random unit
                u = np.where(rng.random(self.N) < wrong, rng.integers(K, size=self.N), world.cls % K); L = np.zeros((K, self.N)); L[u, np.arange(self.N)] = s; return L
            if start == 'noise': return s * rng.standard_normal((K, self.N))
            L = np.zeros((K, self.N)); L[np.arange(K), np.arange(K) % self.N if start == 'identity' else rng.integers(self.N, size=K)] = s; return L   # 'identity': unit i relays code i; 'random': a random code
        self.LL, self.LR, self.TH = pool(), pool(), np.zeros((K, K)); self.M = np.zeros((self.N, self.N)); self.n = dict(teacher=0, own=0, el_teacher=0, el_own=0, rejected=0)

    def tables(self): return softmax(self.LL, 1), softmax(self.LR, 1), softmax(self.TH.ravel()).reshape(self.K, self.K)

    def pairprob(self, tab=None): GL, GR, T = tab or self.tables(); return GL.T @ T @ GR

    def read(self, k, kp, tab, rng):                                                                                               # winner (a, a') of q(.|k,k') and its drive
        GL, GR, T = tab; a = draw((GL[:, k] * (T @ GR)[:, kp]).T, rng, self.hard); ap = draw(T[a] * GR[:, kp].T, rng, self.hard)
        return a, ap, T[a, ap] * GL[a, k] * GR[ap, kp] * self.N ** 2

    def step(self, a, ap, k, kp, d, tab):                                                                                          # three-factor step at the winner: delta x (pre - expected pre)
        GL, GR, T = tab; e = self.eta
        np.add.at(self.LL, (a, k), e * d); self.LL -= e * np.bincount(a, d, self.K)[:, None] * GL
        np.add.at(self.LR, (ap, kp), e * d); self.LR -= e * np.bincount(ap, d, self.K)[:, None] * GR
        np.add.at(self.TH, (a, ap), e * d); self.TH -= e * d.sum() * T

    def decay(self):
        for L in (self.LL, self.LR, self.TH): L *= 1 - self.lam
        if self.clip:                                                                                                              # bounded weights: no entry of a pool further than clip below the unit's strongest
            for L in (self.LL, self.LR): np.maximum(L, L.max(1, keepdims=True) - self.clip, out=L)
            np.maximum(self.TH, self.TH.max() - self.clip - 3, out=self.TH)
        if self.even:                                                                                                              # even use: every unit equally often on each side (rows and columns of T rescaled)
            for _ in range(4): self.TH -= logsumexp(self.TH, 1, keepdims=True); self.TH -= logsumexp(self.TH, 0, keepdims=True)

    def sample(self, n, rng, tab=None):
        GL, GR, T = tab or self.tables(); c = rng.choice(self.K ** 2, n, p=T.ravel()); a, ap = c // self.K, c % self.K
        return draw(GL[a], rng), draw(GR[ap], rng), a, ap

    def watch(self, k, kp, rng, wake=True):
        np.add.at(self.M, (k, kp), 1); self.n['teacher'] += len(k)
        if not wake: return
        tab = self.tables(); a, ap, drive = self.read(k, kp, tab, rng); el = drive > self.theta; self.n['el_teacher'] += el.sum()
        self.step(a[el], ap[el], k[el], kp[el], np.ones(el.sum()), tab); self.decay()

    def practice(self, n, rng, source=None):                                                                                       # source: pairs (k, k') of a foreign generator
        tab = self.tables(); k, kp, a0, ap0 = self.sample(n, rng, tab) if source is None else (*source, None, None)
        a, ap, drive = self.read(k, kp, tab, rng)
        if self.cred == 'gen': a, ap = a0, ap0
        seen = self.M[k, kp] > 0; d = dict(punish=self.gain * (seen - 1.0), signed=2.0 * seen - 1, reward=seen * 1.0)[self.rule]; el = drive > self.theta
        self.n['own'] += n; self.n['el_own'] += el.sum(); self.n['rejected'] += (~seen).sum(); self.step(a[el], ap[el], k[el], kp[el], d[el], tab); self.decay()

    def score(self):
        w = self.w; GL, GR, T = tab = self.tables(); P = self.pairprob(tab); pis = (T.sum(1), T.sum(0))
        pur = [float(sum(pi[a] * np.bincount(w.cls, G[a], w.v).max() for a in range(self.K))) for G, pi in zip((GL, GR), pis)]            # mass a unit puts on its main class, weighted by use
        whole = [np.mean([((G[:, w.members[c]] > 0.5 / w.m).all(1) & (pi > 0.25 / w.v)).any() for c in range(w.v)]) for G, pi in zip((GL, GR), pis)]   # classes that a unit in use holds entirely
        return dict(kl=float((w.pstar[w.valid] * (np.log(w.pstar[w.valid]) - np.log(P[w.valid]))).sum()), valid=float(P[w.valid].sum()), purity=float(np.mean(pur)), whole=float(np.mean(whole)),
                    units=float(np.mean([np.exp(-(pi * np.log(pi + 1e-300)).sum()) for pi in pis])))                                              # effective number of units in use


class Grower:
    """Class units start as copies of the shallow codes (unit i relays code i). Weights are additive and bounded in [lo, s].
    watch: M += 1; the units that read the pair (one winner of the reader) are raised by eta_w (imprinting).
    practice: a pair of units from T writes (k, k'); M says seen (v = 1) or not (v = 0); every weight used moves by eta (v - c vbar), vbar the running mean of v.
      credit 'gen': the units that wrote; 'read': the units the reader assigns the pair to. spread False: a unit never tries a code other than its own; veto False: v = 1 always.
    private True: no reuse, one unit per seen pair with its own two pools (T is then the list of those units)."""

    def __init__(self, world, s=6.0, lo=-6.0, eta=0.3, eta_w=1.0, c=0.8, credit='read', veto=True, spread=True, private=False, seed=0):
        self.w, self.N, self.s, self.lo, self.eta, self.eta_w, self.c, self.cred, self.veto, self.spread, self.private = world, world.N, s, lo, eta, eta_w, c, credit, veto, spread, private; self.vbar = 1.0
        N = self.N; self.M = np.zeros((N, N)); self.n = dict(teacher=0, own=0, rejected=0)
        if private: self.LL, self.LR, self.used = np.zeros((0, N)), np.zeros((0, N)), {}
        else: self.LL, self.LR, self.TH = (s - (0 if spread else lo)) * np.eye(N) + (0 if spread else lo), (s - (0 if spread else lo)) * np.eye(N) + (0 if spread else lo), np.zeros((N, N))   # spread False: a unit never tries another code

    def tables(self):
        if self.private: R = len(self.LL); return softmax(self.LL, 1), softmax(self.LR, 1), np.full(R, 1 / max(R, 1))
        return softmax(self.LL, 1), softmax(self.LR, 1), softmax(self.TH.ravel()).reshape(self.N, self.N)

    def pairprob(self, tab=None):
        GL, GR, T = tab or self.tables(); return (GL.T * T) @ GR if self.private else GL.T @ T @ GR

    def read(self, k, kp, tab, rng, hard=False): GL, GR, T = tab; a = draw((GL[:, k] * (T @ GR)[:, kp]).T, rng, hard); return a, draw(T[a] * GR[:, kp].T, rng, hard)

    def bound(self):
        for L in (self.LL, self.LR) + (() if self.private else (self.TH,)): np.clip(L, self.lo, self.s, out=L)

    def watch(self, k, kp, rng):
        np.add.at(self.M, (k, kp), 1); self.n['teacher'] += len(k)
        if self.private:                                                                                                           # a new unit for every pair not seen before
            new = [(a, b) for a, b in zip(k, kp) if (a, b) not in self.used and not self.used.update({(a, b): 1})]
            if new: a, b = np.array(new).T; e = np.eye(self.N); self.LL, self.LR = np.vstack([self.LL, self.s * e[a]]), np.vstack([self.LR, self.s * e[b]])
            return
        a, ap = self.read(k, kp, self.tables(), rng, hard=True); np.add.at(self.LL, (a, k), self.eta_w); np.add.at(self.LR, (ap, kp), self.eta_w); np.add.at(self.TH, (a, ap), self.eta_w); self.bound()

    def sample(self, n, rng):
        GL, GR, T = self.tables(); c = rng.choice(self.N ** 2, n, p=T.ravel()); return draw(GL[c // self.N], rng), draw(GR[c % self.N], rng)

    def practice(self, n, rng, source=None):                                                                                       # source: pairs written by someone else (credit is then by reading)
        GL, GR, T = tab = self.tables()
        if self.private: r = rng.integers(len(GL), size=n); k, kp = draw(GL[r], rng), draw(GR[r], rng)
        elif source is not None: k, kp = source
        else: c = rng.choice(self.N ** 2, n, p=T.ravel()); a, ap = c // self.N, c % self.N; k, kp = draw(GL[a], rng), draw(GR[ap], rng)
        seen = self.M[k, kp] > 0; d = self.eta * ((seen if self.veto else 1.0) - self.c * self.vbar); self.n['own'] += n; self.n['rejected'] += (~seen).sum()   # verdict against a fraction c of its running mean
        self.vbar += 0.02 * (seen.mean() - self.vbar)
        if self.private: np.add.at(self.LL, (r, k), d); np.add.at(self.LR, (r, kp), d); self.bound(); return
        if self.cred == 'read' or source is not None: a, ap = self.read(k, kp, tab, rng)
        np.add.at(self.LL, (a, k), d); np.add.at(self.LR, (ap, kp), d); np.add.at(self.TH, (a, ap), d); self.bound()

    def score(self):
        w = self.w; GL, GR, T = tab = self.tables(); P = self.pairprob(tab); pis = (T, T) if self.private else (T.sum(1), T.sum(0))
        pur = [float(sum(pi[a] * np.bincount(w.cls, G[a], w.v).max() for a in range(len(G)))) for G, pi in zip((GL, GR), pis)]            # mass a unit puts on its main class, weighted by use
        whole = [np.mean([((G[:, w.members[c]] > 0.5 / w.m).all(1) & (G[:, w.cls != c].sum(1) < 0.1)).any() for c in range(w.v)]) for G in (GL, GR)]   # classes held entirely and purely by some unit
        new = w.valid & (self.M == 0)                                                                                                    # valid pairs the teacher has not shown
        return dict(kl=float((w.pstar[w.valid] * (np.log(w.pstar[w.valid]) - np.log(P[w.valid] + 1e-300))).sum()), valid=float(P[w.valid].sum()), purity=float(np.mean(pur)), whole=float(np.mean(whole)), cover=float((P[w.valid] > 0.25 / w.valid.sum()).mean()),
                    new=float(P[new].sum() / max(new.sum(), 1) * w.valid.sum()))                                                         # probability of an unshown valid pair, in units of its population value


def stream(L, w, rng, steps, share, every):                                                                                        # interleave batches of teacher and own pairs; own share of the samples
    acc, out = 0.0, []
    for t in range(1, steps + 1):
        acc += share
        if acc >= 1 - 1e-9: acc -= 1; L.practice(BATCH, rng)
        else: L.watch(*w.teacher(rng, BATCH), rng)
        if t % every == 0: out.append(L.score())
    return out


CHECK = np.unique(np.round(2 ** np.arange(4, 15.01, 0.5)).astype(int))                                                              # practice batches at which a run is scored


def table_kl(w, M):                                                                                                                # the counted table with the best uniform smoothing
    return min(float((w.pstar[w.valid] * (np.log(w.pstar[w.valid]) - np.log(((M + a) / (M + a).sum())[w.valid]))).sum()) for a in 10.0 ** np.arange(-6, 1, 0.5))


def recall_reader(w, M, c=0.8):                                                                                                    # no own samples: code j joins the unit of k if M has seen j with most of k's partners
    seen = M > 0; pools = []
    for X, S in ((M, seen), (M.T, seen.T)):
        A = (X @ S.T) / np.maximum(X.sum(1, keepdims=True), 1); held = np.eye(w.N, dtype=bool)
        for _ in range(8): held = (A >= c * A[held].mean()) | np.eye(w.N, dtype=bool)
        pools.append(held / held.sum(1, keepdims=True))
    P = pools[0].T @ (M / M.sum()) @ pools[1]
    return dict(kl=float((w.pstar[w.valid] * (np.log(w.pstar[w.valid]) - np.log(P[w.valid] + 1e-300))).sum()), valid=float(P[w.valid].sum()))


def practise(L, rng, batches, source=None):                                                                                        # scores at the batch counts in CHECK
    out, t = [], 0
    for c in CHECK[CHECK <= batches]:
        for _ in range(c - t): L.practice(BATCH, rng, **({} if source is None else dict(source=source())))
        t = c; out.append(L.score())
    return out


def watch(L, w, rng, n):
    for i in range(0, n, 256): L.watch(*w.teacher(rng, min(256, n - i)), rng)


def one_ablation(job):
    arm, kw, n, seed = job; w = World(seed=seed); rng = np.random.default_rng(1000 + seed); L = Grower(w, **kw); watch(L, w, rng, n)
    return arm, seed, [L.score()] + practise(L, rng, 32768), table_kl(w, L.M), recall_reader(w, L.M)


def ablation():                                                                                                                    # what M and each property of the prior do, at a fixed teacher set
    arms = dict(full={}, generator_credit=dict(credit='gen'), no_veto=dict(veto=False), no_spread=dict(spread=False), private=dict(private=True))
    jobs = [(arm, kw, 4096, seed) for arm, kw in arms.items() for seed in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for arm, seed, out, tab, rec in tqdm(p.imap_unordered(one_ablation, jobs), total=len(jobs)): res[arm, seed] = out; res['table', seed] = tab; res['recall', seed] = rec
    keys = list(res[('full', 0)][0]); np.savez(OUT / 'ablation.npz', check=CHECK[CHECK <= 32768] * BATCH, keys=keys, table=[res['table', s] for s in range(args.worlds)], recall=[res['recall', s]['kl'] for s in range(args.worlds)],
                                           **{arm: np.array([[[sc[k] for k in keys] for sc in res[arm, s]] for s in range(args.worlds)]) for arm in arms})


def one_efficiency(job):
    n, seed = job; w = World(seed=seed); rng = np.random.default_rng(2000 + seed); out = {}
    for c in (0.7, 0.8):
        L = Grower(w, c=c); r = np.random.default_rng(2000 + seed); watch(L, w, r, n); sc = practise(L, r, 32768)[-1]; out[f'practice{c}'] = sc['kl']; out[f'whole{c}'] = sc['whole']
    out['table'] = table_kl(w, L.M); out['recall'] = recall_reader(w, L.M)['kl']
    E = GradientLearner(w, K=w.v, start='random', lam=0, even=True, seed=seed); k, kp = w.teacher(rng, n); np.add.at(E.M, (k, kp), 1)             # bottleneck learner, wake only, rehearsing the stored set
    for _ in range(20000): i = rng.integers(n, size=BATCH); E.watch(k[i], kp[i], rng)
    out['bottleneck'] = E.score()['kl']
    return n, seed, out


def efficiency():                                                                                                                  # population loss against the size of the teacher set
    ns = 2 ** np.arange(9, 16); jobs = [(int(n), seed) for n in ns for seed in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for n, seed, out in tqdm(p.imap_unordered(one_efficiency, jobs), total=len(jobs)): res[n, seed] = out
    np.savez(OUT / 'efficiency.npz', n=ns, **{k: np.array([[res[int(n), s][k] for s in range(args.worlds)] for n in ns]) for k in res[int(ns[0]), 0]})


def one_stream(job):
    share, seed = job; w = World(seed=seed); rng = np.random.default_rng(3000 + seed); L = Grower(w); steps = 2 ** 16; out, acc = [], 0.0
    for t in range(1, steps + 1):
        acc += share
        if acc >= 1 - 1e-9: acc -= 1; L.practice(BATCH, rng)
        else: L.watch(*w.teacher(rng, BATCH), rng)
        if t & (t - 1) == 0 and t >= 16: out.append(L.score()['kl'])
    return share, seed, out


def heatmap():                                                                                                                     # population loss over total samples and own share, teacher and own batches interleaved
    shares = [0, 0.5, 0.75, 0.9, 0.97, 0.99, 0.997, 0.999]; jobs = [(sh, seed) for sh in shares for seed in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for sh, seed, out in tqdm(p.imap_unordered(one_stream, jobs), total=len(jobs)): res[sh, seed] = out
    np.savez(OUT / 'heatmap.npz', shares=shares, total=2 ** np.arange(4, 17) * BATCH, kl=np.array([[res[sh, s] for s in range(args.worlds)] for sh in shares]))


def one_source(job):
    src, seed = job; w = World(seed=seed); rng = np.random.default_rng(4000 + seed); L = Grower(w); watch(L, w, rng, 4096); twin = Grower(w); watch(twin, w, rng, 4096)   # the twin has its own teacher set
    def foreign(): twin.practice(BATCH, rng); return twin.sample(BATCH, rng)
    out = practise(L, rng, 32768, dict(own=None, twin=foreign, teacher=lambda: w.teacher(rng, BATCH))[src])
    return src, seed, [sc['kl'] for sc in out], twin.score()['kl']


def source():                                                                                                                      # whose samples to practise on, at equal counts
    jobs = [(src, seed) for src in ('own', 'twin', 'teacher') for seed in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for src, seed, out, tw in tqdm(p.imap_unordered(one_source, jobs), total=len(jobs)): res[src, seed] = out; res['twinkl', src, seed] = tw
    np.savez(OUT / 'source.npz', check=CHECK * BATCH, twin_kl=[res['twinkl', 'twin', s] for s in range(args.worlds)], **{src: np.array([res[src, s] for s in range(args.worlds)]) for src in ('own', 'twin', 'teacher')})


def one_gradient(job):
    name, kw, share, steps, seed = job; w = World(seed=seed); rng = np.random.default_rng(5000 + seed); L = GradientLearner(w, K=w.v, lam=0, even=True, seed=seed, **kw); L.watch(*w.teacher(rng, 8192), rng, wake=False)
    sc = stream(L, w, rng, steps, share, steps)[-1]
    return name, seed, sc | dict(el_teacher=L.n['el_teacher'] / max(L.n['teacher'] - 8192, 1), el_own=L.n['el_own'] / max(L.n['own'], 1))


def gradient():                                                                                                                    # the likelihood-gradient learner with v units: starts, threshold, wake with and without practice
    jobs = [(f'{start}|{theta}|{arm}', dict(start=start, s=sh, theta=theta), share, 30000, seed) for start, sh in (('noise', 0.1), ('noise', 1.0), ('random', 6.0)) for theta in (0, 1)
            for arm, share in (('wake', 0), ('practice', 0.5)) for seed in range(args.worlds)]
    jobs = [(n.replace('noise', f'noise{kw["s"]}'), kw, sh, st, sd) for n, kw, sh, st, sd in jobs]; res = {}
    with Pool(args.procs) as p:
        for name, seed, sc in tqdm(p.imap_unordered(one_gradient, jobs), total=len(jobs)): res.setdefault(name, []).append(sc)
    import json; (OUT / 'gradient.json').write_text(json.dumps({name: {k: [float(x[k]) for x in r] for k in r[0]} for name, r in res.items()}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=28); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); globals()[args.run]()
