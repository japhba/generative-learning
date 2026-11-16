"""Model 5, follow-up (16 Nov 2026): why own samples buy nothing against fresh teacher pairs in one block, and where they do.
Runs:
  bound   one block, teacher sets of 256 to 32,768 pairs: the counted table, a table over the TRUE classes (what naming the classes could be worth),
          and a table over classes found from M alone by clustering the codes on their partner profiles (are the classes in M at that size?).
  sparse  one block: practice with the verdict of the note (Grower) and with a verdict pooled over the partner unit (Pooler), by teacher-set size
          (--ns 512,1024,2048,4096,8192,16384,32768 for the figure).
  depth   three levels (strings of four codes, model5_depth.World3) at a fixed budget of strings, of which a share are the learner's own: half the teacher
          strings, practice on both halves of the string, then the other half read through the named units and counted. Share 0: the table of whole strings.
Writes /var/tmp/jbauer/model5_lateral_verdict/own_benefit_<run>.npz.   Usage: python model5_own_benefit.py <run> [--worlds 16] [--procs 28]"""

import argparse
from multiprocessing import Pool

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from tqdm.auto import tqdm

from model5_depth import World3
from model5_lateral_verdict import BATCH, CHECK, OUT, V, Grower, World, draw, practise, table_kl, watch

ALPHAS = 10.0 ** np.arange(-6, 1, 0.5)


def kl(w, P): return float((w.pstar[w.valid] * (np.log(w.pstar[w.valid]) - np.log(P[w.valid] + 1e-300))).sum())


def class_table(w, M, cl, cr):                                                                                                     # counts over pairs of classes (cl, cr: code -> class), uniform within a class, best uniform smoothing
    C = np.zeros((cl.max() + 1, cr.max() + 1)); np.add.at(C, (cl[:, None], cr[None, :]), M); size = np.outer(np.bincount(cl)[cl], np.bincount(cr)[cr])
    return min(kl(w, ((C + a) / (C + a).sum())[np.ix_(cl, cr)] / size) for a in ALPHAS)


def cluster(F, v):                                                                                                                 # v groups of the rows of F by their normalised profiles
    F = F / np.maximum(F.sum(1, keepdims=True), 1); return fcluster(linkage(F, 'ward'), v, 'maxclust') - 1


def found_classes(w, M, rounds=6):                                                                                                 # alternate: left codes by their profile over right groups, right codes by their profile over left groups
    cl, cr = np.arange(w.N), np.arange(w.N)
    for _ in range(rounds):
        cl = cluster(np.stack([M[:, cr == r].sum(1) for r in range(cr.max() + 1)], 1), w.v); cr = cluster(np.stack([M[cl == l].sum(0) for l in range(cl.max() + 1)], 1), w.v)
    return cl, cr


def one_bound(job):
    n, seed = job; w = World(seed=seed); rng = np.random.default_rng(2000 + seed); k, kp = w.teacher(rng, n); M = np.zeros((w.N, w.N)); np.add.at(M, (k, kp), 1); cl, cr = found_classes(w, M)
    right = np.mean([len(set(c[w.members[a]])) == 1 for c in (cl, cr) for a in range(w.v)])                                         # true classes that fall into one found group
    return n, seed, dict(table=table_kl(w, M), oracle=class_table(w, M, w.cls, w.cls), found=class_table(w, M, cl, cr), right=float(right), shown=float((M[w.valid] > 0).mean()))


def bound():
    ns = 2 ** np.arange(8, 16); jobs = [(int(n), seed) for n in ns for seed in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for n, seed, out in tqdm(p.imap_unordered(one_bound, jobs), total=len(jobs)): res[n, seed] = out
    np.savez(OUT / 'own_benefit_bound.npz', n=ns, **{k: np.array([[res[int(n), s][k] for s in range(args.worlds)] for n in ns]) for k in res[int(ns[0]), 0]})
    for n in ns: print(n, {k: round(float(np.median([res[int(n), s][k] for s in range(args.worlds)])), 3) for k in res[int(n), 0]})


class Pooler(Grower):
    """Grower with a verdict that works on a sparse M.
    pooled: a written code is judged against what the partner UNIT stands for, not against the partner's written code: v_L = M has seen k with a code of the pool
      of a', v_R = M has seen k' with a code of the pool of a (pool: the codes a unit writes at least half as often as its most frequent one). At the start a
      pool is one code and this is the verdict of the Grower; once units grow, a valid pair the teacher never showed is accepted through the pair's neighbours.
    calibrated: the bar is c times the rate at which the same verdict accepts TEACHER pairs (stored pairs replayed from M, their own count left out), a running
      mean rho per side, instead of c times the running mean of the verdict on own samples. The table T keeps the bar of the Grower (its own running mean), since a
      pair of units is not to blame for what its pools try."""

    def __init__(self, world, pooled=True, calibrated=True, **kw): super().__init__(world, **kw); self.pooled, self.cal, self.rho, self.Mp = pooled, calibrated, None, None

    def verdict(self, k, kp, a, ap, HL, HR, loo=False):
        if not self.pooled: v = self.M[k, kp] - loo > 0; return v, v
        return (self.M[k] * HR[ap]).sum(1) - (HR[ap, kp] if loo else 0) > 0, (self.M[:, kp].T * HL[a]).sum(1) - (HL[a, k] if loo else 0) > 0

    def practice(self, n, rng, source=None):
        GL, GR, T = tab = self.tables(); N = self.N; HL, HR = GL >= 0.5 * GL.max(1, keepdims=True), GR >= 0.5 * GR.max(1, keepdims=True)
        if self.cal:                                                                                                                # the bar: how often the verdict accepts a teacher pair
            if self.Mp is None: self.Mp = self.M.ravel() / self.M.sum()
            i = rng.choice(N * N, n if self.rho is not None else 4096, p=self.Mp); tk, tkp = i // N, i % N; ta, tap = self.read(tk, tkp, tab, rng, hard=True); vL, vR = self.verdict(tk, tkp, ta, tap, HL, HR, loo=True)
            r = np.array([vL.mean(), vR.mean()]); self.rho = r if self.rho is None else self.rho + 0.05 * (r - self.rho)
        if source is not None: k, kp = source
        else: c = rng.choice(N ** 2, n, p=T.ravel()); a, ap = c // N, c % N; k, kp = draw(GL[a], rng), draw(GR[ap], rng)
        if self.cred == 'read' or source is not None: a, ap = self.read(k, kp, tab, rng)
        vL, vR = self.verdict(k, kp, a, ap, HL, HR); both = vL & vR; self.n['own'] += n; self.n['rejected'] += (~both).sum(); bar = self.c * (np.array([self.rho[0], self.rho[1], self.vbar]) if self.cal else np.full(3, self.vbar)); self.vbar += 0.02 * (both.mean() - self.vbar)   # T: against its own running mean, as the pools explore
        np.add.at(self.LL, (a, k), self.eta * (vL - bar[0])); np.add.at(self.LR, (ap, kp), self.eta * (vR - bar[1])); np.add.at(self.TH, (a, ap), self.eta * (both - bar[2])); self.bound()

    def watch(self, k, kp, rng): super().watch(k, kp, rng); self.Mp = None


ARMS = dict(grower=(Grower, {}), pooled_only=(Pooler, dict(calibrated=False)), pooled=(Pooler, {}), eta1=(Pooler, dict(eta=1.0)), s5=(Pooler, dict(s=5.0, lo=-7.0)),      # eta1: the learner of the note's section 6 and of the depth run
            gen=(Pooler, dict(credit='gen')), s4=(Pooler, dict(s=4.0, lo=-8.0)), s5_eta1=(Pooler, dict(s=5.0, lo=-7.0, eta=1.0)),                                              # probes (8 worlds): no faster, s4 and s5_eta1 fail at 2,048 pairs
            fast=(Pooler, dict(s=4.0, lo=-8.0, eta=0.6, credit='gen')), imprint=(Pooler, dict(eta_w=6.0)))                                                                     # probes: fail (fast), end worse (table imprinted at once)


def one_sparse(job):
    arm, n, seed = job; w = World(seed=seed); rng = np.random.default_rng(2000 + seed); cls, kw = ARMS[arm]; L = cls(w, **kw); watch(L, w, rng, n); sc = [L.score()] + practise(L, rng, args.batches)
    return arm, n, seed, np.array([[s[k] for k in ('kl', 'valid', 'whole', 'new')] for s in sc])


def sparse():                                                                                                                      # one block: practice against the size of the teacher set, verdict variants
    ns = [int(x) for x in args.ns.split(',')]; arms = args.arms.split(','); jobs = [(arm, n, seed) for arm in arms for n in ns for seed in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for arm, n, seed, out in tqdm(p.imap_unordered(one_sparse, jobs), total=len(jobs)): res[arm, n, seed] = out
    x = np.concatenate([[0], CHECK[CHECK <= args.batches] * BATCH]); np.savez(OUT / f'own_benefit_sparse{args.tag}.npz', n=ns, own=x, **{arm: np.array([[res[arm, n, s] for s in range(args.worlds)] for n in ns]) for arm in arms})
    for arm in arms:
        for i, n in enumerate(ns):
            kl = np.median([res[arm, n, s][:, 0] for s in range(args.worlds)], 0); j = int(np.argmax(kl <= 1.2 * kl[-1]))
            print(f'{arm:16s} n={n:6d}  start {kl[0]:.3f}  end {kl[-1]:.3f}  min {kl.min():.3f}  below 0.1 after {int(x[np.argmax(kl <= 0.1)]) if (kl <= 0.1).any() else -1}  within 20% of the end after {int(x[j])} own samples  classes {np.median([res[arm, n, s][-1, 2] for s in range(args.worlds)]):.2f}  worst world {max(res[arm, n, s][-1, 0] for s in range(args.worlds)):.3f}')


def unit_names(G, side):                                                                                                           # code -> name, the reader's most probable group of units. Practice leaves several units per class and a few that
    GL, GR, T = G.tables(); P, pi = (GL, T.sum(1)) if side == 0 else (GR, T.sum(0)); H = P >= 0.5 * P.max(1, keepdims=True)         # mix two classes: units whose pools all overlap by more than half are one group (complete linkage), and a code
    grp = fcluster(linkage(H, 'complete', metric='jaccard'), 0.5, 'distance') - 1; mass = np.zeros((grp.max() + 1, G.N)); np.add.at(mass, grp, P * pi[:, None]); return mass.argmax(0)   # is read as the group with the most posterior mass


def string_loss(W, pop, x, nm):                                                                                                    # strings x read through the names nm (one map per position), pairs of named pairs counted; loss on the population sample
    N = W.w.N; key = lambda s: ((nm[0][s[:, 0]] * N + nm[1][s[:, 1]]) * N + nm[2][s[:, 2]]) * N + nm[3][s[:, 3]]
    u, c = np.unique(key(x), return_counts=True); kp = key(pop); i = np.minimum(np.searchsorted(u, kp), len(u) - 1); cc = np.where(u[i] == kp, c[i], 0)
    pools = -sum(np.log(np.bincount(nm[j], minlength=N)[nm[j][pop[:, j]]]) for j in range(4))                                      # a name writes the codes that are read as it, each equally often
    return min(float(-W.logn - (np.log((cc + a) / (len(x) + a * N ** 4)) + pools).mean()) for a in 10.0 ** np.arange(-12, 0.1, 0.5))


def one_depth(job):                                                                                                                # three levels, a budget of B strings of which a share f are the learner's own
    B, f, seed = job; W = World3(seed); w = W.w; rng = np.random.default_rng(30_000 + seed); pop = W.strings(rng, 20000); P = int(round(f * B)); n = B - P; x = W.strings(rng, n); n1 = n // 2; ident = [np.arange(w.N)] * 4; out = {}
    if f == 0:
        out['flat'] = string_loss(W, pop, x, ident)                                                                                # no offline phase: nothing is named, the pairs of pairs it counts are whole strings
        M = [np.zeros((w.N, w.N)) for _ in range(2)]
        for m, c in zip(M, ((0, 1), (2, 3))): np.add.at(m, (x[:n1, c[0]], x[:n1, c[1]]), 1)
        nm = [c for m in M for c in found_classes(w, m)]; out['clustered'] = string_loss(W, pop, x[n1:], nm)                         # classes from M in closed form after half the strings, the other half counted
        out['oracle'] = string_loss(W, pop, x, [w.cls] * 4); return B, f, seed, out
    gs = [Pooler(w, **DEPTH_KW), Pooler(w, **DEPTH_KW)]
    for g, c in zip(gs, ((0, 1), (2, 3))):
        for i in range(0, n1, 256): g.watch(x[i:min(i + 256, n1), c[0]], x[i:min(i + 256, n1), c[1]], rng)
        for _ in range(P // BATCH): g.practice(BATCH, rng)
    nm = [unit_names(gs[0], 0), unit_names(gs[0], 1), unit_names(gs[1], 0), unit_names(gs[1], 1)]; out['mix'] = string_loss(W, pop, x[n1:], nm)
    out['named'] = float(np.mean([len(set(nm[i][w.members[k]])) == 1 and (nm[i] == nm[i][w.members[k][0]]).sum() == w.m for i in range(4) for k in range(w.v)])); out['names'] = float(np.mean([len(set(nm[i])) for i in range(4)])); out['blocks'] = float(np.mean([g.score()['kl'] for g in gs]))   # named: classes that are exactly one name
    return B, f, seed, out


DEPTH_KW = dict(eta=1.0)


def depth():
    def distinct(B): return len({r.tobytes() for r in B}) == len(B) and len({r.tobytes() for r in B.T}) == len(B)                   # no two classes with the same partners: such classes are one class for a block
    seeds = [s for s in range(64) if distinct(World(seed=s).B)][:args.worlds]                                                       # and differ only one level up
    Bs = 2 ** np.arange(15, 24); fs = [0, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99]; jobs = sorted([(int(B), f, seed) for B in Bs for f in fs for seed in seeds], key=lambda j: -j[0] * (j[1] + 0.3)); res = {}
    with Pool(args.procs) as p:
        for B, f, seed, out in tqdm(p.imap_unordered(one_depth, jobs), total=len(jobs)): res[B, f, seed] = out
    arr = lambda k, f: np.array([[res[int(B), f, s][k] for s in seeds] for B in Bs])
    np.savez(OUT / f'own_benefit_depth{args.tag}.npz', total=Bs, shares=fs, flat=arr('flat', 0), clustered=arr('clustered', 0), oracle=arr('oracle', 0), **{k: np.array([arr(k, f) for f in fs[1:]]) for k in ('mix', 'named', 'names', 'blocks')})
    for B in Bs: print(B, 'flat', round(float(np.median(arr('flat', 0)[list(Bs).index(B)])), 3), 'clustered', round(float(np.median(arr('clustered', 0)[list(Bs).index(B)])), 3), 'mix by share', {f: round(float(np.median([res[int(B), f, s]['mix'] for s in seeds])), 3) for f in fs[1:]}, 'names', {f: round(float(np.median([res[int(B), f, s]['names'] for s in seeds])), 1) for f in fs[1:]})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--ns', default='1024,2048,4096'); ap.add_argument('--arms', default='pooled_only,pooled,eta1,s5'); ap.add_argument('--batches', type=int, default=32768); ap.add_argument('--tag', default=''); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); globals()[args.run]()
