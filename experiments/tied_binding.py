"""Binding under a level above: a two-level winner-take-all network on the tied block world, sampled strings.

World: C checks of three blocks (symbols in Z_q, a check holds if its symbols sum to 0 mod q, its class is s1 - s2). 'tied': checks come in
pairs whose classes agree (a level above writes one class per pair, each check renders it in one of q ways); 'free': checks independent.
Teacher uniform on the valid strings. Single symbols and symbol pairs are uniform in both worlds.

Learner (separate reader R and writer G, weights = log probabilities, weak random start, rule P <- P + eps (onehot - P) at the credited unit):
  level 1: G1 = C groups of K1 = q^2 states, every group wired to every block (the learner is not told which blocks form a check).
  level 2: G2 = C / 2 groups of K2 states, every group wired to every level-1 group.
  writing: level-2 states from a uniform prior; a level-1 group draws its state from softmax(sum over level-2 groups of lift), a block its
           symbol from softmax(sum over level-1 groups of lift), lift = log G - log of the group's average (as in order3_net.py).
  reading: every level-1 group takes softmax_k(sum_b log R1[g, k, b, h_b]) on its own; level 2 reads the level-1 states drawn.
  verifier: one unit per check (local wiring given) that has counted the triples teacher strings showed; a sample passes if every check unit
           has seen its triple. Learned during watching.
Credit of an own sample: the writing states ('sleep'), or the reading of the own string ('strict').

    python tied_binding.py --out /var/tmp/jbauer/tied_binding/main.json
"""
import argparse, itertools, json, os, time
from multiprocessing import Pool
import numpy as np

FLOOR = 1e-4
sm = lambda u, ax=-1: (lambda e: e / e.sum(ax, keepdims=True))(np.exp(u - u.max(ax, keepdims=True)))


class World:
    def __init__(self, C, q, tied):
        self.C, self.q, self.B, self.tied = C, q, 3 * C, tied
        self.checks = np.arange(self.B).reshape(C, 3)

    def teacher(self, rng, n):
        q, C = self.q, self.C
        cls = rng.integers(q, size=(n, C))
        if self.tied: cls[:, 1::2] = cls[:, 0::2]
        a = rng.integers(q, size=(n, C))
        b = (a - cls) % q
        c = (-a - b) % q
        return np.stack([a, b, c], 2).reshape(n, self.B)

    def all_strings(self):
        return np.array(list(itertools.product(range(self.q), repeat=self.B)))

    def valid(self, x):
        ok = (x[:, self.checks].sum(2) % self.q == 0).all(1)
        if self.tied:
            cls = (x[:, self.checks[:, 0]] - x[:, self.checks[:, 1]]) % self.q
            ok &= (cls[:, 1::2] == cls[:, 0::2]).all(1)
        return ok

    def checks_ok(self, x):
        return (x[:, self.checks].sum(2) % self.q == 0)


class Net:
    def __init__(self, rng, w, K2, a):
        self.w, self.K1, self.K2 = w, w.q ** 2, K2
        G1n, G2n = w.C, max(1, w.C // 2)
        n1, n2 = rng.normal(size=(G1n, self.K1, w.B, w.q)), rng.normal(size=(G2n, K2, G1n, self.K1))
        self.G1, self.R1 = sm(a * n1), sm(a * n1)
        self.G2, self.R2 = sm(a * n2), sm(a * n2)
        self.L = np.zeros((G1n, G1n, self.K1, self.K1))                                          # lateral inhibition between reader groups
        self.seen = np.zeros((w.C, w.q, w.q, w.q), bool)                                         # verifier: triples each check unit has counted

    # ---------------- writing
    def lift(self, P):
        return np.log(P + FLOOR) - np.log(P.mean(1, keepdims=True) + FLOOR)

    def write(self, rng, n):
        w, l1, l2 = self.w, self.lift(self.G1), self.lift(self.G2)
        k2 = rng.integers(self.K2, size=(n, len(self.G2)))
        logit1 = sum(l2[g, k2[:, g]] for g in range(len(self.G2)))                              # [n, G1, K1]
        k1 = (logit1 + rng.gumbel(size=logit1.shape)).argmax(-1)
        logit0 = sum(l1[g, k1[:, g]] for g in range(len(self.G1)))                              # [n, B, q]
        x = (logit0 + rng.gumbel(size=logit0.shape)).argmax(-1)
        return k2, k1, x

    def marginal(self, X):
        """p_writer(x) for the strings X, by enumeration of the joint states (small C only)."""
        l1, l2 = self.lift(self.G1), self.lift(self.G2)
        G1n, G2n = len(self.G1), len(self.G2)
        J1 = np.array(list(itertools.product(range(self.K1), repeat=G1n)))                      # [J1, G1]
        J2 = np.array(list(itertools.product(range(self.K2), repeat=G2n)))
        logit1 = sum(l2[g, J2[:, g]] for g in range(G2n))                                       # [J2, G1, K1]
        lp1 = np.log(sm(logit1))
        p_k1 = np.exp(sum(lp1[:, g, J1[:, g]] for g in range(G1n))).mean(0)                     # [J1] (uniform prior on J2)
        logit0 = sum(l1[g, J1[:, g]] for g in range(G1n))                                       # [J1, B, q]
        lp0 = np.log(sm(logit0))
        lpx = lp0[:, np.arange(self.w.B)[None, :], X].sum(2)                                    # [J1, N]
        return p_k1 @ np.exp(lpx)

    # ---------------- reading
    def read1(self, x, rng, inhibit=False):
        u = np.einsum('gkbn->ngk', np.log(self.R1 + FLOOR)[:, :, np.arange(self.w.B)[:, None], x.T])  # [n, G1, K1]
        if inhibit and len(self.G1) == 2:
            v = u[:, 0, :, None] + u[:, 1, None, :] - self.L[0, 1][None]
            flat = (v.reshape(len(x), -1) + rng.gumbel(size=(len(x), self.K1 ** 2))).argmax(1)
            return np.stack([flat // self.K1, flat % self.K1], 1), u
        return (u + rng.gumbel(size=u.shape)).argmax(-1), u

    def read2(self, k1, rng):
        u = sum(np.log(self.R2 + FLOOR)[:, :, g, k1[:, g]] for g in range(len(self.G1)))         # [G2, K2, n]
        u = np.moveaxis(u, 2, 0)
        return (u + rng.gumbel(size=u.shape)).argmax(-1)

    def exact_post(self, x, rng):
        """a draw of (k2, k1) from the writer's own posterior given x (the ideal reader)."""
        l1, l2 = self.lift(self.G1), self.lift(self.G2)
        G1n, G2n = len(self.G1), len(self.G2)
        J1 = np.array(list(itertools.product(range(self.K1), repeat=G1n)))
        J2 = np.array(list(itertools.product(range(self.K2), repeat=G2n)))
        lp1 = np.log(sm(sum(l2[g, J2[:, g]] for g in range(G2n))))                              # [J2, G1, K1]
        lpk1 = sum(lp1[:, g, J1[:, g]] for g in range(G1n))                                     # [J2, J1]
        lp0 = np.log(sm(sum(l1[g, J1[:, g]] for g in range(G1n))))                              # [J1, B, q]
        lpx = lp0[:, np.arange(self.w.B)[None, :], x].sum(2)                                    # [J1, n]
        lj = lpk1[:, :, None] + lpx[None]                                                       # [J2, J1, n]
        flat = lj.reshape(-1, len(x)).T
        idx = (flat + rng.gumbel(size=flat.shape)).argmax(1)
        return J2[idx // len(J1)], J1[idx % len(J1)]


def confidence(net, x):
    u = np.einsum('gkbn->ngk', np.log(net.R1 + FLOOR)[:, :, np.arange(net.w.B)[:, None], x.T])
    return sm(u).max(-1).prod(-1)


def hebb(P, states, symbols, eps):
    """P[g, k, b, s] <- moves toward onehot(symbols[n, b]) at the credited state states[n, g]."""
    G, K, Bb, q = P.shape
    out = P.copy()
    for g in range(G):
        s1 = np.eye(K)[states[:, g]]
        count = s1.sum(0)
        if count.sum() == 0: continue
        target = np.einsum('nk,nbs->kbs', s1, np.eye(q)[symbols]) / np.maximum(count, 1)[:, None, None]
        keep = ((1 - eps) ** count)[:, None, None]
        out[g] = keep * P[g] + (1 - keep) * target
    return out


def binding(net, X, pT, rng):
    """share of each check's triple entropy that each reader level-1 group's posterior holds (normalized MI) [G1, C], over teacher strings."""
    w = net.w
    u = np.einsum('gkbn->ngk', np.log(net.R1 + FLOOR)[:, :, np.arange(w.B)[:, None], X.T])
    y = sm(u)                                                                                  # [N, G1, K1]
    tri = X[:, w.checks[:, 0]] * w.q + X[:, w.checks[:, 1]]                                     # [N, C]
    out = np.zeros((len(net.G1), w.C))
    for g in range(len(net.G1)):
        for c in range(w.C):
            j = np.einsum('n,nk,nt->kt', pT, y[:, g], np.eye(w.q ** 2)[tri[:, c]])
            pk, pt = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
            out[g, c] = np.sum(j * np.log(np.maximum(j, 1e-300) / np.maximum(pk @ pt, 1e-300))) / np.log(w.q ** 2)
    return out


def assign(b):
    """best one-to-one assignment score of groups to checks (mean normalized MI on the matched pairs) and the worst matched pair."""
    G, C = b.shape
    best = max(itertools.permutations(range(C), G), key=lambda p: sum(b[g, p[g]] for g in range(G)))
    vals = [b[g, best[g]] for g in range(G)]
    return float(np.mean(vals)), float(min(vals))


def deep_reading(net, X, pT, rng, n=4000):
    """normalized MI between the level-2 reading of teacher strings and the teacher's pair class (tied world)."""
    w = net.w
    idx = rng.choice(len(X), size=n, p=pT)
    x = X[idx]
    k1, _ = net.read1(x, rng)
    k2 = net.read2(k1, rng)
    cls = (x[:, w.checks[0, 0]] - x[:, w.checks[0, 1]]) % w.q
    j = np.zeros((net.K2, w.q))
    np.add.at(j, (k2[:, 0], cls), 1); j /= j.sum()
    pk, pc = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    return float(np.sum(j * np.log(np.maximum(j, 1e-300) / np.maximum(pk @ pc, 1e-300))) / np.log(w.q))


def evaluate(net, w, X, valid, rng):
    pT = valid / valid.sum()
    p = net.marginal(X)
    kl = float(np.sum(pT[valid] * np.log(pT[valid] / np.maximum(p[valid], 1e-12))))
    b = binding(net, X, pT, rng)
    sc, worst = assign(b)
    k2, k1, x = net.write(rng, 4000)
    chk = w.checks_ok(x)
    out = dict(kl=kl, valid=float(p[valid].sum()), checks=float(chk.mean()), bind=sc, bind_worst=worst, bindmat=b.round(3).tolist())
    if w.tied:
        out['deep'] = deep_reading(net, X, pT, rng)
    # surface: largest deviation of a symbol pair frequency from the teacher's (uniform), own samples
    H = np.eye(w.q)[x]
    out['pairs'] = float(max(abs(np.einsum('ns,nt->st', H[:, i], H[:, j]) / len(x) - 1 / w.q ** 2).max() for i, j in itertools.combinations(range(w.B), 2)))
    return out


ARMS = {
    'watching (teacher only)':           dict(own=0.),
    'classic wake-sleep':                dict(own=.5, sleep=True, verdict=False, ws=True),
    'sleep only':                        dict(own=.5, sleep=True, verdict=False),
    'practice (sleep + verdict)':        dict(own=.5, sleep=True, verdict=True),
    'practice, verdict also on level 2': dict(own=.5, sleep=True, verdict=True, verdict2=True),
    'practice, verdict only':            dict(own=.5, sleep=False, verdict=True),
    'practice, strict credit':           dict(own=.5, sleep=True, verdict=True, strict=True),
    'twin samples (credit by reading)':  dict(own=.5, sleep=True, verdict=True, twin=True),
    'equal-validity foreign (credit by reading)': dict(own=.5, sleep=True, verdict=True, foreign=True),
    'offline practice after watching':   dict(own=1., sleep=True, verdict=True, offline=True),
    'watching + inhibition':             dict(own=0., inhibit=True),
    'watching + inhibition (c 3)':       dict(own=0., inhibit=True, c=3),
    'watching + inhibition (c 300)':     dict(own=0., inhibit=True, c=300),
    'watching, exact posterior':         dict(own=0., exact=True),
}


def run(job):
    arm, world, seed, C, q, K2, a, eps, n_samples, watch = job[:10]
    stage = job[10] if len(job) > 10 else 0
    slow2 = job[11] if len(job) > 11 else 1.
    gate2 = job[12] if len(job) > 12 else 0.
    pass_on = job[13] if len(job) > 13 else 0.             # level 2 plastic once a probe of own samples passes the checks this often
    unlocked, t_unlock = pass_on == 0, None
    cfg = dict(ARMS[arm.split(' @ ')[0]])
    if ' @ ' in arm: cfg['own'] = float(arm.split(' @ ')[1])                               # 'practice (sleep + verdict) @ 0.25'
    if cfg.get('offline'): watch = max(watch, n_samples / 4)
    rng = np.random.default_rng([seed, 7])
    w = World(C, q, world == 'tied')
    X = w.all_strings(); valid = w.valid(X)
    net = Net(np.random.default_rng([seed, 1]), w, K2, a)
    twin = Net(np.random.default_rng([seed, 2]), w, K2, a) if cfg.get('twin') else None
    batch, curve, t, n_own = 64, [], 0, 0
    marks = set(np.unique(np.round(np.geomspace(batch, n_samples, 30) / batch).astype(int)).tolist()) | {0}
    for i in range(int(n_samples / batch) + 1):
        if i in marks:
            curve.append(dict(samples=i * batch, **evaluate(net, w, X, valid, rng)))
        own_now = cfg['own'] > 0 and i * batch >= watch and n_own < cfg['own'] * (i - watch / batch + 1)
        if not unlocked and i % 50 == 0:
            _, _, xp = net.write(rng, 256)
            if (net.seen[np.arange(w.C)[None, :], xp[:, w.checks[:, 0]], xp[:, w.checks[:, 1]], xp[:, w.checks[:, 2]]]).all(1).mean() >= pass_on:
                unlocked, t_unlock = True, i * batch
        e2 = eps * slow2 if (i * batch >= stage and unlocked) else 0.
        for m in [net] + ([twin] if twin is not None else []):
            if own_now and not (m is twin):
                src = twin if twin is not None else m
                k2, k1, x = src.write(rng, batch)
                if cfg.get('foreign'):                                                          # teacher strings, each replaced by a uniform string
                    v_own = w.valid(x).mean()                                                   # with the probability that keeps validity equal to own samples'
                    xf = w.teacher(rng, batch)
                    pu = valid.mean(); junk = rng.random(batch) > max(v_own - pu, 0) / (1 - pu)
                    xf[junk] = rng.integers(w.q, size=(junk.sum(), w.B))
                    x = xf
                V = (m.seen[np.arange(w.C)[None, :], x[:, w.checks[:, 0]], x[:, w.checks[:, 1]], x[:, w.checks[:, 2]]]).all(1)
                if cfg.get('strict') or twin is not None or cfg.get('foreign'):
                    k1, _ = m.read1(x, rng); k2 = m.read2(k1, rng)
                g2 = rng.random(batch) < confidence(m, x) ** gate2 if gate2 else np.ones(batch, bool)   # level 2 learns from confidently read strings
                if cfg.get('sleep'):
                    m.R1, m.R2 = hebb(m.R1, k1, x, eps), hebb(m.R2, k2[g2], k1[g2], e2)
                if cfg.get('verdict') and V.any():
                    m.G1 = hebb(m.G1, k1[V], x[V], eps)
                    if cfg.get('verdict2'): m.G2 = hebb(m.G2, k2[V & g2], k1[V & g2], e2)              # the verdict checks level 1 only
            else:
                x = w.teacher(rng, batch)
                m.seen[np.arange(w.C)[None, :].repeat(batch, 0), x[:, w.checks[:, 0]], x[:, w.checks[:, 1]], x[:, w.checks[:, 2]]] = True
                if cfg.get('exact'):
                    k2, k1 = m.exact_post(x, rng)
                else:
                    k1, u = m.read1(x, rng, inhibit=cfg.get('inhibit', False))
                    k2 = m.read2(k1, rng)
                g2 = rng.random(batch) < confidence(m, x) ** gate2 if gate2 else np.ones(batch, bool)
                if not cfg.get('ws'):
                    m.R1, m.R2 = hebb(m.R1, k1, x, eps), hebb(m.R2, k2[g2], k1[g2], e2)
                m.G1, m.G2 = hebb(m.G1, k1, x, eps), hebb(m.G2, k2[g2], k1[g2], e2)
                if cfg.get('inhibit'):
                    y = sm(u)
                    cov = np.einsum('nk,nl->kl', y[:, 0], y[:, 1]) / batch - np.outer(y[:, 0].mean(0), y[:, 1].mean(0))
                    m.L[0, 1] = np.maximum(0, m.L[0, 1] + eps * cfg.get('c', 30) * m.K1 ** 2 * cov)
        if own_now: n_own += 1
    return dict(arm=arm, world=world, seed=seed, C=C, curve=curve, t_unlock=t_unlock)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/var/tmp/jbauer/tied_binding/main.json')
    ap.add_argument('--samples', type=float, default=6e5); ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--arms', default='|'.join(ARMS)); ap.add_argument('--worlds', default='tied,free')
    ap.add_argument('--a', type=float, default=.1); ap.add_argument('--eps', type=float, default=.003); ap.add_argument('--watch', type=float, default=0); ap.add_argument('--stage', type=float, default=0); ap.add_argument('--slow2', type=float, default=1.); ap.add_argument('--gate2', type=float, default=0.); ap.add_argument('--pass_on', type=float, default=0.)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    jobs = [(arm, world, s, 2, 3, 3, args.a, args.eps, args.samples, args.watch, args.stage, args.slow2, args.gate2, args.pass_on) for world in args.worlds.split(',') for arm in args.arms.split('|') for s in range(args.seeds)]
    t0 = time.time()
    with Pool(min(30, len(jobs))) as pool:
        res = pool.map(run, jobs, chunksize=1)
    json.dump(res, open(args.out, 'w'))
    print(f'{time.time() - t0:.0f} s')
    for world in args.worlds.split(','):
        print(f'\nworld {world}: end of run (median over seeds) and number of seeds with KL < 0.1')
        for arm in args.arms.split('|'):
            rs = [r for r in res if r['arm'] == arm and r['world'] == world]
            e = [r['curve'][-1] for r in rs]
            med = lambda k: np.median([x[k] for x in e]) if k in e[0] else float('nan')
            tu = [r['t_unlock'] for r in rs if r.get('t_unlock') is not None]
            print(f"  {arm:34s} unlocked {len(tu)}/{len(rs)} (median {np.median(tu) / 1e3 if tu else float('nan'):.0f}k)  KL {med('kl'):6.3f}  solved {sum(x['kl'] < .1 for x in e)}/{len(e)}  valid {med('valid'):.3f}  checks {med('checks'):.3f}  bind {med('bind'):.2f} (worst {med('bind_worst'):.2f})  deep {med('deep'):.2f}  pairs {med('pairs'):.3f}")
