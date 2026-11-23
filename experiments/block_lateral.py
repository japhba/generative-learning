"""Binding without the writing state: lateral inhibition between the deep groups of the reader, learned from the prior (exact mean dynamics).

World and learner of block_flywheel.py with two checks and two groups of nine states, every group connected to every block. The writer draws
its two groups independently, so the states read off an own sample should come out independent. Lateral weights L[k_A, k_B] >= 0 between the
states of the two reader groups enter the reading as inhibition, p(k_A, k_B | h) ~ exp(u_A(k_A) + u_B(k_B) - L[k_A, k_B]), and follow the
covariance of the two readings over a source of strings (anti-Hebbian): L <- max(0, L + eta * c * K^2 * (E[y_A y_B] - E[y_A] E[y_B])).
Second world ('tied'): the classes s1 - s2 of the two checks agree in teacher strings (27 of the 81 valid strings; symbol pairs stay uniform), as a
level above would make them. The learner's two groups are drawn independently, so its best fit is log 3 nats above the teacher, and the right
reading has correlated states on teacher strings but independent states on own samples.
Sources for L: own samples (read through the string, the writing states are not used), teacher strings, uniform random strings.
Own samples move the reader's forward weights only in the arms 'strict' (reader's own reading) and 'sleep' (the state that wrote).

    python block_lateral.py            # runs all arms, writes reports/block-lateral.json
    python block_lateral.py --plot     # figure reports/block-lateral.png from the stored numbers
"""
import itertools, json, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from block_flywheel import EPS, FLOOR, Learner, World, measure, sm

K, SAMPLES, ETA, SEEDS = 9, 600000, .3, 8
ARMS = {  # own: share of own samples; fwd: what own samples do to the reader's forward weights; lat: source of the lateral weights; c: their rate
    'teacher only':                              dict(own=0., fwd=None, lat=None),
    'half own, writing state kept (sleep)':      dict(own=.5, fwd='sleep', lat=None),
    'half own, through the string':              dict(own=.5, fwd='strict', lat=None),
    'teacher only + inhibition from teacher strings': dict(own=0., fwd=None, lat='teacher'),
    'teacher only + inhibition from random strings':  dict(own=0., fwd=None, lat='noise'),
    '1/10 own for the inhibition only':          dict(own=.1, fwd=None, lat='own'),
    'half own through the string + inhibition from own': dict(own=.5, fwd='strict', lat='own'),
}


class Tied(World):
    """two checks whose classes s1 - s2 agree in teacher strings."""
    def __init__(self):
        super().__init__(2)
        self.pT = self.valid & ((self.S[:, 0] - self.S[:, 1]) % 3 == (self.S[:, 3] - self.S[:, 4]) % 3); self.pT = self.pT / self.pT.sum()


def meas(m):
    """measures of block_flywheel.py, with the KL taken over the strings the teacher shows and reported above the best fit of two independent groups."""
    w = m.w
    with np.errstate(all='ignore'): out = measure(m)
    p, sup = m.write()[0].sum(0), w.pT > 0
    floor = np.log(w.valid.sum() / sup.sum())
    return dict(out, kl=float(np.sum(w.pT[sup] * np.log(w.pT[sup] / np.maximum(p[sup], 1e-12))) - floor), floor=float(floor))


class Lateral(Learner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.Lat = np.zeros((self.K, self.K))

    def joint(self):
        """p(k_A, k_B | string) [N, K, K] with lateral inhibition."""
        u = np.einsum('gkbs,nbs->gnk', np.log(self.R + FLOOR), self.w.H)
        v = u[0][:, :, None] + u[1][:, None, :] - self.Lat[None]
        e = np.exp(v - v.max((1, 2), keepdims=True))
        return e / e.sum((1, 2), keepdims=True)

    def read(self):
        j = self.joint()
        return np.stack([j.sum(2), j.sum(1)])


def binding(m):
    """share of a check's entropy (log 9) that each group's reading (its posterior, as a channel) holds about each check under teacher strings [group, check]."""
    w, y = m.w, m.read()
    tri = w.S.reshape(-1, 2, 3)[:, :, 0] * 3 + w.S.reshape(-1, 2, 3)[:, :, 1]              # index of the triple of each check (the third symbol follows)
    out = np.zeros((2, 2))
    for g in range(2):
        for c in range(2):
            j = np.einsum('n,nk,nt->kt', w.pT, y[g], np.eye(9)[tri[:, c]])
            out[g, c] = np.sum(j * np.log(np.maximum(j, 1e-300) / (j.sum(1, keepdims=True) @ j.sum(0, keepdims=True)))) / np.log(9)
    return out


def run(arm, seed, c_lat=1., world='independent', a=.1, samples=SAMPLES):
    cfg, w = ARMS[arm], (Tied() if world == 'tied' else World(2))
    m = Lateral(np.random.default_rng([seed, 1]), w, K, a, a)
    per, curve = ETA / EPS, []
    n_steps = int(samples / per)
    marks = set(np.unique(np.round(np.geomspace(1, n_steps, 40)).astype(int)).tolist()) | {0}
    upd = lambda P, joint: np.einsum('nk,nbs->kbs', joint, w.H) - joint.sum(0)[:, None, None] * P
    norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-9))
    for t in range(n_steps + 1):
        if t in marks:
            b = binding(m)
            curve.append(dict(samples=t * per, **meas(m), bind=b.tolist(), lat=float(m.Lat.max())))
        j = m.joint(); y = np.stack([j.sum(2), j.sum(1)])
        pj, states = m.write(); p_own = pj.sum(0)
        dR, dG = np.zeros_like(m.R), np.zeros_like(m.G)
        for g in range(2):
            teacher = w.pT[:, None] * y[g]
            dG[g], dR[g] = (1 - cfg['own']) * upd(m.G[g], teacher), (1 - cfg['own']) * upd(m.R[g], teacher)
            if cfg['fwd'] == 'sleep': dR[g] += cfg['own'] * upd(m.R[g], np.stack([pj[states[:, g] == k].sum(0) for k in range(K)], 1))
            if cfg['fwd'] == 'strict': dR[g] += cfg['own'] * upd(m.R[g], p_own[:, None] * y[g])
        if cfg['lat']:
            src = dict(own=p_own, teacher=w.pT, noise=np.full(len(w.S), 1 / len(w.S)))[cfg['lat']]
            share = cfg['own'] if cfg['lat'] == 'own' else 1 - cfg['own']                   # the lateral weights learn from that share of the samples
            cov = np.einsum('n,nkl->kl', src, j) - np.outer(src @ y[0], src @ y[1])
            m.Lat = np.maximum(0, m.Lat + ETA * share * c_lat * K * K * cov)
        m.R, m.G = norm(m.R + ETA * dR), norm(m.G + ETA * dG)
    return dict(arm=arm, seed=seed, c_lat=c_lat, world=world, a=a, curve=curve)


WORLDS = {'independent': 'two independent checks', 'tied': 'classes of the two checks tied (a level above)'}


def summary(runs):
    for world in WORLDS:
        print(f"\n{WORLDS[world]}; KL above the best fit of two independent groups")
        for arm in ARMS:
            for c in sorted({r['c_lat'] for r in runs if r['arm'] == arm}):
                rs = [r for r in runs if r['arm'] == arm and r['c_lat'] == c and r['world'] == world]
                end = [r['curve'][-1] for r in rs]
                bound = [abs(float(np.linalg.det(np.array(e['bind'])))) for e in end]             # 1: each group holds one check, 0: both on one check or mixed
                first = [next((p['samples'] for p in r['curve'] if p['kl'] < .1), None) for r in rs]
                done = [f for f in first if f is not None]
                print(f"  {arm:52s} c {c:4.1f}  below 0.1 at some point {len(done)} of {len(rs)} (median {np.median(done) / 1000 if done else float('nan'):4.0f}k)  at the end {sum(e['kl'] < .1 for e in end)} of {len(rs)}  end KL {np.round([e['kl'] for e in end], 2)}  binding {np.round(bound, 2)}  valid {np.round(np.median([e['valid'] for e in end]), 2)}")


if __name__ == '__main__':
    if '--plot' in sys.argv:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        runs = json.load(open('reports/block-lateral.json'))
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for row, world in enumerate(WORLDS):
            for arm in ARMS:
                rs = [r for r in runs if r['arm'] == arm and r['c_lat'] == 1. and r['world'] == world]
                x, ls = [p['samples'] for p in rs[0]['curve']], '--' if ARMS[arm]['lat'] else '-'
                ax[row, 0].plot(x, np.median([[max(p['kl'], 1e-3) for p in r['curve']] for r in rs], 0), ls, label=arm)
                ax[row, 1].plot(x, np.median([[abs(np.linalg.det(np.array(p['bind']))) for p in r['curve']] for r in rs], 0), ls, label=arm)
            ax[row, 0].set(xscale='log', yscale='log', ylabel='KL above the best fit, nats (median of 8 starts)', title=WORLDS[world]); ax[row, 0].set_xlim(1e3, SAMPLES)
            ax[row, 1].set(xscale='log', ylabel='binding of the two reader groups to the two checks', title=WORLDS[world])
        for a in ax[1]: a.set_xlabel('samples (teacher and own counted alike)')
        ax[1, 0].legend(loc='lower left')
        fig.tight_layout(); fig.savefig('reports/block-lateral.png', dpi=150); fig.savefig('reports/block-lateral.svg')
        summary(runs); sys.exit()
    w = Tied(); print('tied world: strings shown', int((w.pT > 0).sum()), ' largest deviation of a symbol pair from uniform', float(max(abs(np.einsum('n,ns,nt->st', w.pT, w.H[:, i], w.H[:, j]) - 1 / 9).max() for i, j in itertools.combinations(range(w.L), 2))))
    jobs = [(arm, seed, c, world) for world in WORLDS for arm in ARMS for c in ([.1, 1.] if ARMS[arm]['lat'] else [1.]) for seed in range(SEEDS)]
    with ProcessPoolExecutor(28) as ex: runs = list(tqdm(ex.map(run, *zip(*jobs)), total=len(jobs)))
    thin = lambda o: round(o, 4) if isinstance(o, float) else [thin(v) for v in o] if isinstance(o, list) else {k: thin(v) for k, v in o.items()} if isinstance(o, dict) else o
    json.dump(thin(runs), open('reports/block-lateral.json', 'w'))
    summary(runs)
