"""Read-write loop with gain in a stack of levels: exact mean dynamics on a block world with L levels of checks.

World. Level 1: 3^(L-1) checks of three blocks each; a block shows a symbol in Z_3 and a check holds if its three symbols sum to zero
(9 valid triples = the 9 causes of the check). Every cause has a class in Z_3, class = first symbol minus second (3 causes per class).
Level l > 1: a check over three checks of the level below, which holds if their three classes sum to zero; its 9 causes are the valid
class triples and again have a class (first class minus second). The teacher draws the top cause uniformly and, going down, for every
class one of its 3 causes uniformly. Single blocks and pairs of blocks are uniform; a level-l check is a statistic of order 3 * 2^(l-1)
in the blocks, and in the causes of the level below it is of order three with uniform pairs.

Learner. One winner-take-all group of K = 9 states per check, wired to the three blocks (level 1) or to the three groups of the level
below; P[k, child, value] = probability of the child's value (symbol or state) given state k, separate arrays for reading (R) and writing
(G), weights = log P, as in block_flywheel.py.
  reading (bottom-up): a group draws its state from softmax_k(sum over children of log R[k, child, value read below]).
  writing (top-down):  the top group draws a state from a uniform prior; a child's value is drawn from softmax(beta * log G[k, child, .]):
                       the writer draws more sharply than its weights say (gain beta).
Rule in expectation, P <- P + eta * E[post_k * (1[value] - P)]:
  teacher string: post = the reader's state; moves R and G of every group (values = what the level below read).
  own sample: moves R only. dreams 'top': written from the top group; 'each': every group also dreams from its own uniform prior, so
  a state's share of the own samples does not depend on the levels above. 'kept': post = the state that wrote, values = what the level below reads back from the
  string; 'string': post = the state the group reads from the own string.
All expectations are exact: a group's teacher statistics depend on the levels below only through the distribution of read states given
the class of the true cause (A), its own-sample statistics through the confusion matrix of the loop below (M).

    python block_depth.py
"""
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPORTS = Path('/nfs/nhome/live/jbauer/generative-learning/reports')
K, FLOOR, EPS = 9, float(os.environ.get('BLOCK_FLOOR', 1e-6)), .003                         # FLOOR: the reader's lapse; it makes every weak tilt decay at rate (eps / K) FLOOR / (1 / v + FLOOR)
sm = lambda u: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(u - u.max(-1, keepdims=True)))
norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-12))
CAUSES = np.array([(a, b, (-a - b) % 3) for a in range(3) for b in range(3)])                    # [9, 3]: the three symbols (level 1) or child classes of a cause
CLASS = (CAUSES[:, 0] - CAUSES[:, 1]) % 3
H = {3: np.eye(3)[np.array(list(itertools.product(range(3), repeat=3)))], 9: np.eye(9)[np.array(list(itertools.product(range(9), repeat=3)))]}   # one-hot of all child-value triples
VAL = {v: np.array(list(itertools.product(range(v), repeat=3))) for v in (3, 9)}
VALID27 = (CAUSES * (9, 3, 1)).sum(1)                                                           # index of a level-1 cause among the 27 symbol triples
joint = lambda P, v: np.exp(np.einsum('kbs,nbs->kn', np.log(np.maximum(P, 1e-300)), H[v]))         # [K, 3, v] block-wise -> [K, v^3]


compact = lambda o: [compact(x) for x in o] if isinstance(o, list) else {k: compact(v) for k, v in o.items()} if isinstance(o, dict) else float(f'{o:.4g}') if isinstance(o, float) else o


class Stack:
    def __init__(self, rng, L, a, beta=1., dreams='top'):
        self.L, self.beta, self.dreams = L, beta, dreams
        self.n = [3 ** (L - l) for l in range(1, L + 1)]
        start = lambda n, v: sm(a * rng.normal(size=(n, K, 3, v)))
        self.R = [start(n, 3 if l == 0 else 9) for l, n in enumerate(self.n)]
        self.G = [r.copy() for r in self.R]
        self.frozen = set()

    def writer(self, l, g):
        return sm(self.beta * (np.log(self.G[l][g] + FLOOR) - np.log(self.G[l][g].mean(0) + FLOOR)))

    def flow(self, own, mode):
        """expected change of every R and G per unit of eps * samples, and what a measurement needs."""
        L, v_of = self.L, lambda l: 3 if l == 0 else 9
        y, qc, A = [[None] * n for n in self.n], [[None] * n for n in self.n], [[None] * n for n in self.n]
        dR, dG = [np.zeros_like(r) for r in self.R], [np.zeros_like(g) for g in self.G]
        for l in range(L):                                                                      # teacher strings, bottom-up
            for g in range(self.n[l]):
                v = v_of(l)
                y[l][g] = sm(np.einsum('kbs,nbs->nk', np.log(self.R[l][g] + FLOOR), H[v]))
                if l == 0:
                    qc[l][g] = np.eye(27)[VALID27]
                else:
                    kids = [A[l - 1][3 * g + i] for i in range(3)]
                    qc[l][g] = np.prod([kids[i][CAUSES[:, i]][:, VAL[9][:, i]] for i in range(3)], 0)     # [9 causes, 729]: read child states given the cause
                D = qc[l][g] @ y[l][g]
                A[l][g] = np.stack([D[CLASS == m].mean(0) for m in range(3)])
                w = qc[l][g].mean(0)[:, None] * y[l][g]                                         # weight of (child values, state read) under teacher strings
                for d, P in ((dR, self.R), (dG, self.G)):
                    d[l][g] = (1 - own) * (np.einsum('nk,nbs->kbs', w, H[v]) - w.sum(0)[:, None, None] * P[l][g])
        pi, Pw = [[None] * n for n in self.n], [[self.writer(l, g) for g in range(self.n[l])] for l in range(L)]
        pi[L - 1][0] = np.full(K, 1 / K)
        for l in range(L - 1, 0, -1):                                                           # how often a state is written, top-down
            for g in range(self.n[l]):
                for i in range(3):
                    pi[l - 1][3 * g + i] = pi[l][g] @ Pw[l][g][:, i] if self.dreams == 'top' else np.full(K, 1 / K)
        M, seen, pv = [[None] * n for n in self.n], [[None] * n for n in self.n], [[None] * n for n in self.n]
        for l in range(L):                                                                      # own samples, bottom-up: what comes back from the string
            for g in range(self.n[l]):
                v = v_of(l)
                seen[l][g] = Pw[l][g] if l == 0 else np.stack([Pw[l][g][:, i] @ M[l - 1][3 * g + i] for i in range(3)], 1)
                pv[l][g] = joint(seen[l][g], v)
                M[l][g] = pv[l][g] @ y[l][g]                                                    # [state that wrote, state read]
                if own:
                    if mode == 'kept':
                        dR[l][g] += own * pi[l][g][:, None, None] * (seen[l][g] - self.R[l][g])
                    else:
                        w = (pi[l][g] @ pv[l][g])[:, None] * y[l][g]
                        dR[l][g] += own * (np.einsum('nk,nbs->kbs', w, H[v]) - w.sum(0)[:, None, None] * self.R[l][g])
        return dR, dG, dict(y=y, qc=qc, pi=pi, Pw=Pw, M=M)

    def step(self, eta, own, mode):
        dR, dG, _ = self.flow(own, mode)
        for l in range(self.L):
            if l not in self.frozen:
                self.R[l], self.G[l] = norm(self.R[l] + eta * dR[l]), norm(self.G[l] + eta * dG[l])

    def measure(self):
        _, _, c = self.flow(0., 'kept')
        out, T = {}, [[None] * n for n in self.n]
        for l in range(self.L):
            kl, rec, fid, reach, valid, tilt = [], [], [], [], [], []
            for g in range(self.n[l]):
                Pw = c['Pw'][l][g]
                if l == 0:
                    T[l][g] = joint(Pw, 3)[:, VALID27]                                          # [state, cause written]
                else:
                    cls = [np.stack([T[l - 1][3 * g + i][:, CLASS == m].sum(1) for m in range(3)], 1) for i in range(3)]
                    T[l][g] = np.prod([(Pw[:, i] @ cls[i])[:, CAUSES[:, i]] for i in range(3)], 0)
                p = c['pi'][l][g] @ T[l][g]
                kl.append(float(np.sum(np.log((1 / 9) / np.maximum(p, 1e-12)) / 9))); valid.append(float(p.sum()))
                rec.append(float(c['pi'][l][g] @ np.diag(c['M'][l][g])))
                fid.append(float(np.sort(np.abs(np.linalg.eigvals(c['M'][l][g])))[-2]))         # largest eigenvalue of the loop's confusion matrix besides 1
                reach.append(float(c['qc'][l][g].mean(0) @ c['y'][l][g].max(1)))
                w = np.log(self.R[l][g]); tilt.append(float(np.sqrt(np.mean((w - w.mean(0)) ** 2))))
            out[l + 1] = dict(kl=max(kl), kl_sum=sum(kl), recover=float(np.mean(rec)), fidelity=float(np.mean(fid)), fidelity_min=min(fid), reach_teacher=float(np.mean(reach)), valid=float(np.mean(valid)), tilt=float(np.mean(tilt)))
        return out


def run(job):
    name, L, a, beta, own, mode, seed, t_max = job[:8]
    m = Stack(np.random.default_rng([seed, 1]), L, a, beta, *job[8:])
    eta, t, curve, solved = .3, 0, [], {}
    marks = set(np.unique(np.round(np.geomspace(1, t_max / (20 * eta / EPS), 40)).astype(int)).tolist()) | {0}
    for i in range(int(t_max / (20 * eta / EPS)) + 1):
        now = m.measure()
        for l in range(1, L + 1):
            if l not in solved and all(now[j]['kl'] < .1 for j in range(1, l + 1)):
                solved[l] = t
        if i in marks or len(solved) == L:
            curve.append(dict(samples=t, **{f'{k}{l}': now[l][k] for l in now for k in ('kl', 'recover', 'fidelity_min', 'tilt')}))
        if len(solved) == L:
            break
        for _ in range(20):
            m.step(eta, own, mode)
        t += 20 * eta / EPS
    return dict(name=name, L=L, a=a, beta=beta, seed=seed, solved={str(k): v for k, v in solved.items()}, end={l: now[l]['kl'] for l in now}, curve=curve)


def threshold():
    """level 1 held at a given sharpness on the right code, level 2 diffuse: growth rate of the level-2 tilt against the prediction."""
    out = []
    for beta in (1.5, 2., 4.):
        for s in (.3, .45, .6, .75, .9):
            m = Stack(np.random.default_rng(0), 2, 1e-7, beta)
            code = np.eye(3)[CAUSES]                                                            # [9, 3, 3]
            m.R[0][:] = m.G[0][:] = (1 - s) / 3 + s * code
            m.frozen = {0}
            f = m.measure()[1]['fidelity']
            spread = lambda: m.measure()[2]['tilt']
            t, v = [0], [spread()]
            while 1e-10 < v[-1] < 1e-3 and len(t) < 400:
                for _ in range(20):
                    m.step(.03, .5, 'kept')
                t.append(t[-1] + 20 * .03 / EPS); v.append(spread())
            k = max(2, len(t) // 3)
            out.append(dict(beta=beta, sharpness=s, fidelity=f, predicted=EPS / (2 * K) * (np.sqrt(beta * f) - 1), measured=float(np.polyfit(t[-k:], np.log(v[-k:]), 1)[0])))
            print(f"  gain {beta:3.1f}, level-1 fidelity {f:.2f} (gain x fidelity {beta * f:4.2f}): level-2 rate predicted {1e5 * out[-1]['predicted']:+6.2f}, measured {1e5 * out[-1]['measured']:+6.2f}  (1e-5 per sample)")
    return out


ARMS = {'teacher strings only': (1., 0., 'kept'), 'own, state kept, gain 1.2': (1.2, .5, 'kept'), 'own, state kept, gain 1.5': (1.5, .5, 'kept'), 'own, state kept, gain 2': (2., .5, 'kept')}

if __name__ == '__main__':
    print('1. level-2 loop with level 1 held fixed: growth rate of the level-2 tilt')
    numbers = dict(floor=FLOOR, threshold=threshold())
    jobs = [(name, 2, a, *ARMS[name], s, 3e6 if name.startswith('teacher') else 1.5e6) for name in ARMS for a in (.1, .01) for s in range(16)]
    jobs += [(name, 3, .1, *ARMS[name], s, 3e6) for name in ('teacher strings only', 'own, state kept, gain 1.5') for s in range(8)]
    with Pool(30) as pool:
        res = pool.map(run, jobs, chunksize=1)
    print('2. samples (thousands) until every level up to l is within 0.1 nats per check; median of the starts that get there (number of starts)')
    for L, a in ((2, .1), (2, .01), (3, .1)):
        for name in ARMS:
            rs = [r for r in res if (r['name'], r['L'], r['a']) == (name, L, a)]
            if not rs:
                continue
            cells = []
            for l in range(1, L + 1):
                d = [r['solved'][str(l)] for r in rs if str(l) in r['solved']]
                cells.append(f"level {l}: {np.median(d) / 1000:6.0f}k ({len(d)}/{len(rs)})" if d else f"level {l}:       - (0/{len(rs)})")
            print(f"  start {a:4.2f}, {L} levels, {name:28s} " + '   '.join(cells))
    numbers['runs'] = res
    json.dump(compact(numbers), open(REPORTS / 'block-depth.json', 'w'), separators=(',', ':'))
