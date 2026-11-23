"""Exact mean dynamics (no sampling) of a deep layer of winner-take-all groups on the block world, reader and writer apart.

World: C checks of three blocks each, a block shows a symbol in Z_3, a check holds if its three symbols sum to zero. The teacher is uniform
on the strings in which every check holds; single symbols and symbol pairs are uniform. The shallow layer is taken as given: the deep layer
sees the symbols h of a string, for teacher strings and own samples alike.

Learner: C groups of K states, every group connected to every block. P[g, k, b, s] = probability of symbol s in block b given state k,
separate arrays for reading (R) and writing (G), weights = log P.
  reading: every group takes the posterior y_g(k | h) = softmax_k(sum_b log R[g, k, b, h_b]).
  writing: every group draws a state from a uniform prior, block b draws its symbol from softmax(sum_g (log G - log of the group's average)).
Rule in expectation over the samples of a source, P <- P + eta * E[post_k * (1[h_b = s] - P)]:
  teacher string: post = the reader's posterior; moves R and G.
  own sample:     moves R only. 'strict': post = the reader's posterior of the own string (the writing states are not used);
                  'sleep': post = the state that wrote; 'twin': strict, with the samples of a second learner from another start.
rule 'gated': post as above (the rule of order3_net.py); 'cov': post minus its mean 1 / K, so a diffuse reader moves nothing.
response 'match': a state's activity is the probability of the string under that state alone (no competition inside the group), so a
string fires a state only as far as it fits. beta: gain of the writer (it draws from softmax(beta * its log weights)). prior_rate > 0: the writer's prior over the states of a group follows how often teacher
strings fire each state.
A share `own` of the samples are own samples from `t_own` samples on. One step stands for eta / eps samples at rate eps.

    python block_flywheel.py
"""
import itertools
import os

import numpy as np

FLOOR, EPS = float(os.environ.get('BLOCK_FLOOR', 1e-3)), .003                               # the reader's lapse; 1e-3 as in order3_net.py unless BLOCK_FLOOR is set
sm = lambda u: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(u - u.max(-1, keepdims=True)))


class World:
    def __init__(self, C):
        self.C, self.L = C, 3 * C
        self.S = np.array(list(itertools.product(range(3), repeat=self.L)))
        self.valid = np.all(self.S.reshape(-1, C, 3).sum(2) % 3 == 0, 1)
        self.pT = self.valid / self.valid.sum()
        self.H = np.eye(3)[self.S]                                                              # [N, L, 3]


class Learner:
    def __init__(self, rng, world, K, a, b, response='posterior', beta=1.):
        self.w, self.K, self.response, self.beta = world, K, response, beta
        n = rng.normal(size=(world.C, K, world.L, 3))
        self.R, self.G, self.pi = sm(a * n), sm(b * n), np.full((world.C, K), 1 / K)

    def read(self):
        """activity of every state of every group for every string [C, N, K]: the posterior, or ('match') the probability of the string under
        the state, with no competition inside the group."""
        u = np.einsum('gkbs,nbs->gnk', np.log(self.R + FLOOR), self.w.H)
        return np.exp(u) if self.response == 'match' else sm(u)

    def write(self):
        """p(string, joint state) [K^C, N] and the joint states [K^C, C]; the states of a group are drawn from its prior pi."""
        lift = np.log(self.G + FLOOR) - np.log(self.G.mean(1, keepdims=True) + FLOOR)
        states = np.array(list(itertools.product(range(self.K), repeat=self.w.C)))
        logq = np.log(sm(self.beta * sum(lift[g, states[:, g]] for g in range(self.w.C))))      # [J, L, 3]; beta > 1: the writer draws more sharply than its weights say
        weight = np.prod([self.pi[g, states[:, g]] for g in range(self.w.C)], 0)
        return weight[:, None] * np.exp(np.einsum('jbs,nbs->jn', logq, self.w.H)), states


def step(m, eta, own, mode, other=None, rule='gated'):
    w, y = m.w, m.read()
    base = 1 / m.K if rule == 'cov' else 0.                                                     # 'cov': post = activity minus its mean (covariance rule)
    upd = lambda P, joint: np.einsum('nk,nbs->kbs', joint, w.H) - joint.sum(0)[:, None, None] * P   # joint [N, K] = weight of (string, active state)
    pj, states = (other or m).write()
    p_own = pj.sum(0)
    dR, dG, share = np.zeros_like(m.R), np.zeros_like(m.G), np.zeros_like(m.pi)
    for g in range(w.C):
        teacher = w.pT[:, None] * (y[g] - base)
        dG[g] = (1 - own) * upd(m.G[g], teacher)
        dR[g] = (1 - own) * upd(m.R[g], teacher)
        if own:
            wrote = np.stack([pj[states[:, g] == k].sum(0) for k in range(m.K)], 1)             # [N, K] weight of (string, state of group g that wrote)
            dR[g] += own * upd(m.R[g], wrote - base * p_own[:, None] if mode == 'sleep' else p_own[:, None] * (y[g] - base))
        fired = w.pT @ y[g]
        share[g] = fired / fired.sum() - m.pi[g]                                                # how often teacher strings fire a state, as a share
    return dR, dG, share


def measure(m):
    w, (pj, states), y = m.w, m.write(), m.read()
    p = pj.sum(0)
    mode = pj.argmax(1)                                                                         # the string a joint state writes most often
    used = pj.sum(1) > .5 / len(states)                                                         # joint states the prior still draws
    marg = np.einsum('n,nbs->bs', p, w.H)
    recover = float(np.mean([sum((pj[states[:, g] == k] * y[g, :, k]).sum() for k in range(m.K)) for g in range(w.C)]))   # probability that reading an own sample gives the state that wrote it
    return dict(kl=float(np.sum(w.pT[w.valid] * np.log(w.pT[w.valid] / np.maximum(p[w.valid], 1e-12)))), valid=float(p[w.valid].sum()), recover=recover,
                reach_teacher=float(np.mean(np.einsum('n,gn->g', w.pT, y.max(2)))), reach_own=float(np.mean(np.einsum('n,gn->g', p, y.max(2)))),
                modes_valid=float(w.valid[mode][used].mean()), covered=float(len(set(mode[w.valid[mode] & used].tolist())) / w.valid.sum()), single=float(abs(marg - 1 / 3).max()),
                states_used=float(used.mean()),
                pairs=float(max(abs(np.einsum('n,ns,nt->st', p, w.H[:, i], w.H[:, j]) - 1 / 9).max() for i, j in itertools.combinations(range(w.L), 2))))


def run(C=1, K=9, a=.1, b=None, own=0., mode='strict', t_own=0, samples=400000, eta=.3, seed=0, every=None, rule='gated', response='posterior', prior_rate=0., beta=1.):
    world = World(C)
    m = Learner(np.random.default_rng([seed, 1]), world, K, a, a if b is None else b, response, beta)
    twin = Learner(np.random.default_rng([seed, 2]), world, K, a, a if b is None else b, response, beta) if mode == 'twin' else None
    per, curve = eta / EPS, []
    n_steps = int(samples / per)
    marks = set(np.unique(np.round(np.geomspace(1, n_steps, every or 40)).astype(int)).tolist()) | {0}
    for t in range(n_steps + 1):
        if t in marks:
            curve.append(dict(samples=t * per, **measure(m)))
        o = own if t * per >= t_own else 0.
        moves = [(m, step(m, eta, o, mode, twin, rule))] + ([(twin, step(twin, eta, o, 'strict', None, rule))] if twin else [])
        norm = lambda P: (lambda c: c / c.sum(-1, keepdims=True))(np.maximum(P, 1e-9))
        for net, (dR, dG, share) in moves:
            net.R, net.G, net.pi = norm(net.R + eta * dR), norm(net.G + eta * dG), norm(net.pi + eta * prior_rate * share)
    return curve


if __name__ == '__main__':
    np.set_printoptions(linewidth=250, precision=2, suppress=True)
    for C in (1, 2):
        print(f'\n{C} check(s), groups of 9, weak tied start')
        for name, kw in (('teacher only', {}), ('half own, strict', dict(own=.5)), ('half own, sleep', dict(own=.5, mode='sleep')), ('half own, twin', dict(own=.5, mode='twin'))):
            ends = [run(C=C, seed=s, **kw) for s in range(8)]
            first = lambda c: next((p['samples'] for p in c if p['kl'] < .1), np.inf)
            print(f"  {name:18s} KL at the end {np.round([c[-1]['kl'] for c in ends], 2)}   samples to KL < 0.1: {np.round([first(c) / 1000 for c in ends])} k")
