"""Exact checks (enumeration, no sampling): what a two-factor Hebb rule sees of an order-three world at a diffuse deep unit.

Deep group: K-state softmax y = softmax(a V h) on a code h; drive D = E[y h^T] - E[y] E[h]^T; a = amplitude of the weights.

1. Block world: three blocks with symbols s_b in Z_3, h = three one-hot vectors; teacher uniform on s1 + s2 + s3 = 0, noise uniform on
   all 27 (single and pair statistics of the teacher are those of the noise). Printed: the teacher-minus-noise drive along V and its
   exponent in a; the same for the group's own samples (state z uniform, each block drawn from softmax(a V[z, block]), weights tied);
   the largest deviation of the own samples' pair statistics from uniform. For a group with one state per valid triple (conjunction
   units), a random group and a 'star' group (three states, state z writes symbol z in every block).
2. Ring of three overlapping checks in the code of its check units (9 per site): teacher = the 27 valid components, noise = one check
   unit per site drawn independently (what a blank deep layer writes). Drive on a 3-state group that reads the shared token t0.

    python order3_saddle_check.py
"""
import itertools

import numpy as np

sm = lambda u: np.exp(u - u.max(-1, keepdims=True)) / np.exp(u - u.max(-1, keepdims=True)).sum(-1, keepdims=True)
A = np.array([.01, .02, .04, .08, .16])
slope = lambda v: float(np.polyfit(np.log(A), np.log(np.abs(v)), 1)[0])


def drive(H, p, W):
    y = sm(H @ W.T)
    return (y * p[:, None]).T @ H - np.outer(p @ y, p @ H)


def block_world():
    rng = np.random.default_rng(0)
    S = np.array(list(itertools.product(range(3), repeat=3)))
    H = np.zeros((27, 9)); H[np.arange(27)[:, None], np.arange(3) * 3 + S] = 1
    pT, pU = (S.sum(1) % 3 == 0) / 9., np.full(27, 1 / 27)
    pairs = lambda p: max(abs(np.bincount(S[:, i] * 3 + S[:, j], weights=p, minlength=9) - 1 / 9).max() for i, j in ((0, 1), (0, 2), (1, 2)))

    def own(W):
        q = sm(W.reshape(len(W), 3, 3))
        return np.mean([np.prod(q[k, np.arange(3), S], 1) for k in range(len(W))], 0)
    valid = S[pT > 0]
    V_and = np.zeros((9, 9)); V_and[np.arange(9)[:, None], np.arange(3) * 3 + valid] = 1; V_and -= 1 / 3
    V_star = np.zeros((3, 9)); V_star[np.arange(3)[:, None], np.arange(3) * 3 + np.arange(3)[:, None]] = 1; V_star -= 1 / 3
    V_rand = rng.normal(size=(9, 9)); V_rand -= V_rand.reshape(9, 3, 3).mean(2).repeat(3, 1)
    print('block world, teacher: largest deviation of single / pair statistics from uniform', abs(pT @ H - 1 / 3).max(), pairs(pT))
    for name, V in (('conjunction group', V_and), ('random 9-state group', V_rand), ('star group', V_star)):
        along = lambda p_of: [np.sum((drive(H, p_of(a * V), a * V) - drive(H, pU, a * V)) * V) / np.sum(V * V) for a in A]
        dT, dO, pr = along(lambda W: pT), along(own), [pairs(own(a * V)) for a in A]
        print(f'  {name:22s} teacher-specific drive: exponent {slope(dT):.2f}, at a = 0.16 {dT[-1]:+.2e}   own samples: exponent {slope(dO):.2f}, {dO[-1]:+.2e}   '
              f'pair statistics of own samples off uniform by {pr[-1]:.1e} at a = 0.16, {pairs(own(3 * V)):.1e} at a = 3')


def ring_world():
    T = np.array(list(itertools.product(range(3), repeat=3)))                                   # shared tokens (t0, t2, t4); check unit of a site = its two shared tokens
    codeT = np.stack([T[:, 0] * 3 + T[:, 1], T[:, 1] * 3 + T[:, 2], T[:, 2] * 3 + T[:, 0]], 1)
    codeU = np.array(list(itertools.product(range(9), repeat=3)))
    onehot = lambda c: np.eye(27)[np.arange(3) * 9 + c].sum(1)
    HT, HU = onehot(codeT), onehot(codeU)
    V = np.zeros((3, 27))
    for u in range(9):
        V[u // 3, u] = V[u % 3, 18 + u] = 1                                                     # t0 is the first entry of site A and the second of site C
    V[:, :9] -= 1 / 3; V[:, 18:] -= 1 / 3
    d = [np.sum((drive(HT, np.full(27, 1 / 27), a * V) - drive(HU, np.full(729, 1 / 729), a * V)) * V) / np.sum(V * V) for a in A]
    print(f'ring world, code of the check units: pair statistics teacher vs noise differ by up to {abs(HT.T @ HT / 27 - HU.T @ HU / 729).max():.3f};   '
          f'teacher-specific drive on a group reading a shared token: exponent {slope(d):.2f}, at a = 0.16 {d[-1]:+.2e}')


if __name__ == '__main__':
    block_world()
    ring_world()
