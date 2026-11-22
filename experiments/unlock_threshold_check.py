"""Check of the 'which space is the target in' ladder for unlocking a level of the RHM.
Rows: the v*m s-tuples of level-l symbols (m synonyms per symbol a), read in symbols (pooled).
(a) columns = sibling tuple, also pooled            -> predicted threshold T ~ v m^2, independent of depth
(b) columns = ONE token, k levels below the sibling -> predicted threshold T ~ v m^(k+3/2)
(c) as (a), each tuple read right with prob. r, else a random tuple -> predicted T ~ v m^2 / r^4
Measure: share of the top v-1 left singular vectors of the centred count matrix that lies in the span of the
synonym-class indicators, rescaled so that chance = 0 and perfect = 1. T = number of observed sibling pairs."""
import numpy as np

def rules(v, m, rng):
    cells = rng.choice(v * v, size=v * m, replace=False); return np.stack([cells // v, cells % v], 1).reshape(v, m, 2)

def overlap(n, v, m):
    X = n - np.outer(n.sum(1), n.sum(0)) / n.sum(); U = np.linalg.svd(X, full_matrices=False)[0][:, :v - 1]
    cm = U.reshape(v, m, -1).mean(1, keepdims=True) - U.mean(0); raw = (m * cm ** 2).sum() / (v - 1); chance = (v - 1) / (v * m - 1)
    return (raw - chance) / (1 - chance)

def joint_ab(v, m, rng):
    r = rules(v, m, rng).reshape(-1, 2); P = np.zeros((v, v)); P[r[:, 0], r[:, 1]] = 1 / (v * m); return P

def pooled(v, m, T, rng, r=1.0):
    P = np.repeat(np.repeat(joint_ab(v, m, rng), m, 0), m, 1) / m ** 2; u = np.full(v * m, 1 / (v * m))
    lam = r * r * P + r * (1 - r) * (np.outer(P.sum(1), u) + np.outer(u, P.sum(0))) + (1 - r) ** 2 * np.outer(u, u)
    return rng.poisson(T * lam)

def token_target(v, m, k, T, rng):
    D = np.eye(v)
    for _ in range(k):
        M = np.zeros((v, v)); np.add.at(M, (np.repeat(np.arange(v), m), rules(v, m, rng)[:, :, 0].ravel()), 1 / m); D = D @ M
    return rng.poisson(T * np.repeat(joint_ab(v, m, rng) @ D, m, 0) / m)

xs = [0.25, 0.5, 1, 2, 4, 8, 16, 32]; seeds = 6
print('x = T / predicted threshold:', xs)
print('\n(a) both sides in symbols, T_pred = v m^2')
for v, m in [(32, 4), (32, 8), (64, 8), (64, 16), (128, 16)]:
    print(f'  v={v:3d} m={m:2d} T_pred={v*m*m:7d} ', ' '.join(f'{np.mean([overlap(pooled(v, m, x*v*m*m, np.random.default_rng(s)), v, m) for s in range(seeds)]):5.2f}' for x in xs))
print('\n(b) target = one token k levels below the sibling symbol, T_pred = v m^(k+3/2)')
for v, m in [(32, 4), (32, 8), (64, 8)]:
    for k in range(4):
        Tp = v * m ** (k + 1.5)
        print(f'  v={v:3d} m={m:2d} k={k} T_pred={Tp:10.0f} ', ' '.join(f'{np.mean([overlap(token_target(v, m, k, x*Tp, np.random.default_rng(s)), v, m) for s in range(seeds)]):5.2f}' for x in xs))
print('\n(c) both sides in symbols, each tuple read right with prob. r; x = T r^4 / (v m^2)')
for v, m in [(64, 8)]:
    for r in [1.0, 0.7, 0.5, 0.35]:
        print(f'  v={v:3d} m={m:2d} r={r:4.2f} T_pred={v*m*m/r**4:8.0f} ', ' '.join(f'{np.mean([overlap(pooled(v, m, x*v*m*m/r**4, np.random.default_rng(s), r), v, m) for s in range(seeds)]):5.2f}' for x in xs))
