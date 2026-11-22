"""Does every level of the RHM unlock at the same amount of data when the level below is read in the learner's own codes?
Fixed set of P strings (v symbols, m productions, pairs, depth L); both learners are tied-weight idealisations of wake-sleep
(the reader is the inverse of the learner's own tables, which is what sleep converges to), trained level by level from the tokens up.
'composed': a node's state is the PAIR of its children's symbols (stored when seen); per level one map state -> symbol (v symbols)
    is learned from the co-occurrence of sibling states (spectral start, then hard EM of the block model). Depth 3, 4, 5.
'flat': one unit per node with K = v m states, p(c1, c2) = sum_u pi_u A(c1|u) B(c2|u), fitted by EM; a node's code is its most
    probable state. Children read with the learner's own codes ('own') or given as true states ('given'). Depth 3.
Measures on fresh strings / own samples: quality of the map or code (mutual information with the truth divided by the larger
entropy), share of own samples that parse up to each level, share of valid own samples that are not in the training set.
Usage: python unlock_stack_check.py [report]"""
import sys, numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
from unlock_escape_rate_check import rules, em_pass, start

OUT = '/var/tmp/jbauer/generative_practice_rhm/unlock_stack.npz'; FIG = '/nfs/nhome/live/jbauer/generative-learning/reports/unlock-stack.png'
V, PASSES, NTEST, XS, SEEDS, MS, DEPTHS = 32, 300, 20000, [0.5, 1, 2, 4, 8, 16, 32, 64, 128], 4, [4, 8], [3, 4, 5]

def sample(R, P, rng):
    v, m, _ = R[0].shape; sym = rng.integers(v, size=(P, 1)); states = {}
    for h in range(len(R), 0, -1):
        prod = rng.integers(m, size=sym.shape); states[h] = sym * m + prod; sym = R[h - 1][sym, prod].reshape(P, -1)
    return sym, states

def nmi(a, b):
    a, b = np.unique(a, return_inverse=True)[1].ravel(), np.unique(b, return_inverse=True)[1].ravel(); J = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(J, (a, b), 1); J /= J.sum()
    H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum(); return (H(J.sum(1)) + H(J.sum(0)) - H(J)) / max(H(J.sum(1)), H(J.sum(0)))

def parses(tok, R):
    v = R[0].shape[0]; ok = np.ones(len(tok), bool); out = []
    for h in range(1, len(R) + 1):
        inv = np.full((v, v), -1); inv[R[h - 1][:, :, 0], R[h - 1][:, :, 1]] = np.arange(v)[:, None]; tok = inv[tok[:, 0::2], tok[:, 1::2]]; ok &= (tok >= 0).all(1); out.append(ok.mean()); tok = np.maximum(tok, 0)
    return out, ok

def draw(p, rng):
    c = p.cumsum(1); return (rng.random((len(p), 1)) * c[:, -1:] > c).sum(1).clip(0, p.shape[1] - 1)

def novel(x, ok, tok): train = set(map(bytes, tok.astype(np.uint8))); return np.mean([bytes(s) not in train for s in x[ok].astype(np.uint8)]) if ok.any() else np.nan

# ---- composed reader
def kmeans(X, k, rng, restarts=8, iters=50):
    best = None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        for _ in range(k - 1): d = np.min([((X - c) ** 2).sum(1) for c in C], 0) + 1e-12; C.append(X[rng.choice(len(X), p=d / d.sum())])
        C = np.array(C)
        for _ in range(iters): z = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1); C = np.array([X[z == j].mean(0) if (z == j).any() else X[rng.integers(len(X))] for j in range(k)])
        cost = ((X - C[z]) ** 2).sum()
        if best is None or cost < best[0]: best = (cost, z)
    return best[1]

def merge(l, r, n, k, rng, rounds=10):
    """one map state -> symbol from the co-occurrence of sibling states; states never seen get symbol 0"""
    N = np.zeros((n, n)); np.add.at(N, (l.ravel(), r.ravel()), 1); seen = N.sum(1) + N.sum(0) > 0; M = N[seen][:, seen]
    U, S, Vt = np.linalg.svd(M / np.sqrt(np.outer(M.sum(1) + 1e-9, M.sum(0) + 1e-9))); z = kmeans(np.hstack([U[:, 1:k] * S[1:k], Vt[1:k].T * S[1:k]]), k, rng)
    for _ in range(rounds):
        Z = np.eye(k)[z]; Rm, Cm = M @ Z, M.T @ Z; Tr, Tc = Z.T @ Rm + 0.1, Z.T @ Cm + 0.1; z2 = (Rm @ np.log(Tr / Tr.sum(1, keepdims=True)).T + Cm @ np.log(Tc / Tc.sum(1, keepdims=True)).T).argmax(1)
        if (z2 == z).all(): break
        z = z2
    out = np.zeros(n, int); out[seen] = z; return out

def composed(args):
    m, depth, P, seed = args; rng = np.random.default_rng(seed); R = [rules(V, m, rng) for _ in range(depth)]; tok, st = sample(R, P, rng); tk, sk = sample(R, NTEST, rng); sym, symk, q, gen = tok, tk, [], []
    for h in range(1, depth + 1):
        pair, pairk = sym[:, 0::2] * V + sym[:, 1::2], symk[:, 0::2] * V + symk[:, 1::2]; known = np.bincount(pair.ravel(), minlength=V * V)
        if h == depth: gen.append((known, None)); break
        sig = merge(pair[:, 0::2], pair[:, 1::2], V * V, V, rng); gen.append((known, sig)); sym, symk = sig[pair], sig[pairk]; q.append(nmi(symk[known[pairk] > 0], (sk[h] // m)[known[pairk] > 0]))
    x = draw(np.tile(gen[-1][0].astype(float), (NTEST, 1)), rng)[:, None]     # own samples: a stored top pair, then every symbol -> one of its stored states, down to the tokens
    for h in range(depth - 1, -1, -1):
        x = np.stack([x // V, x % V], 2).reshape(NTEST, -1)
        if h > 0: known, sig = gen[h - 1]; Wt = np.zeros((V, V * V)); Wt[sig, np.arange(V * V)] = known; Wt[Wt.sum(1) == 0] = 1; x = draw(Wt[x.ravel()], rng).reshape(NTEST, -1)
    val, ok = parses(x, R); pad = [np.nan] * (max(DEPTHS) - depth); return dict(c_merge=q + pad + [np.nan], c_valid=val + pad, c_novel=novel(x, ok, tok))

# ---- flat unit
def fit(l, r, n, K, rng):
    N = np.zeros((n, n)); np.add.at(N, (l.ravel(), r.ravel()), 1); k1, k2 = N.sum(1) > 0, N.sum(0) > 0; Nk = N[k1][:, k2]; pi, A, B = start(Nk, K, 1e-2, rng); ll = []
    for _ in range(PASSES): ll.append((Nk * np.log((A * pi[:, None]).T @ B)).sum() / Nk.sum()); pi, A, B = em_pass(Nk, pi, A, B, smooth=1e-2)
    Af = np.full((K, n), 1e-12); Af[:, k1] = A; Bf = np.full((K, n), 1e-12); Bf[:, k2] = B; return pi + 1e-300, Af / Af.sum(1, keepdims=True), Bf / Bf.sum(1, keepdims=True), np.array(ll)

def reader(pi, A, B): lA, lB = np.log(A * pi[:, None]).T, np.log(B).T; return np.stack([(lA[i][None] + lB).argmax(1) for i in range(lA.shape[0])])

def flat(args):
    m, depth, P, seed = args; rng = np.random.default_rng(seed); R = [rules(V, m, rng) for _ in range(depth)]; K = V * m; tok, st = sample(R, P, rng); tk, sk = sample(R, NTEST, rng); res = {}
    for arm in ['own', 'given']:
        c, ck, n, tabs, q, lls = tok, tk, V, [], [], []
        for h in range(1, depth + 1):
            if arm == 'given' and h > 1: c, ck, n = st[h - 1], sk[h - 1], K
            pi, A, B, ll = fit(c[:, 0::2], c[:, 1::2], n, K, rng); best = reader(pi, A, B); c, ck, n = best[c[:, 0::2], c[:, 1::2]], best[ck[:, 0::2], ck[:, 1::2]], K; tabs.append((pi, A, B)); q.append(nmi(ck, sk[h])); lls.append(ll)
        res[f'f_{arm}_code'] = q; res[f'f_{arm}_ll'] = lls
        if arm == 'own':
            x = draw(np.tile(tabs[-1][0], (NTEST, 1)), rng)[:, None]
            for pi, A, B in tabs[::-1]: x = np.stack([draw(A[x.ravel()], rng), draw(B[x.ravel()], rng)], 1).reshape(NTEST, -1)
            val, ok = parses(x, R); res['f_valid'] = val; res['f_novel'] = novel(x, ok, tok)
    return res

def job(a): return (composed if a[0] == 'c' else flat)(a[1:])

def run():
    strings = lambda m, x: int(x * V * m * m / (1 - m / V)); cj = [('c', m, d, strings(m, x), s) for m in MS for d in DEPTHS for x in XS for s in range(SEEDS)]; fj = [('f', m, 3, strings(m, x), s) for m in MS for x in XS for s in range(SEEDS)]
    with Pool(24) as pool: out = list(tqdm(pool.imap(job, cj + fj, chunksize=1), total=len(cj + fj)))
    co, fo = out[:len(cj)], out[len(cj):]; res = dict(xs=np.array(XS), ms=np.array(MS), depths=np.array(DEPTHS), P=np.array([[strings(m, x) for x in XS] for m in MS]))
    for k in co[0]: res[k] = np.array([o[k] for o in co], float).reshape(len(MS), len(DEPTHS), len(XS), SEEDS, *np.shape(co[0][k]))
    for k in fo[0]: res[k] = np.array([o[k] for o in fo], float).reshape(len(MS), len(XS), SEEDS, *np.shape(fo[0][k]))
    np.savez(OUT, **res)

def report():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = np.load(OUT); xs, ms, depths, P = d['xs'], d['ms'], d['depths'], d['P']; f = lambda a: ' '.join(f'{x:5.2f}' for x in a); se = lambda a: np.nanstd(a, 1) / np.sqrt(a.shape[1])
    for im, m in enumerate(ms):
        print(f'\n===== m = {m}: P = x * v m^2 / (1 - m/v), x =', xs, '\n  strings                          ', ' '.join(f'{p:5d}' if p < 1e5 else f'{p/1e3:4.0f}k' for p in P[im]))
        for i, depth in enumerate(depths):
            print(f'  composed reader, depth {depth} (token-loss scaling v m^(L+1)/(1-f) = {V * m ** (depth + 1) / (1 - m / V):.3g} strings)')
            for h in range(depth - 1): print(f'    map state -> symbol, level {h+1}   ', f(np.nanmean(d['c_merge'][im, i, :, :, h], 1)))
            print('    own samples fully valid        ', f(d['c_valid'][im, i, :, :, depth - 1].mean(1)), '\n    valid samples not in the set   ', f(np.nanmean(d['c_novel'][im, i], 1)))
        print('  flat unit, depth 3')
        for h in range(3): print(f'    code, level {h+1}: own codes below ', f(d['f_own_code'][im, :, :, h].mean(1)), '| true states below', f(d['f_given_code'][im, :, :, h].mean(1)))
        for h in range(3): print(f'    own samples parse up to level {h+1}', f(d['f_valid'][im, :, :, h].mean(1)))
        print('    train log-lik. per token pair, level 1, at the end (bound: -ln(v m) =', f'{-np.log(V * m):.3f})', f(d['f_own_ll'][im, :, :, 0, -1].mean(1)))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4), constrained_layout=True); im = len(ms) - 1; m = ms[im]; cd = plt.cm.plasma(np.linspace(0, 0.75, len(depths))); cl = plt.cm.viridis(np.linspace(0, 0.9, 4))
    for i, depth in enumerate(depths):
        y = d['c_valid'][im, i, :, :, depth - 1]; ax[0].errorbar(P[im], y.mean(1), se(y), color=cd[i], marker='o', label=f'depth {depth}'); ax[0].axvline(V * m ** (depth + 1) / (1 - m / V), color=cd[i], ls='--', lw=1)
    ax[0].set_xscale('log'); ax[0].set_xlim(P[im, 0] / 1.5, 2e7); ax[0].set_xlabel('strings in the fixed set, P'); ax[0].set_ylabel('share of own samples that are fully valid'); ax[0].set_title(f'composed reader (m = {m})\ndashed: published token-loss scaling $v\\,m^{{L+1}}$'); ax[0].legend()
    i = len(depths) - 1
    for h in range(depths[i] - 1): y = d['c_merge'][im, i, :, :, h]; ax[1].errorbar(P[im] * 2 ** (depths[i] - h - 2), np.nanmean(y, 1), se(y), color=cl[h], marker='o', label=f'level {h+1}')
    ax[1].axvline(V * m * m / (1 - m / V), color='k', ls=':', lw=1); ax[1].set_xscale('log'); ax[1].set_xlabel('sibling pairs at that level in the fixed set'); ax[1].set_ylabel('quality of the map state -> symbol'); ax[1].set_title(f'composed reader, depth {depths[i]}, per level\ndotted: $v\\,m^2/(1-f)$'); ax[1].legend()
    for h in range(3): y = d['f_valid'][im, :, :, h]; ax[2].errorbar(P[im], y.mean(1), se(y), color=cl[h], marker='o', label=f'parse up to level {h+1}'); ax[2].axhline((m / V) ** (h + 1) if h else np.nan, color=cl[h], ls=':', lw=1)
    ax[2].set_xscale('log'); ax[2].set_xlabel('strings in the fixed set, P'); ax[2].set_ylabel('share of own samples'); ax[2].set_title('flat unit (likelihood only), depth 3\ndotted: random pairs above level 1'); ax[2].legend()
    for h in range(3): ax[3].plot(np.arange(PASSES), d['f_own_ll'][im, -3, 0, h], color=cl[h], label=f'level {h+1}')
    ax[3].axhline(-np.log(V * m), color=cl[0], ls=':', lw=1); ax[3].set_xlabel('EM passes'); ax[3].set_ylabel('log-likelihood per sibling pair (training set)'); ax[3].set_title(f'flat unit, learning curves\nP = {P[im, -3]}, one seed; dotted: best possible at level 1'); ax[3].legend()
    fig.savefig(FIG, dpi=130, facecolor='white'); print(FIG)

if __name__ == '__main__':
    if len(sys.argv) == 1: run()
    report()
