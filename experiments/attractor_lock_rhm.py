"""Fixed points 'laid ready' in a recogniser that runs from the tokens to deeper and deeper latents: pilot in the RHM
(v symbols, m productions per symbol, pairs, depth L, fixed set of P teacher strings; same world and idealisation as unlock_stack_check.py).
Every level has two Hebbian memories, both in the symbols of the level below:
  pair memory:    the sibling pairs of child symbols that occurred. A node 'locks' if its pair is stored; the stored pair is the node's state.
  context memory: counts of which states stand next to which (as left and as right sibling); two states are alike if they keep the same company.
Symbol units (v per level; a right unit holds the m states of one symbol) are learned in four ways:
  watch:   likelihood clustering of the states by sibling context (spectral start, k-means, hard EM of the block model): the learner of unlock_stack_check.py.
  dream:   start from watch. Offline step: every unit writes its most typical state; the context memory completes it to the m states most alike in
           context (one state joins at a time; the unit's own states count BETA more); the unit copies the completed set and the memory unlearns it
           for the rest of the step (those states are no longer available to the units that follow; the next round starts from the full memory). Units whose own states come back go first. Then hard EM (watching again).
  recruit: the same completion and unlearning, but what the units hold is ignored (a random state among those left is completed): the memory alone.
  keep:    dream without unlearning (a state may be claimed twice; the first unit keeps it).
dream, recruit and keep run ROUNDS rounds of (offline step, hard EM).
Part 0 (pair level, flat units of unlock_stack_check.py: K = v m states, factorised emission, EM): do a unit's dreams land on ONE stored pair?
        Offline step there: the unit writes a pair, takes the stored pair it lands on (or the nearest one left), the memory unlearns it.
Part 1 (one level): quality of the symbol units against the number of sibling pairs; does a right unit stay put when it dreams?
Part 2 (stack, depth 3-5, levels learned from the tokens up): own samples valid up to the root, fresh strings read up to the root.
Usage: python attractor_lock_rhm.py [report]   (report = replot from the saved arrays)"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
from unlock_escape_rate_check import rules
from unlock_stack_check import sample, nmi, parses, draw, novel, kmeans, fit, reader, V

OUT = '/var/tmp/jbauer/generative_practice_attractor/attractor_lock_rhm.npz'; FIG = '/nfs/nhome/live/jbauer/generative-learning/reports/attractor-lock-rhm.png'
MS, X1, XS, DEPTHS, SEEDS1, SEEDS, NTEST, ROUNDS, BETA = [4, 8], [0.5, 1, 1.5, 2, 3, 4, 8, 16], [0.5, 1, 2, 4, 8, 16, 32, 64], [3, 4, 5], 8, 4, 5000, 3, 0.05
MODES = ['watch', 'dream', 'recruit', 'keep']
tc = lambda m: V * m * m / (1 - m / V)   # sibling pairs at which the symbols of a level become detectable (unlock_escape_rate_check.py)

def counts(l, r, n): N = np.zeros((n, n)); np.add.at(N, (l.ravel(), r.ravel()), 1); return N

# ---- symbol units
def hard_em(M, z, k, rounds=10):
    for _ in range(rounds):
        Z = np.eye(k)[z]; Rm, Cm = M @ Z, M.T @ Z; Tr, Tc = Z.T @ Rm + 0.1, Z.T @ Cm + 0.1; z2 = (Rm @ np.log(Tr / Tr.sum(1, keepdims=True)).T + Cm @ np.log(Tc / Tc.sum(1, keepdims=True)).T).argmax(1)
        if (z2 == z).all(): break
        z = z2
    return z

def watch(M, k, rng):
    U, S, Vt = np.linalg.svd(M / np.sqrt(np.outer(M.sum(1) + 1e-9, M.sum(0) + 1e-9))); return hard_em(M, kmeans(np.hstack([U[:, 1:k] * S[1:k], Vt[1:k].T * S[1:k]]), k, rng), k)

def similarity(M): X = np.hstack([M, M.T]); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12); return X @ X.T

def grow(S, seed, m, avail, bias):
    h = [seed]; sc = S[seed].copy(); free = avail.copy(); free[seed] = False
    while len(h) < m and free.any(): c = np.argmax(np.where(free, sc / len(h) + bias, -np.inf)); h.append(c); free[c] = False; sc += S[c]
    return np.array(h)

def dream(M, z, k, m, rng, recruit=False, unlearn=True):
    """one offline step; returns the new units and, per unit, the share of its own states that came back when it dreamed first (1 = stays put)"""
    S = similarity(M); n = len(M); avail = np.ones(n, bool); znew = np.full(n, -1); mem = [np.flatnonzero(z == u) for u in range(k)]
    def seed(u):
        own = mem[u][avail[mem[u]]]; cand = np.flatnonzero(avail)
        if recruit or len(mem[u]) == 0: return cand[rng.integers(len(cand))]
        return own[np.argmax(S[np.ix_(own, own)].sum(1))] if len(own) else cand[np.argmax(S[np.ix_(cand, mem[u])].sum(1))]
    bias = lambda u: 0 if recruit else BETA * (z == u)
    first = [grow(S, seed(u), m, avail, bias(u)) for u in range(k)]; stay = np.array([np.isin(first[u], mem[u]).sum() / max(m, len(mem[u])) for u in range(k)])
    for u in (rng.permutation(k) if recruit else np.argsort(-stay + 1e-6 * rng.random(k))):
        if not avail.any(): break
        if unlearn: h = grow(S, seed(u), m, avail, bias(u)); znew[h] = u; avail[h] = False
        else: h = first[u]; znew[h[znew[h] < 0]] = u
    for i in np.flatnonzero(znew < 0): znew[i] = np.bincount(znew[znew >= 0], weights=S[i][znew >= 0], minlength=k).argmax()   # states nobody took: to the unit most alike
    return znew, stay

def units(M, k, m, rng, mode):
    z = watch(M, k, rng)
    for _ in range(0 if mode == 'watch' else ROUNDS): z = hard_em(M, dream(M, z, k, m, rng, recruit=mode == 'recruit', unlearn=mode != 'keep')[0], k)
    return z

# ---- part 1: one level
def level(a):
    m, x, seed = a; rng = np.random.default_rng([seed, m, int(100 * x)]); R = [rules(V, m, rng) for _ in range(2)]; tok, _ = sample(R, int(x * tc(m)), rng)
    pair = tok[:, 0::2] * V + tok[:, 1::2]; N = counts(pair[:, 0], pair[:, 1], V * V); seen = N.sum(1) + N.sum(0) > 0; M = N[seen][:, seen]
    truth = np.full(V * V, -1); truth[R[0][:, :, 0] * V + R[0][:, :, 1]] = np.arange(V)[:, None]; zt = truth[seen]; S = similarity(M); off = ~np.eye(len(S), dtype=bool)
    zw = watch(M, V, rng); pure = np.array([(zw == u).any() and len(np.unique(zt[zw == u])) == 1 and (zw == u).sum() == (zt == zt[zw == u][0]).sum() for u in range(V)]); stay = dream(M, zw, V, m, rng)[1]
    res = dict(nn=float(np.mean(zt[np.argmax(np.where(off, S, -1), 1)] == zt)), truth_kept=nmi(hard_em(M, zt.copy(), V), zt), pure=float(pure.mean()), stay_pure=float(stay[pure].mean()) if pure.any() else np.nan, stay_other=float(stay[~pure].mean()) if (~pure).any() else np.nan)
    for mode in MODES: z = units(M, V, m, rng, mode); res[f'q_{mode}'] = nmi(z, zt); res[f'exact_{mode}'] = float(nmi(z, zt) > 0.9999)
    return res

# ---- part 0: pair level with flat units
def em0(N, pi, A, B):
    """one EM pass without smoothing (cells that never occurred carry no weight)"""
    W = np.where(N > 0, N / np.maximum((A * pi[:, None]).T @ B, 1e-300), 0); An, Bn = pi[:, None] * A * (B @ W.T), pi[:, None] * B * (A @ W); pin = An.sum(1)
    return pin / pin.sum(), An / np.maximum(An.sum(1, keepdims=True), 1e-300), Bn / np.maximum(Bn.sum(1, keepdims=True), 1e-300)

def pair_level(a):
    m, x, seed = a; rng = np.random.default_rng([seed, m, int(100 * x), 7]); K = V * m; R = [rules(V, m, rng)]; tok, _ = sample(R, int(x * tc(m)), rng); tk, sk = sample(R, NTEST, rng)
    l, r = tok[:, 0::2], tok[:, 1::2]; N = counts(l, r, V); pi, A, B, _ = fit(l, r, V, K, rng); code = lambda p, a, b: nmi(reader(p, a, b)[tk[:, 0::2], tk[:, 1::2]], sk[1])
    act = np.arange(K); used = pi > 0.1 / K; d1 = np.stack([draw(np.tile(A[u], (8, 1)), rng) for u in act]); d2 = np.stack([draw(np.tile(B[u], (8, 1)), rng) for u in act])   # 8 dreams per unit
    one = np.array([len(set(zip(a, b))) == 1 for a, b in zip(d1, d2)], float); res = dict(code_em=code(pi, A, B), used=float(used.mean()), dreams_stored=float((N[d1, d2] > 0)[used].mean()), one_pair=float(one[used].mean()))
    avail = N > 0; A2, B2, p2 = np.full((K, V), 1e-12), np.full((K, V), 1e-12), np.full(K, 1e-300)
    for i in np.argsort(-(one + 0.5 * used) + 1e-6 * rng.random(K)):   # units whose dreams are one pair go first, units that EM left unused last
        if not avail.any(): break
        u, c1, c2 = act[i], d1[i, 0], d2[i, 0]
        if not avail[c1, c2]: near = avail * (np.arange(V)[:, None] == c1) + avail * (np.arange(V)[None] == c2); w = (near if near.any() else avail) * (np.outer(A[u], B[u]) + 1e-30); c1, c2 = np.unravel_index(np.argmax(w), w.shape)
        A2[u, c1], B2[u, c2], p2[u] = 1, 1, N[c1, c2] / N.sum(); avail[c1, c2] = False
    res['code_dream'] = code(p2, A2, B2); res['pairs_left'] = float(avail.sum())
    bound = p2 > 1e-100; p2, A2, B2 = p2[bound], A2[bound], B2[bound]   # watching again: EM from the units that hold a pair
    for _ in range(20): p2, A2, B2 = em0(N, p2, A2, B2)
    res['code_dream_then_em'] = code(p2 + 1e-300, np.maximum(A2, 1e-12), np.maximum(B2, 1e-12)); return res

# ---- part 2: the stack
def stack(a):
    mode, m, depth, P, seed = a; rng = np.random.default_rng([seed, m, depth, P]); R = [rules(V, m, rng) for _ in range(depth)]; tok, st = sample(R, P, rng); tk, sk = sample(R, NTEST, rng)
    sym, symk, ns, read, gen, q, rd = tok, tk, V, np.ones(tk.shape, bool), [], [], []
    for h in range(1, depth + 1):
        pair, pairk = sym[:, 0::2] * ns + sym[:, 1::2], symk[:, 0::2] * ns + symk[:, 1::2]; uniq, inv = np.unique(pair, return_inverse=True); state = inv.reshape(pair.shape); n = len(uniq)
        pos = np.searchsorted(uniq, pairk).clip(0, n - 1); read = (uniq[pos] == pairk) & read[:, 0::2] & read[:, 1::2]; rd.append(read.mean()); cnt = np.bincount(state.ravel(), minlength=n)
        if h == depth: gen.append((uniq, cnt, None, ns)); break
        z = units(counts(state[:, 0::2], state[:, 1::2], n), V, m, rng, mode); gen.append((uniq, cnt, z, ns)); sym, symk, ns = z[state], np.where(read, z[pos], V), V + 1
        q.append(nmi(symk[read], (sk[h] // m)[read]) if read.any() else np.nan)
    x = draw(np.tile(gen[-1][1].astype(float), (NTEST, 1)), rng)[:, None]   # own samples: a stored top pair, then every symbol -> one of its unit's states, down to the tokens
    for h in range(depth, 0, -1):
        uniq, cnt, z, ns = gen[h - 1]; code = uniq[x]; x = np.stack([code // ns, code % ns], 2).reshape(NTEST, -1)
        if h > 1: ub, cb, zb, _ = gen[h - 2]; Wt = np.zeros((V + 1, len(ub))); Wt[zb, np.arange(len(ub))] = cb; Wt[Wt.sum(1) == 0] = 1; x = draw(Wt[x.ravel()], rng).reshape(NTEST, -1)
    val, ok = parses(x, R); pad = [np.nan] * (max(DEPTHS) - depth); return dict(valid=val + pad, read=rd + pad, q=q + pad + [np.nan], novel=novel(x, ok, tok))

def job(a): return dict(level=level, pair=pair_level, stack=stack)[a[0]](a[1:])

def run():
    j1 = [('level', m, x, s) for m in MS for x in X1 for s in range(SEEDS1)]; j0 = [('pair', m, x, s) for m in MS for x in (1, 4, 16) for s in range(SEEDS)]
    j2 = [('stack', mode, m, d, int(x * tc(m)), s) for mode in MODES for m in MS for d in DEPTHS for x in XS for s in range(SEEDS)]
    with Pool(20) as pool: out = list(tqdm(pool.imap(job, j2[::-1] + j1 + j0, chunksize=1), total=len(j0 + j1 + j2)))
    o2, o1, o0 = out[:len(j2)][::-1], out[len(j2):len(j2) + len(j1)], out[len(j2) + len(j1):]; res = {}
    for k in o1[0]: res['l_' + k] = np.array([o[k] for o in o1], float).reshape(len(MS), len(X1), SEEDS1)
    for k in o0[0]: res['p_' + k] = np.array([o[k] for o in o0], float).reshape(len(MS), 3, SEEDS)
    for k in o2[0]: res['s_' + k] = np.array([o[k] for o in o2], float).reshape(len(MODES), len(MS), len(DEPTHS), len(XS), SEEDS, *np.shape(o2[0][k]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True); np.savez(OUT, **res)

def report():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = np.load(OUT); f = lambda a: ' '.join(f'{x:5.2f}' for x in a); name = dict(watch='watching only', dream='+ own dreams, a taken state leaves the memory', recruit='memory alone (what the units hold is ignored)', keep='+ own dreams, taken states stay in the memory')
    for im, m in enumerate(MS):
        print(f'\n===== m = {m}')
        print('  pair level, flat units (sibling pairs / T_c = 1, 4, 16): code quality after EM', f(d['p_code_em'][im].mean(1)), '| dreams that are stored pairs', f(d['p_dreams_stored'][im].mean(1)), '| units whose 8 dreams are one pair', f(d['p_one_pair'][im].mean(1)), '| units in use', f(d['p_used'][im].mean(1)),
              '| code quality after the offline step', f(d['p_code_dream'][im].mean(1)), ', then 20 EM passes', f(d['p_code_dream_then_em'][im].mean(1)), '| stored pairs nobody took', f(d['p_pairs_left'][im].mean(1)))
        print('  one level, sibling pairs / T_c =      ', f(X1)); print('    nearest state in context is a synonym', f(d['l_nn'][im].mean(1))); print('    hard EM started at the truth stays   ', f(d['l_truth_kept'][im].mean(1)))
        print('    units right after watching           ', f(d['l_pure'][im].mean(1)), '\n    own states that come back: right units', f(np.nanmean(d['l_stay_pure'][im], 1)), '\n                               other units', f(np.nanmean(d['l_stay_other'][im], 1)))
        for mode in MODES: print(f'    {name[mode]:46s} quality', f(d[f'l_q_{mode}'][im].mean(1)), '| worlds with every unit right', f(d[f'l_exact_{mode}'][im].mean(1)))
        print('  stack, strings / T_c =                ', f(XS))
        for i, depth in enumerate(DEPTHS):
            for k, mode in enumerate(MODES): print(f'    depth {depth} {name[mode]:46s} own samples valid to the root', f(d['s_valid'][k, im, i, :, :, depth - 1].mean(1)), '| fresh strings read to the root', f(d['s_read'][k, im, i, :, :, depth - 1].mean(1)), '| units, top merge', f(np.nanmean(d['s_q'][k, im, i, :, :, depth - 2], 1)), '| valid samples not in the set', f(np.nanmean(d['s_novel'][k, im, i], 1)))
    im = MS.index(8); m = 8; w, h = plt.rcParams['figure.figsize']; fig, ax = plt.subplots(2, 2, figsize=(2.9 * w, 1.9 * h), constrained_layout=True); ax = ax.ravel(); sty = dict(watch='-', dream='-', recruit='--', keep=':')
    for mode in MODES: ax[0].plot(X1, d[f'l_q_{mode}'][im].mean(1), marker='o', ls=sty[mode], label=name[mode]); ax[1].plot(X1, d[f'l_exact_{mode}'][im].mean(1), marker='o', ls=sty[mode], label=name[mode])
    ax[0].plot(X1, d['l_truth_kept'][im].mean(1), color='k', lw=0.8, ls=':', label='hard EM started at the truth'); ax[0].set_ylabel('quality of the symbol units'); ax[0].set_title(f'(a) one level (v = {V}, m = {m}): how right the symbol units are'); ax[1].set_ylabel('share of worlds with every unit right'); ax[1].set_title('(b) one level: every unit right')
    for a in ax[:2]: a.set_xscale('log'); a.set_xticks(X1); a.set_xticklabels(X1); a.minorticks_off(); a.set_xlabel('sibling pairs in the teacher set / $T_c$')
    i = DEPTHS.index(5); P = [int(x * tc(m)) for x in XS]
    for k, mode in enumerate(MODES): ax[2].plot(P, d['s_valid'][k, im, i, :, :, 4].mean(1), marker='o', ls=sty[mode], label=name[mode])
    ax[2].plot(P, d['s_read'][MODES.index('dream'), im, i, :, :, 4].mean(1), color='k', lw=0.8, ls=':', label='fresh strings whose pairs are all stored (dream arm)'); ax[2].set_ylabel('share of own samples valid up to the root'); ax[2].set_title('(c) stack of 5 levels: own samples')
    for i, depth in enumerate(DEPTHS):
        for mode in ('watch', 'dream'): ax[3].plot(P, d['s_valid'][MODES.index(mode), im, i, :, :, depth - 1].mean(1), marker='o', color=plt.cm.viridis(i / 2.5), ls='-' if mode == 'dream' else '--', label=f'depth {depth}, {name[mode]}')
    ax[3].set_ylabel('share of own samples valid up to the root'); ax[3].set_title('(d) stacks of 3, 4 and 5 levels')
    for a in ax[2:]: a.set_xscale('log'); a.set_xlabel('strings in the teacher set, P')
    for a in ax: a.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.savefig(FIG, dpi=150, facecolor='white'); print(FIG)

if __name__ == '__main__':
    if len(sys.argv) == 1: run()
    report()
