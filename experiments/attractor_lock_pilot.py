"""Pilot: a recogniser with fixed points 'laid ready', and own samples that lock into them.
World: D random features f_k in {-1,+1}^N; an example is the majority vote of L = 3 distinct features, x = sign(f_a + f_b + f_c)
(D (D-1) (D-2) / 6 valid examples). Teacher set: P examples drawn uniformly (unbiased, small).
Recogniser: a memory with the covariance rule on the teacher set, J = sum_p (x^p - xbar)(x^p - xbar)^T, zero diagonal; reading =
letting the state settle under h <- sign(J h). No codes, no generator involved.
Generator: a dictionary W (N x D units), sample = sign of the sum of L units. Baseline learner = tied k-sparse autoencoder on the
teacher set (code = the L units with the largest normalised overlap, x_hat = W z, delta rule on the reconstruction error).
Offline step ('own'): every unit writes its own pattern sign(w_k), the memory settles from it, the unit copies the settled state.
Offline step ('own_conj'): the generator writes whole samples (3 units each, 2 D samples), the memory settles from each, and the
unit of the three nearest to the settled state takes it (majority over the settled states a unit receives).
Parts:
 (1) what is laid ready: never-shown features as fixed points of the memory against P; where a fresh example settles; the
     autoencoder's features on the same sets.
 (2) what one settling does to a unit: overlap with a feature before -> after, for a blurred feature and for a blend of two.
 (3) autoencoder epochs interleaved with offline steps against the same number of autoencoder epochs alone (and against three
     times as many), with controls for the offline step: binarise only | per unit as many random flips as its dream changed |
     the same number of flips spread evenly over all units | settle from random states | settle from teacher examples (unit = a
     random one; or each settled example assigned to its nearest unit) | own dream in the memory of another world | own dream +
     Hebb / anti-Hebb on the settled state in the memory. Also: does a unit that stays put when it settles carry a feature?
Usage: python attractor_lock_pilot.py [report]   (report = replot from the saved arrays)"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm

OUT = '/var/tmp/jbauer/generative_practice_attractor/attractor_lock_pilot.npz'; FIG = '/nfs/nhome/live/jbauer/generative-learning/reports/attractor-lock-pilot.png'
N, D, L = 2000, 40, 3
PS = [100, 200, 400, 800, 1600, 3200]; PS_ARMS = [200, 400, 800, 1600]; SEEDS = 8
EP0, EP_ROUND, ROUNDS = 100, 50, 4; EPOCHS = EP0 + EP_ROUND * ROUNDS
LONG = 3   # the long baseline runs LONG * EPOCHS epochs
PROTOCOLS = {'plain': lambda ep, T: 0.05, 'annealed': lambda ep, T: 0.2 * (1 - ep / T)}
ARMS = ['none', 'own', 'own_conj', 'binarise', 'flips', 'flips_uniform', 'random', 'teacher', 'teacher_nearest', 'wrong_world', 'own_hebb', 'own_antihebb']
BLUR = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])   # overlap of the starting state with the feature

def pm(shape, rng): return np.where(rng.random(shape) < 0.5, 1.0, -1.0).astype(np.float32)
def subsets(n, rng): return np.argsort(rng.random((n, D)), 1)[:, :L]
def maj(F, S): return np.sign(F[:, S].sum(-1)).astype(np.float32)
def flip(X, eps, rng): return (X * np.where(rng.random(X.shape) < eps, -1.0, 1.0)).astype(np.float32)
def outer(X, xbar): Xc = X - xbar; return (Xc @ Xc.T).astype(np.float32)
def memory(X): J = outer(X, X.mean(1, keepdims=True)); np.fill_diagonal(J, 0); return J
def relax(J, H, rng, steps=40):
    """asynchronous-like settling: half of the units updated per step, then three full sweeps"""
    H = H.copy()
    for _ in range(steps): H = np.where(rng.random(H.shape) < 0.5, np.where(J @ H >= 0, 1.0, -1.0).astype(np.float32), H)
    for _ in range(3): H = np.where(J @ H >= 0, 1.0, -1.0).astype(np.float32)
    return H
def binar(W): return np.where(W >= 0, 1.0, -1.0).astype(np.float32)

def code(W, X):
    sc = (W.T @ X) / (np.linalg.norm(W, axis=0)[:, None] + 1e-9); win = np.argpartition(-sc, L - 1, 0)[:L]
    Z = np.zeros((W.shape[1], X.shape[1]), np.float32); np.put_along_axis(Z, win, 1.0, 0); return Z
def ae_epochs(W, X, rng, ep_from, ep_to, sched, T=EPOCHS, bs=8):
    for ep in range(ep_from, ep_to):
        for idx in np.array_split(rng.permutation(X.shape[1]), max(X.shape[1] // bs, 1)):
            x = X[:, idx]; Z = code(W, x); W += sched(ep, T) * (x - W @ Z) @ Z.T
    return W

def memory_stats(J, F, Xt, St, rng):
    """share of features that are fixed points (start at the feature / with 30 % / 40 % of sites flipped); share of fresh examples
    that settle on one of their own features, on any feature"""
    fx = [float(np.mean((relax(J, flip(F, e, rng), rng) * F).mean(0) > 0.95)) for e in (0.0, 0.3, 0.4)]
    M = F.T @ relax(J, Xt, rng) / N; own = np.take_along_axis(M, St.T, 0).max(0)
    return fx + [float(np.mean(own > 0.95)), float(np.mean(M.max(0) > 0.95))]
def learner_stats(W, F, Xt, rng):
    """features recovered (some unit overlaps > 0.9), units clean, sign error of the reconstruction of fresh examples, share of
    own samples (3 distinct random units) that are valid examples (overlap > 0.95 with the majority of their 3 nearest features)"""
    O = binar(W).T @ F / N; err = float(np.mean(np.sign(W @ code(W, Xt)) != Xt))
    Xo = np.sign(W[:, subsets(1000, rng)].sum(-1)).astype(np.float32); top = np.argsort(-(F.T @ Xo), 0)[:L].T
    valid = float(np.mean((maj(F, top) * Xo).mean(0) > 0.95))
    return [float(np.mean(O.max(0) > 0.9)), float(np.mean(O.max(1) > 0.9)), err, valid]

def offline(arm, W, mem, X, rng):
    """one offline step; returns the new dictionary. mem = dict(J, xbar, Jw) is changed in place by the Hebb arms."""
    B = binar(W)
    if arm == 'binarise': H = B
    elif arm == 'flips': H = flip(B, np.mean(relax(mem['J'], B, rng) != B, 0, keepdims=True), rng)
    elif arm == 'flips_uniform': H = flip(B, np.mean(relax(mem['J'], B, rng) != B), rng)
    elif arm == 'random': H = relax(mem['J'], pm(B.shape, rng), rng)
    elif arm == 'teacher': H = relax(mem['J'], X[:, rng.choice(X.shape[1], B.shape[1], replace=False)], rng)
    elif arm == 'teacher_nearest':
        Hx = relax(mem['J'], X[:, rng.choice(X.shape[1], min(400, X.shape[1]), replace=False)], rng); k = np.argmax(B.T @ Hx, 0)
        A = np.zeros((B.shape[1], Hx.shape[1]), np.float32); A[k, np.arange(Hx.shape[1])] = 1.0; S = Hx @ A.T
        H = np.where(S == 0, B, np.sign(S)).astype(np.float32)   # unit = majority of the settled examples nearest to it; ties and units without any keep their sign
    elif arm == 'own_conj':
        T3 = subsets(2 * B.shape[1], rng); Hx = relax(mem['J'], np.sign(W[:, T3].sum(-1)).astype(np.float32), rng); k = T3[np.arange(len(T3)), np.argmax(np.stack([(B[:, T3[:, i]] * Hx).sum(0) for i in range(L)]), 0)]
        A = np.zeros((B.shape[1], len(T3)), np.float32); A[k, np.arange(len(T3))] = 1.0; S = Hx @ A.T; H = np.where(S == 0, B, np.sign(S)).astype(np.float32)
    elif arm == 'wrong_world': H = relax(mem['Jw'], B, rng)
    else:
        H = relax(mem['J'], B, rng)
        if arm in ('own_hebb', 'own_antihebb'): dJ = outer(H, mem['xbar']); np.fill_diagonal(dJ, 0); mem['J'] = mem['J'] + (1 if arm == 'own_hebb' else -1) * dJ
    return 0.5 * H

def world(a):
    seed, P = a; rng = np.random.default_rng([seed, P]); F = pm((N, D), rng); X = maj(F, subsets(P, rng)); St = subsets(400, rng); Xt = maj(F, St)
    J = memory(X); xbar = X.mean(1, keepdims=True); res = dict(seed=seed, P=P, mem=memory_stats(J, F, Xt, St, rng))
    # (2) one settling: blurred feature (overlap m0 with the feature), blend of two features (ties broken at random)
    res['blur'] = [float(np.mean((relax(J, flip(F, (1 - m0) / 2, rng), rng) * F).mean(0))) for m0 in BLUR]
    pairs = np.stack([rng.permutation(D) for _ in range(5)]).reshape(-1, 2); S2 = F[:, pairs[:, 0]] + F[:, pairs[:, 1]]; B2 = np.where(S2 == 0, pm(S2.shape, rng), np.sign(S2)).astype(np.float32)
    M2 = F.T @ relax(J, B2, rng) / N; res['blend'] = [float(np.mean(np.maximum(M2[pairs[:, 0], np.arange(len(pairs))], M2[pairs[:, 1], np.arange(len(pairs))]) > 0.95)), float(np.mean(M2.max(0) > 0.95))]
    # (1) and (3): autoencoder alone, and with offline steps
    Jw = memory(maj(pm((N, D), rng), subsets(P, rng))) if P in PS_ARMS else None
    for prot, sched in PROTOCOLS.items():
        Wi = (0.01 * rng.standard_normal((N, D))).astype(np.float32); W0 = ae_epochs(Wi.copy(), X, rng, 0, EP0, sched)
        if P in PS_ARMS:
            res[f'{prot}/none_long'] = learner_stats(ae_epochs(Wi.copy(), X, rng, 0, LONG * EPOCHS, sched, T=LONG * EPOCHS), F, Xt, rng)
            B = binar(W0); stay = (relax(J, B, rng) * B).mean(0) > 0.95; clean = (B.T @ F / N).max(1) > 0.9   # verification: does a unit that stays put carry a feature?
            res[f'{prot}/verify'] = [float(clean.mean()), float(stay[clean].mean()) if clean.any() else np.nan, float(stay[~clean].mean()) if (~clean).any() else np.nan]
        for arm in (ARMS if P in PS_ARMS else ['none']):
            W = W0.copy(); mem = dict(J=J, xbar=xbar, Jw=Jw); r = np.random.default_rng([seed, P, ARMS.index(arm), list(PROTOCOLS).index(prot)]); after = []
            for k in range(ROUNDS):
                if arm != 'none': W = offline(arm, W, mem, X, r); after.append(learner_stats(W, F, Xt, r))
                W = ae_epochs(W, X, r, EP0 + k * EP_ROUND, EP0 + (k + 1) * EP_ROUND, sched)
            res[f'{prot}/{arm}'] = learner_stats(W, F, Xt, r)
            if arm != 'none': res[f'{prot}/{arm}/after_step'] = after
            if arm in ('own_hebb', 'own_antihebb'): res[f'{prot}/{arm}/mem'] = memory_stats(mem['J'], F, Xt, St, r)
    if P in PS_ARMS: res['mem_fresh'] = memory_stats(memory(np.concatenate([X, maj(F, subsets(ROUNDS * D, rng))], 1)), F, Xt, St, rng)   # as many fresh examples as Hebb steps
    return res

def run():
    tasks = [(s, P) for P in PS[::-1] for s in range(SEEDS)]
    with Pool(min(len(tasks), 14)) as pool: res = list(tqdm(pool.imap_unordered(world, tasks), total=len(tasks)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True); np.savez(OUT, res=np.array(res, dtype=object)); return res

def table(res):
    g = lambda key, P: np.array([r[key] for r in res if r['P'] == P and key in r])
    print('memory (features fixed from 0 / 30 % / 40 % flips; fresh example settles on one of its features / on any feature)')
    for P in PS: print(f'  P {P:5d}: ' + ' '.join(f'{v:.2f}' for v in g('mem', P).mean(0)) + '   one settling, start overlap ' + ' '.join(f'{m:.2f}->{v:.2f}' for m, v in zip(BLUR, g('blur', P).mean(0))) + f'   blend of two -> one of the two {g("blend", P).mean(0)[0]:.2f}, any feature {g("blend", P).mean(0)[1]:.2f}')
    for P in PS_ARMS: print(f'  P {P:5d} + {ROUNDS * D} fresh examples: ' + ' '.join(f'{v:.2f}' for v in g('mem_fresh', P).mean(0)) + ' | after own dreams with Hebb (annealed): ' + ' '.join(f'{v:.2f}' for v in g('annealed/own_hebb/mem', P).mean(0)) + ' | anti-Hebb: ' + ' '.join(f'{v:.2f}' for v in g('annealed/own_antihebb/mem', P).mean(0)))
    print('learner (features recovered, units clean, sign error on fresh examples, own samples valid); in brackets: worlds in which the arm recovers more / fewer features than the autoencoder alone')
    for prot in PROTOCOLS:
        for P in PS_ARMS: v = g(f'{prot}/verify', P); print(f'  {prot:8s} P {P:5d} after {EP0} epochs: units that carry a feature {np.nanmean(v[:, 0]):.2f}; stay put when settling: {np.nanmean(v[:, 1]):.2f} of those, {np.nanmean(v[:, 2]):.2f} of the others;  {LONG * EPOCHS} epochs alone: ' + ' '.join(f'{x:.3f}' for x in g(f'{prot}/none_long', P).mean(0)))
    for prot in PROTOCOLS:
        for P in PS:
            for arm in ARMS:
                v = g(f'{prot}/{arm}', P)
                if len(v) == 0: continue
                ref = g(f'{prot}/none', P)[:, 0]; a = g(f'{prot}/{arm}/after_step', P)
                print(f'  {prot:8s} P {P:5d} {arm:16s} ' + ' '.join(f'{x:.3f}' for x in v.mean(0)) + (f'   [{int(np.sum(v[:, 0] > ref))} / {int(np.sum(v[:, 0] < ref))} of {len(ref)}]   right after the last offline step: ' + ' '.join(f'{x:.3f}' for x in a[:, -1].mean(0)) if arm != 'none' else ''))

def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    g = lambda key, P: np.array([r[key] for r in res if r['P'] == P and key in r]); w, h = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(2, 2, figsize=(2.9 * w, 1.9 * h), constrained_layout=True); ax = ax.ravel()
    def line(a, key, col, Ps, label, band=False, **kw):
        v = np.stack([g(key, P)[:, col] for P in Ps]); a.plot(Ps, v.mean(1), label=label, **kw)
        if band: a.fill_between(Ps, np.quantile(v, 0.25, 1), np.quantile(v, 0.75, 1), alpha=0.2, color=a.lines[-1].get_color(), lw=0)
    line(ax[0], 'mem', 1, PS, 'memory: a never-shown feature is a fixed point\n(reached from 30 % flipped sites)', band=True, marker='o')
    line(ax[0], 'mem', 3, PS, 'memory: a fresh example settles on one of its features', band=True, marker='o', ls=':')
    line(ax[0], 'annealed/none', 0, PS, 'autoencoder, annealed rate: features recovered', band=True, marker='s'); line(ax[0], 'plain/none', 0, PS, 'autoencoder, constant rate: features recovered', band=True, marker='s', ls='--')
    ax[0].axvline(D * (D - 1) * (D - 2) / 6, color='k', lw=0.8, ls=':'); ax[0].set_xscale('log'); ax[0].set_xlabel('teacher examples P  (dotted line: all valid examples)'); ax[0].set_ylabel(f'share of the {D} features'); ax[0].set_title('(a) what the teacher set lays ready')
    for P in (400, 800, 1600): ax[1].plot(BLUR, g('blur', P).mean(0), marker='o', label=f'P = {P}')
    ax[1].plot([0, 1], [0, 1], color='k', lw=0.8, ls=':', label='no change'); ax[1].set_xlabel('overlap of a state with a feature before settling'); ax[1].set_ylabel('after settling'); ax[1].set_title('(b) one settling in the memory')
    main = [('none', 'autoencoder alone', '-'), ('own', '+ own dreams', '-'), ('own_antihebb', '+ own dreams, memory unlearns the settled state', '-'), ('own_hebb', '+ own dreams, memory reinforces the settled state', '-')]
    ctrl = [('own_conj', 'own dreams of whole samples (3 units)', '-.'), ('flips', 'control: units reset by as many random flips', '--'), ('flips_uniform', 'control: the same flips spread over all units', '--'), ('teacher_nearest', 'control: settled teacher examples', '--'), ('wrong_world', "control: own dreams in another world's memory", '--')]
    for arm, lab, ls in main + ctrl: line(ax[2], f'annealed/{arm}', 0, PS_ARMS, lab, band=arm in ('none', 'own'), marker='o', ls=ls)
    line(ax[2], 'annealed/none_long', 0, PS_ARMS, f'autoencoder alone, {LONG} times the epochs', marker='x', ls=':', color='k')
    ax[2].set_ylabel('features recovered by the generator'); ax[2].set_title(f'(c) generator after {EPOCHS} autoencoder epochs and {ROUNDS} offline steps')
    for key, col, lab, ls in (('mem', 2, 'as learned from the teacher set', '-'), ('annealed/own_hebb/mem', 2, f'after {ROUNDS * D} own dreams, settled state reinforced', '-'), ('annealed/own_antihebb/mem', 2, f'after {ROUNDS * D} own dreams, settled state unlearned', '-'), ('mem_fresh', 2, f'with {ROUNDS * D} more teacher examples instead', '--')):
        line(ax[3], key, col, PS_ARMS, lab, marker='o', ls=ls)
    ax[3].set_ylabel('features reached from 40 % flipped sites'); ax[3].set_title('(d) memory after the offline steps')
    for a in ax[2:]: a.set_xscale('log'); a.set_xticks(PS_ARMS); a.set_xticklabels(PS_ARMS); a.minorticks_off(); a.set_xlabel('teacher examples P')
    for a in ax: a.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.savefig(FIG, dpi=150, facecolor='white'); print(FIG)

if __name__ == '__main__':
    res = list(np.load(OUT, allow_pickle=True)['res']) if len(sys.argv) > 1 and sys.argv[1] == 'report' else run()
    table(res); plot(res)
