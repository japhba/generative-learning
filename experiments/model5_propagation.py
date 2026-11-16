"""Model 5: how deep a teacher sample and an own sample reach into the learner, over the course of learning (README: own samples should propagate deeper).
A sample reaches level l if the reader's explanation of it is more probable than chance at every level up to l (the drive criterion of section 2, level by level):
  level 1  both codes of a pair are above chance in the pools of the units that read them, N G[a, k] > 1;
  level 2  and the pair of units is above chance in the table, K^2 T[a, a'] > 1 (K units per side);
  level 3  (three levels) and the string, read through the class names, falls on a pair of named pairs that the counted teacher strings also fall on.
Levels 1 and 2 are averaged over the reader's posterior q(a, a' | k, k'), i.e. they are the probability that the winner the rule would credit passes.
Runs:
  stack     the learner of section 6 (Pooler, eta 1) on three levels: n/2 teacher strings watched, practice within each block, the other n/2 teacher strings
            read through the current names and counted at the top. Fresh teacher strings and the learner's own pairs are scored at checkpoints of practice.
            Own samples have no level 3: practice runs within the blocks and the top is counted from teacher strings only.
  gradient  the likelihood-gradient learner of section 2 (v units, sharp random start, teacher and own batches interleaved 1:1), one block.
Writes /var/tmp/jbauer/model5_lateral_verdict/propagation_<run>.npz.   Usage: python model5_propagation.py <run> [--worlds 8] [--procs 28]
Figure: python model5_propagation.py figure <out dir>  (model5-propagation.svg/.png and propagation_numbers.json)."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from model5_depth import World3
from model5_lateral_verdict import BATCH, OUT, GradientLearner, World
from model5_own_benefit import Pooler, string_loss, unit_names

PRACTICE = np.concatenate([[0], 2 ** np.arange(10, 21)])                                                                            # own strings practised at the checkpoints
GSTEPS = np.unique(np.concatenate([[0], np.round(2 ** np.arange(0, np.log2(30000) + 1e-9, 0.5)).astype(int), [30000]]))            # batches (teacher or own) at the checkpoints


def reach(L, k, kp):                                                                                                                # per pair: probability that the read winner passes level 1, and levels 1 and 2
    GL, GR, T = L.tables(); K, N = len(T), L.N; A, B = GL[:, k].T, GR[:, kp].T
    q = A[:, :, None] * T[None] * B[:, None, :]; q /= q.sum((1, 2), keepdims=True)
    p1 = (N * A > 1)[:, :, None] & (N * B > 1)[:, None, :]; p2 = p1 & (K * K * T > 1)[None]
    return (q * p1).sum((1, 2)), (q * p2).sum((1, 2))


def top_seen(W, nm, held, x):                                                                                                       # level 3: the string's pair of named pairs is one the counted teacher strings fall on
    N = W.w.N; key = lambda s: ((nm[0][s[:, 0]] * N + nm[1][s[:, 1]]) * N + nm[2][s[:, 2]]) * N + nm[3][s[:, 3]]
    assert len(held) < np.prod([len(np.unique(n)) for n in nm])                                                                     # then 'above chance' (count / n > 1 / possible pairs) is 'count >= 1'
    u = np.unique(key(held)); kx = key(x); i = np.minimum(np.searchsorted(u, kx), len(u) - 1); return u[i] == kx


def one_stack(job):
    n, seed = job; W = World3(seed); w = W.w; rng = np.random.default_rng(40_000 + seed); pop = W.strings(rng, 20000); x = W.strings(rng, n); n1 = n // 2; held = x[n1:]
    gs = [Pooler(w, eta=1.0), Pooler(w, eta=1.0)]; out = []; done = 0
    for g, c in zip(gs, ((0, 1), (2, 3))):
        for i in range(0, n1, 256): g.watch(x[i:min(i + 256, n1), c[0]], x[i:min(i + 256, n1), c[1]], rng)
    for P in PRACTICE:
        for g in gs:
            for _ in range((P - done) // BATCH): g.practice(BATCH, rng)
        done = P; t = pop[:4096]; r = [reach(g, t[:, c[0]], t[:, c[1]]) for g, c in zip(gs, ((0, 1), (2, 3)))]
        nm = [unit_names(gs[0], 0), unit_names(gs[0], 1), unit_names(gs[1], 0), unit_names(gs[1], 1)]
        o = [reach(g, *g.sample(4096, rng)) for g in gs]                                                                            # an own string: one own pair from each block, as in practice
        out.append(dict(t1=float((r[0][0] * r[1][0]).mean()), t2=float((r[0][1] * r[1][1]).mean()), t3=float((r[0][1] * r[1][1] * top_seen(W, nm, held, t)).mean()),
                        o1=float((o[0][0] * o[1][0]).mean()), o2=float((o[0][1] * o[1][1]).mean()), loss=string_loss(W, pop, held, nm), table=string_loss(W, pop, x, [np.arange(w.N)] * 4),
                        named=float(np.mean([len(set(nm[i][w.members[k]])) == 1 and (nm[i] == nm[i][w.members[k][0]]).sum() == w.m for i in range(4) for k in range(w.v)]))))
    return n, seed, out


def one_gradient(job):
    seed, theta = job; w = World(seed=seed); rng = np.random.default_rng(5000 + seed); L = GradientLearner(w, K=w.v, lam=0, even=True, seed=seed, start='random', s=6.0, theta=theta); L.watch(*w.teacher(rng, 8192), rng, wake=False)
    out, acc = [], 0.0
    for t in range(GSTEPS[-1] + 1):
        if t in GSTEPS:
            tk, tkp = w.teacher(rng, 4096); r = reach(L, tk, tkp); o = reach(L, *L.sample(4096, rng)[:2])
            out.append(dict(t1=float(r[0].mean()), t2=float(r[1].mean()), o1=float(o[0].mean()), o2=float(o[1].mean()), loss=L.score()['kl']))
        if t == GSTEPS[-1]: break
        acc += 0.5
        if acc >= 1 - 1e-9: acc -= 1; L.practice(BATCH, rng)
        else: L.watch(*w.teacher(rng, BATCH), rng)
    return seed, theta, out


def stack():
    def distinct(B): return len({r.tobytes() for r in B}) == len(B) and len({r.tobytes() for r in B.T}) == len(B)                   # the worlds of section 6
    seeds = [s for s in range(64) if distinct(World(seed=s).B)][:args.worlds]; ns = [int(x) for x in args.ns.split(',')]; jobs = [(n, s) for n in ns for s in seeds]; res = {}
    with Pool(args.procs) as p:
        for n, s, out in tqdm(p.imap_unordered(one_stack, jobs), total=len(jobs)): res[n, s] = out
    keys = list(res[jobs[0]][0]); np.savez(OUT / 'propagation_stack.npz', n=ns, own=PRACTICE, **{k: np.array([[[c[k] for c in res[n, s]] for s in seeds] for n in ns]) for k in keys})
    for n in ns:
        for k in keys: print(n, k, np.median([[c[k] for c in res[n, s]] for s in seeds], 0).round(3).tolist())


def gradient():
    jobs = [(s, th) for th in (0.0, 1.0) for s in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for s, th, out in tqdm(p.imap_unordered(one_gradient, jobs), total=len(jobs)): res[s, th] = out
    keys = list(res[jobs[0]][0]); np.savez(OUT / 'propagation_gradient.npz', theta=[0.0, 1.0], samples=GSTEPS * BATCH, **{k: np.array([[[c[k] for c in res[s, th]] for s in range(args.worlds)] for th in (0.0, 1.0)]) for k in keys})
    for th in (0.0, 1.0):
        for k in keys: print(th, k, np.median([[c[k] for c in res[s, th]] for s in range(args.worlds)], 0).round(3).tolist())


def figure():
    import matplotlib.pyplot as plt
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#898781'; LEV = (BLUE, ORANGE, AQUA); W, H = plt.rcParams['figure.figsize']
    S, G = np.load(OUT / 'propagation_stack.npz'), np.load(OUT / 'propagation_gradient.npz'); nums = {}
    def band(ax, x, y, color, label=None, **kw): q1, med, q3 = np.percentile(y, [25, 50, 75], 0); ax.plot(x, med, color=color, label=label, **kw); ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0)
    def levels(ax, x, D, j, top, named=None):
        for l in range(top): band(ax, x, D[f't{l + 1}'][j], LEV[l], f'teacher samples, level {l + 1}', marker='o', ms=4)
        for l in range(2): band(ax, x, D[f'o{l + 1}'][j], LEV[l], f'own samples, level {l + 1}', ls='--', marker='s', ms=4, mfc='white')
        if named is not None: band(ax, x, named, GREY, 'share of classes named', ls=':')
        ax.set_xscale('log', base=2); ax.set_ylim(-0.03, 1.03)
    fig, axes = plt.subplots(2, 2, figsize=(2 * W, 2 * H), sharey=True, constrained_layout=True); x = np.maximum(S['own'], 2 ** 9)
    for c, n in enumerate(S['n']):
        ax = axes[0, c]; levels(ax, x, S, c, 3, S['named'][c]); ax.set_title(f'model 5 learner, {int(n):,} teacher strings'); ax.set_xlabel('own strings practised (first point: none)')
    gx = np.maximum(G['samples'], 2 ** 5)
    for c, th in enumerate(G['theta']):
        ax = axes[1, c]; levels(ax, gx, G, c, 2); ax.set_title('gradient learner, ' + ('every sample learns' if th == 0 else 'only samples read above chance learn')); ax.set_xlabel('samples, teacher and own 1:1 (first point: none)')
    for ax in axes[:, 0]: ax.set_ylabel('share of samples that reach the level')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncols=3); fig.savefig(out / 'model5-propagation.svg', transparent=True); fig.savefig(out / 'model5-propagation.png', dpi=150, facecolor='white'); print(out / 'model5-propagation.png')
    med = lambda a: np.median(a, 0)
    nums['stack'] = {str(int(n)): dict(worlds=int(S['t1'].shape[1]), own=[int(v) for v in S['own']], **{k: med(S[k][c]).round(4).tolist() for k in ('t1', 't2', 't3', 'o1', 'o2', 'loss', 'table', 'named')}) for c, n in enumerate(S['n'])}
    nums['gradient'] = dict(worlds=int(G['t1'].shape[1]), samples=[int(v) for v in G['samples']], **{f'{k}_th{int(th)}': med(G[k][j]).round(4).tolist() for j, th in enumerate(G['theta']) for k in ('t1', 't2', 'o1', 'o2', 'loss')})
    (out / 'propagation_numbers.json').write_text(json.dumps(nums, indent=1)); print(out / 'propagation_numbers.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('out', nargs='?'); ap.add_argument('--worlds', type=int, default=8); ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--ns', default='4096,16384'); args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); globals()[args.run]()
