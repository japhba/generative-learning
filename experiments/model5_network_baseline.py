"""Model 5, the baseline that was never run: a plain network with distributed hidden units, trained by backpropagation on teacher pairs only.
One block of model5_lateral_verdict.py. Network: code k -> embedding u_k (d units) -> logits over the partner, p(k, k') = softmax_{kk'}(u_k . v_k' + a_k + b_k'),
two linear layers with a d-unit bottleneck (the skip-gram form). Full-batch Adam on the cross-entropy of 85% of the teacher pairs from a small random start;
training stops at the lowest loss on the held-out 15%, and the weight decay (0 to 0.03) is chosen by the same held-out loss (no population information is used).
d = 16: bottleneck of the size of the class code; d = 64: no bottleneck. 'hidden': the codes grouped by their hidden vectors (Ward linkage, 16 groups), then pairs of groups counted.
Compared with the counted table and with the table over the true classes (model5_own_benefit.class_table).
Writes /var/tmp/jbauer/model5_lateral_verdict/network_baseline.npz.   Usage: python model5_network_baseline.py [--worlds 16] [--procs 28]; --plot <dir> draws the figure"""

import argparse
from multiprocessing import Pool

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import log_softmax
from tqdm.auto import tqdm

from model5_lateral_verdict import OUT, World, table_kl
from model5_own_benefit import class_table, kl

STEPS, EVERY, LR, INIT, DECAYS = 4000, 10, 0.02, 0.1, (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def groups(X, v):                                                                                                                  # v groups of the rows of X (Ward linkage)
    return fcluster(linkage(X, 'ward'), v, 'maxclust') - 1


def train(w, k, kp, d, decay, rng):                                                                                                # weights penalised by decay/2 |.|^2. Returns the held-out loss at its minimum and, at that step, the population
    N = w.N; val = rng.random(len(k)) < 0.15; Q, Qv = np.zeros((N, N)), np.zeros((N, N)); np.add.at(Q, (k[~val], kp[~val]), 1); np.add.at(Qv, (k[val], kp[val]), 1); Q /= Q.sum(); Qv /= Qv.sum()   # loss and the hidden layers
    P = [INIT * rng.standard_normal((N, d)), INIT * rng.standard_normal((N, d)), np.zeros((N, 1)), np.zeros((1, N))]; m = [np.zeros_like(x) for x in P]; v = [np.zeros_like(x) for x in P]; best = (np.inf,)
    for t in range(1, STEPS + 1):
        U, V, a, b = P; lp = log_softmax((U @ V.T + a + b).ravel()).reshape(N, N); G = np.exp(lp) - Q; g = [G @ V + decay * U, G.T @ U + decay * V, G.sum(1, keepdims=True), G.sum(0, keepdims=True)]
        if t % EVERY == 0 and -(Qv * lp).sum() < best[0]: best = (float(-(Qv * lp).sum()), kl(w, np.exp(lp)), t, U.copy(), V.copy())
        for i in range(4): m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2; P[i] = P[i] - LR * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return best + (kl(w, np.exp(lp)),)


def one(job):
    n, seed = job; w = World(seed=seed); rng = np.random.default_rng(2000 + seed); k, kp = w.teacher(rng, n); M = np.zeros((w.N, w.N)); np.add.at(M, (k, kp), 1); out = dict(table=table_kl(w, M), oracle=class_table(w, M, w.cls, w.cls))
    for d in (16, 64):
        runs = [train(w, k, kp, d, decay, np.random.default_rng(3000 + seed)) for decay in DECAYS]; i = int(np.argmin([r[0] for r in runs])); vl, pop, stop, U, V, end = runs[i]   # the decay with the lowest held-out loss
        cl, cr = groups(U, w.v), groups(V, w.v); out |= {f'net{d}': pop, f'decay{d}': DECAYS[i], f'stop{d}': stop, f'end{d}': end, f'hidden{d}': class_table(w, M, cl, cr),              # hidden: codes grouped by their hidden vectors, pairs of groups counted
                                                        f'right{d}': float(np.mean([len(set(c[w.members[a]])) == 1 for c in (cl, cr) for a in range(w.v)]))}
    return n, seed, out


def plot(out):                                                                                                                     # MPLBACKEND=Agg python model5_network_baseline.py --plot <dir>
    import matplotlib.pyplot as plt
    d = np.load(OUT / 'network_baseline.npz'); fig, ax = plt.subplots(constrained_layout=True)
    for key, label, col, ls in (('table', 'counted table', '#898781', '--'), ('net16', 'network, what its output writes', '#eb6834', '-'), ('hidden16', 'network, its hidden layer read out', '#2a78d6', '-'), ('oracle', 'table over the true classes', '#0b0b0b', ':')):
        q1, med, q3 = np.percentile(d[key], [25, 50, 75], 1); ax.plot(d['n'], med, color=col, ls=ls, marker='o', label=label); ax.fill_between(d['n'], q1, q3, color=col, alpha=0.15, lw=0)
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('teacher pairs'); ax.set_ylabel('population loss (nats)'); ax.legend()
    fig.savefig(f'{out}/model5-network-baseline.svg', transparent=True); fig.savefig(f'{out}/model5-network-baseline.png', dpi=150, facecolor='white'); print(f'{out}/model5-network-baseline.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worlds', type=int, default=16); ap.add_argument('--procs', type=int, default=28); ap.add_argument('--plot'); args = ap.parse_args()
    if args.plot: plot(args.plot); raise SystemExit
    ns = 2 ** np.arange(9, 14); jobs = [(int(n), s) for n in ns for s in range(args.worlds)]; res = {}
    with Pool(args.procs) as p:
        for n, s, out in tqdm(p.imap_unordered(one, jobs), total=len(jobs)): res[n, s] = out
    np.savez(OUT / 'network_baseline.npz', n=ns, **{key: np.array([[res[int(n), s][key] for s in range(args.worlds)] for n in ns]) for key in res[int(ns[0]), 0]})
    for n in ns: print(n, {key: round(float(np.median([res[int(n), s][key] for s in range(args.worlds)])), 4) for key in res[int(n), 0]})
