"""Cross couplings in the RHM: what must a writer hold that a reader need not? Exact counts, no learner.
RHM with pairs: depth L, v symbols per level, m productions per symbol (random, unambiguous). Cross coupling: at every level, two
neighbouring nodes that are not siblings must agree in a feature phi_j(symbol) in {0..rho-1} (rho = 1: the plain RHM). Target = uniform
over the trees that obey every production and every coupling.
  reader  bottom-up parse (one symbol per node) + one local agreement check per neighbouring pair: exact, by construction
  writer  writes a node's two subtrees independently given the node's state. For exact samples the state must name the symbol AND the
          features along both edges of its subtree (one per level below): counted here by transfer matrices,
          N_d[a][l, r] = number of valid depth-d subtrees of a with edge features l (left) and r (right);  N_d[a] <- N_(d-1)[b] @ N_(d-1)[c]
Printed per depth d: states per symbol the writer needs (reader: 1), table entries per symbol (reader: m), share of plain RHM samples that
are valid, chance that two valid halves fit when joined under a production, choices a root has to make (production and the features at
the cut between its halves). For L <= LE by enumeration: rank of the count matrix [left half, right half | root] against those choices
(equal unless two productions of a symbol share a child), and a writer with the reader's states fitted to the target by maximum likelihood
(own table per node): share of its samples that are valid, KL(target || writer).
Usage: python rhm_cross_coupling_gap.py -> reports/rhm-cross-coupling-gap.png"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONFIGS = [(16, 4, 1), (16, 4, 2), (16, 4, 3), (32, 8, 2), (32, 8, 4)]; LMAX, LE, WORLDS = 6, 3, 8      # (v, m, rho); enumeration only for the v 16 configs
FIG = Path(__file__).resolve().parents[1] / "reports/rhm-cross-coupling-gap.png"


def grammar(rng, v, m, rho): r = [rng.permutation(v * v)[:v * m].reshape(v, m) for _ in range(LMAX)]; return [np.stack([x // v, x % v], -1) for x in r], [rng.permutation(np.arange(v) % rho) for _ in range(LMAX)]   # rules[d-1][a, k] = (b, c) for a node at level d; phi[j][symbol]


def counts(rules, phi, v, m, rho):                                                                          # per depth: writer states and table entries per symbol, valid share of plain RHM samples, chance that two valid halves fit, choices at a root (production and features at the cut)
    N, rows, tree = [np.ones((1, 1)) for _ in range(v)], [], 1.0                                            # tree: plain RHM subtrees per symbol
    for d in range(1, LMAX + 1):
        new, entries, fit, tot, root, q = [np.zeros((rho ** d, rho ** d)) for _ in range(v)], 0.0, 0.0, 0.0, 0.0, rho ** (d - 1)
        for a in range(v):
            for b, c in rules[d - 1][a]:
                new[a][phi[d - 1][b] * q:(phi[d - 1][b] + 1) * q, phi[d - 1][c] * q:(phi[d - 1][c] + 1) * q] += N[b] @ N[c]
                entries += ((N[b] > 0).astype(float) @ (N[c] > 0)).sum(); fit += (N[b] @ N[c]).sum(); tot += N[b].sum() * N[c].sum(); root += ((N[b].sum(0) > 0) & (N[c].sum(1) > 0)).sum()
        N, tree = new, m * tree ** 2; rows.append([np.mean([np.count_nonzero(x) for x in N]), entries / v, sum(x.sum() for x in N) / (v * tree), fit / tot, root / v])
    return np.array(rows)


def enumerate_trees(rules, v, m, L):                                                                        # every tree that obeys the productions: symbols per level (root first) and the production chosen at every node
    lev, chs = [np.arange(v)[:, None]], []
    for d in range(L, 0, -1):
        n = lev[-1].shape[1]; ch = np.stack(np.meshgrid(*[np.arange(m)] * n, indexing="ij"), -1).reshape(-1, n); k = len(ch)
        lev, chs = [np.repeat(x, k, 0) for x in lev], [np.repeat(x, k, 0) for x in chs]; ch = np.tile(ch, (len(lev[-1]) // k, 1)); chs.append(ch); lev.append(rules[d - 1][lev[-1], ch].reshape(len(ch), 2 * n))
    return lev, chs


def enumerated(rules, phi, v, m, L):                                                                        # rank at the root cut (summed over root symbols), reader-sized writer: valid share and KL
    lev, chs = enumerate_trees(rules, v, m, L); ok = np.ones(len(lev[0]), bool)
    for j in range(L - 1): s = phi[j][lev[L - j]]; ok &= (s[:, 1:-1:2] == s[:, 2::2]).all(1)
    leaves, h = lev[-1], 2 ** (L - 1); li, ri = [np.unique(leaves[:, sl] @ v ** np.arange(h), return_inverse=True)[1].ravel() for sl in (slice(0, h), slice(h, None))]; rank = 0
    for a in range(v): M = np.zeros((li.max() + 1, ri.max() + 1)); s = ok & (lev[0][:, 0] == a); M[li[s], ri[s]] = 1; rank += np.linalg.matrix_rank(M)
    logp = np.log(np.bincount(lev[0][ok, 0], minlength=v) / ok.sum())[lev[0][:, 0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        for par, ch in zip(lev[:-1], chs):
            for i in range(par.shape[1]): cnt = np.zeros((v, m)); np.add.at(cnt, (par[ok, i], ch[ok, i]), 1); logp += np.log(cnt / cnt.sum(1, keepdims=True))[par[:, i], ch[:, i]]
    p = np.exp(np.nan_to_num(logp, nan=-np.inf)); assert abs(p.sum() - 1) < 1e-9
    return rank / v, p[ok].sum(), -np.log(ok.sum()) - logp[ok].mean()


def main():
    res = {}; fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True); d = np.arange(1, LMAX + 1)
    for ci, (v, m, rho) in enumerate(CONFIGS):
        rng = np.random.default_rng(0); worlds = [grammar(rng, v, m, rho) for _ in range(WORLDS)]; C = np.median([counts(r, f, v, m, rho) for r, f in worlds], 0); res[v, m, rho] = C
        print(f"v {v}, m {m}, rho {rho} (median of {WORLDS} grammars)\n  depth | writer states per symbol (reader 1) | writer table entries per symbol (reader {m}) | plain RHM samples valid | two valid halves fit | choices at a root (plain RHM {m})")
        for i, row in enumerate(C): print(f"  {i + 1:5d} | {row[0]:10.1f} | {row[1]:12.1f} | {row[2]:.2e} | {row[3]:.3f} | {row[4]:8.1f}")
        if v == 16:
            for L in range(2, LE + 1): E = np.median([enumerated(r, f, v, m, L) for r, f in worlds], 0); print(f"  enumeration, depth {L}: rank at the root cut per symbol {E[0]:.1f} (counted choices {C[L - 1, 4]:.1f}) | writer with the reader's states: valid {E[1]:.3f}, KL {E[2]:.2f} nats (plain RHM writer valid {C[L - 1, 2]:.3f})")
        kw = dict(color=f"C{ci}", marker="o", label=f"v {v}, m {m}, {'no coupling' if rho == 1 else f'{rho} feature values'}")
        ax[0].plot(d, C[:, 0], **kw); ax[1].plot(d, C[:, 1] / m, **kw); ax[2].plot(d, C[:, 2], **kw)
    ax[0].set(yscale="log", xlabel="depth of the node", ylabel="states per symbol", title="states a one-pass writer needs (reader: 1)"); ax[1].set(yscale="log", xlabel="depth of the node", ylabel="table entries per symbol / m", title="size of the writer's table (reader: 1)")
    ax[2].set(yscale="log", xlabel="depth of the tree", ylabel="share of valid samples", title="writer that knows every production, no coupling")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3); fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


if __name__ == "__main__": main()
