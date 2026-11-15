"""Parity tree with Ising agreements at every level: the cross-coupled RHM at v = m = rho = 2 with soft agreements. Exact checks by enumeration.

Tokens x_i = +-1, N = 2^L. Symbols s_{0,i} = x_i, s_{l,i} = s_{l-1,2i} s_{l-1,2i+1}. One check per pair of neighbours of height j that are not
siblings, c_{j,k} = s_{j,2k+1} s_{j,2k+2}, and P(x) ~ exp(sum_j J_j sum_k c_{j,k}).
Claims checked (asserts) for L = 2, 3, 4 and random J:
  1. In prefix products P_t = prod_{u<t} x_u every check is a bond P_a P_b between consecutive odd multiples of 2^j, so the checks are independent
     coins with mean tanh J_j: Z, the entropy and the joint law of the checks.
  2. A top-down writer that hands every node only its symbol is uniform: KL(P || writer) = sum over checks of (J tanh J - log cosh J).
  3. Rank of P(x_left, x_right) across the root's cut: 2^(L-1) (and 2^L given the root's symbol).
  4. A left-to-right writer with the reader's units is exact: token t closes at most one check and sets it by one coin.
  5. A top-down one-pass writer is exact if every node is handed the two end spins of each chain inside its block (2l - 1 spins at height l).
Usage: python experiments/parity_chain_twin.py -> prints the checks, writes reports/parity-chain-twin.png"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]; PNG = ROOT / "reports/parity-chain-twin.png"


def world(L, J):
    N = 2 ** L; X = (1 - 2 * ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1)).astype(np.int64)
    S = [X]
    for _ in range(L): S.append(S[-1][:, 0::2] * S[-1][:, 1::2])
    C = [S[j][:, 1:-1:2] * S[j][:, 2::2] for j in range(L - 1)]
    w = np.exp(sum(J[j] * C[j].sum(1) for j in range(L - 1)))
    return N, X, S, C, w / w.sum(), w.sum()


def info(J): return J * np.tanh(J) - np.log(np.cosh(J))  # log 2 minus the entropy of a coin with mean tanh J, in nats


def bonds(L, j): return 2 ** (L - j - 1) - 1


def left_to_right(L, J, X):
    N = 2 ** L; q = np.ones(len(X))
    for t in range(N):
        p = t + 1; j = (p & -p).bit_length() - 1
        if p == N or p < 3 * 2 ** j: q *= 0.5; continue
        q *= np.exp(J[j] * X[:, p - 2 ** (j + 1):p].prod(1)) / (2 * np.cosh(J[j]))
    return q


def top_down(L, J, X):
    """Probability of every string under the one-pass top-down writer, and the largest number of spins a node of each height is handed.
    On every chain a node draws the two spins next to its cut given the two end spins of the chain inside its block (those it was handed)."""
    N = 2 ** L; P = np.concatenate([np.ones((len(X), 1), dtype=np.int64), X.cumprod(1)], 1); q = np.full(len(X), 0.25); handed = np.zeros((L + 1, N), dtype=int)

    def T(d, u, w, t): return (1 + t ** d * u * w) / 2

    def node(V, t, first, last, h, i):  # V: the chain's spins inside the block of node (h, i); first / last: was that end handed down?
        n = V.shape[1]; handed[h, i] += 1 if n == 1 else first + last
        if n == 1: return 1.0
        a, b = n // 2 - 1, n // 2
        if n == 2: pr = 1.0 if first and last else T(1, V[:, 0], V[:, 1], t) * (1 if first or last else 0.5)
        else:
            pr = T(1, V[:, a], V[:, b], t) * (T(a, V[:, 0], V[:, a], t) if first else 1) * (T(a, V[:, b], V[:, -1], t) if last else 1)
            pr = pr / (T(n - 1, V[:, 0], V[:, -1], t) if first and last else (1 if first or last else 2))
        return pr * node(V[:, :b], t, first, True, h - 1, 2 * i) * node(V[:, b:], t, True, last, h - 1, 2 * i + 1)

    for j in range(L - 1): q = q * node(P[:, 2 ** j:N:2 ** (j + 1)], np.tanh(J[j]), False, False, L, 0)
    return q, handed.max(1)


def check(L, rng):
    J = rng.uniform(0.3, 1.5, L - 1); N, X, S, C, p, Z = world(L, J); t = np.tanh(J); n = [bonds(L, j) for j in range(L - 1)]
    assert np.isclose(Z, 2 ** N * np.prod(np.cosh(J) ** n))
    H = -(p * np.log(p)).sum(); assert np.isclose(H, (L + 1) * np.log(2) + sum(n[j] * (np.log(2) - info(J[j])) for j in range(L - 1)))
    Call = np.concatenate(C, 1); tt = np.repeat(t, n); code = (Call < 0) @ (2 ** np.arange(Call.shape[1]))
    joint = np.bincount(code, p, 2 ** Call.shape[1]); bits = (np.arange(len(joint))[:, None] >> np.arange(Call.shape[1])) & 1
    assert np.allclose(joint, np.prod((1 + (1 - 2 * bits) * tt) / 2, 1))
    for l in range(1, L + 1):  # the production bit (left child's symbol) is a fair coin whatever the node's symbol
        for i in range(S[l].shape[1]):
            for s in (1, -1): m = S[l][:, i] == s; assert np.isclose((p[m] * S[l - 1][m, 2 * i]).sum(), 0)
    kl_uniform = N * np.log(2) - H; assert np.isclose(kl_uniform, sum(n[j] * info(J[j]) for j in range(L - 1)))
    M = p.reshape(2 ** (N // 2), 2 ** (N // 2)); rank = np.linalg.matrix_rank(M, tol=1e-12 * M.max())
    rank_root = np.linalg.matrix_rank((p * (S[L][:, 0] == 1)).reshape(M.shape), tol=1e-12 * M.max()); assert rank == 2 ** (L - 1) and rank_root == 2 ** L
    assert np.allclose(left_to_right(L, J, X), p); q, handed = top_down(L, J, X); assert np.allclose(q, p) and all(handed[h] == 2 * h - 1 for h in range(1, L - 1))
    print(f"L {L}  N {N:2d}  checks {sum(n):2d}  J {np.round(J, 2)}  KL(P || reader-state writer) {kl_uniform / np.log(2):.3f} bits  rank across the root's cut {rank} "
          f"({rank_root} given the root)  left-to-right and top-down writers exact  spins handed to a node, by height 1..L: {handed[1:]}")


def figure(L=4):
    N = 2 ** L; fig, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True); colors = [f"C{j}" for j in range(L - 1)]
    def pos(h, i): return (i + 0.5) * 2 ** h, h
    for ax in axes:
        for h in range(1, L + 1):
            for i in range(2 ** (L - h)):
                for c in (2 * i, 2 * i + 1): ax.plot(*zip(pos(h, i), pos(h - 1, c)), color="0.75", zorder=1)
        ax.scatter(*zip(*[pos(h, i) for h in range(L + 1) for i in range(2 ** (L - h))]), facecolor="w", edgecolor="k", zorder=3)
        ax.axvline(N / 2, color="k", linestyle="--", zorder=0); ax.set(ylabel="height", yticks=range(L + 1), xticks=[]); ax.spines[["top", "right", "bottom"]].set_visible(False)
    for j in range(L - 1):
        for k in range(bonds(L, j)):
            axes[0].plot(*zip(pos(j, 2 * k + 1), pos(j, 2 * k + 2)), color=colors[j], zorder=2)
            axes[1].plot(*zip(pos(j + 1, k), pos(j + 1, k + 1)), color=colors[j], zorder=2)
    axes[0].set_title("reader: products up the tree, checks between non-siblings")
    axes[1].set_title("the same distribution: an Ising chain at every height")
    fig.legend(handles=[Line2D([], [], color="0.75", label="a parent is the product of its two children"), Line2D([], [], color=colors[0], label="check (top) = bond of a chain (bottom), one colour per height"),
                        Line2D([], [], color="k", linestyle="--", label="cut between the two halves of the root")], loc="outside lower center", ncols=1, frameon=False)
    fig.savefig(PNG, dpi=150); print(PNG)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for L in (2, 3, 4): check(L, rng)
    print("closed form, all J = 1: L, tokens, KL of the reader-state writer in bits, share of it left when nodes are handed the chains of the k lowest heights (k = 1, 2, 3)")
    for L in range(2, 11):
        kl = [sum(bonds(L, j) * info(1.0) for j in range(k, L - 1)) / np.log(2) for k in range(4)]
        print(f"  {L:2d} {2 ** L:5d} {kl[0]:8.2f}   " + "  ".join(f"{x / kl[0]:.3f}" for x in kl[1:]))
    figure()
