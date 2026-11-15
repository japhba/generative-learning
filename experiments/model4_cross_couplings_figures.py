"""Figures and numbers for tex/model_4_cross_couplings.html (cross couplings in the RHM: why a one-pass writer must hold more than a reader).

  read-write  a small cross-coupled RHM (depth 3): (a) the reader checks a finished string, one lookup per agreement; (b) a top-down writer
              halfway, its children carrying promises for the nodes that will meet across the cut; (c) a writer with the reader's states only
  cut-ising   what has to cross a cut: Ising chain, complete graph, the complete graph after the Hubbard-Stratonovich transform (a star around
              one field), square lattice
  writer-cost v 16, m 4, median of 8 grammars (as experiments/rhm_cross_coupling_gap.py): (a) states per node and symbol a one-pass writer
              needs, rho 2 and 3, against the bound rho^(2l) and the reader's 1; (b) share of valid strings by depth for the exact writer, the
              best writer with the reader's states (one production table per node and symbol, fitted to S exact samples of the target, scored
              exactly by weighted transfer matrices), and the writer that draws productions uniformly (plain RHM)
The schematic tex/figures/model4-cross-couplings.svg comes from experiments/rhm_cross_coupling_schematic.py.
Numbers -> /nfs/gatsbystor/jbauer/generative-learning/model4_cross_couplings/numbers.json (recomputed with --recompute, ~2 min on 1 core).
Usage: MPLBACKEND=Agg python model4_cross_couplings_figures.py [--recompute] -> tex/figures/model4-*.svg (transparent), reports/model4-*.png"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhm_cross_coupling_gap import counts, enumerated, grammar

ROOT = Path(__file__).resolve().parents[1]; OUT = Path('/nfs/gatsbystor/jbauer/generative-learning/model4_cross_couplings'); NUMS = OUT / 'numbers.json'
V, M, RHOS, LMAX, WORLDS, S = 16, 4, (1, 2, 3), 6, 8, 200_000
INK, MUTED, TREE, BOND, CROSS, BAD, GOOD = '#0b0b0b', '#898781', '0.6', '#2a78d6', '#eb6834', '#e34948', '#008300'
FEAT = [('#d4e5f9', INK), ('#2a78d6', 'w')]                                                                        # fill and text colour per feature value (rho 2)
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'


# ---------- exact counts, exact samples, writers ----------
def inside(rules, phi, v, rho, L):                                                                                 # N[d][a, l, r]: valid depth-d subtrees of a with edge features l (left) and r (right), index sum_j f_j rho^j
    N = [np.ones((v, 1, 1))]
    for d in range(1, L + 1):
        q, new = rho ** (d - 1), np.zeros((v, rho ** d, rho ** d))
        for a in range(v):
            for b, c in rules[d - 1][a]: new[a, phi[d - 1][b] * q:(phi[d - 1][b] + 1) * q, phi[d - 1][c] * q:(phi[d - 1][c] + 1) * q] += N[d - 1][b] @ N[d - 1][c]
        N.append(new)
    return N


def exact_samples(rules, phi, v, rho, L, n, rng, chunk=50_000):                                                    # n samples of the target (uniform on valid trees): sym[d] (n, 2^(L-d)), prod[d] (n, 2^(L-d))
    N, R = inside(rules, phi, v, rho, L), rho ** L; parts = []
    for s0 in range(0, n, chunk):
        k0 = min(chunk, n - s0); p = N[L].reshape(-1); idx = rng.choice(p.size, k0, p=p / p.sum()); rows = np.arange(k0)
        state, sym, prod = [(idx // (R * R), (idx // R) % R, idx % R)], {L: (idx // (R * R))[:, None]}, {}
        for d in range(L, 0, -1):
            q, B, C, nxt, ks = rho ** (d - 1), rules[d - 1][..., 0], rules[d - 1][..., 1], [], []
            for a, Lx, Rx in state:                                                                                # node (symbol, left edge, right edge): choose the production and the features F at the cut between its halves
                b, c = B[a], C[a]; ok = (phi[d - 1][b] == (Lx // q)[:, None]) & (phi[d - 1][c] == (Rx // q)[:, None])
                w = (N[d - 1][b, (Lx % q)[:, None]] * N[d - 1][c, :, (Rx % q)[:, None]] * ok[..., None]).reshape(k0, -1)
                cw = np.cumsum(w, 1); j = (cw < (rng.random(k0) * cw[:, -1])[:, None]).sum(1); k, F = j // q, j % q
                nxt += [(b[rows, k], Lx % q, F), (c[rows, k], F, Rx % q)]; ks.append(k)
            sym[d - 1], prod[d], state = np.stack([x[0] for x in nxt], 1), np.stack(ks, 1), nxt
        parts.append((sym, prod))
    return {d: np.concatenate([p[0][d] for p in parts]) for d in parts[0][0]}, {d: np.concatenate([p[1][d] for p in parts]) for d in parts[0][1]}


def valid(sym, phi, L):                                                                                            # every agreement between neighbouring nodes that are not siblings, at every height below the root's children
    ok = np.ones(len(sym[0]), bool)
    for j in range(L - 1): s = phi[j][sym[j]]; ok &= (s[:, 1:-1:2] == s[:, 2::2]).all(1)
    return ok


def fit(sym, prod, v, m, L):                                                                                       # the writer with the reader's states: root table and one production table per node and symbol, maximum likelihood
    q = {}
    for d in range(1, L + 1):
        for i in range(2 ** (L - d)):
            cnt = np.zeros((v, m)); np.add.at(cnt, (sym[d][:, i], prod[d][:, i]), 1); tot = cnt.sum(1, keepdims=True); q[d, i] = np.where(tot > 0, cnt / np.maximum(tot, 1), 1 / m)
    return np.bincount(sym[L][:, 0], minlength=v) / len(sym[L]), q


def writer_valid(rules, phi, v, rho, L, qroot, q):                                                                 # exact probability that a writer drawing the root from qroot and each production from q[d, i][a] writes a valid string
    W = {(0, i): np.ones((v, 1, 1)) for i in range(2 ** L)}
    for d in range(1, L + 1):
        s = rho ** (d - 1)
        for i in range(2 ** (L - d)):
            new = np.zeros((v, rho ** d, rho ** d))
            for a in range(v):
                for k, (b, c) in enumerate(rules[d - 1][a]):
                    if q[d, i][a, k]: new[a, phi[d - 1][b] * s:(phi[d - 1][b] + 1) * s, phi[d - 1][c] * s:(phi[d - 1][c] + 1) * s] += q[d, i][a, k] * (W[d - 1, 2 * i][b] @ W[d - 1, 2 * i + 1][c])
            W[d, i] = new
    return float(qroot @ W[L, 0].sum((1, 2)))


def compute():
    res = {'v': V, 'm': M, 'worlds': WORLDS, 'samples': S}
    for rho in RHOS:
        rng = np.random.default_rng(0); worlds = [grammar(rng, V, M, rho) for _ in range(WORLDS)]; C = np.median([counts(r, f, V, M, rho) for r, f in worlds], 0)
        res[f'rho{rho}'] = {'states': C[:, 0].tolist(), 'entries': C[:, 1].tolist(), 'plain_valid': C[:, 2].tolist(), 'halves_fit': C[:, 3].tolist(), 'root_choices': C[:, 4].tolist()}
        if rho == 2:
            fitted = np.array([[writer_valid(r, f, V, rho, L, *fit(*exact_samples(r, f, V, rho, L, S, np.random.default_rng(100 * w + L)), V, M, L)) for L in range(1, LMAX + 1)] for w, (r, f) in enumerate(tqdm(worlds, desc='fitted writer'))])
            res['rho2']['fitted_valid'] = np.median(fitted, 0).tolist(); res['rho2']['fitted_valid_range'] = [fitted.min(0).tolist(), fitted.max(0).tolist()]
            res['rho2']['enumerated_valid'] = [float(np.median([enumerated(r, f, V, M, L)[1] for r, f in worlds])) for L in (2, 3)]   # exact ML fit by enumeration, check of the sampled fit
    OUT.mkdir(parents=True, exist_ok=True); NUMS.write_text(json.dumps(res, indent=1)); print(NUMS)
    return res


# ---------- figure 1: reading against writing ----------
def example():                                                                                                     # small grammar (v 6, m 2, rho 2, depth 3): one exact sample, and a writer with the reader's states writing the same root and children
    L, v, m, rho = 3, 6, 2, 2; rng = np.random.default_rng(3); rules, phi = grammar(rng, v, m, rho)
    for seed in range(1000):
        sym, prod = exact_samples(rules, phi, v, rho, L, 1, np.random.default_rng(seed), chunk=1); good = {d: sym[d][0] for d in sym}
        if len(set(good[0])) < 4 or phi[0][good[0][3]] == phi[0][good[0][4]] == phi[0][good[0][1]]: continue        # want varied tokens and both feature values
        r2 = np.random.default_rng(seed + 1)
        for _ in range(200):                                                                                       # same root and children; each lower production drawn uniformly (the writer knows every production, carries the symbol only)
            bad = {3: good[3], 2: good[2]}
            for d in (2, 1): bad[d - 1] = np.concatenate([rules[d - 1][a, r2.integers(m)] for a in bad[d]])
            cross = [phi[1][bad[1][1]] != phi[1][bad[1][2]], phi[0][bad[0][3]] != phi[0][bad[0][4]]]; inner = [phi[0][bad[0][1]] == phi[0][bad[0][2]], phi[0][bad[0][5]] == phi[0][bad[0][6]]]
            if all(cross) and all(inner): return phi, good, bad
    raise RuntimeError('no example found')


YS = 1.4                                                                                                           # vertical spacing of the heights in figure 1


def pos(d, i): return (i + 0.5) * 2 ** d - 0.5, YS * d


def draw_tree(ax, phi, sym, L=3, ghost=(), promised=None, bonds=None):
    for d in range(1, L + 1):
        for i in range(2 ** (L - d)):
            for c in (2 * i, 2 * i + 1): ax.plot(*zip(pos(d, i), pos(d - 1, c)), color=TREE, zorder=1, linestyle=':' if (d - 1, c) in ghost else '-')
    for (d, i), (style, mark) in (bonds or {}).items():                                                            # bond between node i and i + 1 at height d
        (x0, y), (x1, _) = pos(d, i), pos(d, i + 1); color, ls = style; ax.plot([x0, x1], [y, y], color=color, linestyle=ls, linewidth=2.5, zorder=2)
        if mark: ax.text((x0 + x1) / 2, y + 0.12, mark, color=GOOD if mark == '✓' else BAD, ha='center', va='bottom', fontweight='bold', bbox=dict(facecolor='w', edgecolor='none', pad=0.3))
    for d in range(L + 1):
        for i in range(2 ** (L - d)):
            x, y = pos(d, i); feat = phi[d][sym[d][i]] if d < L - 1 and (d, i) not in ghost else None
            if (d, i) in ghost:
                f = (promised or {}).get((d, i)); ax.add_patch(Circle((x, y), 0.32, facecolor=FEAT[f][0] if f is not None else 'w', edgecolor=INK if f is not None else MUTED, linestyle='--', zorder=3))
                if f is not None: ax.text(x, y, '?', ha='center', va='center', color=FEAT[f][1], zorder=4)
            else:
                ax.add_patch(Circle((x, y), 0.32, facecolor=FEAT[feat][0] if feat is not None else 'w', edgecolor=INK, zorder=3)); ax.text(x, y, str(sym[d][i] + 1), ha='center', va='center', color=FEAT[feat][1] if feat is not None else INK, zorder=4)
    ax.axvline(3.5, color=INK, linestyle='--', linewidth=1, zorder=0); ax.set(xlim=(-0.6, 7.6), ylim=(-0.5, YS * L + 0.5), aspect='equal'); ax.axis('off')


def fig_read_write():
    phi, good, bad = example(); fig, axes = plt.subplots(3, 2, figsize=(9, 8.6), width_ratios=(1.25, 1), constrained_layout=True)
    mark = lambda s, d, i: '✓' if phi[d][s[d][i]] == phi[d][s[d][i + 1]] else '✗'
    pairs = [(0, 1), (0, 3), (0, 5), (1, 1)]; crossing = {(0, 3), (1, 1)}
    ghost = {(1, i) for i in range(4)} | {(0, i) for i in range(8)}; promised = {(1, 1): phi[1][good[1][1]], (1, 2): phi[1][good[1][2]], (0, 3): phi[0][good[0][3]], (0, 4): phi[0][good[0][4]]}
    draw_tree(axes[0, 0], phi, good, bonds={(d, i): ((CROSS if (d, i) in crossing else BOND, '-'), mark(good, d, i)) for d, i in pairs})
    draw_tree(axes[1, 0], phi, good, ghost=ghost, promised=promised, bonds={(d, i): ((CROSS, '--') if (d, i) in crossing else (MUTED, ':'), None) for d, i in pairs})
    draw_tree(axes[2, 0], phi, bad, bonds={(d, i): ((CROSS if (d, i) in crossing else BOND, '-'), mark(bad, d, i)) for d, i in pairs})
    notes = [('(a) Reading a finished string', 'Parse bottom-up: every node gets its\nsymbol, one state out of $v$. Then check\nevery agreement by one lookup of two\nfeatures, in any order. No check needs\nto know about any other.'),
             ('(b) Writing top-down, halfway', 'The root and its children are written;\nnothing below them exists yet. The two\nhalves will be written independently, but\nthe agreements across the cut join nodes\nthat are not there yet. So their features\nare fixed now and handed down: each child\ncarries its symbol and one promised\nfeature per height below it, on each edge.'),
             ("(c) Writing with the reader's states", 'Same root and children. Every production\nis known, but each node is told only its\nsymbol. An agreement between nodes\nwritten apart then holds by luck: here\nthe two across the cut fail.')]
    for ax, (title, body) in zip(axes[:, 1], notes):
        ax.axis('off'); ax.text(0, 0.95, title, transform=ax.transAxes, va='top', fontweight='bold'); ax.text(0, 0.8, body, transform=ax.transAxes, va='top', linespacing=1.4)
    handles = [Line2D([], [], color=TREE, label='production'), Line2D([], [], color=BOND, linewidth=2.5, label='agreement inside a half'), Line2D([], [], color=CROSS, linewidth=2.5, label='agreement across the cut'),
               Patch(facecolor=FEAT[0][0], edgecolor=INK, label='feature value 1'), Patch(facecolor=FEAT[1][0], edgecolor=INK, label='feature value 2'),
               Patch(facecolor='w', edgecolor=MUTED, linestyle='--', label='not yet written'), Patch(facecolor=FEAT[1][0], edgecolor=INK, linestyle='--', label='?  promised feature')]
    fig.legend(handles=handles, loc='outside lower center', ncols=4, frameon=False, handlelength=1.6)
    save(fig, 'model4-read-write')


# ---------- figure 2: what crosses a cut, Ising ----------
def frame(ax, x0, x1, top):                                                                                        # same box shape in every panel: drawing in the upper 70 %, text below
    w = x1 - x0; ax.set(xlim=(x0, x1), ylim=(top - 1.25 * w, top), aspect='equal'); ax.axis('off')


def note(ax, text): ax.text(0.5, 0.25, text, transform=ax.transAxes, ha='center', va='top', linespacing=1.4)


def fig_cut_ising():
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.2), constrained_layout=True); axes = axes.ravel(); node = dict(s=110, facecolor='w', edgecolor=INK, zorder=3)
    ax = axes[0]                                                                                                   # chain
    for i in range(7): ax.plot([i, i + 1], [0, 0], color=CROSS if i == 3 else TREE, linewidth=2.5 if i == 3 else 1.5, zorder=1)
    ax.scatter(np.arange(8), np.zeros(8), **node); ax.plot([3.5, 3.5], [-2.2, 2.2], color=INK, linestyle='--', linewidth=1); frame(ax, -0.8, 7.8, 3.9)
    ax.set_title('(a) chain', loc='left'); note(ax, 'one bond crosses\n$h$ = the boundary spin\n2 states')
    ang = np.pi / 2 - 2 * np.pi * (np.arange(8) + 0.5) / 8; P = np.c_[np.cos(ang), np.sin(ang)]; left = P[:, 0] < 0   # complete graph on a circle
    ax = axes[1]
    for i in range(8):
        for j in range(i + 1, 8): cr = left[i] != left[j]; ax.plot(*zip(P[i], P[j]), color=CROSS if cr else TREE, linewidth=1.3 if cr else 1, zorder=2 if cr else 1)
    ax.scatter(*P.T, **node); ax.plot([0, 0], [-1.25, 1.25], color=INK, linestyle='--', linewidth=1); frame(ax, -1.5, 1.5, 1.4)
    ax.set_title('(b) complete graph', loc='left'); note(ax, '$N^2/4$ bonds cross, all through\nthe magnetisation $M_L$ of a half\n$h = M_L$: $N/2+1$ states')
    ax = axes[2]                                                                                                   # star after Hubbard-Stratonovich
    for i in range(8): ax.plot(*zip(P[i], (0, 0)), color=CROSS, linewidth=1.5, zorder=1)
    ax.scatter(*P.T, **node); ax.scatter([0], [0], s=420, marker='D', facecolor='#fbd9c9', edgecolor=INK, zorder=4); ax.text(0, 0, r'$\phi$', ha='center', va='center', zorder=5)
    ax.plot([0, 0], [-1.25, 1.25], color=INK, linestyle='--', linewidth=1, zorder=0); frame(ax, -1.5, 1.5, 1.4)
    ax.set_title('(c) complete graph, rewritten', loc='left'); note(ax, 'Hubbard–Stratonovich: every spin\ncouples only to one field $\\phi$, and given\n$\\phi$ the spins are independent: $h = \\phi$')
    ax = axes[3]; n = 6                                                                                            # square lattice
    for i in range(n):
        for j in range(n):
            if i + 1 < n: cr = i == n // 2 - 1; ax.plot([i, i + 1], [j, j], color=CROSS if cr else TREE, linewidth=2.5 if cr else 1.5, zorder=2 if cr else 1)
            if j + 1 < n: ax.plot([i, i], [j, j + 1], color=TREE, linewidth=1.5, zorder=1)
    g = np.array([(i, j) for i in range(n) for j in range(n)]); ax.scatter(*g.T, s=60, facecolor='w', edgecolor=INK, zorder=3); ax.plot([n / 2 - 0.5] * 2, [-0.6, n - 0.4], color=INK, linestyle='--', linewidth=1)
    frame(ax, -1.5, n + 0.5, n + 0.2); ax.set_title(r'(d) square lattice, $N\times N$', loc='left'); note(ax, '$N$ bonds cross, on $N$ different spins\n$h$ = the $N$ boundary spins\n$2^N$ states')
    fig.legend(handles=[Line2D([], [], color=CROSS, linewidth=2.5, label='coupling across the cut'), Line2D([], [], color=TREE, linewidth=1.5, label='other couplings'), Line2D([], [], color=INK, linestyle='--', linewidth=1, label='cut: left half | right half')],
               loc='outside lower center', ncols=3, frameon=False)
    save(fig, 'model4-cut-ising')


# ---------- figure 4: what the writer pays ----------
def fig_writer_cost(res):
    l = np.arange(1, LMAX + 1); fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    r2 = res['rho2']
    a.plot(l, r2['states'], color=C1, marker='o', label='states per symbol (counted)'); a.plot(l, 4.0 ** l, color=C1, linestyle=':', label='bound on the states, $\\rho^{2l}=4^l$')
    a.plot(l, np.array(r2['entries']) / M, color=C2, marker='o', label='table entries per symbol $/\\,m$ (counted)')
    a.plot(l, res['rho1']['states'], color=MUTED, marker='o', label='reader (both measures), and a writer without agreements')
    a.set(yscale='log', xlabel='height $l$ of the node ($2^l$ tokens below it)', ylabel="multiple of the reader's", title='(a) what a writer holds per node ($\\rho=2$)', xticks=l, ylim=(0.5, 1e4)); a.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    lo, hi = r2['fitted_valid_range']
    b.plot(l, np.ones(LMAX), color=MUTED, marker='o', label='writer with promises: symbol + edge features')
    b.fill_between(l, lo, hi, color=C1, alpha=0.15, linewidth=0); b.plot(l, r2['fitted_valid'], color=C1, marker='o', label="best writer with the reader's states (fitted)")
    b.plot(l, r2['plain_valid'], color=C2, marker='o', label='writer that knows the productions, draws them uniformly')
    b.set(yscale='log', xlabel='depth $L$ of the tree ($2^L$ tokens)', ylabel='share of valid strings', title='(b) share of valid strings ($\\rho=2$)', xticks=l, ylim=(1e-19, 3)); b.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    for ax in (a, b): ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'model4-writer-cost')


def save(fig, name):
    svg, png = ROOT / 'tex/figures' / f'{name}.svg', ROOT / 'reports' / f'{name}.png'
    fig.savefig(svg, transparent=True); fig.savefig(png, dpi=150, facecolor='w'); plt.close(fig); print(svg, png, sep='\n')


def main():
    args = argparse.ArgumentParser(description=__doc__); args.add_argument('--recompute', action='store_true'); args = args.parse_args()
    res = compute() if args.recompute or not NUMS.exists() else json.loads(NUMS.read_text())
    for k in ('states', 'plain_valid', 'halves_fit'): print(k, np.round(res['rho2'][k], 4))
    print('fitted writer valid', np.round(res['rho2']['fitted_valid'], 6), 'enumerated depth 2, 3', np.round(res['rho2']['enumerated_valid'], 4))
    fig_read_write(); fig_cut_ising(); fig_writer_cost(res)


if __name__ == '__main__': main()
