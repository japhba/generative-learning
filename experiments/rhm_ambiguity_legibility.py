"""When is a teacher string harder to read than an own sample is to check? Exact counts on a small RHM with cross couplings, no learner.
Grammar as in rhm_cross_coupling_gap.py (pairs, agreement feature with rho values between neighbouring nodes that are not siblings), but a
pair of children may be a production of several symbols ("ambiguous": each symbol draws its m pairs independently) or of one only.
Target = uniform over the valid derivations of depth L (all enumerated); a teacher example shows the leaves only. Every rule is known.
  local reader   bottom-up, one symbol per node from the two symbols read below it (the most frequent parent of that pair in the target)
  exact reader   the most probable symbol of every node given the whole string (all derivations with the same leaves)
  own sample     the derivation is known; a join of two valid halves is kept if the features at the cut agree (chance from the counts)
Printed per level: nodes read right, whole subtrees read right (both readers), chance that an own join is kept, strings with one derivation.
Usage: python rhm_ambiguity_legibility.py -> reports/rhm-ambiguity-legibility.png"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rhm_cross_coupling_gap import LMAX, counts, enumerate_trees

L, M, RHO, WORLDS, VS = 3, 4, 2, 8, [8, 12, 16]
FIG = Path(__file__).resolve().parents[1] / "reports/rhm-ambiguity-legibility.png"


def grammar(rng, v, ambiguous):                                                                             # rules[d-1][a, k] = (b, c); phi[j][symbol]
    r = [np.stack([rng.permutation(v * v)[:M] for _ in range(v)]) if ambiguous else rng.permutation(v * v)[:v * M].reshape(v, M) for _ in range(LMAX)]
    return [np.stack([x // v, x % v], -1) for x in r], [rng.permutation(np.arange(v) % RHO) for _ in range(LMAX)]


def subtree(right, d):                                                                                      # right[k]: [N, nodes at level k] -> [N, nodes at level d] with every node below read right as well
    out = right[d].copy()
    for k in range(d - 1, 0, -1): out &= right[k].reshape(len(out), out.shape[1], -1).all(2)
    return out


def legibility(rules, phi, v):
    lev, _ = enumerate_trees(rules, v, M, L); ok = np.ones(len(lev[0]), bool)
    for j in range(L - 1): s = phi[j][lev[L - j]]; ok &= (s[:, 1:-1:2] == s[:, 2::2]).all(1)
    true = {d: lev[L - d][ok] for d in range(L + 1)}; n = ok.sum(); g = np.unique(true[0] @ v ** np.arange(2 ** L), return_inverse=True)[1].ravel(); G = g.max() + 1
    local, exact, read = {}, {}, true[0]
    for d in range(1, L + 1):
        code = lambda x: x[:, 0::2] * v + x[:, 1::2]; cnt = np.zeros((v * v, v)); np.add.at(cnt, (code(true[d - 1]).ravel(), true[d].ravel()), 1)
        read = cnt.argmax(1)[code(read)]; local[d] = read == true[d]; exact[d] = np.zeros_like(local[d])
        for i in range(true[d].shape[1]): c = np.zeros((G, v)); np.add.at(c, (g, true[d][:, i]), 1); exact[d][:, i] = c.argmax(1)[g] == true[d][:, i]
    return np.array([[local[d].mean(), subtree(local, d).mean(), exact[d].mean(), subtree(exact, d).mean()] for d in range(1, L + 1)]), (np.bincount(g)[g] == 1).mean(), n


def main():
    fig, ax = plt.subplots(1, len(VS), figsize=(16, 4.6), constrained_layout=True, sharey=True); d = np.arange(1, L + 1)
    for a, v in zip(ax, VS):
        for ambiguous in (False, True):
            rng = np.random.default_rng(0); worlds = [grammar(rng, v, ambiguous) for _ in range(WORLDS)]; out = [legibility(r, f, v) for r, f in worlds]
            R, uniq, n, keep = np.median([o[0] for o in out], 0), np.median([o[1] for o in out]), np.median([o[2] for o in out]), np.median([counts(r, f, v, M, RHO)[:L, 3] for r, f in worlds], 0)
            print(f"v {v}, m {M}, rho {RHO}, {'ambiguous' if ambiguous else 'unambiguous'} (median of {WORLDS} grammars, {n:.0f} valid derivations, strings with one derivation {uniq:.3f})\n  level | local reader: node, subtree | exact reader: node, subtree | own join kept")
            for i in range(L): print(f"  {i + 1:5d} | {R[i, 0]:.3f}  {R[i, 1]:.3f} | {R[i, 2]:.3f}  {R[i, 3]:.3f} | {keep[i]:.3f}")
            if ambiguous: a.plot(d, R[:, 1], "o-", color="C3", label="teacher string, read bottom-up: subtree read right"); a.plot(d, R[:, 3], "o--", color="C0", label="teacher string, read with the whole string: subtree read right"); a.plot(d, keep, "s-", color="C2", label="own sample: join of two valid halves kept")
            else: a.plot(d, R[:, 1], "o:", color="gray", label="unambiguous grammar: teacher subtree read right")
        a.set(xlabel="level of the node", title=f"v {v}, m {M}: a pair of children has several parents" if v == VS[0] else f"v {v}, m {M}", xticks=d)
    ax[0].set_ylabel("share"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2); fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


if __name__ == "__main__": main()
