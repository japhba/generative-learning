"""Exact count for the tokens-only version of cross_coupling_depth3_pilot.py: can the reader find the level-1 symbols of teacher strings from the
company a token pair keeps? Two token pairs at a place get the same class iff they occur with the same partners over the whole population (the
limit of infinitely many teacher strings). Partners: the sibling token pair, or the sibling and the neighbouring pair across the root's cut.
Then the same again with partners replaced by their classes, until nothing changes. Reported per place: number of classes, whether a class ever
mixes two symbols, and what a class is (symbol and the token features at the cuts). Control: the same grammar without the agreements.
Usage: python experiments/cross_coupling_context_classes.py (seconds)"""

import numpy as np

from cross_coupling_depth3_pilot import M, V, World


def classes(columns, partners):   # columns: (strings, places) token pairs; partners[place] = places whose pairs are the company of this place
    label = [np.unique(c, return_inverse=True)[1] for c in columns.T]   # start: every token pair its own class
    while True:
        new = []
        for i, c in enumerate(columns.T):
            company = {}
            for row, pair in zip(np.stack([label[j] for j in partners[i]], 1), c): company.setdefault(pair, set()).add(tuple(row))
            keys = {pair: frozenset(s) for pair, s in company.items()}; ids = {k: n for n, k in enumerate(set(keys.values()))}; new.append(np.array([ids[keys[pair]] for pair in c]))
        if all(len(set(a)) == len(set(b)) for a, b in zip(new, label)): return new
        label = new


def describe(w, trees, label, name):
    a, p3, pL, pR, *ts = trees.T; ys = (pL // V, pL % V, pR // V, pR % V); print(name)
    for i, place in enumerate(("outer left", "inner left", "inner right", "outer right")):
        n = len(set(label[i])); mixed = sum(len(set(ys[i][label[i] == c])) > 1 for c in set(label[i])); sizes = sorted({len(set(ts[i][label[i] == c])) for c in set(label[i])})
        print(f"   {place:12s} {len(set(ts[i]))} token pairs, {len(set(ys[i]))} symbols -> {n} classes of {sizes} token pairs, {mixed} classes mix symbols")


if __name__ == "__main__":
    w = World(np.random.default_rng(0)); cols = w.trees[:, 4:]; sibling, both = {0: [1], 1: [0], 2: [3], 3: [2]}, {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    describe(w, w.trees, classes(cols, sibling), "with agreements, company = sibling pair")
    describe(w, w.trees, classes(cols, both), "with agreements, company = sibling pair and the pair across the root's cut")
    free = np.array([(a, p3, pL, pR, t1, t2, t3, t4) for a in range(V) for p3 in w.R3[a] for pL in w.R2[p3 // V] for pR in w.R2[p3 % V] for t1 in w.R1[pL // V] for t2 in w.R1[pL % V] for t3 in w.R1[pR // V]
                     for t4 in w.R1[pR % V]])
    describe(w, free, classes(free[:, 4:], sibling), f"without agreements ({len(free)} strings), company = sibling pair")
