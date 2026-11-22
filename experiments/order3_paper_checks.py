"""Exact checks for the order-three world, before any learner is run.

1. Marginals. Every arrangement is enumerated in full and each marginal of order 1 to 5 compared with the uniform one:
   one triple a + b + c = 0 (mod q); three triples on six tokens that overlap in one token each; the same with q = 4;
   a ring of two triples (the arrangement that does tie two positions); and the three-level block (three triples whose
   symbols s = a - b sum to zero).
2. Additive writer. A token that is minus the sum of two independently chosen latent values cannot be the most probable
   output of a softmax whose logits add a term per latent: a linear program for the best margin, for q = 2, 3, 5.
3. Per-check rule on one block of three tokens (weights w[token, value], samples from softmax(w); if the triple passes
   the sampled weights are multiplied by 1 + eta, otherwise by 1 - lam; bounded to [floor, wmax]): the exact expected
   change of the log weights at a diffuse block and at a sharp block (valid or not), and the number of samples until
   the block is sharp on a valid triple, by simulation, over a grid of eta and lam. The same grid for the rule with two
   factors (sampled weights of a passed triple times 1 + eta, every weight times 1 - lam on every sample).

    python order3_paper_checks.py --out /nfs/nhome/live/jbauer/generative-learning/reports/order3-paper-checks.json
"""
import argparse
import itertools
import json

import numpy as np
from scipy.optimize import linprog
from tqdm.auto import tqdm


def marginals(q, n, constraints):
    """constraints: list of {position: coefficient}; returns for each order the largest deviation of a marginal from uniform and the position sets that deviate."""
    x = np.array(list(itertools.product(range(q), repeat=n)))
    ok = np.ones(len(x), bool)
    for c in constraints:
        ok &= sum(coef * x[:, pos] for pos, coef in c.items()) % q == 0
    x = x[ok]
    out = {}
    for order in range(1, min(n, 6)):
        dev, sets = 0., []
        for pos in itertools.combinations(range(n), order):
            counts = np.bincount((x[:, pos] * q ** np.arange(order)).sum(1), minlength=q ** order) / len(x)
            d = float(np.abs(counts - q ** -order).max())
            dev = max(dev, d)
            if d > 1e-12:
                sets.append(pos)
        out[order] = dict(max_deviation=dev, n_sets=len(sets), sets=sets if len(sets) <= 12 else sets[:12])
    return dict(q=q, tokens=n, strings=int(len(x)), by_order=out)


def additive_margin(q):
    """largest margin with which logits u[a, b] + v[a, c] can make a = -(b + c) the argmax for all b, c, with |u|, |v| <= 1."""
    nvar = 2 * q * q + 1                                                # u, v, margin
    rows, rhs = [], []
    for b, c, a in itertools.product(range(q), repeat=3):
        t = (-b - c) % q
        if a == t:
            continue
        row = np.zeros(nvar)                                            # u[a,b] + v[a,c] - u[t,b] - v[t,c] + margin <= 0
        row[a * q + b] += 1; row[q * q + a * q + c] += 1; row[t * q + b] -= 1; row[q * q + t * q + c] -= 1; row[-1] = 1
        rows.append(row); rhs.append(0.)
    cost = np.zeros(nvar); cost[-1] = -1
    res = linprog(cost, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(-1, 1)] * (nvar - 1) + [(None, None)])
    return float(res.x[-1])


def drift(w, q, eta, lam):
    """exact expected change of log w[token, value] per sample, and the pass probability."""
    p = np.exp(w - w.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    d, passed = np.zeros_like(w), 0.
    for a, b in itertools.product(range(q), repeat=2):
        for c in range(q):
            pr = p[0, a] * p[1, b] * p[2, c]
            ok = (a + b + c) % q == 0
            passed += pr * ok
            for tok, val in enumerate((a, b, c)):
                d[tok, val] += pr * (np.log1p(eta) if ok else np.log1p(-lam))
    return d, float(passed)


def time_to_valid(rng, q, eta, lam, start, w0, wmax, floor, trials, horizon, rule='three'):
    """samples until the block writes one valid triple with probability > 0.9; inf if not within the horizon.
    rule 'hebb': sampled weights of a passed check times 1 + eta, then every weight times 1 - lam (the unit fires on every sample)."""
    w = np.full((trials, 3, q), floor)
    if start == 'invalid':
        w[:, 0, 0] = w[:, 1, 0] = w[:, 2, 1] = w0
    t_done = np.full(trials, np.inf)
    idx = np.arange(trials)
    for t in range(1, horizon + 1):
        p = np.exp(w - w.max(2, keepdims=True)); p /= p.sum(2, keepdims=True)
        x = (p.cumsum(2) < rng.random((trials, 3, 1))).sum(2).clip(max=q - 1)
        ok = x.sum(1) % q == 0
        for tok in range(3):
            w[idx, tok, x[:, tok]] = w[idx, tok, x[:, tok]] * np.where(ok, 1 + eta, 1 if rule == 'hebb' else 1 - lam)
        w = np.clip(w * (1 - lam if rule == 'hebb' else 1), floor, wmax)
        top = w.argmax(2)
        sharp = (np.take_along_axis(p, top[..., None], 2)[..., 0].prod(1) > .9) & (top.sum(1) % q == 0)
        t_done = np.where(sharp & np.isinf(t_done), t, t_done)
        if np.isfinite(t_done).all():
            break
    return t_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    tri = lambda *pos: {p: 1 for p in pos}
    out['marginals'] = {
        'one triple, q=3': marginals(3, 3, [tri(0, 1, 2)]),
        'three overlapping triples on six tokens, q=3': marginals(3, 6, [tri(0, 1, 2), tri(2, 3, 4), tri(4, 5, 0)]),
        'three overlapping triples on six tokens, q=4': marginals(4, 6, [tri(0, 1, 2), tri(2, 3, 4), tri(4, 5, 0)]),
        'two overlapping triples on four tokens, q=3': marginals(3, 4, [tri(0, 1, 2), tri(2, 3, 0)]),
        'three-level block (symbols a - b sum to zero), q=3': marginals(3, 9, [tri(0, 1, 2), tri(3, 4, 5), tri(6, 7, 8), {0: 1, 1: -1, 3: 1, 4: -1, 6: 1, 7: -1}]),
        'three-level block, q=4': marginals(4, 9, [tri(0, 1, 2), tri(3, 4, 5), tri(6, 7, 8), {0: 1, 1: -1, 3: 1, 4: -1, 6: 1, 7: -1}]),
    }
    out['additive_writer_best_margin'] = {q: additive_margin(q) for q in (2, 3, 5)}
    q, w0, wmax, floor = 3, 4., 6., .05
    states = dict(diffuse=np.full((3, q), floor), sharp_invalid=np.full((3, q), floor), sharp_valid=np.full((3, q), floor))
    states['sharp_invalid'][[0, 1, 2], [0, 0, 1]] = w0
    states['sharp_valid'][[0, 1, 2], [0, 0, 0]] = w0
    out['drift'] = {}
    for name, w in states.items():
        d, passed = drift(w, q, 1., .2)
        out['drift'][name] = dict(pass_probability=passed, log_weight_drift=d.round(4).tolist())
    grid = []
    for eta, lam in tqdm(list(itertools.product((.1, .3, 1., 3.), (.05, .1, .2, .4)))):
        row = dict(eta=eta, lam=lam, diffuse_drift=float(np.log1p(eta) / q + (1 - 1 / q) * np.log1p(-lam)))
        for start in ('invalid', 'diffuse'):
            t = time_to_valid(rng, q, eta, lam, start, w0, wmax, floor, 512, 20000)
            row[start] = dict(median=float(np.median(t)), share_done=float(np.isfinite(t).mean()))
        grid.append(row)
        print(json.dumps(row), flush=True)
    out['time_to_valid'] = grid
    grid = []
    for eta, lam in tqdm(list(itertools.product((.3, 1., 3.), (.01, .03, .1, .3)))):
        row = dict(eta=eta, lam=lam, diffuse_drift=float(np.log1p(eta) / q ** 2 + np.log1p(-lam)))
        for start in ('invalid', 'diffuse'):
            t = time_to_valid(rng, q, eta, lam, start, w0, wmax, floor, 512, 20000, rule='hebb')
            row[start] = dict(median=float(np.median(t)), share_done=float(np.isfinite(t).mean()))
        grid.append(row)
    out['time_to_valid_hebb'] = grid
    json.dump(out, open(args.out, 'w'), indent=1)
    for name, m in out['marginals'].items():
        print(name, m['strings'], {o: (round(v['max_deviation'], 4), v['n_sets']) for o, v in m['by_order'].items()})
    print(out['additive_writer_best_margin'])
    print(json.dumps(out['drift'], indent=1))


if __name__ == '__main__':
    main()
