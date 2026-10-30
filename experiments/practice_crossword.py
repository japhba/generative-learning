"""Crossword of two random hierarchy models: every row must parse under grammar A, every column under grammar B.
Checking a grid is easy (parse each row and each column); filling one is hard. Does practice (fill a grid, check it with
one's own learned parsers, learn) teach the filler more than watching lecture grids?"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from practice_rhm import RHM, Learner, onehot, sample_cat, softmax


class Crossword:
    """n x n grids (n = 2^L) whose rows are valid strings of grammar A and whose columns are valid strings of grammar B.
    Strings are indexed by their base-v number; a grid is stored as the indices of its n rows."""
    def __init__(self, v, nc, m, L, rng):
        self.v, self.n, self.L = v, 2 ** L, L
        self.A, self.B = RHM(v, nc, m, 2, L, rng), RHM(v, nc, m, 2, L, rng)
        self.S = np.array(list(itertools.product(range(v), repeat=self.n)))
        self.okA, self.okB = self.A.parses(self.S)[:, -1], self.B.parses(self.S)[:, -1]
        rows = np.flatnonzero(self.okA); combos = np.array(list(itertools.product(range(len(rows)), repeat=self.n))); keep = []
        for c in np.array_split(combos, 16):
            g = self.S[rows[c]]; keep.append(c[np.all([self.okB[self.index(g[:, :, j])] for j in range(self.n)], 0)])
        self.grids = rows[np.concatenate(keep)]
        self.key = {tuple(g) for g in self.grids}

    def index(self, x): return x @ self.v ** np.arange(self.n)[::-1]

    def columns(self, rows): return self.index(self.S[rows].transpose(0, 2, 1).reshape(-1, self.n)).reshape(len(rows), self.n)

    def truth(self, rows):
        """Rows all valid; columns all parsing up to height h = 1..L."""
        g = self.S[rows]
        return self.okA[rows].all(1), np.stack([self.B.parses(g[:, :, j]) for j in range(self.n)], 1).all(1)


def parse_score(ln, X):
    """The learner's check of a string: recognise it bottom-up (most likely state at every node), then score every step
    with its own generator, log p(root state) + sum over nodes of log p(children's states | node's state)."""
    codes = [None] * ln.L + [X]
    for k in reversed(range(ln.L)):
        ch = codes[k + 1].reshape(len(X), -1, 2); codes[k] = (sum(ln.R[k][j][ch[..., j]] for j in range(2)) + ln.r[k]).argmax(-1)
    score = np.log(softmax(ln.g0))[codes[0][:, 0]]
    for k in range(ln.L):
        for j in range(2): score = score + np.log(softmax(ln.G[k][j])[codes[k], codes[k + 1].reshape(len(X), -1, 2)[..., j]]).sum(1)
    return score


def learn_checker(strings, cw, args, rng):
    """Row or column checker learned from the lecture strings by wake-sleep (or exact inference); accepts a string if its
    score is no worse than that of the worst lecture string minus a margin. Returns the verdict on every string."""
    K = [args.nc * args.m * args.kappa] + [args.v * args.m * args.kappa] * (cw.L - 1)
    ln = Learner(K, args.v, 2, "delta", rng, args.init)
    for _ in tqdm(range(args.checker_updates), leave=False, desc="checker"):
        ln.wake(strings[rng.integers(len(strings), size=args.batch)], rng, args.checker_lr, "exact" if args.checker == "exact" else False)
        if args.checker == "practice": ln.sleep(args.batch, rng, args.checker_lr)
    score = parse_score(ln, cw.S)
    return score >= score[cw.index(strings)].min() - args.margin


class Filler:
    """Fills a grid row by row: p(row t = x | rows above) is proportional to
    exp(beta * acc(x) + sum_j psi[t][prefix of column j so far, x_j]), over all strings x; acc is the learner's row verdict."""
    def __init__(self, cw, acc, beta):
        self.cw, self.n, self.v, self.bias = cw, cw.n, cw.v, beta * acc
        self.psi = [np.zeros((cw.v ** t, cw.v)) for t in range(cw.n)]
        self.E = [onehot(cw.S[:, j], cw.v) for j in range(cw.n)]  # value of column j in each candidate row

    def prefix(self, rows, t): return (self.cw.S[rows[:, :t]] * (self.v ** np.arange(t)[::-1])[None, :, None]).sum(1)

    def probs(self, rows, t):
        pre = self.prefix(rows, t)
        return softmax(self.bias + sum(self.psi[t][pre[:, j]] @ self.E[j].T for j in range(self.n)))

    def sample(self, B, rng):
        rows = np.zeros((B, self.n), int)
        for t in range(self.n): rows[:, t] = sample_cat(self.probs(rows, t), rng)
        return rows

    def logq(self, rows): return sum(np.log(self.probs(rows, t)[np.arange(len(rows)), rows[:, t]]) for t in range(self.n))

    def update(self, rows, w, lr):
        """Delta rule, each grid weighted by w (w > 0 towards it, w < 0 away): teacher forcing when w = 1."""
        for t in range(self.n):
            pre = self.prefix(rows, t); p = self.probs(rows, t)
            for j in range(self.n):
                err = onehot(self.cw.S[rows[:, t], j], self.v) - p @ self.E[j]
                np.add.at(self.psi[t], pre[:, j], lr * w[:, None] * err / len(rows))


def evaluate(cw, f, lect, accept, rng, n):
    rows = f.sample(n, rng); rows_ok, cols_ok = cw.truth(rows); ok = rows_ok & cols_ok[:, -1]
    seen = {tuple(g) for g in lect}; novel = np.array([tuple(g) not in seen for g in rows])
    lq = f.logq(cw.grids); is_seen = np.array([tuple(g) in seen for g in cw.grids])
    return dict(valid=float(ok.mean()), rows=float(rows_ok.mean()), cols=[float(c) for c in (rows_ok[:, None] & cols_ok).mean(0)],
                novel_valid=float((ok & novel).mean()), believed=float(accept(rows).mean()),
                kl=float(-np.log(len(cw.grids)) - lq.mean()), logq_seen=float(lq[is_seen].mean()), logq_unseen=float(lq[~is_seen].mean()),
                coverage=len({tuple(g) for g in rows[ok]}) / min(n, len(cw.grids)))


METHODS = ["watch", "fresh", "practice", "practice_true", "practice_rft", "practice_is"]


def run(args, seed):
    rng = np.random.default_rng(seed); cw = Crossword(args.v, args.nc, args.m, args.L, rng)
    lect = cw.grids[rng.choice(len(cw.grids), args.N, replace=False)]
    rng = np.random.default_rng(1000 + seed)
    rows_str, cols_str = cw.S[lect].reshape(-1, cw.n), cw.S[lect].transpose(0, 2, 1).reshape(-1, cw.n)
    if args.checker == "true": accA, accB = cw.okA, cw.okB
    elif args.checker == "lexicon": accA, accB = np.isin(cw.index(cw.S), cw.index(rows_str)), np.isin(cw.index(cw.S), cw.index(cols_str))
    else: accA, accB = learn_checker(rows_str, cw, args, rng), learn_checker(cols_str, cw, args, rng)
    checker = dict(rows_tpr=float(accA[cw.okA].mean()), rows_tnr=float(1 - accA[~cw.okA].mean()),
                   cols_tpr=float(accB[cw.okB].mean()), cols_tnr=float(1 - accB[~cw.okB].mean()))
    accept = lambda rows: accA[rows].all(1) & accB[cw.columns(rows)].all(1)
    truth = lambda rows: cw.okA[rows].all(1) & cw.okB[cw.columns(rows)].all(1)
    f = Filler(cw, accA, args.beta); out = dict(n_grids=len(cw.grids), checker=checker, curves={})
    for t in tqdm(range(args.watch_updates), leave=False, desc="watch"): f.update(lect[rng.integers(args.N, size=args.batch)], np.ones(args.batch), args.lr)
    ref = Filler(cw, accA, args.beta)
    for t in tqdm(range(args.watch_updates), leave=False, desc="reference"): ref.update(cw.grids[rng.integers(len(cw.grids), size=args.batch)], np.ones(args.batch), args.lr)
    out["reference"] = evaluate(cw, ref, lect, accept, rng, args.eval_n)
    start = [p.copy() for p in f.psi]
    for method in args.methods.split(","):
        f.psi = [p.copy() for p in start]; rng = np.random.default_rng(2000 + seed); curve = []
        for t in tqdm(range(args.updates + 1), leave=False, desc=method):
            if t % args.every == 0: curve.append(dict(t=t, **evaluate(cw, f, lect, accept, rng, args.eval_n)))
            if method == "watch": f.update(lect[rng.integers(args.N, size=args.batch)], np.ones(args.batch), args.lr)
            elif method == "fresh": f.update(cw.grids[rng.integers(len(cw.grids), size=args.batch)], np.ones(args.batch), args.lr)
            else:
                rows = f.sample(args.batch, rng); ok = (truth if method.endswith("_true") else accept)(rows)
                if method == "practice_rft": w = ok.astype(float)
                elif method in ("practice_is", "practice_is_true"):  # imitate own accepted grids, weighted by 1/q: covers all accepted grids
                    lw = np.where(ok, -f.logq(rows), -np.inf); w = len(rows) * softmax(lw) if ok.any() else np.zeros(len(rows))
                else: R = -args.penalty * ~ok - args.entropy * f.logq(rows); w = (R - R.mean()) / (R.std() + 1e-8)
                f.update(rows, w, args.lr)
        out["curves"][method] = curve
    out["lexicon_grids"] = float(np.mean(accept(cw.grids)))  # share of all valid grids the learner's own checker accepts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=4); ap.add_argument("--nc", type=int, default=4); ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--L", type=int, default=2); ap.add_argument("--N", type=int, default=8, help="stored lecture grids")
    ap.add_argument("--checker", default="lexicon", choices=["lexicon", "practice", "exact", "true"],
                    help="lexicon: accept the rows and columns seen in lectures; practice/exact: a learned grammar; true: the world's grammar"); ap.add_argument("--kappa", type=int, default=2)
    ap.add_argument("--init", type=float, default=0.5); ap.add_argument("--checker_lr", type=float, default=0.5)
    ap.add_argument("--checker_updates", type=int, default=20000); ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=5.0, help="filler's preference for rows its own row checker accepts")
    ap.add_argument("--lr", type=float, default=0.3); ap.add_argument("--penalty", type=float, default=10.0); ap.add_argument("--entropy", type=float, default=1.0)
    ap.add_argument("--watch_updates", type=int, default=2000); ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--every", type=int, default=200); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--eval_n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--methods", default=",".join(METHODS)); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_dir = Path("/var/tmp/jbauer/generative_practice_crossword") / f"{args.tag}{time.strftime('%Y%m%d_%H%M%S')}"; out_dir.mkdir(parents=True)
    results = [run(args, seed) for seed in tqdm(range(args.seeds), desc="seeds")]
    json.dump(dict(args=vars(args), results=results), open(out_dir / "results.json", "w"))
    for r in results: print("grids", r["n_grids"], "checker", {k: round(v, 3) for k, v in r["checker"].items()}, "reference kl", round(r["reference"]["kl"], 2))
    for m in args.methods.split(","):
        last = [r["curves"][m][-1] for r in results]; first = [r["curves"][m][0] for r in results]
        f = lambda k, c: np.mean([x[k] for x in c])
        print(f"{m:14s} valid {f('valid', first):.3f}->{f('valid', last):.3f}  novel valid {f('novel_valid', first):.3f}->{f('novel_valid', last):.3f}  "
              f"kl {f('kl', first):.2f}->{f('kl', last):.2f}  logq unseen {f('logq_unseen', last):.1f}  believed {f('believed', last):.3f}")
    print(out_dir)


if __name__ == "__main__": main()
