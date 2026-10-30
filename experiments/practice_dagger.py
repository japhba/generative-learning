"""Watching vs rederiving: teacher forcing on clean lectures vs practice checked by the learner's own step model.

World: a derivation tree (branching b, depth D). Root = axioms, leaves = targets. Actions: apply one of the b lemmas
(go to a child) or undo the last step (go to the parent). Every executed action is replaced by a uniformly random
available one with probability eps (a slip), both when the student acts and when it practises in its head.
Lectures: clean root-to-target derivations, so they never contain a slip or an undo.
Student: linear softmax policy over its own recognition features, computed from a step model learned from noisy
atomic observations: child feature = "this lemma leads towards the target", back feature = "the target is not below me".
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange


class Tree:
    def __init__(self, b, D):
        self.b, self.D = b, D
        depth, parent, children = [0], [-1], []
        for n in range(sum(b ** d for d in range(D))):  # internal nodes in BFS order
            children.append([len(depth) + i for i in range(b)])
            for _ in range(b): depth.append(depth[n] + 1); parent.append(n)
        self.N = len(depth); self.depth = np.array(depth); self.parent = np.array(parent)
        self.nxt = -np.ones((self.N, b + 1), int)  # actions 0..b-1 children, b = back
        for n, ch in enumerate(children): self.nxt[n, :b] = ch
        self.nxt[1:, b] = self.parent[1:]
        self.valid = self.nxt >= 0
        self.leaves = np.where(self.depth == D)[0]


def learn_model(tree, rng, k=5, rho=0.05):
    """Step model from k noisy atomic observations per valid (node, action): majority vote."""
    nxt = tree.nxt.copy()
    for n, a in zip(*np.where(tree.valid)):
        obs = np.where(rng.random(k) < rho, rng.integers(0, tree.N, k), tree.nxt[n, a])
        vals, counts = np.unique(obs, return_counts=True); nxt[n, a] = vals[counts.argmax()]
    return nxt


def lecture_model(tree, targets):
    """What a student who only watched lectures knows about steps: the child moves shown in lectures, no undo."""
    nxt = -np.ones_like(tree.nxt)
    for t in targets:
        n = t
        while n != 0: p = tree.parent[n]; a = list(tree.nxt[p, :tree.b]).index(n); nxt[p, a] = n; n = p
    return nxt


def features(tree, nxt):
    """F[n, t, a] for leaves t: child a leads towards t (by following the model's child links); back: t not below n."""
    below = np.zeros((tree.N, tree.N), bool)  # below[n, m]: m reachable from n through model child links
    for n in range(tree.N):
        frontier = [n]
        for _ in range(tree.D + 1):
            below[n, frontier] = True
            frontier = [c for m in frontier for c in nxt[m, :tree.b] if c >= 0 and not below[n, c]]
    sub = below[:, tree.leaves]  # (N, L)
    F = np.zeros((tree.N, len(tree.leaves), tree.b + 1))
    for a in range(tree.b):
        ok = nxt[:, a] >= 0; F[ok, :, a] = sub[nxt[ok, a]]
    F[:, :, tree.b] = 1.0 - sub
    return F


def policy(theta, F, valid, eps):
    """Intended policy pi and executed (slip-mixed) policy P for features F (..., A) with action mask valid (..., A)."""
    wc, bb, wb = theta
    logit = np.where(np.arange(F.shape[-1]) < F.shape[-1] - 1, wc * F, bb + wb * F)
    logit = np.where(valid, logit, -np.inf)
    pi = np.exp(logit - logit.max(-1, keepdims=True)); pi /= pi.sum(-1, keepdims=True)
    u = valid / valid.sum(-1, keepdims=True)
    return pi, (1 - eps) * pi + eps * u


def success(theta, F, tree, nxt, eps, T):
    """Exact probability of reaching each leaf target from the root within T steps, executing in transitions nxt."""
    _, P = policy(theta, F, tree.valid[:, None, :], eps)
    L = len(tree.leaves); at = np.zeros((tree.N, L)); at[tree.leaves, np.arange(L)] = 1
    V = at.copy(); safe = np.where(nxt >= 0, nxt, 0); ok = (nxt >= 0)[:, None, :]
    for _ in range(T): V = np.maximum(at, (P * V[safe].transpose(0, 2, 1) * ok).sum(2))
    return float(V[0].mean())


def grads(theta, F, n, t, a, valid):
    """d log pi(a | n, t) / d theta for arrays of contexts (intended policy)."""
    f = F[n, t]; p, _ = policy(theta, f, valid[n], 0.0); b = F.shape[2] - 1
    g = np.zeros((len(n), F.shape[2], 3)); g[:, :b, 0] = f[:, :b]; g[:, b, 1] = 1; g[:, b, 2] = f[:, b]
    return g[np.arange(len(n)), a] - (p[:, :, None] * g).sum(1)


def rollouts(theta, F, tree, nxt, eps, T, targets, rng):
    """Free-running derivations in the student's head (transitions nxt); unknown transitions end the attempt."""
    B = len(targets); node = np.zeros(B, int); alive = np.ones(B, bool); done = np.zeros(B, bool)
    steps = []
    for _ in range(T):
        act = alive & ~done
        if not act.any(): break
        _, p = policy(theta, F[node, targets], tree.valid[node], eps); a = (p.cumsum(1) < rng.random((B, 1))).sum(1).clip(0, tree.b)
        idx = np.where(act)[0]; steps.append((idx, node[idx].copy(), a[idx]))
        new = nxt[node, a]; alive &= ~(act & (new < 0)); node = np.where(act & (new >= 0), new, node)
        done |= alive & (node == tree.leaves[targets])
    return steps, done


def run(method, theta, F, tree, model, eps, T, rng, args, lec_targets, F_lec=None, model_lec=None):
    th = np.array(theta, float); curve = []; L = len(tree.leaves)
    for u in range(args.updates + 1):
        if u % args.every == 0: curve.append((success(th, F, tree, tree.nxt, eps, T), *th))
        if u == args.updates: break
        tg = rng.integers(0, L, args.batch)
        if method == "watch_more":  # teacher forcing on as many fresh lectures as practice rollouts
            th += args.lr * bc_grad(th, F, tree, tg)
        elif method == "practice_rft":  # rederive, keep what your own step model says reached the target, imitate it
            steps, ok = rollouts(th, F, tree, model, eps, T, tg, rng)
            th += args.lr * sum(grads(th, F, n, tg[i], a, tree.valid)[ok[i]].sum(0) for i, n, a in steps) / args.batch
        elif method == "practice_unverified":  # imitate all own derivations, no verdict
            steps, _ = rollouts(th, F, tree, model, eps, T, tg, rng)
            th += args.lr * sum(grads(th, F, n, tg[i], a, tree.valid).sum(0) for i, n, a in steps) / args.batch
        elif method == "practice_lecture_checker":  # step model knows only what lectures showed
            steps, ok = rollouts(th, F_lec, tree, model_lec, eps, T, tg, rng)
            th += args.lr * sum(grads(th, F_lec, n, tg[i], a, tree.valid)[ok[i]].sum(0) for i, n, a in steps) / args.batch
        elif method == "practice_reinforce":  # paired baseline, gradient of the slip-mixed generator
            tt = np.concatenate([tg, tg]); steps, ok = rollouts(th, F, tree, model, eps, T, tt, rng)
            adv = ok.astype(float) - np.concatenate([ok[args.batch:], ok[:args.batch]])
            g = 0
            for i, n, a in steps:
                pi, P = policy(th, F[n, tt[i]], tree.valid[n], eps); k = np.arange(len(n))
                g = g + (adv[i, None] * (1 - eps) * pi[k, a][:, None] / P[k, a][:, None] * grads(th, F, n, tt[i], a, tree.valid)).sum(0)
            th += args.lr * g / len(tt)
        elif method == "sleep_hindsight":  # dream a derivation, recognize what it proves (first leaf reached), imitate it for that result
            steps, _ = rollouts(th, F, tree, model, eps, T, tg, rng)
            first, proved = np.full(args.batch, len(steps)), np.zeros(args.batch, int); leaf_idx = np.full(tree.N, -1); leaf_idx[tree.leaves] = np.arange(L)
            for k, (i, n, a) in enumerate(steps):
                nn = model[n, a]; hit = (first[i] == len(steps)) & (nn >= 0) & (leaf_idx[np.maximum(nn, 0)] >= 0)
                first[i[hit]] = k; proved[i[hit]] = leaf_idx[nn[hit]]
            g = np.zeros(3)
            for k, (i, n, a) in enumerate(steps):
                use = first[i] >= k
                if use.any(): g += grads(th, F, n[use], proved[i[use]], a[use], tree.valid).sum(0)
            th += args.lr * g / args.batch
        elif method == "dagger_self":  # label own states with the own planner's action (child towards target, else undo)
            steps, _ = rollouts(th, F, tree, model, eps, T, tg, rng)
            for i, n, a in steps:
                lab = np.where(F[n, tg[i], :tree.b].max(1) > 0, F[n, tg[i], :tree.b].argmax(1), tree.b)
                keep = tree.valid[n, lab]
                th += args.lr * grads(th, F, n[keep], tg[i][keep], lab[keep], tree.valid).sum(0) / args.batch
        else: raise ValueError(method)
    return curve


def bc_grad(th, F, tree, targets):
    """Teacher forcing on clean lecture derivations to the given leaf targets."""
    n_all, t_all, a_all = [], [], []
    for ti in targets:
        n = tree.leaves[ti]
        while n != 0: p = tree.parent[n]; n_all.append(p); t_all.append(ti); a_all.append(list(tree.nxt[p, :tree.b]).index(n)); n = p
    return grads(th, F, np.array(n_all), np.array(t_all), np.array(a_all), tree.valid).sum(0) / len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=3); ap.add_argument("--D", type=int, default=4); ap.add_argument("--slack", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.15); ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--watch_steps", type=int, default=2000); ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=0.2); ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--methods", default="watch_more,practice_rft,practice_reinforce,practice_unverified,practice_lecture_checker,dagger_self")
    ap.add_argument("--out", default="/var/tmp/jbauer/generative_practice_dagger"); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / f"b{args.b}_D{args.D}_eps{args.eps}_M{args.M}{args.tag}_{stamp}"; out.mkdir(parents=True)
    tree = Tree(args.b, args.D); T = args.D + args.slack; methods = args.methods.split(",")
    res = {m: [] for m in ["watch"] + methods}; ref = []
    for s in trange(args.seeds, desc="seeds"):
        rng = np.random.default_rng(s); model = learn_model(tree, rng); F = features(tree, model)
        lec = rng.integers(0, len(tree.leaves), args.M); model_lec = lecture_model(tree, tree.leaves[lec]); F_lec = features(tree, model_lec)
        th = np.zeros(3); curve = []
        for u in range(args.watch_steps + 1):  # watching: teacher forcing on the M lectures
            if u % args.every == 0: curve.append((success(th, F, tree, tree.nxt, args.eps, T), *th))
            if u < args.watch_steps: th = th + args.lr * bc_grad(th, F, tree, lec)
        best = int(np.argmax([c[0] for c in curve])); th0 = np.array(curve[best][1:])  # best checkpoint, chosen on true success
        res["watch"].append(curve)
        ref.append(dict(planner=success(np.array([30.0, -30.0, 60.0]), F, tree, tree.nxt, args.eps, T), watch_best=curve[best][0], theta0=th0.tolist()))
        for m in tqdm(methods, desc="methods", leave=False):
            res[m].append(run(m, th0, F, tree, model, args.eps, T, np.random.default_rng([s, methods.index(m)]), args, lec, F_lec, model_lec))
    json.dump(dict(args=vars(args), results=res, ref=ref), open(out / "results.json", "w"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5)); color = {m: f"C{i}" for i, m in enumerate(["watch"] + methods)}
    for m in ["watch"] + methods:
        c = np.array(res[m]); xx = (0 if m == "watch" else args.watch_steps) + np.arange(c.shape[1]) * args.every
        axes[0].plot(xx, c[:, :, 0].mean(0), color=color[m], label=m); axes[1].plot(xx, c[:, :, 3].mean(0), color=color[m], label=m)
    axes[0].axhline(np.mean([r["planner"] for r in ref]), color="k", ls=":", label="planner at test time (reference)")
    axes[0].set_xlabel("update"); axes[0].set_ylabel("success, root to target, with slips"); axes[0].legend()
    axes[1].set_xlabel("update"); axes[1].set_ylabel("weight on 'off track' for undo")
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=120)
    summary = {m: float(np.mean([c[-1][0] for c in res[m]])) for m in methods}
    summary.update(watch_best=float(np.mean([r["watch_best"] for r in ref])), planner=float(np.mean([r["planner"] for r in ref])))
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps(summary, indent=1)); print(out / "curves.png")


if __name__ == "__main__": main()
