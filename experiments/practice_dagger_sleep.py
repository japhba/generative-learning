"""Wake-sleep's sleep phase in the DAgger toy: which half of practice does the wake-sleep rule supply?

Reading the DAgger toy as a Helmholtz machine: the student's step model is the generator (actions -> where a derivation
ends), the policy is the recognizer (target -> actions). Sleep = dream a derivation, read off what it proves (the first
leaf it reaches) and train the recognizer to produce the dream's actions for that result (hindsight relabeling): exact
labels of the dream's own causes, no verdict, no search. DAgger = label the student's states with its own planner.

2 x 2 design from watching's best checkpoint (as in practice_dagger.py):
  states  own policy with slips (practice) | babbling (uniform actions)
  labels  hindsight = the dream's own actions for what it proved (sleep) | own planner's action for the task (DAgger)
plus watching, extra lectures and the paired comparison as references, and hindsight with a per-state verdict
(*_hindsight_optimal): keep a dream step only if the rest of the dream, from that state, is a shortest path in the step
model to what the dream proved. Logged per update: how many training contexts are
off track (feature f_back = 1) and how often their label is undo.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_dagger import Tree, bc_grad, features, grads, learn_model, rollouts, run as run_ref, success

METHODS = ["watch_more", "practice_reinforce", "own_planner", "own_hindsight", "babble_planner", "babble_hindsight", "own_hindsight_optimal", "babble_hindsight_optimal"]


def model_dist(tree, model):
    """Shortest number of steps from every node to every leaf through the step model's transitions (BFS)."""
    dist = np.full((tree.N, len(tree.leaves)), 10 ** 6)
    for j, leaf in enumerate(tree.leaves):
        d = np.full(tree.N, 10 ** 6); d[leaf] = 0; frontier = {leaf}; pred = [[] for _ in range(tree.N)]
        for n, a in zip(*np.where(model >= 0)): pred[model[n, a]].append(n)
        k = 0
        while frontier:
            k += 1; frontier = {m for x in frontier for m in pred[x] if d[m] > k}
            for m in frontier: d[m] = k
        dist[:, j] = d
    return dist


def contexts(method, th, F, tree, model, eps, T, tg, rng, dist=None):
    """Training contexts (node, target, label action) for one batch, under a state source x label rule."""
    babble = method.startswith("babble")
    steps, _ = rollouts(np.zeros(3) if babble else th, F, tree, model, 1.0 if babble else eps, T, tg, rng)
    if method.endswith("planner"):  # own planner: child towards the task's target, else undo
        n = np.concatenate([s[1] for s in steps]); t = np.concatenate([tg[s[0]] for s in steps])
        lab = np.where(F[n, t, :tree.b].max(1) > 0, F[n, t, :tree.b].argmax(1), tree.b); keep = tree.valid[n, lab]
        return n[keep], t[keep], lab[keep]
    leaf_idx = np.full(tree.N, -1); leaf_idx[tree.leaves] = np.arange(len(tree.leaves))  # hindsight: what the dream proved
    first, proved = np.full(len(tg), len(steps)), np.full(len(tg), -1)
    for k, (i, n, a) in enumerate(steps):
        nn = model[n, a]; hit = (proved[i] < 0) & (nn >= 0); hit[hit] = leaf_idx[nn[hit]] >= 0
        first[i[hit]] = k; proved[i[hit]] = leaf_idx[nn[hit]]
    out = [(n[u], proved[i[u]], a[u]) for k, (i, n, a) in enumerate(steps) for u in [(proved[i] >= 0) & (first[i] >= k)]]
    if method.endswith("optimal"):  # per-state verdict: the rest of the dream is a shortest route to what it proved
        out = [(n[u], proved[i[u]], a[u]) for k, (i, n, a) in enumerate(steps)
               for u in [(proved[i] >= 0) & (first[i] >= k) & (first[i] - k + 1 == dist[n, np.maximum(proved[i], 0)])]]
    return tuple(np.concatenate(x) for x in zip(*out))


def run(task):
    seed, method, args = task
    tree = Tree(args["b"], args["D"]); T = args["D"] + args["slack"]; eps = args["eps"]
    rng = np.random.default_rng(seed); model = learn_model(tree, rng); F = features(tree, model)
    lec = rng.integers(0, len(tree.leaves), args["M"])
    th = np.zeros(3); curve = []
    for u in range(args["watch_steps"] + 1):  # watching, identical to practice_dagger.py
        if u % args["every"] == 0: curve.append((success(th, F, tree, tree.nxt, eps, T), *th))
        if u < args["watch_steps"]: th = th + args["lr"] * bc_grad(th, F, tree, lec)
    best = int(np.argmax([c[0] for c in curve])); th0 = np.array(curve[best][1:])
    ref = dict(watch=curve, watch_best=curve[best][0], planner=success(np.array([30.0, -30.0, 60.0]), F, tree, tree.nxt, eps, T))
    r = np.random.default_rng([seed, METHODS.index(method)])
    if method in ("watch_more", "practice_reinforce"):
        ns = argparse.Namespace(**args); return seed, method, run_ref(method, th0, F, tree, model, eps, T, r, ns, lec), [], ref
    th = th0.copy(); curve, diag = [], []; L = len(tree.leaves); dist = model_dist(tree, model) if method.endswith("optimal") else None
    for u in range(args["updates"] + 1):
        if u % args["every"] == 0: curve.append((success(th, F, tree, tree.nxt, eps, T), *th))
        if u == args["updates"]: break
        n, t, a = contexts(method, th, F, tree, model, eps, T, r.integers(0, L, args["batch"]), r, dist)
        off = F[n, t, tree.b] > 0
        if u % args["every"] == 0: diag.append((len(n) / args["batch"], off.mean(), (a[off] == tree.b).mean() if off.any() else np.nan, (a[~off] == tree.b).mean()))
        th = th + args["lr"] * grads(th, F, n, t, a, tree.valid).sum(0) / args["batch"]
    return seed, method, curve, diag, ref


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", type=int, default=3); ap.add_argument("--D", type=int, default=4); ap.add_argument("--slack", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.15); ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--watch_steps", type=int, default=2000); ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=0.2); ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--methods", default=",".join(METHODS)); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="/var/tmp/jbauer/generative_practice_dagger/sleep"); ap.add_argument("--tag", default="")
    args = vars(ap.parse_args())
    out = Path(args["out"]) / f"b{args['b']}_D{args['D']}_eps{args['eps']}{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    tasks = [(s, m, args) for s in range(args["seeds"]) for m in args["methods"].split(",")]
    res, diag, ref = {}, {}, {}
    with ProcessPoolExecutor(args["workers"]) as ex:
        for s, m, c, d, rf in tqdm(ex.map(run, tasks), total=len(tasks), desc="runs"):
            res.setdefault(m, {})[s] = c; diag.setdefault(m, {})[s] = d; ref[s] = rf
    S = sorted(ref)
    res = {m: [v[s] for s in S] for m, v in res.items()}; diag = {m: [v[s] for s in S] for m, v in diag.items()}
    json.dump(dict(args=args, results=res, diag=diag, ref=[ref[s] for s in S]), open(out / "results.json", "w"))
    summary = {m: dict(success=float(np.mean([c[-1][0] for c in res[m]])), se=float(np.std([c[-1][0] for c in res[m]]) / np.sqrt(len(S))),
                       w_lemma=float(np.mean([c[-1][1] for c in res[m]])), b_undo=float(np.mean([c[-1][2] for c in res[m]])), w_undo=float(np.mean([c[-1][3] for c in res[m]])),
                       **({} if not diag[m][0] else dict(contexts_per_task=float(np.mean([d[-1][0] for d in diag[m]])), off_track_share=float(np.mean([np.mean([x[1] for x in d]) for d in diag[m]])),
                                                       undo_label_off=float(np.nanmean([np.nanmean([x[2] for x in d]) for d in diag[m]])), undo_label_on=float(np.mean([np.mean([x[3] for x in d]) for d in diag[m]])))))
               for m in res}
    summary |= dict(watch_best=float(np.mean([ref[s]["watch_best"] for s in S])), planner=float(np.mean([ref[s]["planner"] for s in S])))
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps(summary, indent=1)); print(out)


if __name__ == "__main__": main()
