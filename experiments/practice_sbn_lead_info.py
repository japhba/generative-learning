"""Reading as information, in the deep sigmoid belief net of practice_sbn_lead.py: does the learner's code carry the world's
latent (in any labelling) before its samples are valid one level up, as in the tree Helmholtz machine on the RHM?

Same worlds, learner, wake-sleep training and seeds as practice_sbn_lead.run. Added measure per layer l:
  read_info  I(u_l; h_l) / H(h_l): u_l the learner's sampled bottom-up code (its exact distribution given x, enumerated over
             all configurations of each layer), h_l the world's code of the same sample; the same with h_l shuffled across
             samples is subtracted (plug-in bias). Invariant to any relabelling of the learner's configurations, as the
             Helmholtz machine's I(u; a) / H(a). A random or zero recognizer's sampled code carries nothing.
Kept: read (strict: signed-permutation-matched mean code), write (strict parse of own samples), KL(world || learner).
Usage: python practice_sbn_lead_info.py --conds expand:2-4-8:8,expand:2-4-8:32 --worlds 4"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from practice_sbn_lead import OUT, Learner, configs, logpdf, lsig, make_world, matched, strict_parse


def entropy(p): p = p[p > 0]; return float(-(p * np.log(p)).sum())


def code_info(L, X, H, rng):
    n, out, q = len(X), [], None
    for l, (R, b) in enumerate(zip(L.R, L.b)):
        C = configs(R.shape[0])
        if l == 0: q = np.exp(lsig(2 * (X @ R.T + b)[:, None, :] * C[None]).sum(2))  # (n, C_l) exact q(u_1 | x)
        else: q = q @ np.exp(lsig(2 * (configs(R.shape[1]) @ R.T + b)[:, None, :] * C[None]).sum(2))  # marginalise the layer below
        a = ((H[l] > 0) * 2 ** np.arange(H[l].shape[1])[::-1]).sum(1)  # index in configs() order
        def mi(a):
            vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True); Qa = np.zeros((len(vals), q.shape[1])); np.add.at(Qa, inv, q)
            return entropy(q.mean(0)) - sum(c / n * entropy(Qa[i] / c) for i, c in enumerate(cnt))
        out.append((mi(a) - mi(rng.permutation(a))) / entropy(np.unique(a, return_counts=True)[1] / n))
    return out


def run(task):
    name, K, D, w, args = task
    rng = np.random.default_rng([args["seed"], w, D, *K]); world = make_world(K, D, args["sigma"], args["beta"], rng)
    ht, Xt = world.sample(args["n_eval"], rng); lp_true = logpdf(world, Xt)
    L = Learner(K, D, args["sigma"], np.random.default_rng([args["seed"], w, 1]), args["init_scale"]); r = np.random.default_rng([args["seed"], w, 2])
    evals = sorted(set(np.unique(np.round(np.geomspace(1, args["steps"], args["n_points"])).astype(int)).tolist()) | {0}); curve = []; irng = np.random.default_rng([args["seed"], w, 7])
    for t in range(args["steps"] + 1):
        if t in evals:
            m = L.read(Xt); _, Xo = L.G.sample(args["n_eval"], np.random.default_rng([args["seed"], w, 3, t])); ov, _ = strict_parse(world, Xo)
            curve.append(dict(step=t, kl=float(np.mean(lp_true - logpdf(L.G, Xt))), read=[matched(m[l], ht[l])[1] for l in range(len(K))],
                              read_info=code_info(L, Xt, ht, irng), write=ov))
        if t == args["steps"]: break
        L.wake(world.sample(args["batch"], r)[1], r, args["lr"]); L.sleep(r, args["lr"] * args["sleep_lr_scale"], args["batch"])
    ref_valid, _ = strict_parse(world, Xt)
    return dict(name=name, K=K, D=D, world=w, ref_write=ref_valid, curve=curve)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conds", default="expand:2-4-8:8,expand:2-4-8:32,expand:2-4-8:128"); ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--n_points", type=int, default=24); ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--sleep_lr_scale", type=float, default=10.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--init_scale", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(); a = vars(args); tasks = []
    for c in args.conds.split(","):
        name, ks, d = c.split(":"); K = [int(k) for k in ks.split("-")][::-1]
        tasks += [(f"{name}_D{d}", K, int(d), w, a) for w in range(args.worlds)]
    out = OUT / f"info_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    with ProcessPoolExecutor(args.workers) as ex: res = list(tqdm(ex.map(run, tasks), total=len(tasks), desc="runs"))
    json.dump(dict(args=a, results=res), open(out / "results.json", "w")); print(out)


if __name__ == "__main__": main()
