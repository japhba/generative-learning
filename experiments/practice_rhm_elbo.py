"""Tree Helmholtz machine on the RHM trained jointly on the exact ELBO, as the wake-sleep arms of practice_rhm.py are trained
by wake and sleep. Same world, same generator and recognition model, same initialisation (seeded as in practice_rhm.run),
same evaluation (practice_rhm.evaluate); only the learning rule differs.

Recognition q(z | x) = prod_i q(z_i | z_children(i)), each a softmax of additive child votes (the recognition model of
practice_rhm.Learner), with the leaves fixed to x. Because children's subtrees are independent under q, the node marginals
and every parent-child joint follow bottom-up exactly (sums over the K^s child configurations, s = 2), so
  ELBO = E_q[log p(z_root)] + sum_edges E_q[log p(child | parent)] + sum_i E_q[H(z_i | z_children(i))]
is computed exactly, with no sampling. Plain SGD on the ELBO; each table's step is scaled by 1 / (number of nodes it is
applied at), so the generator's step is exactly the expectation under q of the wake delta rule (lr 0.5); only how
recognition learns differs (ELBO gradient instead of sleep).
Usage: python practice_rhm_elbo.py --L 3 --kappa 2 --updates 60000 --every 250 --tag NAME"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from practice_rhm import RHM, Learner, evaluate


def elbo(P, X, s, L):
    """P: torch parameters g0, G[k] (s, K_k, K_k+1), R[k] (s, K_k+1, K_k), r[k] (K_k); X: (n, s^L) leaves. Per-string ELBO (n,)."""
    assert s == 2
    n = len(X); ls = torch.log_softmax; lc = X.reshape(n, -1, s)
    lq = ls(P["R"][L - 1][0][lc[..., 0]] + P["R"][L - 1][1][lc[..., 1]] + P["r"][L - 1], -1); q = lq.exp()  # (n, nodes, K)
    H = -(q * lq).sum((1, 2)); lG = ls(P["G"][L - 1], -1)
    E = sum((q * lG[j].T[lc[..., j]]).sum((1, 2)) for j in range(s))
    for k in reversed(range(L - 1)):
        qc = q.reshape(n, -1, s, q.shape[-1])
        lP = ls(P["R"][k][0][:, None, :] + P["R"][k][1][None, :, :] + P["r"][k], -1); Pp = lP.exp()  # (Kc, Kc, Kp)
        W = qc[:, :, 0, :, None] * qc[:, :, 1, None, :]  # joint of the two children (n, nodes, Kc, Kc)
        J0, J1 = torch.einsum("nibc,bca->niab", W, Pp), torch.einsum("nibc,bca->niac", W, Pp)  # parent-child joints
        q = J0.sum(-1); H = H - torch.einsum("nibc,bc->n", W, (Pp * lP).sum(-1)); lG = ls(P["G"][k], -1)
        E = E + (J0 * lG[0]).sum((1, 2, 3)) + (J1 * lG[1]).sum((1, 2, 3))
    return E + (q[:, 0] * ls(P["g0"], -1)).sum(-1) + H


def run(args, seed):
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed); world = RHM(args.v, args.nc, args.m, args.s, args.L, rng)
    K = [args.nc * args.m * args.kappa] + [args.v * args.m * args.kappa] * (args.L - 1)
    ln = Learner(K, args.v, args.s, "delta", np.random.default_rng(1000 + seed), args.init); rng = np.random.default_rng(2000 + seed)
    t64 = lambda a: torch.tensor(np.array(a), dtype=torch.float64, requires_grad=True)
    P = dict(g0=t64(ln.g0), G=[t64(G) for G in ln.G], R=[t64(R) for R in ln.R], r=[t64(r) for r in ln.r])
    nodes = [args.s ** k for k in range(args.L)]  # parents at tree level k (0 = root)
    groups = [dict(params=[P["g0"]], lr=args.lr)] + [dict(params=[P["G"][k]], lr=args.lr / nodes[k]) for k in range(args.L)] \
        + [dict(params=[P["R"][k], P["r"][k]], lr=args.lr * args.rec_scale / nodes[k]) for k in range(args.L)]
    opt = torch.optim.SGD(groups); curve = []
    def sync():
        ln.g0 = P["g0"].detach().numpy().copy(); ln.G = [G.detach().numpy().copy() for G in P["G"]]
        ln.R = [R.detach().numpy().copy() for R in P["R"]]; ln.r = [r.detach().numpy().copy() for r in P["r"]]
    for t in tqdm(range(args.updates + 1), leave=False, desc=f"elbo seed {seed}", mininterval=10):
        if t % args.every == 0:
            sync(); rec = evaluate(world, ln, rng, args.eval_n, False)
            Xe = world.sample(args.eval_n, rng)[-1]
            with torch.no_grad(): rec["elbo_gap"] = float(np.mean(ln.loglik(Xe)) - elbo(P, torch.as_tensor(Xe), args.s, args.L).mean().item())
            curve.append(dict(t=t, **rec))
        loss = -elbo(P, torch.as_tensor(world.sample(args.batch, rng)[-1]), args.s, args.L).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=8); ap.add_argument("--nc", type=int, default=8); ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--s", type=int, default=2); ap.add_argument("--L", type=int, default=3); ap.add_argument("--kappa", type=int, default=2)
    ap.add_argument("--init", type=float, default=0.5); ap.add_argument("--lr", type=float, default=0.5); ap.add_argument("--rec_scale", type=float, default=1.0)
    ap.add_argument("--updates", type=int, default=60000); ap.add_argument("--every", type=int, default=250); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_n", type=int, default=2000); ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path("/var/tmp/jbauer/generative_practice_rhm") / f"{args.tag}{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True)
    with ProcessPoolExecutor(args.seeds) as ex: results = {"elbo": list(ex.map(partial(run, args), range(args.seeds)))}
    json.dump(dict(args=vars(args), results=results), open(out / "results.json", "w")); print(out)


if __name__ == "__main__": main()
