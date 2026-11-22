"""After own practice, are world examples read better, and does the gain survive when watching resumes? Stroke toy without RPM
(what_how_sharpen.py) and the exact prototype model (prototype_sharpen.py)."""
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys; sys.path.insert(0, "/nfs/nhome/live/jbauer/generative-learning/experiments")
from itertools import product
from multiprocessing import Pool
import numpy as np
import what_how_sharpen as S
import prototype_sharpen as P
from what_how_pilot import ALPHA, B, D, J, K, M, N, dream, hebb, softmax, unit, world_batch

def strokes(args):
    nu, seed = args; rng = np.random.default_rng(seed); a, n, u, cause = unit(rng, J, K, D), unit(rng, J, B, D), ALPHA * unit(rng, J, M, D), -np.ones((J, M), int)
    for j in range(J): idx = rng.permutation(M)[:K]; u[j, idx] = ALPHA * a[j]; cause[j, idx] = np.arange(K)
    wb = lambda: world_batch(rng, a, n, nu, N)[1]; ce, xe = world_batch(rng, a, n, nu, 4000); L = 0.1 * rng.standard_normal((J, K, M))
    def step(L, x, mode, strokes=None):
        lu, q = S.read(L, x, u); return L + S.LR * hebb(L, S.elig(lu) if mode == "elig" else softmax(lu), q if strokes is None else strokes)
    def watch(L, T, mode="plain"):
        for _ in range(2 * T): L = step(L, wb(), mode)
        return L
    def practise(L, T):
        for _ in range(2 * T): xo, s = dream(rng, L, u, N); L = step(L, xo, "elig", s)
        return L
    sc = lambda L: S.score(L, u, cause, ce, xe); out = {}
    L = watch(L, 3000); out["1 watched"] = sc(L); Lp = practise(L.copy(), 1500); out["2 then own practice"] = sc(Lp)
    out["3a then plain watching again"] = sc(watch(Lp.copy(), 1500)); out["3b then world + eligibility"] = sc(watch(Lp.copy(), 1500, "elig"))
    out["(world + eligibility without the practice)"] = sc(watch(L.copy(), 3000, "elig"))
    return out

def proto(args):
    rho, seed = args; rng = np.random.default_rng(seed); s, n = rng.choice([-1.0, 1.0], (P.K, P.D)), rng.choice([-1.0, 1.0], (P.B, P.D))
    Pk = np.stack([np.mean([np.prod((1 - rho) * (P.X == s[k]) + rho * (P.X == n[b]), 1) for b in range(P.B)], 0) for k in range(P.K)], 1) / P.K; Pw = Pk.sum(1)
    clip = lambda m: np.clip(m, -1 + 1e-6, 1 - 1e-6); m = np.clip(0.2 * rng.standard_normal((P.K, P.D)), -0.9, 0.9); out = {}
    for _ in range(400): m = P.step(m, Pw, False, 1.0)
    out["1 watched"] = P.score(m, s, Pk); mp = m.copy()
    for _ in range(200): mp = clip(P.step(mp, P.own(mp), True, 0.5))
    out["2 then own practice"] = P.score(mp, s, Pk); mw = mp.copy()
    for _ in range(400): mw = P.step(mw, Pw, False, 1.0)
    out["3a then plain watching again"] = P.score(mw, s, Pk); out["back at the first fixed point (max |dm|)"] = {"right": np.abs(np.sort(mw.ravel()) - np.sort(m.ravel())).max()}
    return out

if __name__ == "__main__":
    for name, fn, jobs, keys in [("STROKE TOY, no RPM (8 worlds)", strokes, list(product([2.0, 3.0, 4.0], range(8))), ["write", "valid", "read_world"]),
                                 ("PROTOTYPE MODEL, exact (20 worlds)", proto, list(product([0.2, 0.3, 0.4], range(20))), ["right", "valid", "read"])]:
        with Pool(24) as pool: res = pool.map(fn, jobs)
        levels = sorted({j[0] for j in jobs}); print(name)
        for stage in res[0]:
            for k in keys:
                if k in res[0][stage]: print(f"  {stage:44s} {k:10s}", "  ".join(f"{lv:g}: {np.mean([r[stage][k] for r, j in zip(res, jobs) if j[0] == lv]):.3f}" for lv in levels))
        if "read_world" in keys:
            for lv in levels:
                R = [r for r, j in zip(res, jobs) if j[0] == lv]
                print(f"  nu {lv:g}: worlds (of {len(R)}) where reading of world examples rose with own practice: {sum(r['2 then own practice']['read_world'] > r['1 watched']['read_world'] for r in R)};"
                      f" valid after 3a above stage 1: {sum(r['3a then plain watching again']['valid'] > r['1 watched']['valid'] + 0.02 for r in R)}; valid after 3b above world+elig alone: {sum(r['3b then world + eligibility']['valid'] > r['(world + eligibility without the practice)']['valid'] for r in R)}")
