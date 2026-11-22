"""Can own examples improve generation when recognition is no better informed than generation? (user's sketch in tex/talk/slides.html, 10 Nov 2026)
Same world and stroke repertoire as what_how_pilot.py, but no RPM: the only reader is the learner's own model,
  p(x) = sum_c p(c) prod_j sum_m pi_j(m | c) N(x_j; u_jm, sigma^2),
read exactly (what a perfectly sleep-trained recogniser returns). Its code for an example is u_c = p(c) prod_j p(x_j | c) / p(x_j),
W = sum_c u_c = p(x) / prod_j p(x_j) (how much better one shared cause explains the views than independent views).
Phase 1 (all arms): plain watching = EM on world examples, d logit_j(c, .) ~ q(c | x) (q(m | x_j, c) - pi_j(. | c)).
Phase 2 (branches from the same state, 2 batches per step each):
  watch       plain watching continued
  watch_elig  world examples with the eligibility: q(c | x) replaced by u_c / <W>
  own_elig    own examples with the eligibility; stroke = the one just drawn
  own_plain   own examples, normalised code (zero-signal control)
  both        one plain watch batch and one own_elig batch
Scores as in what_how_pilot.py (labels matched by the best permutation of the learner's reading of world examples);
mode = share of (view, cause) whose most likely stroke is the right one; blur = mean entropy of pi_j(. | c) in nats.
Usage: python what_how_sharpen.py -> reports/what-how-sharpen.png, /var/tmp/jbauer/generative_practice_whathow/sharpen.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from what_how_pilot import ALPHA, B, D, J, K, M, N, OUT, PERMS, SIGMA, dream, hebb, logsumexp, own_loglik, softmax, unit, world_batch

NUS, SEEDS, T1, T2, EVAL, LR = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0], 8, 4000, 3000, 100, 2.0
ARMS = ["watch", "watch_elig", "own_elig", "own_plain", "both"]
FIG = Path(__file__).resolve().parents[1] / "reports/what-how-sharpen.png"


def read(L, x, u):
    """Own-model reading: log u_c (N, K) and stroke posteriors q(m | x_j, c) (N, J, K, M)."""
    a = np.log(softmax(L))[None] + own_loglik(x, u)[:, :, None]; lpc = logsumexp(a, -1)                # N, J, K: log p(x_j | c) + const
    return (lpc - (logsumexp(lpc, -1) - np.log(K))[..., None]).sum(1) - np.log(K), softmax(a)


def elig(lu): return np.exp(lu - np.log(np.exp(logsumexp(lu, 1)).mean()))                              # u_c / batch mean of W


def score(L, u, cause, ce, xe):
    lu, _ = read(L, xe, u); C = np.zeros((K, K)); np.add.at(C, (lu.argmax(1), ce), 1); lab = PERMS[C[np.arange(K), PERMS].sum(1).argmax()]
    p = softmax(L); ok = cause[:, None, :] == lab[None, :, None]; right = (p * ok).sum(-1)
    return {"read_world": (lab[lu.argmax(1)] == ce).mean(), "write": right.mean(), "valid": right.prod(0).mean(),
            "mode": np.take_along_axis(ok, p.argmax(-1)[..., None], -1).mean(), "blur": -(p * np.log(p + 1e-300)).sum(-1).mean()}


def run(args):
    nu, seed = args; rng = np.random.default_rng(seed); a, n, u, cause = unit(rng, J, K, D), unit(rng, J, B, D), ALPHA * unit(rng, J, M, D), -np.ones((J, M), int)
    for j in range(J): idx = rng.permutation(M)[:K]; u[j, idx] = ALPHA * a[j]; cause[j, idx] = np.arange(K)
    wb = lambda: world_batch(rng, a, n, nu, N)[1]; ce, xe = world_batch(rng, a, n, nu, 4000); L0 = 0.1 * rng.standard_normal((J, K, M)); log = {"phase1": []}

    def step(L, x, mode, strokes=None):
        lu, q = read(L, x, u)
        return L + LR * hebb(L, elig(lu) if mode == "elig" else softmax(lu), q if strokes is None else strokes)

    for t in range(T1 + 1):
        if t % EVAL == 0: log["phase1"].append(score(L0, u, cause, ce, xe))
        for _ in range(2): L0 = step(L0, wb(), "plain")
    L = {arm: L0.copy() for arm in ARMS}; log.update({arm: [] for arm in ARMS})
    for t in range(T2 + 1):
        if t % EVAL == 0:
            for arm in ARMS: log[arm].append(score(L[arm], u, cause, ce, xe))
        for _ in range(2): L["watch"] = step(L["watch"], wb(), "plain"); L["watch_elig"] = step(L["watch_elig"], wb(), "elig")
        for arm, mode in [("own_elig", "elig"), ("own_plain", "plain")]:
            for _ in range(2): xo, s = dream(rng, L[arm], u, N); L[arm] = step(L[arm], xo, mode, s)
        L["both"] = step(L["both"], wb(), "plain"); xo, s = dream(rng, L["both"], u, N); L["both"] = step(L["both"], xo, "elig", s)
    return {f"{arm}/{k}": np.array([row[k] for row in rows]) for arm, rows in log.items() for k in rows[0]}


def main():
    jobs = list(product(NUS, range(SEEDS)))
    with Pool(24) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs), desc="worlds"))
    arr = {k: np.array([r[k] for r in res]).reshape(len(NUS), SEEDS, -1) for k in res[0]}; np.savez(OUT / "sharpen.npz", nus=NUS, **arr)
    end = lambda arm, k: arr[f"{arm}/{k}"][:, :, -1]
    for k in ["read_world", "write", "valid", "mode", "blur"]:
        print(f"{k}: mean over {SEEDS} worlds at the end (columns: background strength nu)")
        for arm in ["phase1"] + ARMS: print(f"  {arm:11s}", "  ".join(f"nu {nu:g}: {end(arm, k)[i].mean():6.3f}" for i, nu in enumerate(NUS)))
    for arm in ["watch_elig", "own_elig", "both"]:
        print(f"worlds (of {SEEDS}) where {arm} ends above plain watching, write / valid:", "  ".join(f"nu {nu:g}: {(end(arm, 'write')[i] > end('watch', 'write')[i]).sum()} / {(end(arm, 'valid')[i] > end('watch', 'valid')[i]).sum()}" for i, nu in enumerate(NUS)))
    print("worlds where own_elig ends above watch_elig, write / valid:", "  ".join(f"nu {nu:g}: {(end('own_elig', 'write')[i] > end('watch_elig', 'write')[i]).sum()} / {(end('own_elig', 'valid')[i] > end('watch_elig', 'valid')[i]).sum()}" for i, nu in enumerate(NUS)))

    col = {"watch": "#eb6834", "watch_elig": "#b8860b", "own_elig": "#2a6fdb", "own_plain": "0.5", "both": "#8a4fd8"}; i2 = NUS.index(2.0)
    name = {"watch": "keep watching (plain)", "watch_elig": "world examples with eligibility", "own_elig": "own examples with eligibility", "own_plain": "own examples, no eligibility", "both": "plain watching + own with eligibility"}
    fig, AX = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True); t1 = np.arange(0, T1 + 1, EVAL); t2 = T1 + np.arange(0, T2 + 1, EVAL)
    ax = AX[0]; ax.plot(t1, arr["phase1/valid"][i2].mean(0), color="k", label="phase 1: plain watching")
    for arm in ARMS:
        ax.plot(t2, arr[f"{arm}/valid"][i2].mean(0), color=col[arm], label=name[arm])
        for s in range(SEEDS): ax.plot(t2, arr[f"{arm}/valid"][i2, s], color=col[arm], alpha=0.2)
    ax.axvline(T1, color="0.6", ls=":"); ax.set_xlabel("updates"); ax.set_ylabel("share of products with all strokes right"); ax.set_title("(a) background strength 2", loc="left"); ax.legend()
    for ax, k, yl, ttl in [(AX[1], "write", "share of right strokes at the end", "(b) strokes"), (AX[2], "valid", "share of products with all strokes right", "(c) whole products")]:
        for arm in ARMS:
            ax.plot(NUS, end(arm, k).mean(1), "o-", color=col[arm], label=name[arm])
            for s in range(SEEDS): ax.plot(NUS, end(arm, k)[:, s], ".", color=col[arm], alpha=0.4)
        if k == "write": ax.plot(NUS, end("phase1", "mode").mean(1), "k^:", label="most likely stroke is right, after phase 1"); ax.legend()
        ax.set_xlabel("background strength nu (content = 1)"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left")
    fig.savefig(FIG, dpi=150); print(FIG)


if __name__ == "__main__":
    main()
