"""Does reinforcing own examples speed up ordinary wake-sleep while its recogniser is still blind to world examples?
(user's sketch, tex/talk/slides.html: the shared feature is learnt quicker from own examples because that is where R starts.)
World and generator as in rpm_headstart.py (free direction w_jc per cause and view), no RPM. Each update uses one world batch
and one batch of own dreams. The recogniser q(c | x) ~ prod_j f_j(c | x_j), f_j = softmax(U_j x_j), learns only from the dreams (sleep).
  ws            wake on the world batch with q; sleep on the dreams
  ws_own        the same, plus the wake rule on the same dreams with the unnormalised code u_c = p(c) prod_j f_j / F_j (eligibility)
  ws_own_plain  the same, plus the wake rule on the dreams with the normalised code q (control)
  em            wake with the learner's own exact posterior (reference, no recogniser to learn)
Scores: read = recogniser on world examples; cos_spec = cosine between the cause-specific parts of w and a; write_fixed = ideal
reader recovers the intended cause from own products with the cause-specific part rescaled to world strength; along = w . a.
Usage: python ws_own_bootstrap.py -> reports/ws-own-bootstrap.png, /var/tmp/jbauer/generative_practice_whathow/ws_own_bootstrap.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from rpm_wake_simple import match, own_logp, upd
from what_how_pilot import ALPHA, B, D, J, K, N, OUT, SIGMA, Adam, logsumexp, softmax, unit, world_batch

NUS, SEEDS, STEPS = [0.0, 0.5, 1.0, 1.5, 2.0], 16, 2000
T = np.unique(np.r_[0, np.round(np.geomspace(1, STEPS, 50))].astype(int)); ARMS = ["ws", "ws_own", "ws_own_plain", "em"]
FIG = Path(__file__).resolve().parents[1] / "reports/ws-own-bootstrap.png"


def run(args):
    nu, seed = args; rng = np.random.default_rng(seed); a, n = unit(rng, J, K, D), unit(rng, J, B, D); ce, xe = world_batch(rng, a, n, nu, 4000)
    w0, U0 = 0.01 * rng.standard_normal((J, K, D)), 0.01 * rng.standard_normal((J, K, D))                 # the same start for every arm
    w = {arm: w0.copy() for arm in ARMS}; R = {arm: [U0.copy(), np.zeros((J, K)), Adam(0.02), Adam(0.02)] for arm in ARMS[:3]}; log = {arm: [] for arm in ARMS}
    views = lambda arm, x: np.einsum("njd,jkd->njk", x, R[arm][0]) + R[arm][1]                             # N, J, K per-view logits
    logit = lambda arm, x: own_logp(w["em"], x) if arm == "em" else views(arm, x).sum(1)
    def elig(arm, x): f = softmax(views(arm, x)); lu = (np.log(f) - np.log(f.mean(0))).sum(1) - np.log(K); return np.exp(lu - np.log(np.exp(logsumexp(lu, 1)).mean()))
    def ideal(x): return logsumexp(-((x[:, :, None, None] - (ALPHA * a[None, :, :, None] + nu * n[None, :, None])) ** 2).sum(-1) / (2 * SIGMA ** 2), -1).sum(1).argmax(1)
    for t in range(STEPS + 1):
        if t in T:
            for arm in ARMS:
                pred = logit(arm, xe).argmax(1); lab = match(pred, ce); at = a[:, lab]; co = np.repeat(np.arange(K), 50); noise = SIGMA * rng.standard_normal((len(co), J, D))
                sp, spt = w[arm] - w[arm].mean(1, keepdims=True), at - at.mean(1, keepdims=True)
                wf = w[arm].mean(1, keepdims=True) + sp * np.linalg.norm(spt, axis=-1).mean() / np.linalg.norm(sp, axis=-1).mean()
                log[arm].append({"read": (lab[pred] == ce).mean(), "cos_spec": ((sp * spt).sum(-1) / (np.linalg.norm(sp, axis=-1) * np.linalg.norm(spt, axis=-1))).mean(),
                                 "write_fixed": (ideal(wf[:, co].transpose(1, 0, 2) + noise) == lab[co]).mean(), "write": (ideal(w[arm][:, co].transpose(1, 0, 2) + noise) == lab[co]).mean(),
                                 "along": (w[arm] * at).sum(-1).mean()})
        x = world_batch(rng, a, n, nu, N)[1]; c = rng.integers(K, size=N); xi = SIGMA * rng.standard_normal((N, J, D))   # same world batch, dream causes and noise for every arm
        for arm in ARMS: w[arm] = upd(w[arm], softmax(logit(arm, x)), x)                                    # wake on the world batch
        for arm in ARMS[:3]:
            xo = w[arm][:, c].transpose(1, 0, 2) + xi; g = np.eye(K)[c] - softmax(views(arm, xo).sum(1)); U, ub, oU, oub = R[arm]
            U += oU(np.einsum("nk,njd->jkd", g, xo) / N); ub += oub(np.broadcast_to(g.mean(0), (J, K)))    # sleep on the dreams
            if arm == "ws_own": w[arm] = upd(w[arm], elig(arm, xo), xo)                                     # reinforce own examples, weighted by how well they are recognised
            if arm == "ws_own_plain": w[arm] = upd(w[arm], softmax(views(arm, xo).sum(1)), xo)
    return {f"{arm}/{k}": np.array([row[k] for row in rows]) for arm, rows in log.items() for k in rows[0]}


def main():
    jobs = list(product(NUS, range(SEEDS)))
    with Pool(24) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs), desc="worlds"))
    arr = {k: np.array([r[k] for r in res]).reshape(len(NUS), SEEDS, -1) for k in res[0]}; np.savez(OUT / "ws_own_bootstrap.npz", nus=NUS, t=T, **arr)
    first = lambda y, th: np.array([T[np.argmax(v >= th)] if (v >= th).any() else np.inf for v in y])
    for k in ["read", "cos_spec", "write_fixed", "along"]:
        print(f"{k} at the end (mean over {SEEDS} worlds)")
        for arm in ARMS: print(f"  {arm:13s}", "  ".join(f"nu {nu:g}: {arr[f'{arm}/{k}'][i, :, -1].mean():.3f}" for i, nu in enumerate(NUS)))
    for k, th in [("read", 0.5), ("read", 0.9), ("cos_spec", 0.5), ("cos_spec", 0.9), ("write_fixed", 0.9)]:
        print(f"updates until {k} >= {th}: median over worlds (worlds that get there)")
        for arm in ARMS: print(f"  {arm:13s}", "  ".join(f"nu {nu:g}: {np.median(first(arr[f'{arm}/{k}'][i], th)):6.0f} ({np.isfinite(first(arr[f'{arm}/{k}'][i], th)).sum():2d}/{SEEDS})" for i, nu in enumerate(NUS)))
        print("   worlds where ws_own gets there before ws / after ws:", "  ".join(f"nu {nu:g}: {(first(arr[f'ws_own/{k}'][i], th) < first(arr[f'ws/{k}'][i], th)).sum()} / {(first(arr[f'ws_own/{k}'][i], th) > first(arr[f'ws/{k}'][i], th)).sum()}" for i, nu in enumerate(NUS)))

    col = {"ws": "#eb6834", "ws_own": "#2a6fdb", "ws_own_plain": "0.5", "em": "#b8860b"}
    name = {"ws": "wake-sleep", "ws_own": "wake-sleep + own examples with eligibility", "ws_own_plain": "wake-sleep + own examples, normalised code", "em": "own exact posterior (reference)"}
    fig, AX = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, nu in zip(AX, [0.0, 1.0, 1.5]):
        i = NUS.index(nu)
        for arm in ARMS:
            ax.plot(T[1:], arr[f"{arm}/cos_spec"][i].mean(0)[1:], color=col[arm], label=name[arm]); ax.plot(T[1:], arr[f"{arm}/read"][i].mean(0)[1:], color=col[arm], ls=":")
        ax.set_xscale("log"); ax.set_xlabel("updates (one world batch + one batch of own dreams)"); ax.set_ylabel("direction of the generator (solid), reading world examples (dotted)"); ax.set_title(f"background strength {nu:g}", loc="left")
    AX[0].legend(); fig.savefig(FIG, dpi=150); print(FIG)


if __name__ == "__main__":
    main()
