"""Does a recogniser that already reads world examples (a wake-aligned head start) speed up learning to generate? (user, 10 Nov 2026)
World and generator as in rpm_wake_simple.py: x_j = a_jc + nu n_{j,b_j} + sigma eps, one free direction w_jc per cause and view,
wake rule w_jc <- w_jc + lr <code_c (x_j - w_jc)> on ONE world batch per update. Only the source of the code differs:
  ws          ordinary wake-sleep: read-out q(c | x) ~ prod_j softmax(U_j x_j), trained by sleep on the learner's own dreams (one batch per update)
  em          the learner's own exact posterior (wake-sleep with perfect sleep)
  rpm_tandem  RPM read-out learned on the same world batches, starting together with the generator
  rpm_head    RPM read-out trained on HEAD world batches before the generator starts (the head start)
Scores: write = share of own products whose intended cause an ideal reader recovers (chance 1 / K); read = the arm's recogniser
on world examples; cos_spec = cosine between the cause-specific parts of w and a.
Usage: python rpm_headstart.py -> reports/rpm-headstart.png, /var/tmp/jbauer/generative_practice_whathow/rpm_headstart.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from rpm_wake_simple import match, own_logp, upd
from what_how_pilot import ALPHA, B, D, J, K, N, OUT, SIGMA, Adam, logsumexp, rpm_f, rpm_grad, rpm_logu, softmax, unit, world_batch

NUS, SEEDS, STEPS, HEAD = [0.0, 1.0, 2.0, 4.0, 8.0], 8, 3000, 1000
T = np.unique(np.r_[0, np.round(np.geomspace(1, STEPS, 45))].astype(int)); ARMS = ["ws", "em", "rpm_tandem", "rpm_head"]
FIG = Path(__file__).resolve().parents[1] / "reports/rpm-headstart.png"


def run(args):
    nu, seed = args; rng = np.random.default_rng(seed); a, n = unit(rng, J, K, D), unit(rng, J, B, D); wb = lambda: world_batch(rng, a, n, nu, N)[1]
    ce, xe = world_batch(rng, a, n, nu, 4000); new = lambda: (0.01 * rng.standard_normal((J, K, D)), np.zeros((J, K)), Adam(0.02), Adam(0.02))
    Vh, bh, o1, o2 = new()
    for _ in range(HEAD): g1, g2, _, _ = rpm_grad(Vh, bh, wb()); Vh += o1(g1); bh += o2(g2)
    V, bias, oV, ob = new(); U, ub, oU, oub = new(); w = {arm: 0.01 * rng.standard_normal((J, K, D)) for arm in ARMS}; log = {arm: [] for arm in ARMS}
    rpm = lambda V, b, x: (lambda f: rpm_logu(f, f.mean(0)))(rpm_f(V, b, x)); amort = lambda x: (np.einsum("njd,jkd->njk", x, U) + ub).sum(1)
    logit = {"ws": amort, "em": lambda x: own_logp(w["em"], x), "rpm_tandem": lambda x: rpm(V, bias, x), "rpm_head": lambda x: rpm(Vh, bh, x)}
    def ideal(x): return logsumexp(-((x[:, :, None, None] - (ALPHA * a[None, :, :, None] + nu * n[None, :, None])) ** 2).sum(-1) / (2 * SIGMA ** 2), -1).sum(1).argmax(1)
    for t in range(STEPS + 1):
        if t in T:
            for arm in ARMS:
                pred = logit[arm](xe).argmax(1); lab = match(pred, ce); at = a[:, lab]; co = np.repeat(np.arange(K), 50)
                sp, spt = w[arm] - w[arm].mean(1, keepdims=True), at - at.mean(1, keepdims=True)
                log[arm].append({"write": (ideal(w[arm][:, co].transpose(1, 0, 2) + SIGMA * rng.standard_normal((len(co), J, D))) == lab[co]).mean(), "read": (lab[pred] == ce).mean(),
                                 "cos_spec": ((sp * spt).sum(-1) / (np.linalg.norm(sp, axis=-1) * np.linalg.norm(spt, axis=-1))).mean()})
        x = wb()
        for arm in ARMS: w[arm] = upd(w[arm], softmax(logit[arm](x)), x)                                   # wake: one world batch, the arm's own code
        g1, g2, _, _ = rpm_grad(V, bias, x); V += oV(g1); bias += ob(g2)                                   # the tandem RPM learns from the same batch
        c = rng.integers(K, size=N); xo = w["ws"][:, c].transpose(1, 0, 2) + SIGMA * rng.standard_normal((N, J, D)); g = np.eye(K)[c] - softmax(amort(xo))
        U += oU(np.einsum("nk,njd->jkd", g, xo) / N); ub += oub(np.broadcast_to(g.mean(0), (J, K)))        # sleep: delta rule on own dreams
    return {f"{arm}/{k}": np.array([row[k] for row in rows]) for arm, rows in log.items() for k in rows[0]}


def main():
    jobs = list(product(NUS, range(SEEDS)))
    with Pool(20) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs), desc="worlds"))
    arr = {k: np.array([r[k] for r in res]).reshape(len(NUS), SEEDS, -1) for k in res[0]}; np.savez(OUT / "rpm_headstart.npz", nus=NUS, t=T, **arr)
    first = lambda y, th: np.array([T[np.argmax(v >= th)] if (v >= th).any() else np.inf for v in y])
    for k in ["write", "read", "cos_spec"]:
        print(f"{k} at the end (mean over {SEEDS} worlds)")
        for arm in ARMS: print(f"  {arm:10s}", "  ".join(f"nu {nu:g}: {arr[f'{arm}/{k}'][i, :, -1].mean():.3f}" for i, nu in enumerate(NUS)))
    for th in [0.5, 0.9]:
        print(f"world batches until write >= {th}: median over worlds (worlds that get there)")
        for arm in ARMS: print(f"  {arm:10s}", "  ".join(f"nu {nu:g}: {np.median(first(arr[f'{arm}/write'][i], th)):6.0f} ({np.isfinite(first(arr[f'{arm}/write'][i], th)).sum()}/{SEEDS})" for i, nu in enumerate(NUS)))
    print("world batches until the recogniser reads >= 0.9:")
    for arm in ARMS: print(f"  {arm:10s}", "  ".join(f"nu {nu:g}: {np.median(first(arr[f'{arm}/read'][i], 0.9)):6.0f} ({np.isfinite(first(arr[f'{arm}/read'][i], 0.9)).sum()}/{SEEDS})" for i, nu in enumerate(NUS)))

    col = {"ws": "#eb6834", "em": "#b8860b", "rpm_tandem": "#8a4fd8", "rpm_head": "#2a6fdb"}
    name = {"ws": "wake-sleep (recogniser trained on own dreams)", "em": "the learner's own exact posterior", "rpm_tandem": "RPM learned alongside", "rpm_head": f"RPM with a head start ({HEAD} batches)"}
    fig, AX = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, nu in zip(AX, [1.0, 2.0, 4.0]):
        i = NUS.index(nu)
        for arm in ARMS:
            ax.plot(T[1:], arr[f"{arm}/write"][i].mean(0)[1:], color=col[arm], label=name[arm]); ax.plot(T[1:], arr[f"{arm}/read"][i].mean(0)[1:], color=col[arm], ls=":")
        ax.set_xscale("log"); ax.set_xlabel("world batches seen by the generator"); ax.set_ylabel("writing (solid), reading world examples (dotted)"); ax.set_title(f"background strength {nu:g}", loc="left")
    AX[0].legend(); fig.savefig(FIG, dpi=150); print(FIG)


if __name__ == "__main__":
    main()
