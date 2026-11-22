"""What/how pilot for the stretch goal (own examples teach generation better than foreign ones of higher quality).
World: cause c (K-way), J views x_j = alpha a_jc + nu n_{j,b_j} + sigma eps (b_j: private background per view, strength nu).
Learner: a fixed repertoire of M strokes per view (K of them draw the content patterns alpha a_jc, the rest are junk);
its products are strokes plus noise, so they never contain the world's background (KL(world || own) is unbounded).
  what: a K-way RPM read-out f_j(c | x_j) = softmax(V_j x_j), learned on world examples only (local rule, no generator).
        Its code for an example is the unnormalised posterior u_c = p(c) prod_j f_j(c | x_j) / F_j(c); W = sum_c u_c.
  how:  a policy pi_j(stroke | c) with one Hebbian rule, d logit_j(c, .) ~ u_c (stroke - pi_j(. | c)), fed by
    watch          world examples: stroke from the exact posterior of the learner's own model, q(m | x_j, c) ~ pi_j(m | c) N(x_j; u_jm, sigma^2)
                   (what a perfectly sleep-trained parser returns); 2 batches per step
    practice       own examples: stroke = the one just drawn, u_c = the RPM's code of the product
    both           one watch batch and one practice batch per step
    practice_norm  practice with the normalised posterior u_c / W (control: the eligibility W removed)
    self           practice read by the learner's own posterior over c instead of the RPM (zero-signal control)
Scores (labels of the RPM matched to the world's causes by the best permutation):
  write = share of strokes that draw the intended cause; valid = all J strokes right;
  fav_rpm = share of (view, cause) whose stroke with the highest RPM read-out is the right one (predicted end point of practice);
  fav_parser = share of (view, cause) whose stroke most often parsed from world patches is the right one (same for watching).
Usage: python what_how_pilot.py -> reports/what-how-pilot.png, /var/tmp/jbauer/generative_practice_whathow/pilot.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from itertools import permutations, product
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

K, J, D, M, B, ALPHA, SIGMA = 8, 4, 32, 16, 4, 1.0, 0.5
NUS, SEEDS, STEPS, N, EVAL = [0.0, 1.0, 2.0, 4.0, 8.0], 8, 4000, 256, 100
ARMS = ["watch", "practice", "both", "practice_norm", "self"]
PERMS = np.array(list(permutations(range(K))))
OUT = Path("/var/tmp/jbauer/generative_practice_whathow"); FIG = Path(__file__).resolve().parents[1] / "reports/what-how-pilot.png"


def softmax(a): e = np.exp(a - a.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def logsumexp(a, axis): m = a.max(axis, keepdims=True); return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)
def unit(rng, *shape): v = rng.standard_normal(shape); return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Adam:
    def __init__(s, lr): s.lr, s.m, s.v, s.t = lr, 0.0, 0.0, 0
    def __call__(s, g): s.t += 1; s.m = 0.9 * s.m + 0.1 * g; s.v = 0.999 * s.v + 0.001 * g * g; return s.lr * (s.m / (1 - 0.9 ** s.t)) / (np.sqrt(s.v / (1 - 0.999 ** s.t)) + 1e-8)


def world_batch(rng, a, n, nu, size):
    c = rng.integers(K, size=size); b = rng.integers(B, size=(size, J))
    return c, ALPHA * a[:, c].transpose(1, 0, 2) + nu * n[np.arange(J), b] + SIGMA * rng.standard_normal((size, J, D))


def rpm_f(V, bias, x): return softmax(np.einsum("njd,jkd->njk", x, V) + bias)                       # N, J, K
def rpm_logu(f, F): return (np.log(f) - np.log(F)).sum(1) - np.log(K)                               # N, K: log of p(c) prod_j f_j / F_j


def rpm_grad(V, bias, x):
    """Local rule = gradient of mean_n log W_n, W_n = sum_c p(c) prod_j f_j(c | x_nj) / F_j(c), F_j = batch mean of f_j."""
    f = rpm_f(V, bias, x); F = f.mean(0); lu = rpm_logu(f, F); q = softmax(lu); r = q.mean(0) / F
    delta = q[:, None] - f * (1 + r - (f * r).sum(-1, keepdims=True))
    return np.einsum("njk,njd->jkd", delta, x) / len(x), delta.mean(0), F, logsumexp(lu, 1).mean()


def check_gradient():
    rng = np.random.default_rng(0); a, n = unit(rng, J, K, D), unit(rng, J, B, D); _, x = world_batch(rng, a, n, 2.0, 300)
    V, bias = 0.3 * rng.standard_normal((J, K, D)), 0.1 * rng.standard_normal((J, K)); g = rpm_grad(V, bias, x)[0]; e = 1e-5; worst = 0
    for idx in [(0, 0, 0), (1, 3, 5), (3, 7, 31), (2, 4, 17)]:
        dV = np.zeros_like(V); dV[idx] = e; worst = max(worst, abs((rpm_grad(V + dV, bias, x)[3] - rpm_grad(V - dV, bias, x)[3]) / (2 * e) - g[idx]))
    print(f"K-way RPM local rule vs finite differences: max |diff| = {worst:.1e}")


def own_loglik(x, u): return (np.einsum("njd,jmd->njm", x, u) - 0.5 * (u ** 2).sum(-1)) / SIGMA ** 2      # N, J, M (up to a constant)


def dream(rng, L, u, size):
    c = rng.integers(K, size=size); p = softmax(L)[:, c].transpose(1, 0, 2)                           # N, J, M
    z = (p.cumsum(-1) > rng.random((size, J, 1))).argmax(-1)
    return u[np.arange(J), z] + SIGMA * rng.standard_normal((size, J, D)), np.eye(M)[z]


def hebb(L, code, s):
    """d logits (J, K, M) for cause code (N, K) and stroke responsibilities s (N, J, K, M) or one-hot strokes (N, J, M)."""
    s = s[:, :, None] if s.ndim == 3 else s
    return (code[:, None, :, None] * (s - softmax(L))).mean(0)


def run(args, lr_rpm=0.02, lr_pol=2.0):
    nu, seed = args; rng = np.random.default_rng(seed); a, n, u, cause = unit(rng, J, K, D), unit(rng, J, B, D), ALPHA * unit(rng, J, M, D), -np.ones((J, M), int)
    for j in range(J): idx = rng.permutation(M)[:K]; u[j, idx] = ALPHA * a[j]; cause[j, idx] = np.arange(K)
    inv = np.argsort(np.where(cause >= 0, cause, K), 1)[:, :K]; jj = np.arange(J)                    # inv[j, c] = stroke that draws cause c in view j
    V, bias, oV, ob = 0.01 * rng.standard_normal((J, K, D)), np.zeros((J, K)), Adam(lr_rpm), Adam(lr_rpm)
    L = {arm: 0.1 * rng.standard_normal((J, K, M)) for arm in ARMS}; log = []; ce, xe = world_batch(rng, a, n, nu, 4000)
    code = lambda x, F: (lambda lu: np.exp(lu - np.log(np.exp(logsumexp(lu, 1)).mean())))(rpm_logu(rpm_f(V, bias, x), F))   # u_c / batch mean of W
    for t in range(STEPS + 1):
        if t % EVAL == 0:
            f = rpm_f(V, bias, xe); F = f.mean(0); lu = rpm_logu(f, F); C = np.zeros((K, K)); np.add.at(C, (lu.argmax(1), ce), 1)
            lab = PERMS[C[np.arange(K), PERMS].sum(1).argmax()]                                      # RPM label k means world cause lab[k]
            co = np.repeat(np.arange(K), 100); xo = u[jj, inv[:, co].T] + SIGMA * rng.standard_normal((len(co), J, D))     # the right product of every cause
            xs = u[None] + SIGMA * rng.standard_normal((40, J, M, D)); fs = softmax(np.einsum("sjmd,jkd->sjmk", xs, V) + bias[None, :, None])
            fav = (np.log(fs) - np.log(F)[None, :, None]).mean(0).argmax(1)                          # J, K: stroke the RPM reads most strongly as label k
            parsed = softmax(own_loglik(xe, u)); favp = np.stack([parsed[ce == c].mean(0).argmax(-1) for c in range(K)], 1)   # J, K (world causes)
            row = {"t": t, "read_world": (lab[lu.argmax(1)] == ce).mean(), "read_own": (lab[rpm_logu(rpm_f(V, bias, xo), F).argmax(1)] == co).mean(),
                   "parse_world": (cause[jj, parsed.argmax(-1)] == ce[:, None]).mean(), "fav_rpm": (fav == inv[jj[:, None], lab[None]]).mean(), "fav_parser": (favp == inv).mean(),
                   "logW_world": np.median(logsumexp(lu, 1)), "logW_own": np.median(logsumexp(rpm_logu(rpm_f(V, bias, dream(rng, L["practice"], u, 2000)[0]), F), 1))}
            for arm in ARMS:
                right = (softmax(L[arm]) * (cause[:, None, :] == lab[None, :, None])).sum(-1)        # J, K
                row[f"write_{arm}"], row[f"valid_{arm}"] = right.mean(), right.prod(0).mean()
            log.append(row)
        _, x = world_batch(rng, a, n, nu, N); gV, gb, F, _ = rpm_grad(V, bias, x); V += oV(gV); bias += ob(gb)   # what: RPM on world examples
        for arm, xs in [("watch", [x, world_batch(rng, a, n, nu, N)[1]]), ("both", [x])]:               # how, from world examples
            for xw in xs: L[arm] += lr_pol * hebb(L[arm], code(xw, F), softmax(np.log(softmax(L[arm]))[None] + own_loglik(xw, u)[:, :, None]))
        for arm in ["practice", "both", "practice_norm", "self"]:                                       # how, from own examples
            xo, s = dream(rng, L[arm], u, N)
            if arm == "self": r = softmax(logsumexp(np.log(softmax(L[arm]))[None] + own_loglik(xo, u)[:, :, None], -1).sum(1))
            elif arm == "practice_norm": r = softmax(rpm_logu(rpm_f(V, bias, xo), F))
            else: r = code(xo, F)
            L[arm] += lr_pol * hebb(L[arm], r, s)
    return {k: np.array([row[k] for row in log]) for k in log[0]}


def main():
    check_gradient(); OUT.mkdir(parents=True, exist_ok=True); jobs = list(product(NUS, range(SEEDS)))
    with Pool(16) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs), desc="worlds"))
    arr = {k: np.array([r[k] for r in res]).reshape(len(NUS), SEEDS, -1) for k in res[0]}; np.savez(OUT / "pilot.npz", nus=NUS, **arr)
    print(f"final values, mean over {SEEDS} worlds (columns: background strength nu)")
    for k in list(arr)[1:]: print(f"  {k:20s}", "  ".join(f"nu {nu:g}: {arr[k][i, :, -1].mean():6.3f}" for i, nu in enumerate(NUS)))
    print("worlds (of", SEEDS, ") in which practice ends above watching, write / valid:", "  ".join(f"nu {nu:g}: {(arr['write_practice'][i, :, -1] > arr['write_watch'][i, :, -1]).sum()} / {(arr['valid_practice'][i, :, -1] > arr['valid_watch'][i, :, -1]).sum()}" for i, nu in enumerate(NUS)))

    col = {"watch": "#eb6834", "practice": "#2a6fdb", "both": "#8a4fd8", "practice_norm": "#59a14f", "self": "0.5"}; t = arr["t"][0, 0]; i4 = NUS.index(4.0)
    name = {"watch": "watch: world examples (2 batches)", "practice": "practice: own examples, RPM code", "both": "both (1 + 1 batch)",
            "practice_norm": "practice, normalised code (no eligibility)", "self": "own examples read by the learner itself"}
    fig, AX = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ax = AX[0]; ax.plot(t[1:], arr["read_world"][i4].mean(0)[1:], color="k", ls="--", label="RPM reads world examples")
    for arm in ARMS:
        ax.plot(t[1:], arr[f"write_{arm}"][i4].mean(0)[1:], color=col[arm], label=name[arm])
        for s in range(SEEDS): ax.plot(t[1:], arr[f"write_{arm}"][i4, s, 1:], color=col[arm], alpha=0.2)
    ax.set_xscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("share of strokes that draw the intended cause"); ax.set_title("(a) background strength 4", loc="left"); ax.legend()
    for ax, key, yl, ttl in [(AX[1], "write", "share of right strokes at the end", "(b) writing against background strength"), (AX[2], "valid", "share of products with all strokes right", "(c) whole products")]:
        for arm in ARMS:
            y = arr[f"{key}_{arm}"][:, :, -1]; ax.plot(NUS, y.mean(1), "o-", color=col[arm], label=name[arm])
            for s in range(SEEDS): ax.plot(NUS, y[:, s], ".", color=col[arm], alpha=0.4)
        if key == "write":
            ax.plot(NUS, arr["fav_rpm"][:, :, -1].mean(1), "^:", color="#2a6fdb", label="RPM's favourite stroke is the right one")
            ax.plot(NUS, arr["fav_parser"][:, :, -1].mean(1), "v:", color="#eb6834", label="parser's most seen stroke is the right one")
            ax.legend()
        ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("background strength nu (content = 1)"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left")
    fig.savefig(FIG, dpi=150); print(FIG)


if __name__ == "__main__":
    main()
