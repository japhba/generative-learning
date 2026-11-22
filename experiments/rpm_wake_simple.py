"""The plain RPM + wake combination of the README stretch goal in a simple model: does it give an own advantage? (user, 10 Nov 2026)
World as in what_how_pilot.py: cause c (K-way), J views x_j = a_jc + nu n_{j,b_j} + sigma eps (private background b_j per view).
Recognition: the K-way RPM read-out, learned on world examples only (the generation-free "alt sleep" step).
Generation: one free direction per cause and view, x~_j = w_jc + sigma xi. One rule, w_jc <- w_jc + lr <code_c (x_j - w_jc)>, fed by
  em        world examples, code = exact posterior of the learner's own model (wake-sleep with perfect sleep, no RPM)
  rpm_wake  world examples, code = RPM posterior                                 (2 batches per step, like every arm)
  rpm_own   own examples, code = unnormalised RPM code u_c / <W> (eligibility);  no world example reaches the generator
  rpm_both  one rpm_wake batch and one rpm_own batch per step ("equal number of samples")
Scores: gap = E_world log p(x) of the class-mean generator (w = a_jc + nu <n_j>, the best one that codes the cause) minus the
  learner's, in nats per example (negative: the learner fits better, by coding backgrounds); write = share of own products whose
  intended cause an ideal reader (true world model) recovers; write_fixed = the same after rescaling what distinguishes the
  causes to the world's strength (removes exaggeration); cos_spec = cosine between the cause-specific parts w_jc - <w_j> and
  a_jc - <a_j>; along = w_jc . a_jc (content amplitude, 1 in the world).
Usage: python rpm_wake_simple.py -> reports/rpm-wake-simple.png, /var/tmp/jbauer/generative_practice_whathow/rpm_wake_simple.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from what_how_pilot import ALPHA, B, D, J, K, N, OUT, PERMS, SIGMA, Adam, logsumexp, rpm_f, rpm_grad, rpm_logu, softmax, unit, world_batch

NUS, SEEDS, STEPS, EVAL, LR = [0.0, 1.0, 2.0, 4.0, 8.0], 8, 3000, 100, 2.0
ARMS = ["em", "rpm_wake", "rpm_own", "rpm_both"]; FIG = Path(__file__).resolve().parents[1] / "reports/rpm-wake-simple.png"


def own_logp(w, x): return -((x[:, :, None] - w[None]) ** 2).sum((1, 3)) / (2 * SIGMA ** 2)             # N, K: log p(x | c) + const
def upd(w, code, x): return w + LR * (np.einsum("nk,njd->jkd", code, x) - code.sum(0)[None, :, None] * w) / len(x)
def match(pred, ce): C = np.zeros((K, K)); np.add.at(C, (pred, ce), 1); return PERMS[C[np.arange(K), PERMS].sum(1).argmax()]


def run(args):
    nu, seed = args; rng = np.random.default_rng(seed); a, n = unit(rng, J, K, D), unit(rng, J, B, D); jj = np.arange(J)
    V, bias, oV, ob = 0.01 * rng.standard_normal((J, K, D)), np.zeros((J, K)), Adam(0.02), Adam(0.02)
    w = {arm: 0.01 * rng.standard_normal((J, K, D)) for arm in ARMS}; log = {arm: [] for arm in ARMS}; ce, xe = world_batch(rng, a, n, nu, 4000)
    ref = ALPHA * a + nu * n.mean(1)[:, None]; ll = lambda w: (logsumexp(own_logp(w, xe), 1) - np.log(K)).mean(); ll_ref = ll(ref)
    wb = lambda: world_batch(rng, a, n, nu, N)[1]
    def dream(w): c = rng.integers(K, size=N); return w[:, c].transpose(1, 0, 2) + SIGMA * rng.standard_normal((N, J, D))
    def ideal(x): return logsumexp(-((x[:, :, None, None] - (ALPHA * a[None, :, :, None] + nu * n[None, :, None])) ** 2).sum(-1) / (2 * SIGMA ** 2), -1).sum(1).argmax(1)   # N: cause read by the true model
    for t in range(STEPS + 1):
        x = wb(); gV, gb, F, _ = rpm_grad(V, bias, x)
        post = lambda x: softmax(rpm_logu(rpm_f(V, bias, x), F)); elig = lambda x: (lambda lu: np.exp(lu - np.log(np.exp(logsumexp(lu, 1)).mean())))(rpm_logu(rpm_f(V, bias, x), F))
        if t % EVAL == 0:
            lab_rpm = match(rpm_logu(rpm_f(V, bias, xe), F).argmax(1), ce); read = (lab_rpm[rpm_logu(rpm_f(V, bias, xe), F).argmax(1)] == ce).mean()
            for arm in ARMS:
                lab = match(own_logp(w[arm], xe).argmax(1), ce) if arm == "em" else lab_rpm; at = a[:, lab]             # J, K, D: content of the cause each label stands for
                co = np.repeat(np.arange(K), 50); noise = SIGMA * rng.standard_normal((len(co), J, D)); xo = w[arm][:, co].transpose(1, 0, 2) + noise
                sp, spt = w[arm] - w[arm].mean(1, keepdims=True), at - at.mean(1, keepdims=True)                           # what distinguishes the causes: learner, world
                wf = w[arm].mean(1, keepdims=True) + sp * np.linalg.norm(spt, axis=-1).mean() / np.linalg.norm(sp, axis=-1).mean()   # same directions, the world's content strength
                log[arm].append({"gap": ll_ref - ll(w[arm]), "write": (ideal(xo) == lab[co]).mean(), "along": (w[arm] * at).sum(-1).mean(),
                                 "write_fixed": (ideal(wf[:, co].transpose(1, 0, 2) + noise) == lab[co]).mean(), "cos_spec": ((sp * spt).sum(-1) / (np.linalg.norm(sp, axis=-1) * np.linalg.norm(spt, axis=-1))).mean(),
                                 "cos": ((w[arm] * at).sum(-1) / np.linalg.norm(w[arm], axis=-1)).mean(), "norm": np.linalg.norm(w[arm], axis=-1).mean(), "read_rpm": read,
                                 "read_own_model": (lab[own_logp(w[arm], xe).argmax(1)] == ce).mean()})
        V += oV(gV); bias += ob(gb)
        for xw in [x, wb()]: w["em"] = upd(w["em"], softmax(own_logp(w["em"], xw)), xw); w["rpm_wake"] = upd(w["rpm_wake"], post(xw), xw)
        for _ in range(2): xo = dream(w["rpm_own"]); w["rpm_own"] = upd(w["rpm_own"], elig(xo), xo)
        w["rpm_both"] = upd(w["rpm_both"], post(x), x); xo = dream(w["rpm_both"]); w["rpm_both"] = upd(w["rpm_both"], elig(xo), xo)
    return {f"{arm}/{k}": np.array([row[k] for row in rows]) for arm, rows in log.items() for k in rows[0]}


def main():
    jobs = list(product(NUS, range(SEEDS)))
    with Pool(20) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs), desc="worlds"))
    arr = {k: np.array([r[k] for r in res]).reshape(len(NUS), SEEDS, -1) for k in res[0]}; np.savez(OUT / "rpm_wake_simple.npz", nus=NUS, **arr)
    end = lambda arm, k: arr[f"{arm}/{k}"][:, :, -1]
    print("RPM reads world examples:", "  ".join(f"nu {nu:g}: {end('rpm_wake', 'read_rpm')[i].mean():.3f}" for i, nu in enumerate(NUS)))
    for k, label in [("gap", "log-likelihood of world examples: class-mean generator minus learner, nats per example"), ("write", "ideal reader recovers the intended cause from own products"),
                     ("write_fixed", "the same with the cause-specific part rescaled to the world's strength"), ("cos_spec", "cosine between the cause-specific parts of w and of a"), ("along", "content amplitude w . a (world: 1)"),
                     ("cos", "cosine(w, a)"), ("norm", "length of w"), ("read_own_model", "learner's own model reads world examples")]:
        print(label)
        for arm in ARMS: print(f"  {arm:9s}", "  ".join(f"nu {nu:g}: {end(arm, k)[i].mean():8.3f}" for i, nu in enumerate(NUS)))
    for k, sign in [("gap", -1), ("write", 1), ("write_fixed", 1), ("cos_spec", 1)]:
        print(f"worlds (of {SEEDS}) where rpm_both ends better than rpm_wake in {k}:", "  ".join(f"nu {nu:g}: {(sign * end('rpm_both', k)[i] > sign * end('rpm_wake', k)[i]).sum()}" for i, nu in enumerate(NUS)),
              "| rpm_wake better than em:", "  ".join(f"nu {nu:g}: {(sign * end('rpm_wake', k)[i] > sign * end('em', k)[i]).sum()}" for i, nu in enumerate(NUS)))
    t = np.arange(0, STEPS + 1, EVAL); i4 = NUS.index(4.0)
    print("updates until write >= 0.9 at nu 4 (median over worlds; inf = never):", "  ".join(f"{arm}: {np.median([t[np.argmax(y >= 0.9)] if (y >= 0.9).any() else np.inf for y in arr[f'{arm}/write'][i4]])}" for arm in ARMS))

    col = {"em": "#eb6834", "rpm_wake": "#b8860b", "rpm_own": "#2a6fdb", "rpm_both": "#8a4fd8"}
    name = {"em": "wake with the learner's own exact posterior (no RPM)", "rpm_wake": "RPM + wake on world examples", "rpm_own": "RPM + own examples only", "rpm_both": "RPM + wake + own examples (1 + 1)"}
    fig, AX = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ax = AX[0]; ax.plot(t[1:], arr["rpm_wake/read_rpm"][i4].mean(0)[1:], "k--", label="RPM reads world examples")
    for arm in ARMS: ax.plot(t[1:], arr[f"{arm}/write"][i4].mean(0)[1:], color=col[arm], label=name[arm])
    ax.set_xscale("log"); ax.set_xlabel("updates"); ax.set_ylabel("ideal reader recovers the intended cause"); ax.set_title("(a) background strength 4", loc="left"); ax.legend()
    for ax, k, yl, ttl in [(AX[1], "write_fixed", "ideal reader recovers the cause, content at world strength", "(b) writing with the exaggeration removed"), (AX[2], "along", "content amplitude w . a", "(c) how strongly the content is drawn (world: 1)")]:
        for arm in ARMS:
            ax.plot(NUS, end(arm, k).mean(1), "o-", color=col[arm], label=name[arm])
            for s in range(SEEDS): ax.plot(NUS, end(arm, k)[:, s], ".", color=col[arm], alpha=0.4)
        if k == "along": ax.axhline(1, color="0.5", ls=":"); ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("background strength nu (content = 1)"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left")
    fig.savefig(FIG, dpi=150); print(FIG)


if __name__ == "__main__":
    main()
