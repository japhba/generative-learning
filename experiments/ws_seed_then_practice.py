"""Is the learner seeded by watching, so that practice alone finishes the job? Two-level word model of ws_two_level_chimera.py
(cause -> S symbols -> patches of L tokens; dictionary = shallow level, top table = deep level), blank start.
Phase 1: plain wake-sleep for T0 world batches (one own batch per world batch). Then the teacher is cut off (no stored examples either):
  teacher   reference: wake-sleep goes on with world batches
  plain     own batch read by the reader, delta rule on both tables with the read codes (wake on own samples) + a sleep batch
  judged    own batch, delta rule with the codes it was written from, each weighted by the probability the reader gives that code
            (symbol from its patch for the dictionary, cause from the symbols for the top table) + a sleep batch
Scores as in ws_two_level_chimera.py: surface = share of own patches that are valid, deep = share of valid combinations among samples
whose patches are all valid, kl = KL(teacher || student).
Usage: python ws_seed_then_practice.py [report] -> reports/ws-seed-then-practice.png, /var/tmp/jbauer/generative_practice_reading/ws_seed_then_practice.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import ws_two_level_chimera as m
from ws_reading_gain import B, K, LR, SEEDS, V, draw, sm

T0S, AFTER, ARMS, RUNS, KEYS = [50, 100, 200, 300, 500, 1000], 2000, ["teacher", "plain", "judged"], 2, ["kl", "surface", "deep", "read2"]
TA = np.unique(np.r_[0, np.round(np.geomspace(1, AFTER, 30))].astype(int)); N, SR, LR_ = m.N, m.SR, m.LR_
OUT = Path("/var/tmp/jbauer/generative_practice_reading/ws_seed_then_practice.npz"); FIG = Path(__file__).resolve().parents[1] / "reports/ws-seed-then-practice.png"


def practise(rng, A, D, R1, b1, R2, b2, judged):
    c, mm, x = m.dream(rng, A, D, B)
    if judged: w1 = np.take_along_axis(sm(m.read1(R1, b1, x)), mm[..., None], 3)[..., 0]; w2 = np.take_along_axis(sm(m.read2(R2, b2, mm)), c[..., None], 2)[..., 0]; kh, mh = c, mm   # w1 [n, S, b], w2 [n, b]
    else: mh = draw(rng, m.read1(R1, b1, x)); kh = draw(rng, m.read2(R2, b2, mh)); w1, w2 = np.ones(mh.shape), np.ones(kh.shape)
    om, ok = np.eye(m.M)[mh] * w1[..., None], np.eye(K)[kh] * w2[..., None]
    D += LR * (np.einsum("njbm,njlbv->nlmv", om, np.eye(V)[x]) - om.sum((1, 2))[:, None, :, None] * sm(D)) / (m.S * B)
    A += LR * (np.einsum("nbk,njbm->njkm", ok, np.eye(m.M)[mh]) - ok.sum(1)[:, None, :, None] * sm(A)) / B


def run(job):
    t0, arm, seed = job; rng = np.random.default_rng(seed); tau, sigma = np.argsort(rng.random((SEEDS, m.S, K)), -1), np.argsort(rng.random((SEEDS, m.L, m.M)), -1)
    A, D, _, _ = m.start(rng, tau, sigma, None, None); A0, D0 = sm(A), sm(D); R1, b1, R2, b2, log = np.zeros((SEEDS, m.L, m.M, V)), np.zeros((SEEDS, m.M)), np.zeros((SEEDS, m.S, K, m.M)), np.zeros((SEEDS, K)), []
    for _ in range(m.PRESLEEP): m.sleep(rng, A, D, R1, b1, R2, b2, "top")
    for _ in range(t0): m.wake(rng, tau, sigma, A, D, R1, b1, R2, b2, True, "bottom-up"); m.sleep(rng, A, D, R1, b1, R2, b2, "top")
    for t in range(AFTER + 1):
        if t in TA: log.append(m.score(rng, tau, sigma, A, D, R1, b1, R2, b2, A0, D0))
        if arm == "teacher": m.wake(rng, tau, sigma, A, D, R1, b1, R2, b2, True, "bottom-up")
        else: practise(rng, A, D, R1, b1, R2, b2, arm == "judged")
        m.sleep(rng, A, D, R1, b1, R2, b2, "top")
    return {k: np.array([row[k] for row in log]) for k in KEYS}                                             # [time after the cut, world]


def report():
    z = np.load(OUT); res = {(t0, arm): {k: z[f"{t0}_{arm}/{k}"] for k in KEYS} for t0 in T0S for arm in ARMS}; med = lambda y: np.median(y, -1); names = {"teacher": "teacher examples go on", "plain": "own samples only, read by the reader", "judged": "own samples only, weighted by the reader"}
    print(f"{RUNS * SEEDS} worlds (two groups: about half hold one symbol twice, so changes are taken per world). At the cut: median patches valid, combinations valid | then, {AFTER} batches later, median change per world: patches, combinations, KL (nats) | worlds with lower KL")
    for t0 in T0S:
        for arm in ARMS:
            r = res[t0, arm]; d = lambda k: med(r[k][-1] - r[k][0]); print(f"  watched {t0:4d}: {med(r['surface'][0]):.2f}, {med(r['deep'][0]):.2f} | {arm:8s} {d('surface'):+.2f}  {d('deep'):+.2f}  {d('kl'):+8.2f} | {(r['kl'][-1] < r['kl'][0]).sum():2d}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for arm, c in [("teacher", "C0"), ("plain", "C1"), ("judged", "C3")]:
        for a, k in zip(ax[:2], ["surface", "deep"]): a.plot(T0S, [med(res[t0, arm][k][-1] - res[t0, arm][k][0]) for t0 in T0S], "o-", color=c, label=names[arm])
        ax[2].plot(T0S, [(res[t0, arm]["kl"][-1] < res[t0, arm]["kl"][0]).mean() for t0 in T0S], "o-", color=c, label=names[arm])
    for a, lab in zip(ax, ["change in the share of valid patches", "change in the share of valid combinations", "share of worlds whose KL fell"]): a.set(xscale="log", xlabel="world batches watched before the cut", ylabel=lab); a.axhline(0, color="gray", lw=plt.rcParams["lines.linewidth"] / 2)
    ax[0].set_title(f"shallow level, {AFTER} batches after the cut"); ax[1].set_title("deep level"); ax[2].set_title("fit to the teacher"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3); fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


def main():
    jobs = [(t0, arm, seed) for t0 in T0S for arm in ARMS for seed in range(RUNS)]
    with Pool(min(len(jobs), os.cpu_count())) as pool: out = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
    np.savez(OUT, t=TA, **{f"{t0}_{arm}/{k}": np.concatenate([o[k] for (a, b, _), o in zip(jobs, out) if (a, b) == (t0, arm)], 1) for t0 in T0S for arm in ARMS for k in KEYS}); report()


if __name__ == "__main__": report() if sys.argv[1:] == ["report"] else main()
