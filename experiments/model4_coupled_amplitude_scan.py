"""Is a random but coupled deep start doomed? Follow-up to model4_coupled_init_check.py, whose "random" start was a sharp, complete grammar
with no entry right. Here the deep (top) table starts as Gaussian logits of standard deviation AMP (0.1 = the "diffuse" start of the check,
4 = nearly one-hot rows), the dictionary is right, and the deep reader is
  coupled      PRESLEEP own batches written from the top with the generator frozen (R2 inverts the random top table)
  not coupled  the same number of own batches that label patches only (R2 stays at zero until sleep trains it during the run)
then plain wake-sleep, one own batch per world batch, as in the check.
Usage: python model4_coupled_amplitude_scan.py [report] -> reports/model4-coupled-amplitude-scan.png, /var/tmp/jbauer/generative_practice_reading/model4_coupled_amplitude_scan.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import ws_two_level_chimera as m
from ws_reading_gain import K, SEEDS, sm

RUNS, AMPS, PROCS = 2, [0.1, 0.3, 1.0, 2.0, 4.0], 20                                                        # RUNS x SEEDS worlds per arm
KEYS = ["kl", "deep", "read2", "top"]
OUT = Path("/var/tmp/jbauer/generative_practice_reading/model4_coupled_amplitude_scan.npz"); OLD = OUT.with_name("model4_coupled_init.npz"); FIG = Path(__file__).resolve().parents[1] / "reports/model4-coupled-amplitude-scan.png"


def run(job):
    amp, coupled, seed = job; rng = np.random.default_rng(seed); tau, sigma = np.argsort(rng.random((SEEDS, m.S, K)), -1), np.argsort(rng.random((SEEDS, m.L, m.M)), -1)
    A, D, _, _ = m.start(rng, tau, sigma, None, m.L); A *= amp / 0.1; A0, D0 = sm(A), sm(D); R1, b1, R2, b2, log = np.zeros((SEEDS, m.L, m.M, m.V)), np.zeros((SEEDS, m.M)), np.zeros((SEEDS, m.S, K, m.M)), np.zeros((SEEDS, K)), []
    for _ in range(m.PRESLEEP): m.sleep(rng, A, D, R1, b1, R2, b2, "top" if coupled else "syllable")
    for t in range(m.STEPS + 1):
        if t in m.T: log.append(m.score(rng, tau, sigma, A, D, R1, b1, R2, b2, A0, D0) | {"top": sm(A).max(-1).mean((1, 2))})   # top: mean largest probability in a row of the top table (1 / M = flat, 1 = one-hot)
        m.wake(rng, tau, sigma, A, D, R1, b1, R2, b2, True, "bottom-up"); m.sleep(rng, A, D, R1, b1, R2, b2, "top")
    return {k: np.array([row[k] for row in log]) for k in KEYS}                                             # [time, world]


def report():
    z, old = np.load(OUT), np.load(OLD); T = m.T; at = lambda y, t: y[np.searchsorted(T, t)]; res = {(a, c): {k: z[f"{a}/{c}/{k}"] for k in KEYS} for a in AMPS for c in (True, False)}
    print(f"{RUNS * SEEDS} worlds per arm | sharpest entry of a top row at the start | teacher strings read at the start | world batches until half the combinations are valid (median, worlds that get there) | median KL at 100, 300, 1000, 3000 | worlds with KL > 1 at 1000, 3000")
    for (a, c), r in res.items():
        o = m.onset(r["deep"]); print(f"  logit sd {a:3.1f}, {'coupled    ' if c else 'not coupled'} {np.median(at(r['top'], 0)):.2f} | {np.median(at(r['read2'], 0)):.2f} | {m.nmed(o):5.0f} ({np.isfinite(o).sum():2d}) | " + "  ".join(f"{np.median(at(r['kl'], t)):5.2f}" for t in (100, 300, 1000, 3000)) + " | " + "  ".join(f"{(at(r['kl'], t) > 1).sum():2d}" for t in (1000, 3000)))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True); x = np.arange(len(AMPS) + 1); lab = [f"{a:g}" for a in AMPS] + ["sharp,\nno entry right"]
    for c, col in [(True, "C3"), (False, "C0")]:
        name = "coupled" if c else "not coupled"; o = [m.onset(res[a, c]["deep"]) for a in AMPS] + [m.onset(old[f"random, {name}/deep"])]; kl = [res[a, c]["kl"] for a in AMPS] + [old[f"random, {name}/kl"]]
        ax[0].plot(x, [np.median(at(res[a, c]["read2"], 0)) for a in AMPS] + [np.median(at(old[f"random, {name}/read2"], 0))], "o-", color=col, label=f"deep reader {name}")
        ax[1].plot(x, [m.nmed(v) for v in o], "o-", color=col, label=f"deep reader {name}"); ax[2].plot(x, [(at(k, 3000) > 1).mean() for k in kl], "o-", color=col, label=f"deep reader {name}")
    ax[0].axhline(1 / K, color="gray", lw=plt.rcParams["lines.linewidth"] / 2, label="chance"); ax[0].set(ylabel="teacher strings read into one code per cause, at the start", title="what coupling gives at the start")
    ax[1].set(ylabel="world batches until half the combinations are valid (median)", title="speed"); ax[2].set(ylabel="share of worlds with KL > 1 nat after 3000 world batches", title="worlds that stay wrong")
    for a in ax: a.set_xticks(x, lab); a.set_xlabel("random deep start: standard deviation of the top table's logits"); a.legend()
    fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


def main():
    jobs = [(a, c, seed) for a in AMPS for c in (True, False) for seed in range(RUNS)]
    with Pool(PROCS) as pool: out = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
    OUT.parent.mkdir(parents=True, exist_ok=True); np.savez(OUT, t=m.T, **{f"{a}/{c}/{k}": np.concatenate([o[k] for (a_, c_, _), o in zip(jobs, out) if (a_, c_) == (a, c)], 1) for a in AMPS for c in (True, False) for k in KEYS}); report()


if __name__ == "__main__": report() if sys.argv[1:] == ["report"] else main()
