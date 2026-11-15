"""Check of the first assumption in tex/archive/model_4.md: deep encoder and decoder start random but coupled, and that helps.
Two-level word model of ws_two_level_chimera.py (cause -> S symbols -> patches of L tokens; top table = deep layer, dictionary = shallow
layer), plain wake-sleep with one own batch per world batch. Two dials:
  deep start   top table sharp and unrelated to the teacher's ("random") | diffuse | 4 of 8 entries right per code
  deep reader  coupled: PRESLEEP own batches written from the top with the generator frozen (R2 inverts the top table)
               not coupled: the same number of own batches that label patches only (R1 inverts the dictionary, R2 stays at zero)
with the dictionary right at the start (the state model 4 expects after watching), and the literal start of model 4 (dictionary blank).
Usage: python model4_coupled_init_check.py [report] -> reports/model4-coupled-init.png, /var/tmp/jbauer/generative_practice_reading/model4_coupled_init.npz"""

import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import ws_two_level_chimera as m
from ws_reading_gain import K, SEEDS, sm

RUNS = 4                                                                                                    # RUNS x SEEDS worlds per arm
ARMS = {"random, coupled": (0, m.L, True), "random, not coupled": (0, m.L, False), "diffuse, coupled": (None, m.L, True), "diffuse, not coupled": (None, m.L, False),
        "4 of 8 right, coupled": (4, m.L, True), "4 of 8 right, not coupled": (4, m.L, False), "dictionary blank: deep random, coupled": (0, None, True), "dictionary blank: deep diffuse": (None, None, True)}   # (top, dictionary, deep reader coupled)
KEYS = ["kl", "deep", "surface", "read2"]
OUT = Path("/var/tmp/jbauer/generative_practice_reading/model4_coupled_init.npz"); FIG = Path(__file__).resolve().parents[1] / "reports/model4-coupled-init.png"


def run(job):
    name, seed = job; top, bottom, coupled = ARMS[name]; rng = np.random.default_rng(seed); tau, sigma = np.argsort(rng.random((SEEDS, m.S, K)), -1), np.argsort(rng.random((SEEDS, m.L, m.M)), -1)
    A, D, _, _ = m.start(rng, tau, sigma, top, bottom); A0, D0 = sm(A), sm(D); R1, b1, R2, b2, log = np.zeros((SEEDS, m.L, m.M, m.V)), np.zeros((SEEDS, m.M)), np.zeros((SEEDS, m.S, K, m.M)), np.zeros((SEEDS, K)), []
    for _ in range(m.PRESLEEP): m.sleep(rng, A, D, R1, b1, R2, b2, "top" if coupled else "syllable")
    for t in range(m.STEPS + 1):
        if t in m.T: log.append(m.score(rng, tau, sigma, A, D, R1, b1, R2, b2, A0, D0))
        m.wake(rng, tau, sigma, A, D, R1, b1, R2, b2, True, "bottom-up"); m.sleep(rng, A, D, R1, b1, R2, b2, "top")
    return {k: np.array([row[k] for row in log]) for k in KEYS}                                             # [time, world]


def report():
    z = np.load(OUT); res = {name: {k: z[f"{name}/{k}"] for k in KEYS} for name in ARMS}; T = m.T; at = lambda y, t: y[np.searchsorted(T, t)]
    print(f"{RUNS * SEEDS} worlds per arm | strings read before any world example | world batches until half the combinations are valid (median, worlds that get there) | median KL at 0, 30, 100, 300, 1000, 3000 | worlds with KL > 1 at 300, 1000, 3000")
    for name, r in res.items():
        o = m.onset(r["deep"]); print(f"  {name:42s} {np.median(at(r['read2'], 0)):.2f} | {m.nmed(o):5.0f} ({np.isfinite(o).sum():2d}) | " + "  ".join(f"{np.median(at(r['kl'], t)):6.2f}" for t in (0, 30, 100, 300, 1000, 3000)) + " | " + "  ".join(f"{(at(r['kl'], t) > 1).sum():2d}" for t in (300, 1000, 3000)))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True); col = {"random": "C3", "diffuse": "C0", "4 of 8 right": "C2"}
    for name in list(ARMS)[:6]:
        kind, cpl = name.split(", "); kw = dict(color=col[kind], ls="-" if cpl == "coupled" else "--", label=f"deep layer {kind}, reader {cpl}")
        ax[0].plot(T + 1, np.median(res[name]["kl"], 1), **kw); ax[1].plot(T + 1, np.median(res[name]["deep"], 1), **kw); ax[2].plot(T + 1, (res[name]["kl"] > 1).mean(1), **kw)
    for name, c in [("dictionary blank: deep random, coupled", "C3"), ("dictionary blank: deep diffuse", "C0")]: ax[0].plot(T + 1, np.median(res[name]["kl"], 1), color=c, ls=":", label=name); ax[2].plot(T + 1, (res[name]["kl"] > 1).mean(1), color=c, ls=":", label=name)
    ax[0].set(xscale="log", yscale="log", xlabel="world batches + 1", ylabel="KL(teacher || student), median", title="learning curve"); ax[1].set(xscale="log", xlabel="world batches + 1", ylabel="share of valid combinations of patches, median", title="what the deep layer writes")
    ax[2].set(xscale="log", xlabel="world batches + 1", ylabel="share of worlds with KL > 1 nat", title="worlds not yet right")
    for a in ax: a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(FIG, dpi=150, bbox_inches="tight"); print(FIG)


def main():
    jobs = [(name, seed) for name in ARMS for seed in range(RUNS)]
    with Pool(min(len(jobs), os.cpu_count())) as pool: out = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
    OUT.parent.mkdir(parents=True, exist_ok=True); np.savez(OUT, t=m.T, **{f"{name}/{k}": np.concatenate([o[k] for (n, _), o in zip(jobs, out) if n == name], 1) for name in ARMS for k in KEYS}); report()


if __name__ == "__main__": report() if sys.argv[1:] == ["report"] else main()
