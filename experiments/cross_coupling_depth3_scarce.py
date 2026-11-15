"""Teacher-sample efficiency of the depth-3 pilot when teacher samples are scarce and own samples are free (README, Baselines: classic wake-sleep
need not be beaten at convergence, only in teacher samples).
A blank learner watches N0 teacher samples (wake step read by the blank naming layers, so every code is credited alike; the checker memories fill).
Then, with no further teacher sample:
  practice alone      OWN own samples (naming + accepted if legible, per try)
  wake-sleep, stored  the N0 stored samples re-read in turn (credited through the naming layers) in alternation with own samples that train the
                      naming layers, OWN steps in total
Reference from cross_coupling_depth3_maps.py (sorted): N0 fresh teacher samples in one pass, wake step read by the reader that needs no own samples.
Scores exact, as in cross_coupling_depth3_pilot.py.
Usage: python experiments/cross_coupling_depth3_scarce.py [report] -> /var/tmp/jbauer/generative_practice_reading/cross_coupling_depth3_scarce.npz,
reports/cross-coupling-depth3-scarce.png (about 20 minutes on 32 cores)"""

import copy
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cross_coupling_depth3_maps import NPZ as MAPS, POINTS
from cross_coupling_depth3_pilot import Layers, Reader, World, Writer, practice, score, teacher

SETS, OWN, MARKS, WORLDS, FORMS, ARMS = (1024, 2048, 4096, 8192, 16384), 300000, [round(300000 * 2 ** (-i / 2)) for i in range(12, -1, -1)], 16, ("tables", "layers"), ("practice alone", "wake-sleep, stored")
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_depth3_scarce.npz"); PNG = ROOT / "reports/cross-coupling-depth3-scarce.png"


def run(job):
    seed, form, n0 = job; rng = np.random.default_rng(seed); w = World(rng); wr, rd = (Layers if form == "layers" else Writer)(rng), Reader(); r = np.random.default_rng(seed + 1000); stored = w.trees[r.integers(len(w.trees), size=n0)]
    for tree in stored: teacher(wr, rd, SimpleNamespace(trees=tree[None]), r, "named")
    out = []
    for arm in ARMS:
        wr2, rd2, r2, curve = wr.copy(), copy.deepcopy(rd), np.random.default_rng(seed + 2000), [score(wr, rd, w)[:3]]
        for t in range(OWN):
            if arm == "practice alone": practice(wr2, rd2, w, r2, "naming+keep legible per try")
            elif t % 2: practice(wr2, rd2, w, r2, "naming")
            else: teacher(wr2, rd2, SimpleNamespace(trees=stored[(t // 2) % n0][None]), r2, "named")
            if t + 1 in MARKS: curve.append(score(wr2, rd2, w)[:3])
        out.append(curve)
    return np.array(out)


def report():
    d, b = np.load(NPZ), np.load(MAPS.with_name(MAPS.stem + "_sorted.npz")); fig, axes = plt.subplots(1, len(FORMS), sharey=True, constrained_layout=True, figsize=(6.4 * 2, 4.8 * 1.2)); at = [POINTS.index(n) for n in SETS]
    for ax, form in zip(axes, FORMS):
        s = d[form]; lines = [("N0 fresh teacher samples, units named without own samples", b[form][:, at, 2]), ("N0 teacher samples, then wake-sleep on the stored set", s[:, :, 1, -1, 2]), ("N0 teacher samples, then practice alone", s[:, :, 0, -1, 2])]
        print(f"{form}: after watching, median KL {np.round(np.median(s[:, :, 0, 0, 2], 0), 2)} for N0 = {SETS}")
        for n, (name, kl) in enumerate(lines):
            ax.plot(SETS, np.median(kl, 0), "o-", color=f"C{n}", label=name); ax.fill_between(SETS, *np.quantile(kl, [0.25, 0.75], 0), color=f"C{n}", alpha=0.15, linewidth=0)
            print(f"   {name:58s} median KL {np.round(np.median(kl, 0), 3)}   worlds below 0.2: {(kl < 0.2).sum(0)} of {len(kl)}")
        k = s[:, :, 0, :, :]; print(f"   practice alone, valid {np.round(np.median(k[:, :, -1, 0], 0), 3)} written {np.round(np.median(k[:, :, -1, 1], 0), 2)}; KL over own samples for N0 = {SETS[0]}: {np.round(np.median(k[:, 0, :, 2], 0), 2)}")
        ax.set(xscale="log", yscale="log", xlabel="teacher samples seen (N0)", title=f"writer as {form}")
    axes[0].set_ylabel(f"population KL after {OWN} further steps, nats"); handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=1, frameon=False); fig.savefig(PNG, dpi=130); print(PNG)


if __name__ == "__main__":
    if sys.argv[1:] != ["report"]:
        jobs = [(seed, form, n0) for form in FORMS for n0 in SETS for seed in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        np.savez(NPZ, **{form: np.array([r for r, j in zip(res, jobs) if j[1] == form]).reshape(len(SETS), WORLDS, len(ARMS), len(MARKS) + 1, 3).transpose(1, 0, 2, 3, 4) for form in FORMS})
    report()
