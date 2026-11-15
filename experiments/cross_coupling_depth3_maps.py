"""Sample-resolved maps for the depth-3 pilot (README, Strategy): population loss as a function of total samples and own share, and the gain
per further teacher sample and per further own sample at every point of that map.
World, writer, reader and steps are those of cross_coupling_depth3_pilot.py (a teacher sample is a derivation).
Protocol: blank learner (random writer, blank reader), no separate Watching stage; teacher and own samples are interleaved at a fixed own share y from
the first sample (sample t is a teacher sample iff floor((t + 1)(1 - y)) > floor(t (1 - y))). Teacher samples are credited through the naming layers
(the wake step; with a blank reader this credits every code alike). At every checkpoint P the learner is copied twice: one copy gets P / 16 further
teacher samples, the other P / 16 further own samples; gain = (loss before - loss after) / number of further samples, in nats per sample.
Learners: "wake-sleep" (own samples train the naming layers only), "practice" (own samples also change the writer: accepted if legible, per try).
At own share 0 both are Watching with a wake step and a reader that never learns.
Baseline ("sorted"): teacher samples only, wake step read by a reader that names the writer's units without own samples (Sorter in the pilot).
Second figure: loss against the number of teacher samples seen, for that baseline, own share 0, wake-sleep at own share 1/2 and practice at three shares.
Usage: python experiments/cross_coupling_depth3_maps.py [sorted | report] -> /var/tmp/jbauer/generative_practice_reading/cross_coupling_depth3_maps{,_sorted}.npz,
reports/cross-coupling-depth3-maps-{tables,layers}.png, reports/cross-coupling-depth3-teacher-samples.png (maps: about 20 minutes on 32 cores; sorted: 2 minutes)"""

import copy
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from tqdm.auto import tqdm

from cross_coupling_depth3_pilot import Layers, Reader, Sorter, World, Writer, practice, score, teacher

SHARES, POINTS, WORLDS, FORMS = (0, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 15 / 16, 31 / 32), [round(2 ** (10 + i / 2)) for i in range(15)], 16, ("tables", "layers")
LEARNERS = {"practice": "naming+keep legible per try", "wake-sleep": "naming"}
ROOT = Path(__file__).resolve().parents[1]; NPZ = Path("/var/tmp/jbauer/generative_practice_reading/cross_coupling_depth3_maps.npz"); PNG = ROOT / "reports/cross-coupling-depth3-maps.png"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"]); DIVERGING = LinearSegmentedColormap.from_list("blue-red", ["#7a1414", "#e34948", "#f0efec", "#3987e5", "#0d366b"])


def run(job):
    seed, form, learner, share = job; rng = np.random.default_rng(seed); w = World(rng); wr, rd = (Layers if form == "layers" else Writer)(rng), Reader(); r = np.random.default_rng(seed + 1000)
    out, done = [], 0
    for P in POINTS:
        for t in range(done, P):
            if int((t + 1) * (1 - share)) > int(t * (1 - share)): teacher(wr, rd, w, r, "named")
            else: practice(wr, rd, w, r, LEARNERS[learner])
        done, now, further, after = P, score(wr, rd, w)[:3], max(P // 16, 64), []
        for own in (False, True):
            wr2, rd2, r2 = wr.copy(), copy.deepcopy(rd), np.random.default_rng([seed, P, own])
            for _ in range(further):
                if own: practice(wr2, rd2, w, r2, LEARNERS[learner])
                else: teacher(wr2, rd2, w, r2, "named")
            after.append(score(wr2, rd2, w)[2])
        out.append((*now, (now[2] - after[0]) / further, (now[2] - after[1]) / further))
    return np.array(out)


def run_sorted(job):
    seed, form = job; rng = np.random.default_rng(seed); w = World(rng); wr, rd = (Layers if form == "layers" else Writer)(rng), Sorter(); r = np.random.default_rng(seed + 1000); out, done = [], 0
    for P in POINTS:
        for t in range(done, P): teacher(wr, rd, w, r, "sorted")
        done = P; out.append(score(wr, rd, w)[:3])
    return np.array(out)


def teacher_samples():
    d, b = np.load(NPZ), np.load(NPZ.with_name(NPZ.stem + "_sorted.npz")); fig, axes = plt.subplots(1, len(FORMS), sharey=True, constrained_layout=True, figsize=(6.4 * 2, 4.8 * 1.2)); P = np.array(POINTS)
    for ax, form in zip(axes, FORMS):
        lines = [("teacher only, reader never learns", P, d[form][:, 0, 0, :, 2]), ("teacher only, units named without own samples", P, b[form][:, :, 2]), ("wake-sleep, own share 1/2", P / 2, d[form][:, 1, 3, :, 2])]
        lines += [(f"practice, own share {round(SHARES[j] * 32)}/32", P * (1 - SHARES[j]), d[form][:, 0, j, :, 2]) for j in (3, 6, 7)]; print(f"{form}: teacher samples until the loss is first below 0.2, median over worlds")
        for n, (name, x, kl) in enumerate(lines):
            ax.plot(x, np.median(kl, 0), color=f"C{n}", label=name); ax.fill_between(x, *np.quantile(kl, [0.25, 0.75], 0), color=f"C{n}", alpha=0.15, linewidth=0)
            hit = kl < 0.2; t = np.where(hit.any(1), x[hit.argmax(1)], np.inf); print(f"   {name:46s} {np.median(t):8.0f}  ({np.isinf(t).sum()} of {len(t)} worlds never)   loss at the end {np.median(kl[:, -1]):.3f}")
        ax.set(xscale="log", yscale="log", xlabel="teacher samples seen", title=f"writer as {form}")
    axes[0].set_ylabel("population KL, nats"); handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    png = PNG.with_name("cross-coupling-depth3-teacher-samples.png"); fig.savefig(png, dpi=130); print(png)


def report():
    d = np.load(NPZ); population = 8192; xt = np.arange(0, len(POINTS), 2)
    for form in FORMS:
        fig, axes = plt.subplots(len(LEARNERS), 4, sharex=True, sharey=True, constrained_layout=True, figsize=(6.4 * 2.8, 4.8 * 1.5)); m = np.median(d[form], 0)   # (learner, share, point, value)
        top = np.abs(m[..., 3:]).max(); gain = SymLogNorm(linthresh=top / 1e3, vmin=-top, vmax=top); loss = LogNorm(m[..., 2].min(), m[..., 2].max())
        for k, (learner, row) in enumerate(zip(LEARNERS, axes)):
            panels = ((m[k, :, :, 2], "population KL, nats", SEQUENTIAL, loss), (m[k, :, :, 3], "gain per further teacher sample", DIVERGING, gain), (m[k, :, :, 4], "gain per further own sample", DIVERGING, gain),
                      (m[k, :, :, 4] - m[k, :, :, 3], "own minus teacher", DIVERGING, gain))
            for ax, (z, title, cmap, norm) in zip(row, panels):
                im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, norm=norm); ax.set(xticks=xt, xticklabels=[f"{POINTS[i] / population:g}" for i in xt], yticks=range(len(SHARES)), yticklabels=["0"] + [f"{round(s * 32)}/32" for s in SHARES[1:]])
                if k == 0: ax.set_title(title)
                if k == len(LEARNERS) - 1: ax.set_xlabel("total samples / number of valid derivations")
                if ax is row[0]: ax.set_ylabel(f"{learner}\nown share of the samples")
                if ax is row[0] or ax is row[-1]: fig.colorbar(im, ax=ax if ax is row[0] else row[1:], label="nats" if ax is row[0] else "nats per sample (blue: loss falls)")
        fig.suptitle(f"writer as {form}"); png = PNG.with_name(f"{PNG.stem}-{form}.png"); fig.savefig(png, dpi=130); print(png)
        for k, learner in enumerate(LEARNERS):
            print(f"{form}, {learner}: median KL by own share (rows) at total samples {[POINTS[i] for i in xt]}")
            for j, s in enumerate(SHARES): print(f"   own share {s:7.5f}  " + "  ".join(f"{m[k, j, i, 2]:6.3f}" for i in xt) + f"   | teacher samples to KL < 0.2: {first(d[form][:, k, j, :, 2], s)}")


def first(kl, share):   # median over worlds of the teacher samples seen when the loss first falls below 0.2
    hit = kl < 0.2; t = np.where(hit.any(1), np.array(POINTS)[hit.argmax(1)] * (1 - share), np.inf); return f"{np.median(t):.0f} ({np.isinf(t).sum()} of {len(t)} worlds never)"


if __name__ == "__main__":
    if sys.argv[1:] == ["sorted"]:
        jobs = [(seed, form) for form in FORMS for seed in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run_sorted, jobs), total=len(jobs)))
        np.savez(NPZ.with_name(NPZ.stem + "_sorted.npz"), **{form: np.array([r for r, j in zip(res, jobs) if j[1] == form]) for form in FORMS}); teacher_samples(); sys.exit()
    if sys.argv[1:] != ["report"]:
        jobs = [(seed, form, learner, share) for form in FORMS for learner in LEARNERS for share in SHARES for seed in range(WORLDS)]
        with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
        np.savez(NPZ, **{form: np.array([r for r, j in zip(res, jobs) if j[1] == form]).reshape(len(LEARNERS), len(SHARES), WORLDS, len(POINTS), 5).transpose(2, 0, 1, 3, 4) for form in FORMS})
    report()
    if NPZ.with_name(NPZ.stem + "_sorted.npz").exists(): teacher_samples()
