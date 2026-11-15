"""Why does practice on own samples help next to a stored teacher set (cross_coupling_depth3_scarce.py), and does it survive a fair comparison?
Every arm sees the stored set once (watching) and then exactly T further samples (exposures), each one either a re-read stored teacher sample or an
own sample. Scores exact, as in cross_coupling_depth3_pilot.py, at log-spaced numbers of exposures.

lead    exact counts, no learning: the share of the valid derivations for which a stored set of N0 samples has shown every fact that
          the checker needs      productions per level and what met at a cut, wherever they occurred
          a layer writer needs   productions per node (the agreement weights are shared by all symbols)
          a table writer needs   productions per node and per value of the agreement features handed down to that node (its rows)
rates   own share 1/2 (teacher only: 0), teacher step in units of ETA (x1 as before, x4 = about the largest practice step, ETA_TRY * legibility):
          teacher only           the reader that needs no own samples (Sorter) re-reads the stored set
          wake-sleep             own samples train the naming layers only
          practice               own samples also train the writer (accepted if legible, per try)
          practice, checker accepts everything / checker complete: the same with a checker that never rejects / that knows every fact of the world
        At the end: log-probability deficit of a valid derivation by the number of its table rows that the stored set never showed.
shares  practice and wake-sleep at other own shares, same T (argument: teacher step in units of ETA)
Usage: python experiments/cross_coupling_depth3_mechanism.py lead | rates | shares <step> | report
-> /var/tmp/jbauer/generative_practice_reading/cross_coupling_depth3_mechanism_{lead,rates,shares}.npz, reports/cross-coupling-depth3-mechanism-{lead,rates,rows,shares,map}.png"""

import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import matplotlib; matplotlib.use("Agg")   # figures are only saved; a stale X forward makes the default backend hang
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cross_coupling_depth3_pilot import ETA, RHO, U2, U3, V, Layers, Reader, Sorter, World, Writer, practice, score, soft, teacher

SETS, T, WORLDS, FORMS, OWN = (256, 1024, 4096), 200000, 16, ("tables", "layers"), "naming+keep legible per try"
MARKS = sorted({round(1000 * 2 ** (i / 2)) for i in range(16)} | {T}); GRID = [round(2 ** (4 + i / 2)) for i in range(19)]
# arm: (reader, rule for own samples, teacher step in units of ETA, own share of the T exposures)
RATES = {"teacher only": ("sorted", None, 1, 0), "teacher only, step x4": ("sorted", None, 4, 0), "wake-sleep": ("named", "naming", 1, 1 / 2), "wake-sleep, step x4": ("named", "naming", 4, 1 / 2),
         "practice": ("named", OWN, 1, 1 / 2), "practice, step x4": ("named", OWN, 4, 1 / 2), "practice, checker accepts everything": ("lenient", OWN, 1, 1 / 2), "practice, checker complete": ("complete", OWN, 1, 1 / 2)}
SHARES = (1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 15 / 16, 1)
OUT = Path("/var/tmp/jbauer/generative_practice_reading"); PNG = Path(__file__).resolve().parents[1] / "reports"


class Lenient(Reader):
    def verdicts(self, tree): return [True] * 7


def reader(kind, w):
    rd = {"sorted": Sorter, "lenient": Lenient}.get(kind, Reader)()
    if kind == "complete":
        for tree in w.trees: rd.see(tree)
    return rd


def facts(w, trees, who):   # one integer per fact that `who` needs for a derivation
    a, p3, pL, pR, t1, t2, t3, t4 = trees.T; ys, ts, f0, f1, z = (pL // V, pL % V, pR // V, pR % V), (t1, t2, t3, t4), w.phi0[t2 % V], w.phi1[pL % V], np.zeros_like(a)
    if who == "checker": cols = [(0, a, p3, z, z), (1, p3 // V, pL, z, z), (1, p3 % V, pR, z, z), *((2, ys[i], ts[i], z, z) for i in range(4)), *((3, ts[i] % V, ts[i + 1] // V, z, z) for i in range(3)), (4, ys[1], ys[2], z, z)]
    else:
        ctx = [(z, z), (f0, z), (f0, z), (f0, f1), (f1, z), (f1, z), (f0, f1)] if who == "tables" else [(z, z)] * 7   # the handed-down features a table row is addressed by, beyond what its production fixes
        cols = [(n, sym, p, *c) for n, (sym, p), c in zip(range(7), ((a, p3), (p3 // V, pL), (p3 % V, pR), *zip(ys, ts)), ctx)]
    return np.stack([(((n * V + sym) * V * V + p) * RHO + x) * RHO + y for n, sym, p, x, y in cols], 1)


def logq(wr, w):   # log of the writer's probability of every valid derivation, relative to the target's
    a, p3, pL, pR, *ts = w.trees.T; ys = (pL // V, pL % V, pR // V, pR % V); F2, F1 = wr.full(); A3, B2, C1 = soft(wr.D3).mean(1), soft(F2).mean(3), soft(F1).mean(4)
    leaf = [C1[i][ys[i], :, :, ts[i]].transpose(0, 2, 1) for i in range(4)]
    q = (A3[a, p3] / V / U3)[:, None, None, None] * (B2[0][p3 // V, :, pL] * B2[1][p3 % V, :, pR])[:, :, None, None] * (leaf[0] * leaf[1] / U2)[:, :, :, None] * (leaf[2] * leaf[3] / U2)[:, :, None, :]
    return np.log(q.sum((1, 2, 3)) * len(q))


def lead(seed):
    w = World(np.random.default_rng(seed)); r = np.random.default_rng(seed + 1000); out = []
    for n0 in GRID:
        stored = w.trees[r.integers(len(w.trees), size=n0)]; out.append([np.isin(facts(w, w.trees, who), facts(w, stored, who)).all(1).mean() for who in ("checker", "layers", "tables")])
    return np.array(out)


def run(job):
    seed, form, n0, arms = job; rng = np.random.default_rng(seed); w = World(rng); start = (Layers if form == "layers" else Writer)(rng); marks = set(MARKS)
    stored = w.trees[np.random.default_rng(seed + 1000).integers(len(w.trees), size=n0)]; missing = np.minimum((~np.isin(facts(w, w.trees, "tables"), facts(w, stored, "tables"))).sum(1), 3); curves, ends = [], []
    for kind, own, step, share in arms.values():
        wr, rd, r, how, n, curve = start.copy(), reader(kind, w), np.random.default_rng(seed + 2000), "sorted" if kind == "sorted" else "named", 0, []
        for tree in stored: teacher(wr, rd, SimpleNamespace(trees=tree[None]), r, how, ETA * step)
        for t in range(T):
            if int((t + 1) * (1 - share)) > int(t * (1 - share)): teacher(wr, rd, SimpleNamespace(trees=stored[n % n0][None]), r, how, ETA * step); n += 1
            else: practice(wr, rd, w, r, own)
            if t + 1 in marks: curve.append(score(wr, rd, w))
        lq = logq(wr, w); curves.append(curve); ends.append([-lq[missing == k].mean() if (missing == k).any() else np.nan for k in range(4)] + [(missing == k).mean() for k in range(4)])
    return np.array(curves), np.array(ends)


def sweep(arms, sets, name, **extra):
    jobs = [(seed, form, n0, arms) for form in FORMS for n0 in sets for seed in range(WORLDS)]
    with Pool(32) as pool: res = list(tqdm(pool.imap(run, jobs), total=len(jobs)))
    np.savez(OUT / f"cross_coupling_depth3_mechanism_{name}.npz", **extra, **{f"{form} {k}": np.array([r[i] for r, j in zip(res, jobs) if j[1] == form]).reshape(len(sets), WORLDS, len(arms), *res[0][i].shape[1:]) for form in FORMS for i, k in enumerate(("curves", "ends"))})


def slope(y): return np.polyfit(np.log(MARKS[-5:]), np.log(y[-5:]), 1)[0]   # d log KL / d log exposures over the last factor of four


def report_lead():
    d = np.load(OUT / "cross_coupling_depth3_mechanism_lead.npz")["lead"]; fig, ax = plt.subplots(constrained_layout=True)
    for n, who in enumerate(("the checker can verify", "a layer writer has been shown (productions per node)", "a table writer has been shown (all of its rows)")):
        ax.plot(GRID, np.median(d[:, :, n], 0), "o-", color=f"C{n}", label=who); ax.fill_between(GRID, *np.quantile(d[:, :, n], [0.25, 0.75], 0), color=f"C{n}", alpha=0.15, linewidth=0)
        print(f"lead: {who:55s} half of the population after {GRID[int((np.median(d[:, :, n], 0) > 0.5).argmax())]} samples, 95% after {GRID[int((np.median(d[:, :, n], 0) > 0.95).argmax())]}")
    ax.set(xscale="log", xlabel="teacher samples in the stored set", ylabel="share of the valid derivations"); ax.legend(frameon=False); fig.savefig(PNG / "cross-coupling-depth3-mechanism-lead.png", dpi=130); print(PNG / "cross-coupling-depth3-mechanism-lead.png")


def report_rates():
    d = np.load(OUT / "cross_coupling_depth3_mechanism_rates.npz"); style = {"teacher only": ("C0", "-"), "wake-sleep": ("C1", "-"), "practice": ("C2", "-"), "practice, checker accepts everything": ("C3", ":"), "practice, checker complete": ("C4", ":")}
    fig, axes = plt.subplots(len(FORMS), len(SETS), sharex=True, sharey=True, constrained_layout=True, figsize=(6.4 * 2.2, 4.8 * 1.7)); fig2, axes2 = plt.subplots(1, len(FORMS), sharey=True, constrained_layout=True, figsize=(6.4 * 2, 4.8))
    for row, form in enumerate(FORMS):
        c, e = d[f"{form} curves"], d[f"{form} ends"]   # (set, world, arm, mark, metric), (set, world, arm, 8)
        for col, n0 in enumerate(SETS):
            print(f"{form}, stored set of {n0}, {T} exposures: median KL | of which invalid mass | rest | valid derivations written | worlds below 0.2 | slope of log KL against log exposures at the end")
            for k, arm in enumerate(RATES):
                kl, inv = c[col, :, k, :, 2], -np.log(c[col, :, k, :, 0]); color, line = style[arm.removesuffix(", step x4")]; line = "--" if arm.endswith("x4") else line
                axes[row, col].plot(MARKS, np.median(kl, 0), color=color, linestyle=line, label=arm)
                print(f"   {arm:38s} {np.median(kl[:, -1]):.3f} | {np.median(inv[:, -1]):.3f} | {np.median(kl[:, -1] - inv[:, -1]):.3f} | {np.median(c[col, :, k, -1, 1]):.3f} | {(kl[:, -1] < 0.2).sum():2d} of {len(kl)} | {slope(np.median(kl, 0)):+.2f}")
            axes[row, col].set(xscale="log", yscale="log", title=f"writer as {form}, stored set of {n0}" if row == 0 else f"{form}, {n0}")
        print(f"{form}, stored set of {SETS[0]}: share of the valid derivations with 0, 1, 2, 3+ table rows never shown: {np.round(np.nanmean(e[0, :, 0, 4:], 0), 3)}; mean deficit (nats) by that number")
        for k, arm in enumerate(RATES):
            print(f"   {arm:38s} {np.round(np.nanmedian(e[0, :, k, :4], 0), 2)}")
            if not arm.endswith("x4"): axes2[row].plot(range(4), np.nanmedian(e[0, :, k, :4], 0), marker="o", color=style[arm][0], linestyle=style[arm][1], label=arm)
        axes2[row].set(xlabel="rows of the derivation that the stored set never showed", xticks=range(4), xticklabels=["0", "1", "2", "3 or more"], title=f"writer as {form}, stored set of {SETS[0]}")
    axes[-1, 1].set_xlabel("samples after watching (re-read teacher samples and own samples)")
    for ax in axes[:, 0]: ax.set_ylabel("population KL, nats")
    axes2[0].set_ylabel("log-probability missing, nats"); handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    handles, labels = axes2[0].get_legend_handles_labels(); fig2.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    for f, name in ((fig, "rates"), (fig2, "rows")): f.savefig(PNG / f"cross-coupling-depth3-mechanism-{name}.png", dpi=130); print(PNG / f"cross-coupling-depth3-mechanism-{name}.png")


def report_shares():
    d, rates = np.load(OUT / "cross_coupling_depth3_mechanism_shares.npz"), np.load(OUT / "cross_coupling_depth3_mechanism_rates.npz"); step, sets = int(d["step"]), tuple(int(n) for n in d["stored"]); learners = ("practice", "wake-sleep")
    fig, axes = plt.subplots(len(FORMS), len(sets), sharex=True, sharey=True, constrained_layout=True, figsize=(6.4 * 1.1 * len(sets), 4.8 * 1.7))
    for row, form in enumerate(FORMS):
        c = d[f"{form} curves"]; ref = rates[f"{form} curves"][:, :, list(RATES).index("teacher only" + (", step x4" if step == 4 else "")), -1, 2]   # (set, world)
        for col, n0 in enumerate(sets):
            print(f"{form}, stored set of {n0}, {T} exposures, teacher step x{step}: median KL (worlds below 0.2) by own share {[str(s) for s in SHARES]}; teacher only: {np.median(ref[SETS.index(n0)]):.3f}")
            for n, name in enumerate(learners):
                kl = c[col, :, n * len(SHARES):(n + 1) * len(SHARES), -1, 2]; axes[row, col].plot(range(len(SHARES)), np.median(kl, 0), "o-", color=f"C{2 - n}", label=name); axes[row, col].fill_between(range(len(SHARES)), *np.quantile(kl, [0.25, 0.75], 0), color=f"C{2 - n}", alpha=0.15, linewidth=0)
                print(f"   {name:12s} " + "  ".join(f"{np.median(kl[:, i]):.3f} ({(kl[:, i] < 0.2).sum()})" for i in range(len(SHARES))))
            axes[row, col].axhline(np.median(ref[SETS.index(n0)]), color="C0", linestyle="--", label="teacher only (own share 0)"); axes[row, col].set(yscale="log", title=f"writer as {form}, stored set of {n0}", xticks=range(len(SHARES)), xticklabels=["1/8", "1/4", "1/2", "3/4", "7/8", "15/16", "1"])
    for ax in axes[-1]: ax.set_xlabel(f"own share of the {T} samples after watching")
    for ax in axes[:, 0]: ax.set_ylabel("population KL, nats")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False); fig.savefig(PNG / "cross-coupling-depth3-mechanism-shares.png", dpi=130); print(PNG / "cross-coupling-depth3-mechanism-shares.png")


def report_map():   # the same scan resolved in exposures: x = samples after watching, y = own share, colour = median population KL (README, Strategy)
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    d = np.load(OUT / "cross_coupling_depth3_mechanism_shares.npz"); sets, learners = tuple(int(n) for n in d["stored"]), ("practice", "wake-sleep"); ticks = [MARKS.index(m) for m in (1000, 8000, 64000, T)]
    fig, axes = plt.subplots(len(FORMS), len(learners) * len(sets), sharex=True, sharey=True, constrained_layout=True, figsize=(6.4 * 2.4, 4.8 * 1.5)); cmap, norm = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#0d366b"]), LogNorm(0.005, 20)
    for row, form in enumerate(FORMS):
        for n, name in enumerate(learners):
            for col, n0 in enumerate(sets):
                ax = axes[row, n * len(sets) + col]; kl = np.median(d[f"{form} curves"][col, :, n * len(SHARES):(n + 1) * len(SHARES), :, 2], 0); im = ax.pcolormesh(np.arange(len(MARKS) + 1) - 0.5, np.arange(len(SHARES) + 1) - 0.5, kl, cmap=cmap, norm=norm)
                ax.set(title=f"{name}, {form}, stored set of {n0}", xticks=ticks, xticklabels=["1k", "8k", "64k", "200k"], yticks=range(len(SHARES)), yticklabels=["1/8", "1/4", "1/2", "3/4", "7/8", "15/16", "1"])
    for ax in axes[-1]: ax.set_xlabel("samples after watching")
    for ax in axes[:, 0]: ax.set_ylabel("own share")
    fig.colorbar(im, ax=axes, label="population KL, nats"); fig.savefig(PNG / "cross-coupling-depth3-mechanism-map.png", dpi=130); print(PNG / "cross-coupling-depth3-mechanism-map.png")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "lead":
        with Pool(32) as pool: np.savez(OUT / "cross_coupling_depth3_mechanism_lead.npz", lead=np.array(pool.map(lead, range(64))))
        report_lead()
    if mode == "rates": sweep(RATES, SETS, "rates"); report_rates()
    if mode == "shares":
        step = int(sys.argv[2]); sets = SETS[:2]; sweep({f"{name} {s}": ("named", rule, step, s) for name, rule in (("practice", OWN), ("wake-sleep", "naming")) for s in SHARES}, sets, "shares", step=step, stored=sets); report_shares(); report_map()
    if mode == "report": report_lead(); report_rates(); report_shares(); report_map()
