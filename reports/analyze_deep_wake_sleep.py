"""Aggregate all commissioned cases, retaining floors, failures, and censored times."""

import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports"
RESULTS = ROOT / "results/deep_wake_sleep"
COLORS = {"elbo": "C0", "sleep": "C1", "forward": "C2"}
LABELS = {"elbo": "ELBO", "sleep": "ELBO + sleep", "forward": "Exact posterior teaching"}


def read_stage(stage):
    path = RESULTS / stage
    protocol = json.loads((path / "protocol.json").read_text())
    assert hashlib.sha256((path / "runtime.py").read_bytes()).hexdigest() == (path / "runtime.sha256").read_text().strip()
    def read(case): return json.loads((path / case["name"] / "run.json").read_text())
    with ThreadPoolExecutor(max_workers=16) as pool: runs = list(tqdm(pool.map(read, protocol["cases"]), total=len(protocol["cases"]), desc=stage))
    assert all((path / r["config"]["name"] / "weights.npz").exists() for r in runs), f"Unfinished {stage}"
    return runs


def key(c): return (len(c["layers"]), c["recognition_depth"], c["neglect"], c["optimizer"], c["sampled"], c["coupled"])


def curve(run, grid, metric="posterior_kl", axis="wake_equivalent"):
    rows = run["rows"][1:]
    return np.exp(np.interp(np.log(grid), np.log([r[axis] for r in rows]), np.log([max(r[metric], 1e-12) for r in rows])))


def matrix(runs, grid, metric="posterior_kl", axis="wake_equivalent"):
    return np.stack([curve(r, grid, metric, axis) for r in sorted(runs, key=lambda r: r["config"]["seed"])])


def plot_group(ax, group, metric="posterior_kl"):
    unit = group[0]["costs"]["wake"]
    grid = np.geomspace(16, 16384, 100)
    for arm in ("elbo", "sleep", "forward"):
        ys = matrix([r for r in group if r["config"]["arm"] == arm], grid, metric)
        mean, err = ys.mean(axis=0), ys.std(axis=0, ddof=1) / np.sqrt(len(ys))
        ax.plot(grid * unit, mean, label=LABELS[arm], color=COLORS[arm])
        ax.fill_between(grid * unit, np.maximum(mean - err, 1e-12), mean + err, color=COLORS[arm], alpha=.15)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Allocated dense FLOPs (approx.)")
    ax.set_ylabel("Held-out posterior KL" if metric == "posterior_kl" else "Generator marginal KL")
    ax.grid(True, alpha=.2)


def main():
    mainruns = read_stage("confirmation")
    followup = read_stage("rate_followup")
    replacement = {r["config"]["name"]: r for r in followup}
    runs = [replacement[r["config"]["name"]] if r["config"]["name"] in replacement else r for r in mainruns]
    groups = {k: [r for r in runs if key(r["config"]) == k] for k in sorted({key(r["config"]) for r in runs})}
    rng = np.random.default_rng(9599)
    bootstrap = rng.integers(6, size=(2000, 6))
    summaries = []
    for k, group in groups.items():
        stats = {"group": list(k), "budgets": {}, "slopes": {}}
        for budget in (1024, 4096, 16384):
            v = {arm: matrix([r for r in group if r["config"]["arm"] == arm], [budget])[:, 0] for arm in ("elbo", "sleep", "forward")}
            ratios = v["elbo"][bootstrap].mean(axis=1) / v["sleep"][bootstrap].mean(axis=1)
            stats["budgets"][str(budget)] = {"means": {a: float(b.mean()) for a, b in v.items()},
                "elbo_over_sleep": float(v["elbo"].mean() / v["sleep"].mean()), "ratio_ci": np.quantile(ratios, [.025, .975]).tolist(),
                "sleep_wins": int(np.sum(v["sleep"] < v["elbo"]))}
        for lo, hi in ((256, 2048), (1024, 8192), (4096, 16384)):
            grid = np.geomspace(lo, hi, 25)
            stats["slopes"][f"{lo}-{hi}"] = {arm: float(-np.polyfit(np.log(grid), np.log(matrix([r for r in group if r["config"]["arm"] == arm], grid).mean(axis=0)), 1)[0]) for arm in ("elbo", "sleep", "forward")}
        summaries.append(stats)
    for sampled, filename in ((False, "deep-wake-sleep-population.png"), (True, "deep-wake-sleep-sampled.png")):
        fig, axs = plt.subplots(1, 2, layout="constrained")
        for ax, depth in zip(axs, (1, 3), strict=True):
            group = groups[(depth, depth, 6., "adam", sampled, False)]
            plot_group(ax, group); ax.set_title(f"{depth} layer(s), initial neglect = 6")
        axs[0].legend(); fig.savefig(DEST / filename); plt.close(fig)
    fig, axs = plt.subplots(1, 2, layout="constrained")
    for ax, neglect in zip(axs, (0., 6.), strict=True):
        plot_group(ax, groups[(3, 3, neglect, "adam", True, True)], "marginal_kl")
        ax.set_title(f"Coupled deep model, neglect = {neglect:g}")
    axs[0].legend(); fig.savefig(DEST / "deep-wake-sleep-coupled.png"); plt.close(fig)
    size_runs = read_stage("state_scaling")
    size_summary = []
    fig, axs = plt.subplots(1, 2, layout="constrained")
    for ax, depth in zip(axs, (1, 2), strict=True):
        for arm in ("elbo", "sleep"):
            xs, medians = [], []
            for bits in (4, 6, 8, 10):
                matches = [r for r in size_runs if len(r["config"]["layers"]) == depth and sum(r["config"]["layers"]) == bits and r["config"]["arm"] == arm]
                times, censored = [], []
                for r in matches:
                    hit = [v for v in r["rows"] if v["posterior_kl"] <= .1]
                    times.append(hit[0]["compute"] if hit else r["config"]["budget"] * r["costs"]["wake"])
                    censored.append(not bool(hit))
                xs.append(2**bits); medians.append(np.nan if any(censored) else float(np.median(times)))
                for value, failed in zip(times, censored, strict=True): ax.scatter(2**bits, value, color=COLORS[arm], marker="^" if failed else "o")
                size_summary.append({"depth": depth, "states": 2**bits, "arm": arm, "compute_to_kl_0p1_or_limit": times, "censored": censored})
            ax.plot(xs, medians, color=COLORS[arm], label=LABELS[arm])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Latent configurations"); ax.set_ylabel("FLOPs to posterior KL ≤ 0.1")
        ax.set_title(f"{depth} latent layer(s)"); ax.grid(True, alpha=.2)
    fig.suptitle("Triangles mark budget limits, not target attainment")
    axs[0].legend(); fig.savefig(DEST / "deep-wake-sleep-states.png"); plt.close(fig)
    result = {"confirmation_runs": len(mainruns), "rate_followup_runs": len(followup), "state_scaling_runs": len(size_runs),
              "note": "Mean +/- one seed SE; stationary or converged runs held at final error for larger allocated budgets, not claimed to receive additional training. Slope fits include floors/crossovers and are not asymptotic exponents.",
              "groups": summaries, "state_scaling": size_summary}
    (RESULTS / "analysis.json").write_text(json.dumps(result, indent=2))
    for s in summaries:
        if s["group"] in ([3,3,6.,"adam",False,False], [3,3,6.,"adam",True,False], [3,3,6.,"adam",True,True], [1,1,6.,"adam",False,False]): print(json.dumps(s))
    print(json.dumps({"state_scaling": size_summary}))


if __name__ == "__main__": main()
