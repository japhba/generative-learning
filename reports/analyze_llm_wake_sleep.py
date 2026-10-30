"""Paired-seed, matched-compute analysis of the actual trained LLM adapters."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("elbo", "sleep", "forward")
LABELS = {"elbo": "ELBO", "sleep": "ELBO + sleep", "forward": "Exact posterior teaching"}


def interpolate(run, grid, key):
    rows = run["rows"][1:]
    return np.interp(np.log(grid), np.log([r["flops"] for r in rows]), [r[key] for r in rows])


def main():
    directory = ROOT / "results/llm_wake_sleep"
    runs = {}
    for arm in ARMS:
        runs[arm] = []
        for seed in (9610, 9611, 9612):
            path = directory / f"confirmation_{arm}_s{seed}"
            run = json.loads((path / "run.json").read_text())
            assert hashlib.sha256((path / "runtime.py").read_bytes()).hexdigest() == run["protocol"]["source_sha256"]
            assert run["rows"][-1]["step"] == 512 or run["stop_reason"] != "budget", f"Unfinished {path}"
            resume = json.loads((path / "resume.json").read_text())
            assert Path(resume["persistent"]).exists()
            runs[arm].append(run)
    maximum = min(r["rows"][-1]["flops"] for group in runs.values() for r in group)
    minimum = max(r["rows"][1]["flops"] for group in runs.values() for r in group)
    grid = np.geomspace(minimum, maximum, 100)
    result = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "seeds": [9610,9611,9612], "matched_compute_maximum": maximum,
              "scope": "Coupled rank-four adapters, normalized finite output supports, temperature one, no latent labels/checker in updates; exploratory task, independent optimization seeds.",
              "curves": {}, "final_at_matched_compute": {}, "target_times": {}, "effective_slopes": {}}
    fig, axs = plt.subplots(1, 2, layout="constrained")
    for arm, color in zip(ARMS, ("C0", "C1", "C2"), strict=True):
        metrics = {}
        for key in ("compatible_mass", "accuracy", "posterior_kl", "generator_correct_probability", "marginal_nll"):
            ys = np.stack([interpolate(r, grid, key) for r in runs[arm]])
            metrics[key] = ys
        result["curves"][arm] = {"flops": grid.tolist(), **{k:v.tolist() for k,v in metrics.items()}}
        result["final_at_matched_compute"][arm] = {k: {"mean": float(v[:, -1].mean()), "per_seed": v[:, -1].tolist()} for k,v in metrics.items()}
        for ax, key in zip(axs, ("compatible_mass", "posterior_kl"), strict=True):
            ys = 1 - metrics[key] if key == "compatible_mass" else metrics[key]
            mean, se = ys.mean(axis=0), ys.std(axis=0, ddof=1) / np.sqrt(3)
            ax.plot(grid, mean, color=color, label=LABELS[arm]); ax.fill_between(grid, mean-se, mean+se, color=color, alpha=.15)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Training FLOPs (approx.)"); ax.grid(True, alpha=.2)
        target_times = {}
        for target in (.5, .7, .75):
            target_times[str(target)] = []
            for r in runs[arm]:
                hit = [v for v in r["rows"] if v["compatible_mass"] >= target]
                target_times[str(target)].append({"flops": hit[0]["flops"] if hit else None, "last_observed_flops": r["rows"][-1]["flops"]})
        result["target_times"][arm] = target_times
        unit = runs["elbo"][0]["rows"][1]["flops"]
        result["effective_slopes"][arm] = {}
        for lo, hi in ((32,128), (64,256), (128,512)):
            end = min(hi*unit, maximum)
            x = np.geomspace(lo*unit, end, 25)
            error = 1 - np.stack([interpolate(r, x, "compatible_mass") for r in runs[arm]]).mean(axis=0)
            result["effective_slopes"][arm][f"{lo}-{end/unit:g} wake units"] = float(-np.polyfit(np.log(x), np.log(error), 1)[0])
    axs[0].set_ylabel("Probability of incompatible rule"); axs[1].set_ylabel("True-posterior KL")
    fig.legend(*axs[0].get_legend_handles_labels(),loc="outside lower center",ncol=3)
    fig.suptitle("Qwen2.5-1.5B: three paired optimization seeds")
    fig.savefig(ROOT / "reports/llm-wake-sleep-learning.png"); plt.close(fig)
    rng = np.random.default_rng(9690); samples = rng.integers(3, size=(4000, 3))
    for fraction in (.25, .5, 1.):
        c = fraction * maximum
        e = {a: 1 - np.array([interpolate(r, [c], "compatible_mass")[0] for r in runs[a]]) for a in ARMS}
        ratio = e["elbo"][samples].mean(axis=1) / e["sleep"][samples].mean(axis=1)
        result[f"budget_fraction_{fraction:g}"] = {"flops": c, "elbo_error": float(e["elbo"].mean()), "sleep_error": float(e["sleep"].mean()),
            "error_ratio": float(e["elbo"].mean()/e["sleep"].mean()), "ratio_seed_ci": np.quantile(ratio,[.025,.975]).tolist(), "sleep_wins": int((e["sleep"]<e["elbo"]).sum())}
    (directory / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "curves"}, indent=2))


if __name__ == "__main__": main()
