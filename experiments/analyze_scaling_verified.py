"""Prespecified sample-efficiency comparisons, without requiring exponent changes."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


THRESHOLDS = (.1, .05, .02, .01, .005, .002)


def crossing(x, y, threshold):
    indices = np.flatnonzero(y <= threshold)
    if not len(indices): return {"status": "not_reached", "lower_bound": float(x[-1])}
    index = indices[0]
    if index == 0: return {"status": "before_first_positive_checkpoint", "upper_bound": float(x[0])}
    fraction = np.log(threshold / y[index - 1]) / np.log(y[index] / y[index - 1])
    return {"status": "interpolated", "budget": float(np.exp(np.log(x[index - 1]) + fraction * np.log(x[index] / x[index - 1])))}


def read_runs(output):
    protocol = json.loads((output / "protocol.json").read_text())
    runs = []
    for cell in protocol["cells"]:
        root = output / str(cell["seed"]) / cell["name"]
        path = root / "complete.json"
        complete = path.exists()
        if not complete: path = root / "progress.json"
        if path.exists(): runs.append({"cell": cell, "complete": complete, "history": json.loads(path.read_text())})
    return protocol, runs


def pilot(output):
    protocol, runs = read_runs(output)
    fig, ax = plt.subplots()
    summaries = []
    styles = ("-", "--", ":")
    rates = sorted({c["eta"] for c in protocol["cells"]})
    for run in runs:
        cell, history = run["cell"], run["history"]
        records = [r for r in history["evaluations"] if r["step"] > 0]
        x = np.array([64 * r["step"] for r in records]); y = np.array([r["error"] for r in records])
        ax.loglog(x, y, linestyle=styles[rates.index(cell["eta"])], color="C0" if cell["source"] == "current" else "C1", label=f"{cell['source']}, rate {cell['eta']:.1e}")
        eligible = run["complete"] and history["step"] == protocol["steps"]
        auc = None
        if eligible:
            keep = x >= 128 * 64
            auc = float(np.trapezoid(np.log(y[keep]), np.log(x[keep])) / np.log(x[keep][-1] / x[keep][0]))
        summaries.append({**cell, "complete": run["complete"], "steps": history["step"], "final_error": float(y[-1]), "log_error_auc": auc, "eligible": eligible,
                          "crossings": {str(t): crossing(x, y, t) for t in THRESHOLDS}})
    ax.set(xlabel="Generated candidate tokens", ylabel="Monitoring error probability", title="Verified-only pilot: all rates and both sources")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-pilot.png"); plt.close(fig)
    result = {"updated_at_uk": stamp(), "stage": "exploratory pilot", "runs": summaries}
    if len(runs) == len(protocol["cells"]) and all(r["complete"] for r in runs):
        result["selected"] = {source: min((r for r in summaries if r["source"] == source and r["eligible"]), key=lambda r: r["log_error_auc"])["eta"] for source in ("current", "frozen")}
    save(output / "analysis.json", result); print(json.dumps(result, indent=2))


def tensors(runs, split):
    sources = ("current", "frozen")
    seeds = sorted({r["cell"]["seed"] for r in runs})
    curves = {}
    for source in sources:
        selected = sorted([r for r in runs if r["cell"]["source"] == source], key=lambda r: r["cell"]["seed"])
        assert [r["cell"]["seed"] for r in selected] == seeds
        if split == "monitor": records = [r["history"]["evaluations"] for r in selected]
        else: records = [r["test"]["evaluations"] for r in selected]
        steps = sorted(set.intersection(*[{r["step"] for r in rows} for rows in records]))
        rows = [[next(r for r in cohort if r["step"] == step) for step in steps] for cohort in records]
        ids = [r["id"] for r in rows[0][0]["records"]]
        assert all([r["id"] for r in row["records"]] == ids for cohort in rows for row in cohort)
        error = np.array([[[r["error"] for r in row["records"]] for row in cohort] for cohort in rows])
        greedy = np.array([[[not r["greedy_correct"] for r in row["records"]] for row in cohort] for cohort in rows])
        curves[source] = {"steps": steps, "ids": ids, "errors": error, "greedy": greedy}
    common = sorted(set(curves["current"]["steps"]) & set(curves["frozen"]["steps"]))
    assert curves["current"]["ids"] == curves["frozen"]["ids"]
    error = np.stack([curves[s]["errors"][:, [curves[s]["steps"].index(step) for step in common]] for s in sources])
    greedy = np.stack([curves[s]["greedy"][:, [curves[s]["steps"].index(step) for step in common]] for s in sources])
    return seeds, np.array(common), error, greedy


def bootstrap(steps, error, greedy, labels, repetitions=2000):
    rng = np.random.default_rng(198271)
    strata = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    weights = np.empty(len(labels))
    for indices in strata: weights[indices] = len(labels) / (len(strata) * len(indices))
    error = error * weights[None, None, None, :]; greedy = greedy * weights[None, None, None, :]
    mean = error.mean(axis=(1, 3)); gmean = greedy.mean(axis=(1, 3))
    draws = np.empty((repetitions, 2, len(steps))); greedy_draws = np.empty_like(draws)
    for i in tqdm(range(repetitions), desc="Paired training-seed / held-out-question bootstrap"):
        seeds = rng.integers(error.shape[1], size=error.shape[1])
        questions = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        draws[i] = error[:, seeds][:, :, :, questions].mean(axis=(1, 3))
        greedy_draws[i] = greedy[:, seeds][:, :, :, questions].mean(axis=(1, 3))
    pointwise = []
    for index, step in enumerate(steps):
        ratios = draws[:, 1, index] / draws[:, 0, index]
        pointwise.append({"step": int(step), "candidates": int(64 * step), "current_error": float(mean[0, index]), "frozen_error": float(mean[1, index]),
                          "frozen_over_current": float(mean[1, index] / mean[0, index]), "ratio_ci95": np.quantile(ratios, [.025, .975]).tolist(),
                          "current_greedy_error": float(gmean[0, index]), "frozen_greedy_error": float(gmean[1, index]),
                          "greedy_error_difference": float(gmean[1, index] - gmean[0, index]),
                          "greedy_difference_ci95": np.quantile(greedy_draws[:, 1, index] - greedy_draws[:, 0, index], [.025, .975]).tolist()})
    positive = steps > 0; x = 64 * steps[positive]
    crossings = []
    for threshold in THRESHOLDS:
        current, frozen = [crossing(x, mean[i, positive], threshold) for i in range(2)]
        record = {"error": threshold, "current": current, "frozen": frozen}
        if current["status"] == frozen["status"] == "interpolated":
            values = []
            for draw in draws:
                a, b = [crossing(x, draw[i, positive], threshold) for i in range(2)]
                if a["status"] == b["status"] == "interpolated": values.append(b["budget"] / a["budget"])
            record.update({"candidate_speedup": frozen["budget"] / current["budget"], "uncensored_bootstraps": len(values)})
            if len(values) == repetitions: record["speedup_ci95"] = np.quantile(values, [.025, .975]).tolist()
        crossings.append(record)
    windows = []
    for start, end in ((128, 1024), (256, 2048), (512, 4096), (1024, 4096), (2048, 4096)):
        keep = (steps >= start) & (steps <= end)
        if start not in steps or end not in steps or keep.sum() < 3: continue
        betas = np.stack([-np.polyfit(np.log(steps[keep]), np.log(draws[:, i, keep]).T, 1)[0] for i in range(2)], axis=1)
        estimates = [-np.polyfit(np.log(steps[keep]), np.log(mean[i, keep]), 1)[0] for i in range(2)]
        windows.append({"start": start, "end": end, "current_beta": float(estimates[0]), "frozen_beta": float(estimates[1]),
                        "current_ci95": np.quantile(betas[:, 0], [.025, .975]).tolist(), "frozen_ci95": np.quantile(betas[:, 1], [.025, .975]).tolist(),
                        "beta_difference": float(estimates[0] - estimates[1]), "difference_ci95": np.quantile(betas[:, 0] - betas[:, 1], [.025, .975]).tolist()})
    cross_budget = []
    for current_step, frozen_step in ((512, 1024), (512, 2048), (512, 4096), (1024, 2048), (1024, 4096)):
        if current_step not in steps or frozen_step not in steps: continue
        a = int(np.flatnonzero(steps == current_step)[0]); b = int(np.flatnonzero(steps == frozen_step)[0])
        delta = draws[:, 1, b] - draws[:, 0, a]
        cross_budget.append({"current_updates": current_step, "frozen_updates": frozen_step, "budget_ratio": frozen_step / current_step,
                             "current_error": float(mean[0, a]), "frozen_error": float(mean[1, b]),
                             "frozen_minus_current_error": float(mean[1, b] - mean[0, a]), "difference_ci95": np.quantile(delta, [.025, .975]).tolist()})
    return {"pointwise": pointwise, "crossings": crossings, "windows": windows, "cross_budget": cross_budget, "curve_ci95": np.quantile(draws, [.025, .975], axis=0).tolist()}


def confirmation(output, split):
    protocol, runs = read_runs(output)
    assert len(runs) == len(protocol["cells"])
    if split != "monitor":
        for run in runs:
            cell = run["cell"]; root = output / str(cell["seed"]) / cell["name"]
            run["test"] = json.loads((root / "test.json").read_text())
    seeds, steps, error, greedy = tensors(runs, "monitor" if split == "monitor" else "test")
    cohort = json.loads((output / ("monitor.json" if split == "monitor" else "test.json")).read_text())
    if split == "strict":
        allowed = {r["id"] for r in json.loads((output / "strict_test.json").read_text())}
        keep = np.array([r["id"] in allowed for r in cohort]); error = error[..., keep]; greedy = greedy[..., keep]
        cohort = [r for r in cohort if r["id"] in allowed]
    labels = np.array([r["label"] for r in cohort])
    result = bootstrap(steps, error, greedy, labels)
    result.update({"updated_at_uk": stamp(), "split": split, "seeds": seeds, "all_training_complete": all(r["complete"] for r in runs),
                   "questions": len(labels), "class_counts": {str(k): int(np.sum(labels == k)) for k in np.unique(labels)},
                   "bootstrap_design": "Resample training seeds and common whole questions within answer strata; preserve method/checkpoint pairing and the designed class balance.",
                   "protocol_sha256": digest(output / "protocol.json"), "analysis_sha256": digest(ROOT / "experiments/analyze_scaling_verified.py")})
    fig, ax = plt.subplots()
    balanced = np.stack([error[..., labels == value].mean(-1) for value in np.unique(labels)], axis=-1)
    balanced_greedy = np.stack([greedy[..., labels == value].mean(-1) for value in np.unique(labels)], axis=-1)
    mean = balanced.mean(axis=(1, 3)); positive = steps > 0; ci = np.array(result["curve_ci95"])
    for i, source in enumerate(("current", "frozen")):
        x = 64 * steps[positive]
        line, = ax.loglog(x, mean[i, positive], "o-", label=f"{source} generator; trained student")
        ax.fill_between(x, ci[0, i, positive], ci[1, i, positive], alpha=.2, color=line.get_color())
        for seed_curve in balanced[i].mean(-1): ax.loglog(x, seed_curve[positive], color=line.get_color(), alpha=.2)
    ax.set(xlabel="Generated candidate tokens", ylabel="Held-out error probability", title=f"Verified-only self-training: {split}, three fresh seeds")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / f"reports/scaling-verified-{split}.png"); plt.close(fig)
    fig, ax = plt.subplots()
    for i, source in enumerate(("current", "frozen")):
        ax.semilogx(64 * steps[positive], balanced_greedy[i].mean(axis=(0, 2))[positive], "o-", label=source)
    ax.set(xlabel="Generated candidate tokens", ylabel="Greedy decision error", title=f"Decision quality, separately from sampling confidence ({split})")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / f"reports/scaling-verified-{split}-greedy.png"); plt.close(fig)
    save(output / f"analysis_{split}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "curve_ci95"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("pilot", "confirmation")); parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("monitor", "test", "strict"), default="monitor")
    args = parser.parse_args(); output = args.output if args.output else ROOT / "results/scaling_r5s" / args.stage
    if args.stage == "pilot": pilot(output)
    else: confirmation(output, args.split)
