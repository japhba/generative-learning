"""Analyze immutable local probes; update plots as confirmations finish."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/bestn_b2k"
REPORTS = ROOT / "reports"


def interval(values, seed):
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(values), size=(10000, len(values)))
    return np.quantile(np.asarray(values)[samples].mean(axis=1), [.025, .975]).tolist()


def local_results():
    records = []
    for regime in ("low", "high"):
        root = RESULTS / "success_probe" / regime / "run"
        measurements = json.loads((root / "complete.json").read_text())["measurements"]
        whole = {r["n"]: r for r in measurements if r["fraction"] == 1}
        null = np.array(whole[1]["importance"]["per_prompt_delta"])
        for n in (1, 2, 4):
            raw = np.array(whole[n]["importance"]["per_prompt_delta"])
            corrected = raw - null / n
            records.append({"regime": regime, "n": n, "forecast": whole[n]["forecast_success_gain"],
                            "actual": float(raw.mean()), "actual_ci95": interval(raw, 87001),
                            "corrected": float(corrected.mean()), "corrected_ci95": interval(corrected, 87002),
                            "fresh_original": whole[n]["fresh"]["success"], "ess_fraction": whole[n]["importance"]["ess_fraction"]})
    for corrected in (False, True):
        rows = [r for r in records if r["n"] > 1]
        fig, ax = plt.subplots(layout="constrained")
        x = np.arange(len(rows))
        mean = np.array([r["corrected" if corrected else "actual"] for r in rows]) * 100
        ci = np.array([r["corrected_ci95" if corrected else "actual_ci95"] for r in rows]).T * 100
        ax.plot(x, [r["forecast"] * 100 for r in rows], "x", label="Forecast from independent bank")
        ax.errorbar(x, mean, yerr=np.stack([mean - ci[0], ci[1] - mean]), fmt="o", label="Measured importance estimate")
        ax.set_xticks(x, [("Hard" if r["regime"] == "low" else "Easy") + f"\nN={r['n']}" for r in rows])
        ax.set_ylabel("Success gain (percentage points)")
        ax.set_title("Post-hoc selection diagnostic" if corrected else "FP32 LLM: total gains at equal rollouts")
        ax.legend()
        path = REPORTS / ("bestn-local-selection.png" if corrected else "bestn-local-total.png")
        fig.savefig(path)
        plt.close(fig)
        print(path)
    return records


def fresh_results():
    records = []
    for regime in ("low", "high"):
        paths = {n: RESULTS / f"fresh_confirmation/{regime}_n{n}/independent.json" for n in (0, 1, 2, 4)}
        if not all(path.exists() for path in paths.values()): continue
        data = {n: json.loads(path.read_text()) for n, path in paths.items()}
        ids = sorted(data[0]["per_prompt"])
        values = {n: np.array([d["per_prompt"][key] for key in ids]) for n, d in data.items()}
        for n in (1, 2, 4):
            raw = values[n] - values[0]
            corrected = raw - (values[1] - values[0]) / n
            records.append({"regime": regime, "n": n, "baseline": data[0]["success"], "success": data[n]["success"],
                            "gain": float(raw.mean()), "gain_ci95": interval(raw, 87101),
                            "corrected_gain": float(corrected.mean()), "corrected_gain_ci95": interval(corrected, 87102)})
    fig, ax = plt.subplots(layout="constrained")
    x = np.arange(len(records))
    means = np.array([r["gain"] for r in records]) * 100
    limits = np.array([r["gain_ci95"] for r in records]).T * 100
    ax.errorbar(x, means, yerr=np.stack([means - limits[0], limits[1] - means]), fmt="o", label="Independent fresh samples")
    forecasts = {(r["regime"], r["n"]): r["forecast"] for r in local_results_data()}
    ax.plot(x, [100 * forecasts[r["regime"], r["n"]] for r in records], "x", label="Pre-update gradient forecast")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xticks(x, [("Hard" if r["regime"] == "low" else "Easy") + f"\nN={r['n']}" for r in records])
    ax.set_ylabel("Test accuracy gain (percentage points)")
    ax.set_title("One SGD update: 1,024 fresh responses per condition")
    ax.legend()
    path = REPORTS / "bestn-fresh.png"
    fig.savefig(path)
    plt.close(fig)
    print(path)
    return records


def local_results_data():
    return [{"regime": regime, "n": r["n"], "forecast": r["forecast_success_gain"]}
            for regime in ("low", "high")
            for r in json.loads((RESULTS / f"success_probe/{regime}/run/complete.json").read_text())["measurements"] if r["fraction"] == 1]


def paired_endpoint(histories, step):
    rows = []
    for seed in (86001, 86002, 86003):
        pair = {h["source"]: h for h in histories if h["seed"] == seed}
        if len(pair) < 2: continue
        evaluations = {source: next((r for r in h["evaluations"] if r["step"] == step), None) for source, h in pair.items()}
        if any(r is None for r in evaluations.values()): continue
        ids = sorted(evaluations["current"]["per_prompt"])
        assert ids == sorted(evaluations["frozen"]["per_prompt"])
        values = {source: np.array([r["per_prompt"][key] for key in ids]) for source, r in evaluations.items()}
        delta = values["current"] - values["frozen"]
        rows.append({"seed": seed, "current": float(values["current"].mean()), "frozen": float(values["frozen"].mean()),
                     "delta": float(delta.mean()), "ci95": interval(delta, 87301 + seed), "per_prompt_delta": delta.tolist()})
    if not rows: return None
    rng = np.random.default_rng(87301)
    draws = []
    for row in rows:
        values = np.array(row["per_prompt_delta"])
        draws.append(values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1))
    draws = np.array(draws).T
    fixed = draws.mean(axis=1)
    hierarchical = np.take_along_axis(draws, rng.integers(len(rows), size=draws.shape), axis=1).mean(axis=1)
    return {"step": step, "candidates": 32 * step, "seeds": rows,
            "current": float(np.mean([r["current"] for r in rows])), "frozen": float(np.mean([r["frozen"] for r in rows])),
            "delta": float(np.mean([r["delta"] for r in rows])), "ci95_fixed_runs": np.quantile(fixed, [.025, .975]).tolist(),
            "ci95_hierarchical": np.quantile(hierarchical, [.025, .975]).tolist(),
            "uncertainty": "Fixed-run interval resamples whole evaluation questions within each run; hierarchical interval additionally resamples the training seeds. Only three training seeds are available."}


def detailed_training(histories):
    common = set.intersection(*[{r["step"] for r in h["evaluations"]} for h in histories])
    comparisons = [paired_endpoint(histories, step) for step in sorted(common)]
    forecasts, costs, matching = [], [], []
    for h in histories:
        for f in h["forecasts"]:
            before = next(r for r in h["evaluations"] if r["step"] == f["step"])
            after = next(r for r in h["evaluations"] if r["step"] == f["end_step"])
            forecasts.append({"seed": h["seed"], "source": h["source"], "step": f["step"], "end_step": f["end_step"],
                              "forecast": f["predicted_success_gain"], "importance_gain": f["measured_importance_gain"]["delta"],
                              "fresh_gain": after["accuracy"] - before["accuracy"], "ess_fraction": f["measured_importance_gain"]["ess_fraction"],
                              "written_before_updates_at_uk": f["written_before_updates_at_uk"]})
        for e in h["evaluations"]:
            updates = [u for u in h["updates"] if u["step"] <= e["step"]]
            rollout_tokens = sum(u["usage"]["prompt_tokens"] + u["usage"]["completion_tokens"] for u in updates)
            selected_completion_tokens = 0
            for u in updates:
                bank = [json.loads(line) for line in Path(u["path"]).read_text().splitlines()]
                chosen = {(r["id"], r["sample"]) for r in u["selected"]}
                selected_completion_tokens += sum(r["completion_tokens"] for r in bank if (r["id"], r["sample"]) in chosen)
            training_input_tokens = selected_completion_tokens + sum(u["usage"]["prompt_tokens"] / 2 for u in updates)
            costs.append({"seed": h["seed"], "source": h["source"], "step": e["step"], "candidates": e["candidates"],
                          "accuracy": e["accuracy"], "rollout_tokens": rollout_tokens, "selected_completion_tokens": selected_completion_tokens,
                          "training_input_tokens": training_input_tokens, "compute_proxy": rollout_tokens + 3 * training_input_tokens})
    for seed in (86001, 86002, 86003):
        pair = {h["source"]: h for h in histories if h["seed"] == seed}
        if len(pair) < 2 or any(not h["updates"] for h in pair.values()): continue
        first = {source: h["updates"][0] for source, h in pair.items()}
        candidates = {source: [json.loads(line) for line in Path(u["path"]).read_text().splitlines()] for source, u in first.items()}
        same_candidates = candidates["current"] == candidates["frozen"]
        same_selection = first["current"]["selected"] == first["frozen"]["selected"]
        assert same_candidates and same_selection, "The paired sources must start from identical candidate and target banks."
        matching.append({"seed": seed, "first_candidates_identical": same_candidates, "first_selection_identical": same_selection,
                         "first_loss_difference": first["current"]["loss"] - first["frozen"]["loss"]})
    summaries = []
    for source in ("current", "frozen"):
        for seeds, scope in (((86001, 86002, 86003), "all"), ((86002, 86003), "confirmation")):
            rows = [r for r in forecasts if r["source"] == source and r["seed"] in seeds]
            if not rows: continue
            for measurement in ("importance_gain", "fresh_gain"):
                actual = np.array([r[measurement] for r in rows])
                predicted = np.array([r["forecast"] for r in rows])
                summaries.append({"source": source, "scope": scope, "measurement": measurement, "blocks": len(rows),
                                  "mae": float(np.abs(actual - predicted).mean()), "rmse": float(np.sqrt(np.mean((actual - predicted)**2))),
                                  "zero_forecast_rmse": float(np.sqrt(np.mean(actual**2))), "bias_actual_minus_forecast": float(np.mean(actual - predicted)),
                                  "sign_matches": int(np.sum(np.sign(actual) == np.sign(predicted)))})
    fig, ax = plt.subplots(layout="constrained")
    for index, source in enumerate(("current", "frozen")):
        for seed in (86001, 86002, 86003):
            rows = [r for r in costs if r["source"] == source and r["seed"] == seed]
            if rows: ax.plot([r["compute_proxy"] for r in rows], [r["accuracy"] for r in rows], "o-", color=f"C{index}", alpha=.5,
                             label=source.capitalize() + " source" if seed == 86001 else None)
    ax.set_xlabel("Token compute proxy: generation + 3 × training input")
    ax.set_ylabel("Held-out sampled accuracy")
    ax.set_title("Actual token lengths; diagnostics excluded from this axis")
    ax.legend()
    path = REPORTS / "bestn-training-compute.png"
    fig.savefig(path)
    plt.close(fig)
    print(path)
    fig, ax = plt.subplots(layout="constrained")
    for index, source in enumerate(("current", "frozen")):
        rows = [r for r in forecasts if r["source"] == source]
        ax.scatter([100 * r["forecast"] for r in rows], [100 * r["importance_gain"] for r in rows], color=f"C{index}", label=source.capitalize())
    values = [100 * r[k] for r in forecasts for k in ("forecast", "importance_gain")]
    ax.plot([min(values), max(values)], [min(values), max(values)], color="gray", linestyle="--")
    ax.set_xlabel("Forecast gain, saved before updates (percentage points)")
    ax.set_ylabel("Measured importance gain (percentage points)")
    ax.set_title("Prospective four-update forecasts")
    ax.legend()
    path = REPORTS / "bestn-block-forecasts.png"
    fig.savefig(path)
    plt.close(fig)
    print(path)
    return {"common_budget_comparisons": comparisons, "forecasts": forecasts, "forecast_summary": summaries,
            "costs": costs, "first_update_matching": matching, "all_six_complete": all((RESULTS / f"training/{seed}/{source}/complete.json").exists()
                for seed in (86001, 86002, 86003) for source in ("current", "frozen"))}


def final_evaluation():
    root = RESULTS / "final_evaluation"
    records = [json.loads(p.read_text()) for p in sorted(root.glob("*/complete.json"))]
    if len(records) != 7: return {"complete": False, "completed_conditions": len(records), "records": records}
    baseline = next(r for r in records if r["source"] == "baseline")
    ids = sorted(baseline["per_prompt"])
    values = {(r["seed"], r["source"]): np.array([r["per_prompt"][key]["accuracy"] for key in ids]) for r in records}
    seeds = (86001, 86002, 86003)
    delta = np.stack([values[seed, "current"] - values[seed, "frozen"] for seed in seeds])
    rng = np.random.default_rng(890201)
    prompt_draws = rng.integers(len(ids), size=(10000, len(ids)))
    seed_draws = rng.integers(len(seeds), size=(10000, len(seeds)))
    fixed_draws = delta.mean(axis=0)[prompt_draws].mean(axis=1)
    hierarchical_draws = delta[seed_draws[:, :, None], prompt_draws[:, None, :]].mean(axis=(1, 2))
    summary = {"step": 16, "candidates": 512, "tasks": len(ids), "samples_per_task": 8,
               "baseline": baseline["accuracy"], "current": float(np.mean([values[s, "current"].mean() for s in seeds])),
               "frozen": float(np.mean([values[s, "frozen"].mean() for s in seeds])), "delta": float(delta.mean()),
               "ci95_fixed_runs": np.quantile(fixed_draws, [.025, .975]).tolist(),
               "ci95_hierarchical": np.quantile(hierarchical_draws, [.025, .975]).tolist(),
               "seeds": [{"seed": seed, "current": float(values[seed, "current"].mean()), "frozen": float(values[seed, "frozen"].mean()),
                          "delta": float(delta[i].mean()), "ci95": interval(delta[i], 890202 + i)} for i, seed in enumerate(seeds)],
               "uncertainty": "The same128 unseen questions are shared across all conditions. Whole-question resampling preserves this pairing. The hierarchical interval additionally resamples the three training seeds."}
    fig, ax = plt.subplots(layout="constrained")
    for index, source in enumerate(("baseline", "frozen", "current")):
        per_prompt = values[0, "baseline"] if source == "baseline" else np.mean([values[s, source] for s in seeds], axis=0)
        mean = per_prompt.mean()
        limits = np.quantile(per_prompt[prompt_draws].mean(axis=1), [.025, .975])
        ax.errorbar(index, mean, yerr=np.array([[mean - limits[0]], [limits[1] - mean]]), fmt="o", color=f"C{index}")
    for seed in seeds:
        ax.plot([1, 2], [values[seed, "frozen"].mean(), values[seed, "current"].mean()], color="gray", alpha=.4)
    ax.set_xticks([0, 1, 2], ["Untrained", "Frozen source", "Current source"])
    ax.set_ylabel("Accuracy on untouched test questions")
    ax.set_title("512 training rollouts; 128 new questions × 8 responses")
    path = REPORTS / "bestn-final-test.png"
    fig.savefig(path)
    plt.close(fig)
    print(path)
    return {"complete": True, "completed_conditions": 7, "records": records, "summary": summary}


def training_results():
    histories = []
    for seed in (86001, 86002, 86003):
        for source in ("current", "frozen"):
            root = RESULTS / "training" / str(seed) / source
            path = root / "complete.json"
            if not path.exists(): path = root / "progress.json"
            if path.exists(): histories.append(json.loads(path.read_text()))
    if histories:
        fig, ax = plt.subplots(layout="constrained")
        for index, source in enumerate(("current", "frozen")):
            rows = [h for h in histories if h["source"] == source]
            for row in rows:
                ax.plot([e["candidates"] for e in row["evaluations"]], [e["accuracy"] for e in row["evaluations"]],
                        color=f"C{index}", alpha=.5, label=source.capitalize() + " source" if row == rows[0] else None)
        ax.set_xlabel("Training candidate rollouts")
        ax.set_ylabel("Held-out sampled accuracy")
        ax.set_title("Qwen2.5-1.5B: current versus frozen source")
        ax.legend()
        path = REPORTS / "bestn-training.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)
    return histories


def main():
    result = {"local": local_results(), "fresh": fresh_results(), "training": training_results()}
    result["training_analysis"] = detailed_training(result["training"])
    result["final_evaluation"] = final_evaluation()
    (RESULTS / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
