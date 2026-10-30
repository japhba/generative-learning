"""Paired cross-model comparison of the frozen Zipf ICL sample exponents."""

import argparse
import json
from pathlib import Path

from analyze_zipf import CONDITIONS, FIT_WINDOWS, LABELS, effective_fit, plt, np

MODEL_LABELS = {"google/gemini-2.5-flash-lite": "Gemini 2.5\nFlash Lite", "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3 30B\nirregular curve", "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2\n24B"}


def compare(run_dirs, output, bootstrap_samples=1500, seed=20260905):
    rows = [json.loads(line) for directory in run_dirs for line in (directory / "results.jsonl").read_text().splitlines()]
    models = sorted({row["model"] for row in rows})
    worlds = sorted({row["world_seed"] for row in rows})
    budgets = np.array(sorted({row["budget"] for row in rows}))
    lookup = {(row["model"], row["world_seed"], row["condition"], row["budget"]): row for row in rows}
    assert len(lookup) == len(rows) == len(models) * len(worlds) * len(CONDITIONS) * len(budgets)
    for world in worlds:
        reference = lookup[models[0], world, CONDITIONS[0], int(budgets[0])]
        for model in models:
            for condition in CONDITIONS:
                for budget in budgets:
                    row = lookup[model, world, condition, int(budget)]
                    assert all(row[key] == reference[key] for key in ("keys", "test_weights", "targets"))
    errors = {model: {condition: np.array([[lookup[model, world, condition, int(budget)]["error"] for budget in budgets] for world in worlds]) for condition in CONDITIONS} for model in models}
    indices = np.random.default_rng(seed).integers(0, len(worlds), size=(bootstrap_samples, len(worlds)))
    reference_model = "google/gemini-2.5-flash-lite"
    result = {"world_seeds": worlds, "bootstrap_samples": bootstrap_samples, "reference_model": reference_model, "interpretation": "Same-world paired cross-model effective exponents. Similarity of point estimates or an interval covering zero does not establish exponent equivalence. Windows summarize finite ranges; model failures and nonmonotonic curves remain included.", "windows": []}
    primary = None
    for minimum, maximum in FIT_WINDOWS:
        fitted, draws = {}, {}
        for model in models:
            fitted[model], draws[model] = {}, {}
            for condition in CONDITIONS:
                fitted[model][condition], draws[model][condition] = effective_fit(budgets, errors[model][condition], indices, minimum, maximum)
        contrasts = {}
        for model in models:
            if model == reference_model: continue
            contrasts[model] = {}
            for condition in CONDITIONS:
                delta = draws[model][condition] - draws[reference_model][condition]
                assert np.all(np.isfinite(delta))
                contrasts[model][condition] = {"mean": fitted[model][condition]["alpha"] - fitted[reference_model][condition]["alpha"], "ci95": np.quantile(delta, [.025, .975]).tolist()}
        result["windows"].append({"budget_range": [minimum, maximum], "models": fitted, "exponent_minus_reference": contrasts})
        if minimum == 8 and maximum == 128: primary = fitted
    figure, axis = plt.subplots()
    positions = np.arange(len(models))
    for condition, offset in zip(CONDITIONS, [-.2, 0, .2], strict=True):
        estimates = np.array([primary[model][condition]["alpha"] for model in models])
        limits = np.array([primary[model][condition]["ci95"] for model in models])
        axis.errorbar(positions + offset, estimates, yerr=np.array([estimates - limits[:, 0], limits[:, 1] - estimates]), fmt="o", label=LABELS[condition])
    axis.set_xticks(positions, [MODEL_LABELS[model] for model in models])
    axis.set(ylabel="Effective sample exponent, 8–128 examples", title="Cross-model replication: 32 paired dictionaries")
    axis.legend(loc="upper left")
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "cross_model_exponents.png")
    plt.close(figure)
    (output / "cross_model_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output / "cross_model_exponents.png")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.run_dirs, args.output)


if __name__ == "__main__": main()
