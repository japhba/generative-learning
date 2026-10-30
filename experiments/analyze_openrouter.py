"""Plot OpenRouter learning curves with paired, world-cluster bootstrap intervals."""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


CONDITIONS = ("study", "generate_feedback")
LABELS = {"study": "Study", "generate_feedback": "Generate + feedback", "bayes": "Bayes MAP (same examples)"}


def estimate(values, bootstrap_indices):
    """Each row is one independently sampled world; columns are practice budgets."""
    mean = values.mean(axis=0)
    lower, upper = np.quantile(values[bootstrap_indices].mean(axis=1), [0.025, 0.975], axis=0)
    return {"mean": mean, "lower": lower, "upper": upper}


def point_summary(estimates, index):
    return {"mean": float(estimates["mean"][index]), "ci95": [float(estimates["lower"][index]), float(estimates["upper"][index])]}


def draw_curve(axis, x, estimates, label):
    line, = axis.plot(x, estimates["mean"], marker="o", label=label)
    axis.fill_between(x, estimates["lower"], estimates["upper"], color=line.get_color(), alpha=0.2)


def make_plots(run_dir, model, budgets, estimates, mean_prompt_tokens):
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    paths = {name: run_dir / f"{slug}_{name}.png" for name in ("learning_curve", "paired_difference", "context_curve")}

    figure, axis = plt.subplots()
    for condition in (*CONDITIONS, "bayes"):
        draw_curve(axis, budgets, estimates[condition], LABELS[condition])
    axis.set(xlabel="Practice examples with answers or feedback", ylabel="Held-out accuracy", title=model, ylim=(0, 1), xticks=budgets)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["learning_curve"])
    plt.close(figure)

    figure, axis = plt.subplots()
    draw_curve(axis, budgets, estimates["difference"], "Generate + feedback − study")
    axis.axhline(0, color="black", linestyle="--")
    axis.set(xlabel="Practice examples with answers or feedback", ylabel="Paired accuracy difference", title=model, xticks=budgets)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["paired_difference"])
    plt.close(figure)

    figure, axis = plt.subplots()
    for condition in CONDITIONS:
        draw_curve(axis, mean_prompt_tokens[condition], estimates[condition], LABELS[condition])
    axis.set(xlabel="Evaluation prompt tokens (mean per world)", ylabel="Held-out accuracy", title=model, ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["context_curve"])
    plt.close(figure)
    return {name: str(path.resolve()) for name, path in paths.items()}


def analyze(run_dir, bootstrap_samples=5000, seed=20260905):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    worlds = json.loads((run_dir / "worlds.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    models = sorted({row["model"] for row in rows} - {"bayes"})
    if not models:
        raise ValueError("No LLM results found; run the OpenRouter experiment before plotting.")
    indexed = {(row["model"], row["condition"], row["world_seed"], row["budget"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate model/condition/world/budget result rows.")
    summary = {
        "schema_version": 1,
        "run_directory": str(run_dir.resolve()),
        "manifest": manifest,
        "n_worlds_in_manifest": len(worlds),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "world", "interval": "pointwise percentile 95%", "paired": True},
        "estimand": "Mean held-out accuracy over worlds, weighting worlds equally. Entire world learning curves are resampled together.",
        "context_axis": "Observed evaluation prompt tokens, averaged over worlds at each practice budget; this is not total compute or FLOPs.",
        "models": {},
    }
    rng = np.random.default_rng(seed)
    for model in tqdm(models, desc="Analyzing models"):
        model_rows = [row for row in rows if row["model"] == model]
        world_seeds = sorted({row["world_seed"] for row in model_rows})
        budgets = sorted({row["budget"] for row in model_rows})
        required_keys = {(model, condition, world, budget) for condition in CONDITIONS for world in world_seeds for budget in budgets}
        actual_keys = {(row["model"], row["condition"], row["world_seed"], row["budget"]) for row in model_rows}
        if actual_keys != required_keys:
            raise ValueError(f"{model}: incomplete paired world × condition × budget results.")
        if len(world_seeds) < 2:
            raise ValueError(f"{model}: at least two independent worlds are required for uncertainty estimates.")
        groups = {
            condition: [[indexed[(model, condition, world, budget)] for budget in budgets] for world in world_seeds]
            for condition in CONDITIONS
        }
        groups["bayes"] = [[indexed[("bayes", "bayes", world, budget)] for budget in budgets] for world in world_seeds]
        for world_index, world in enumerate(world_seeds):
            targets = groups["study"][world_index][0]["targets"]
            for condition, group in groups.items():
                for row in group[world_index]:
                    if row["targets"] != targets:
                        raise ValueError(f"{model}, world {world}: evaluation targets differ across conditions or budgets.")
                    if len(row["correct"]) != len(targets) or not np.isclose(np.mean(row["correct"]), row["accuracy"]):
                        raise ValueError(f"{model}, world {world}, {condition}: accuracy is inconsistent with per-item correctness.")
        bootstrap_indices = rng.integers(0, len(world_seeds), size=(bootstrap_samples, len(world_seeds)))
        accuracies = {condition: np.array([[row["accuracy"] for row in world_rows] for world_rows in group]) for condition, group in groups.items()}
        estimates = {condition: estimate(values, bootstrap_indices) for condition, values in accuracies.items()}
        estimates["difference"] = estimate(accuracies["generate_feedback"] - accuracies["study"], bootstrap_indices)
        expected_bayes = estimate(np.array([[row["expected_accuracy"] for row in world_rows] for world_rows in groups["bayes"]]), bootstrap_indices)
        prompt_tokens = {condition: np.array([[row["prompt_tokens"] for row in world_rows] for world_rows in groups[condition]]) for condition in CONDITIONS}
        mean_prompt_tokens = {condition: values.mean(axis=0) for condition, values in prompt_tokens.items()}
        budget_summaries = []
        for budget_index, budget in enumerate(budgets):
            item_counts = [len(world_rows[budget_index]["targets"]) for world_rows in groups["study"]]
            result = {
                "budget": budget,
                "n_independent_worlds": len(world_seeds),
                "n_evaluation_items_per_world": item_counts,
                "n_evaluation_predictions_per_condition": sum(item_counts),
                "paired_generation_minus_study": point_summary(estimates["difference"], budget_index),
                "bayes": {"accuracy": point_summary(estimates["bayes"], budget_index), "posterior_expected_accuracy": point_summary(expected_bayes, budget_index)},
            }
            for condition in CONDITIONS:
                condition_rows = [world_rows[budget_index] for world_rows in groups[condition]]
                result[condition] = {"accuracy": point_summary(estimates[condition], budget_index)}
                for field in ("prompt_tokens", "completion_tokens", "eval_cost_usd", "practice_cost_usd", "practice_completion_tokens"):
                    result[condition][f"mean_{field}"] = float(np.mean([row[field] for row in condition_rows]))
            budget_summaries.append(result)
        calls = {}
        for row in model_rows:
            call_id = row["call_id"]
            cost = row["eval_cost_usd"]
            if call_id in calls and calls[call_id] != cost:
                raise ValueError(f"{model}: conflicting costs for shared evaluation call {call_id}.")
            calls[call_id] = cost
        evaluation_cost = sum(calls.values())
        practice_cost = sum(world_rows[-1]["practice_cost_usd"] for condition in CONDITIONS for world_rows in groups[condition])
        figures = make_plots(run_dir, model, budgets, estimates, mean_prompt_tokens)
        summary["models"][model] = {
            "world_seeds": world_seeds,
            "n_independent_worlds": len(world_seeds),
            "budget_levels": budgets,
            "budgets": budget_summaries,
            "costs": {"unique_evaluation_calls": len(calls), "evaluation_usd": evaluation_cost, "practice_usd": practice_cost, "total_usd": evaluation_cost + practice_cost},
            "figures": figures,
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for result in summary["models"].values():
        for path in result["figures"].values():
            print(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    analyze(args.run_dir, args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
