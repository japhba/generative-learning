"""Analyze paired Zipf learning curves; fitted exponents are finite-range summaries."""

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


CONDITIONS = ("iid_study", "active_generate", "active_yoked")
LABELS = {"iid_study": "IID study", "active_generate": "Active generation", "active_yoked": "Yoked study"}
FIT_WINDOWS = ((8, 128), (4, 64), (16, 128))


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def summarize_bootstrap(mean, replicates):
    """Undefined conditional errors retain NaN internally and become null in JSON."""
    lower, upper, valid_counts = [], [], []
    for column in replicates.T:
        valid = column[np.isfinite(column)]
        valid_counts.append(len(valid))
        interval = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
        lower.append(interval[0])
        upper.append(interval[1])
    return {"mean": np.asarray(mean), "lower": np.array(lower), "upper": np.array(upper), "n_valid_bootstrap": valid_counts}


def estimate(values, indices):
    return summarize_bootstrap(values.mean(axis=0), values[indices].mean(axis=1))


def conditional_estimate(numerators, denominators, indices):
    totals = denominators.sum(axis=0)
    mean = np.divide(numerators.sum(axis=0), totals, out=np.full(totals.shape, np.nan), where=totals > 0)
    bootstrap_totals = denominators[indices].sum(axis=1)
    replicates = np.divide(numerators[indices].sum(axis=1), bootstrap_totals, out=np.full(bootstrap_totals.shape, np.nan), where=bootstrap_totals > 0)
    result = summarize_bootstrap(mean, replicates)
    result["total_test_weight"] = totals
    return result


def point_summary(estimate, index):
    return {"mean": finite_number(estimate["mean"][index]), "ci95": [finite_number(estimate["lower"][index]), finite_number(estimate["upper"][index])], "n_valid_bootstrap": estimate["n_valid_bootstrap"][index]}


def effective_fit(budgets, errors, indices, minimum, maximum):
    mask = (budgets >= minimum) & (budgets <= maximum)
    x = np.log(budgets[mask])
    means = errors.mean(axis=0)[mask]
    result = {"budget_range_requested": [minimum, maximum], "budgets_used": budgets[mask].tolist(), "n_points": len(x)}
    if len(x) < 3 or np.any(means <= 0):
        result.update({"status": "unidentified: fewer than three budgets or nonpositive mean error", "alpha": None, "ci95": [None, None]})
        return result, np.full(len(indices), np.nan)
    centered_x = x - x.mean()
    denominator = centered_x @ centered_x
    log_means = np.log(means)
    alpha = -(centered_x @ log_means) / denominator
    intercept = log_means.mean() + alpha * x.mean()
    residual = log_means - (intercept - alpha * x)
    bootstrap_means = errors[indices].mean(axis=1)[:, mask]
    valid = np.all(bootstrap_means > 0, axis=1)
    alphas = np.full(len(indices), np.nan)
    alphas[valid] = -(np.log(bootstrap_means[valid]) @ centered_x) / denominator
    interval = np.quantile(alphas[valid], [0.025, 0.975]) if valid.any() else (np.nan, np.nan)
    result.update({"status": "finite-range effective exponent", "alpha": float(alpha), "amplitude": float(np.exp(intercept)), "ci95": [finite_number(value) for value in interval], "n_valid_bootstrap": int(valid.sum()), "n_bootstrap": len(indices), "log_error_rmse": float(np.sqrt(np.mean(residual**2))), "budget_span_factor": float(budgets[mask][-1] / budgets[mask][0])})
    return result, alphas


def exponent_summary(budgets, errors, indices):
    windows = []
    for minimum, maximum in FIT_WINDOWS:
        fits, draws = {}, {}
        for condition in CONDITIONS:
            fits[condition], draws[condition] = effective_fit(budgets, errors[condition], indices, minimum, maximum)
        differences = {}
        for name, first, second in (("active_minus_iid", "active_generate", "iid_study"), ("yoked_minus_iid", "active_yoked", "iid_study"), ("active_minus_yoked", "active_generate", "active_yoked")):
            difference = draws[first] - draws[second]
            valid = difference[np.isfinite(difference)]
            interval = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
            point = fits[first]["alpha"] - fits[second]["alpha"] if fits[first]["alpha"] is not None and fits[second]["alpha"] is not None else None
            differences[name] = {"mean": point, "ci95": [finite_number(value) for value in interval], "n_valid_bootstrap": len(valid)}
        windows.append({"budget_range": [minimum, maximum], "conditions": fits, "paired_alpha_differences": differences})
    return {"interpretation": "Preliminary effective exponents from least squares on log mean error. These are not asymptotic exponents. These slopes alone do not distinguish floors or other curve families; see supplementary model checks. Intervals exclude bootstrap draws with nonpositive mean error; inspect their valid counts. Sensitivity windows and outcomes are not independent tests.", "windows": windows}


def token_exponent_summary(budgets, errors, token_clocks, indices):
    results = {}
    mask = (budgets >= 8) & (budgets <= 128)
    for clock, tokens in token_clocks.items():
        fits, draws = {}, {}
        for condition in CONDITIONS:
            x, y = tokens[condition].mean(axis=0)[mask], errors[condition].mean(axis=0)[mask]
            bootstrap_x = tokens[condition][indices].mean(axis=1)[:, mask]
            bootstrap_y = errors[condition][indices].mean(axis=1)[:, mask]
            valid = np.all((bootstrap_x > 0) & (bootstrap_y > 0), axis=1)
            alphas = np.full(len(indices), np.nan)
            if len(x) < 3 or np.any(y <= 0) or np.any(x <= 0):
                fits[condition] = {"alpha": None, "ci95": [None, None], "status": "unidentified"}
            else:
                log_x, log_y = np.log(x), np.log(y)
                centered_x = log_x - log_x.mean()
                alpha = -(centered_x @ log_y) / (centered_x @ centered_x)
                centered_bootstrap_x = np.log(bootstrap_x[valid])
                centered_bootstrap_x -= centered_bootstrap_x.mean(axis=1, keepdims=True)
                alphas[valid] = -(centered_bootstrap_x * np.log(bootstrap_y[valid])).sum(axis=1) / (centered_bootstrap_x**2).sum(axis=1)
                interval = np.quantile(alphas[valid], [0.025, 0.975])
                fits[condition] = {"alpha": float(alpha), "ci95": interval.tolist(), "n_valid_bootstrap": int(valid.sum()), "token_span_factor": float(x[-1] / x[0]), "budgets_used": budgets[mask].tolist()}
            draws[condition] = alphas
        differences = {}
        for name, first, second in (("active_minus_iid", "active_generate", "iid_study"), ("yoked_minus_iid", "active_yoked", "iid_study"), ("active_minus_yoked", "active_generate", "active_yoked")):
            valid = draws[first] - draws[second]
            valid = valid[np.isfinite(valid)]
            interval = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
            point = fits[first]["alpha"] - fits[second]["alpha"] if fits[first]["alpha"] is not None and fits[second]["alpha"] is not None else None
            differences[name] = {"mean": point, "ci95": [finite_number(value) for value in interval], "n_valid_bootstrap": len(valid)}
        results[clock] = {"conditions": fits, "paired_alpha_differences": differences}
    return {"interpretation": "Descriptive log-error slopes over checkpoints8–128 against observed mean API-reported tokens per logical world; not FLOPs, weight-training rates, or asymptotic laws. Tokens and errors are resampled jointly by world; shared-cache amortization is excluded. Deployment tokens exclude earlier evaluation checkpoints; yoked deployment excludes the active learner's label-acquisition calls.", "clocks": results}


def curve_design(budgets, family):
    shapes = np.geomspace(0.005, 4, 512) if "power" in family else np.geomspace(0.0001, 1, 512)
    features = (budgets[None, :] / budgets[0]) ** -shapes[:, None] if "power" in family else np.exp(-shapes[:, None] * (budgets - budgets[0]))
    return shapes, features


def fit_curve(mean_errors, shapes, features, floor_allowed):
    """Profile the shape over a fixed grid; solve amplitude/floor on their triangle."""
    no_floor_amplitude = np.clip((features @ mean_errors) / (features**2).sum(axis=1), 0, 1)
    amplitudes, floors = [no_floor_amplitude], [np.zeros(len(shapes))]
    if floor_allowed:
        feature_mean = features.mean(axis=1)
        centered_features = features - feature_mean[:, None]
        free_amplitude = (centered_features @ mean_errors) / (centered_features**2).sum(axis=1)
        free_floor = mean_errors.mean() - free_amplitude * feature_mean
        feasible = (free_amplitude >= 0) & (free_floor >= 0) & (free_amplitude + free_floor <= 1)
        amplitudes.append(np.where(feasible, free_amplitude, np.nan))
        floors.append(np.where(feasible, free_floor, np.nan))
        amplitudes.append(np.zeros(len(shapes)))
        floors.append(np.full(len(shapes), np.clip(mean_errors.mean(), 0, 1)))
        boundary_floor = np.clip(((1 - features) * (mean_errors - features)).sum(axis=1) / ((1 - features)**2).sum(axis=1), 0, 1)
        amplitudes.append(1 - boundary_floor)
        floors.append(boundary_floor)
    amplitudes, floors = np.array(amplitudes), np.array(floors)
    predictions = floors[:, :, None] + amplitudes[:, :, None] * features[None, :, :]
    rss = ((predictions - mean_errors)**2).sum(axis=2)
    boundary, shape_index = np.unravel_index(np.nanargmin(rss), rss.shape)
    return {"shape": float(shapes[shape_index]), "floor": float(floors[boundary, shape_index]), "amplitude_at_first_budget": float(amplitudes[boundary, shape_index]), "rmse": float(np.sqrt(rss[boundary, shape_index] / len(mean_errors))), "shape_hits_grid_boundary": bool(shape_index in (0, len(shapes) - 1)), "predictions": predictions[boundary, shape_index]}


def supplementary_model_checks(budgets, errors, indices):
    mask = (budgets >= 8) & (budgets <= 128)
    selected_budgets = budgets[mask]
    if len(selected_budgets) < 4:
        return {"status": "At least four budgets are needed for supplementary three-parameter curve fits."}
    result = {"interpretation": "Supplementary finite-range curve checks on8–128. Grid-profiled least squares compares power, exponential, and both with a nonnegative floor. Leave-one-world-out prediction error is descriptive validation at the same budgets, not future-budget extrapolation or a formal model-selection test. Floor/shape bootstrap intervals can expose confounding; finite-range fits cannot establish asymptotic floors. Primary log-error exponent estimates are unchanged.", "budgets": selected_budgets.tolist(), "parameter_bounds": {"floor": [0, 1], "amplitude": "0 to1−floor", "power_alpha": [0.005, 4], "exponential_rate": [0.0001, 1], "shape_grid_points": 512}, "conditions": {}}
    for condition in CONDITIONS:
        world_errors = errors[condition][:, mask]
        mean_errors = world_errors.mean(axis=0)
        bootstrap_means = world_errors[indices].mean(axis=1)
        models = {}
        for family in ("power", "exponential", "power_floor", "exponential_floor"):
            shapes, features = curve_design(selected_budgets, family)
            floor_allowed = family.endswith("_floor")
            fit = fit_curve(mean_errors, shapes, features, floor_allowed)
            fit["predictions"] = fit["predictions"].tolist()
            prediction_errors = []
            for world_index in range(len(world_errors)):
                training_mean = (world_errors.sum(axis=0) - world_errors[world_index]) / (len(world_errors) - 1)
                prediction = fit_curve(training_mean, shapes, features, floor_allowed)["predictions"]
                prediction_errors.append(float(np.mean((world_errors[world_index] - prediction)**2)))
            fit["leave_one_world_out_mse"] = float(np.mean(prediction_errors))
            fit["leave_one_world_out_mse_by_world"] = prediction_errors
            bootstrap_fits = [fit_curve(values, shapes, features, floor_allowed) for values in bootstrap_means]
            for parameter in ("shape", "floor"):
                fit[f"{parameter}_ci95"] = np.quantile([sample[parameter] for sample in bootstrap_fits], [0.025, 0.975]).tolist()
            fit["bootstrap_shape_boundary_fraction"] = float(np.mean([sample["shape_hits_grid_boundary"] for sample in bootstrap_fits]))
            models[family] = fit
        result["conditions"][condition] = models
    return result


def draw_curve(axis, x, values, label, linestyle="-", color=None):
    line, = axis.plot(x, values["mean"], marker="o", label=label, linestyle=linestyle, color=color)
    axis.fill_between(x, values["lower"], values["upper"], color=line.get_color(), alpha=0.2)
    return line.get_color()


def make_plots(run_dir, model, budgets, estimates, theory, token_clocks):
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    paths = {name: run_dir / f"{slug}_{name}.png" for name in ("learning_curve", "coverage_curve", "paired_difference", "conditional_error", "posterior_expected_error", "token_curve", "deployment_token_curve")}

    figure, axis = plt.subplots()
    for condition in CONDITIONS:
        draw_curve(axis, budgets, estimates[condition]["error"], LABELS[condition])
    axis.plot(theory["budgets"], theory["iid_errors"], linestyle="--", label="Ideal IID coverage")
    axis.plot(theory["budgets"], theory["ranked_errors"], linestyle="--", label="Ideal ranked coverage")
    axis.set(xlabel="Feedback or study examples", ylabel="Weighted test error", title=model, xscale="log", yscale="log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["learning_curve"])
    plt.close(figure)

    figure, axis = plt.subplots()
    for condition in CONDITIONS[:2]:
        color = draw_curve(axis, budgets, estimates[condition]["error"], LABELS[condition])
        draw_curve(axis, budgets, estimates[condition]["coverage_error"], LABELS[condition] + " coverage error", linestyle="--", color=color)
    axis.plot(theory["budgets"], theory["iid_errors"], linestyle=":", label="Ideal IID coverage")
    axis.plot(theory["budgets"], theory["ranked_errors"], linestyle=":", label="Ideal ranked coverage")
    axis.set(xlabel="Feedback or study examples", ylabel="Weighted test error", title=model, xscale="log", yscale="log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["coverage_curve"])
    plt.close(figure)

    figure, axis = plt.subplots()
    draw_curve(axis, budgets, estimates["active_minus_yoked"], "Active generation − yoked study")
    draw_curve(axis, budgets, estimates["yoked_minus_iid"], "Yoked study − IID study")
    axis.axhline(0, color="black", linestyle="--")
    axis.set(xlabel="Feedback or study examples", ylabel="Paired weighted error difference", title=model, xscale="log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["paired_difference"])
    plt.close(figure)

    figure, axis = plt.subplots()
    for condition in CONDITIONS:
        color = draw_curve(axis, budgets, estimates[condition]["seen_error"], LABELS[condition] + ": seen")
        draw_curve(axis, budgets, estimates[condition]["unseen_error"], LABELS[condition] + ": unseen", linestyle="--", color=color)
    axis.axhline(0.75, color="black", linestyle=":", label="Unseen random-guess error")
    axis.set(xlabel="Feedback or study examples", ylabel="Conditional weighted test error", title=model, xscale="log", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["conditional_error"])
    plt.close(figure)

    figure, axis = plt.subplots()
    for condition in CONDITIONS:
        draw_curve(axis, budgets, estimates[condition]["posterior_expected_error"], LABELS[condition])
    axis.plot(theory["budgets"], theory["iid_errors"], linestyle="--", label="Ideal IID coverage")
    axis.plot(theory["budgets"], theory["ranked_errors"], linestyle="--", label="Ideal ranked coverage")
    axis.set(xlabel="Feedback or study examples", ylabel="Coverage error + weighted seen mistakes", title=model, xscale="log", yscale="log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["posterior_expected_error"])
    plt.close(figure)

    for clock, filename, label in (("experiment", "token_curve", "Cumulative experiment tokens"), ("deployment", "deployment_token_curve", "Practice + current evaluation tokens")):
        figure, axis = plt.subplots()
        for condition in CONDITIONS:
            draw_curve(axis, token_clocks[clock][condition].mean(axis=0), estimates[condition]["error"], LABELS[condition])
        axis.set(xlabel=label, ylabel="Weighted test error", title=model, xscale="log", yscale="log")
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths[filename])
        plt.close(figure)
    return {name: str(path.resolve()) for name, path in paths.items()}


def matrix(groups, condition, field):
    return np.array([[row[field] for row in world_rows] for world_rows in groups[condition]])


def validate_groups(groups, world_seeds):
    for index, world in enumerate(world_seeds):
        reference = groups["iid_study"][index][0]
        for condition in CONDITIONS:
            for row in groups[condition][index]:
                for field in ("keys", "targets", "test_weights"):
                    if row[field] != reference[field]:
                        raise ValueError(f"World {world}: {field} differ across conditions or budgets.")
                weights, correct, seen = np.array(row["test_weights"]), np.array(row["correct"]), np.array(row["seen"])
                if not np.isclose(weights.sum(), 1) or np.any(weights < 0):
                    raise ValueError(f"World {world}: test weights must be nonnegative and sum to one.")
                if len(weights) != len(correct) or len(weights) != len(seen) or len(weights) != len(row["targets"]):
                    raise ValueError(f"World {world}: inconsistent evaluation array lengths.")
                if not np.isclose(weights @ (1 - correct), row["error"]):
                    raise ValueError(f"World {world}: stored weighted error disagrees with correctness.")
                if not np.isclose(row["known_mass"], 1 - row["coverage_error"] / 0.75):
                    raise ValueError(f"World {world}: known mass disagrees with coverage error.")
        for active, yoked in zip(groups["active_generate"][index], groups["active_yoked"][index], strict=True):
            if active["query_keys"] != yoked["query_keys"] or active["seen"] != yoked["seen"]:
                raise ValueError(f"World {world}: active and yoked examples differ.")


def analyze(run_dir, bootstrap_samples=1500, seed=20260905):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    worlds = json.loads((run_dir / "worlds.json").read_text())
    theory = json.loads((run_dir / "theory.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("No model results found; run the Zipf experiment before plotting.")
    indexed = {(row["model"], row["condition"], row["world_seed"], row["budget"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate model/condition/world/budget result rows.")
    summary = {
        "schema_version": 1,
        "run_directory": str(run_dir.resolve()),
        "manifest": manifest,
        "n_worlds_in_manifest": len(worlds),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "world", "interval": "pointwise percentile 95%", "paired": True},
        "estimand": "Mean weighted test error over worlds, with equal weight per world. Conditional errors divide pooled weighted mistakes by pooled seen/unseen test mass; intervals resample entire paired world curves.",
        "token_axis": "Cumulative API-reported practice prompt/completion tokens per logical world plus evaluation prompt/completion tokens at all checkpoints through this budget. Includes evaluation overhead; shared-cache amortization is excluded; not FLOPs or optimizer steps.",
        "deployment_token_axis": "Cumulative billed practice tokens plus only the current evaluation call. Earlier measurement checkpoints are excluded. Yoked replay excludes the active learner's cost of acquiring labels.",
        "coverage_error": "Exact probability mass of unqueried keys times 0.75, assuming perfect recall of queried keys and random guessing otherwise.",
        "posterior_expected_error": "Exact coverage error plus the stratified-test estimate of weighted mistakes on seen keys. Marginalizes random unknown labels; raw weighted test error remains the primary outcome.",
        "theory": theory,
        "models": {},
    }
    rng = np.random.default_rng(seed)
    for model in tqdm(sorted({row["model"] for row in rows}), desc="Analyzing Zipf models"):
        model_rows = [row for row in rows if row["model"] == model]
        world_seeds = sorted({row["world_seed"] for row in model_rows})
        budgets = np.array(sorted({row["budget"] for row in model_rows}))
        required = {(model, condition, world, int(budget)) for condition in CONDITIONS for world in world_seeds for budget in budgets}
        actual = {(row["model"], row["condition"], row["world_seed"], row["budget"]) for row in model_rows}
        if required != actual or budgets.tolist() != manifest["budgets"]:
            raise ValueError(f"{model}: incomplete paired world × condition × budget results.")
        if len(world_seeds) < 2:
            raise ValueError(f"{model}: at least two independent worlds are required for intervals.")
        groups = {condition: [[indexed[(model, condition, world, int(budget))] for budget in budgets] for world in world_seeds] for condition in CONDITIONS}
        validate_groups(groups, world_seeds)
        indices = rng.integers(0, len(world_seeds), size=(bootstrap_samples, len(world_seeds)))
        errors, estimates = {}, {}
        token_clocks = {"experiment": {}, "deployment": {}}
        for condition in CONDITIONS:
            errors[condition] = matrix(groups, condition, "error")
            estimates[condition] = {field: estimate(matrix(groups, condition, field), indices) for field in ("error", "coverage_error", "known_mass")}
            weights = matrix(groups, condition, "test_weights")
            seen = matrix(groups, condition, "seen").astype(bool)
            incorrect = 1 - matrix(groups, condition, "correct")
            for name, selection in (("seen_error", seen), ("unseen_error", ~seen)):
                numerators = (weights * incorrect * selection).sum(axis=2)
                denominators = (weights * selection).sum(axis=2)
                estimates[condition][name] = conditional_estimate(numerators, denominators, indices)
            expected_error = matrix(groups, condition, "coverage_error") + (weights * incorrect * seen).sum(axis=2)
            estimates[condition]["posterior_expected_error"] = estimate(expected_error, indices)
            practice_tokens = matrix(groups, condition, "practice_prompt_tokens") + matrix(groups, condition, "practice_completion_tokens")
            evaluation_tokens = matrix(groups, condition, "prompt_tokens") + matrix(groups, condition, "completion_tokens")
            token_clocks["experiment"][condition] = practice_tokens + np.cumsum(evaluation_tokens, axis=1)
            token_clocks["deployment"][condition] = practice_tokens + evaluation_tokens
        for name, first, second in (("active_minus_iid", "active_generate", "iid_study"), ("yoked_minus_iid", "active_yoked", "iid_study"), ("active_minus_yoked", "active_generate", "active_yoked")):
            estimates[name] = estimate(errors[first] - errors[second], indices)
        budget_results = []
        for index, budget in enumerate(budgets):
            result = {"budget": int(budget), "n_independent_worlds": len(world_seeds), "n_test_items_per_world": [len(world_rows[index]["targets"]) for world_rows in groups["iid_study"]], "paired_error_differences": {name: point_summary(estimates[name], index) for name in ("active_minus_iid", "yoked_minus_iid", "active_minus_yoked")}}
            for condition in CONDITIONS:
                result[condition] = {field: point_summary(estimates[condition][field], index) for field in ("error", "coverage_error", "known_mass", "seen_error", "unseen_error", "posterior_expected_error")}
                for field in ("seen_error", "unseen_error"):
                    result[condition][field]["total_test_weight"] = float(estimates[condition][field]["total_test_weight"][index])
                result[condition]["mean_cumulative_tokens"] = float(token_clocks["experiment"][condition][:, index].mean())
                result[condition]["mean_deployment_tokens"] = float(token_clocks["deployment"][condition][:, index].mean())
                for field in ("prompt_tokens", "completion_tokens", "eval_cost_usd", "practice_cost_usd", "practice_prompt_tokens", "practice_completion_tokens"):
                    result[condition][f"mean_{field}"] = float(matrix(groups, condition, field)[:, index].mean())
            budget_results.append(result)
        calls = {}
        for row in model_rows:
            call_id, cost = row["call_id"], row["eval_cost_usd"]
            if call_id in calls and calls[call_id] != cost:
                raise ValueError(f"{model}: conflicting costs for shared call {call_id}.")
            calls[call_id] = cost
        evaluation_cost = sum(calls.values())
        practice_cost = sum(world_rows[-1]["practice_cost_usd"] for condition in CONDITIONS for world_rows in groups[condition])
        figures = make_plots(run_dir, model, budgets, estimates, theory, token_clocks)
        summary["models"][model] = {"world_seeds": world_seeds, "n_independent_worlds": len(world_seeds), "budgets": budget_results, "effective_exponents": exponent_summary(budgets, errors, indices), "effective_token_exponents": token_exponent_summary(budgets, errors, token_clocks, indices), "supplementary_model_checks": supplementary_model_checks(budgets, errors, indices), "costs": {"unique_evaluation_calls": len(calls), "evaluation_usd": evaluation_cost, "practice_usd": practice_cost, "total_usd": evaluation_cost + practice_cost}, "figures": figures}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for result in summary["models"].values():
        for path in result["figures"].values():
            print(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    analyze(args.run_dir, args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
