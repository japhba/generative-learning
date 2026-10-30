"""Describe paired weight-training experiments without assuming a scaling law."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from training_task import _paths, identity, shortest_plan, split_for, teacher_solution, transition, verify_solution


ARMS = ("sft", "rollout_distill", "yoked_sft")
LABELS = {"sft": "Teacher SFT", "rollout_distill": "Verified rollout distillation", "yoked_sft": "Yoked teacher SFT"}
CLOCK_LABELS = {"problems_presented": "New training prompts after warm-start", "training_examples": "Training examples after warm-start",
                "optimizer_updates": "Optimizer updates after warm-start", "reported_tokens": "Training + rollout tokens after warm-start",
                "phase_seconds": "Training + rollout seconds after warm-start"}
STRATUM_LABELS = {"easy": "Easy: shortest distance 1–2", "in_distribution": "In distribution: shortest distance 3–4",
                  "long_horizon": "Long horizon: shortest distance 5–6"}
_MOVE = re.compile(r"([ABC])[ \t]+(0|[1-9][0-9]*)")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def estimate(values, bootstrap_samples=2000):
    """Resample independent training seeds, never correlated examples or batches."""
    values = np.asarray(values, dtype=float)
    if not len(values): raise ValueError("At least one observed seed is required for an estimate.")
    interval = None
    if len(values) >= 2:
        indices = np.random.default_rng(88210).integers(0, len(values), size=(bootstrap_samples, len(values)))
        interval = np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()
    return {"mean": float(values.mean()), "ci95": interval, "n_seeds": len(values), "by_seed": values.tolist()}


def clocks(counters):
    return {"problems_presented": counters["problems_presented"], "training_examples": counters["training_examples"],
            "optimizer_updates": counters["optimizer_updates"],
            "reported_tokens": counters["training_tokens"] + counters["rollout_prompt_tokens"] + counters["rollout_completion_tokens"],
            "phase_seconds": counters["training_seconds"] + counters["rollout_seconds"]}


def load_runs(directories):
    runs = []
    for directory in directories:
        config = read_json(directory / "config.json")
        metrics = {}
        for arm in ARMS:
            path = directory / f"{arm}_metrics.jsonl"
            if path.exists():
                records = read_jsonl(path)
                budgets = [record["budget"] for record in records]
                if len(set(budgets)) != len(budgets):
                    raise ValueError(f"Duplicate {arm} checkpoints in {directory}.")
                if any(record["seed"] != config["seed"] or record["arm"] != arm for record in records):
                    raise ValueError(f"Mislabeled metrics in {directory}.")
                if any(record["counters"]["problems_presented"] != record["budget"] for record in records):
                    raise ValueError(f"Prompt-budget accounting mismatch in {directory}.")
                metrics[arm] = {record["budget"]: record for record in records}
        if not metrics:
            raise ValueError(f"No training-arm metrics in {directory}; run at least one arm first.")
        baselines = [records[0]["metrics"] for records in metrics.values()]
        if any(baseline != baselines[0] for baseline in baselines[1:]):
            raise ValueError(f"Arms have different warm-start baseline predictions in {directory}.")
        runs.append({"directory": directory, "config": config, "metrics": metrics, "seed": config["seed"],
                     "dataset": read_json(directory / "dataset.json"), "warm_counters": read_json(directory / "warm_counters.json"),
                     "base": read_jsonl(directory / "base_metrics.jsonl")[0], "warm": read_jsonl(directory / "warm_metrics.jsonl")[0]})
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Each run must have a distinct training seed for seed-level uncertainty.")
    fields = ("model", "revision", "warm_size", "warm_epochs", "warm_learning_rate", "learning_rate", "batch_size", "query_batch_size", "rollouts", "temperature", "max_new_tokens", "training", "rollout_method", "control", "evaluation")
    for run in runs[1:]:
        differences = [field for field in fields if run["config"][field] != runs[0]["config"][field]]
        if differences:
            raise ValueError(f"Incompatible experiments cannot be pooled: {differences}.")
        task_fields = ("task", "moduli", "increments", "train_distances", "eval_strata", "max_steps")
        if any(run["dataset"]["metadata"][field] != runs[0]["dataset"]["metadata"][field] for field in task_fields):
            raise ValueError("Task distributions or held-out difficulty definitions differ between runs.")
    return runs


def summarize_curves(runs, bootstrap_samples):
    indexed = defaultdict(list)
    for run in runs:
        for arm, records in run["metrics"].items():
            for budget, record in records.items(): indexed[arm, budget].append((run["seed"], record))
    result = {arm: [] for arm in ARMS if any(key[0] == arm for key in indexed)}
    for (arm, budget), records in sorted(indexed.items()):
        records.sort(key=lambda pair: pair[0])
        point = {"budget": budget, "seeds": [seed for seed, _ in records], "clocks": {}, "metrics": {}, "weight_probes": {}}
        for clock in CLOCK_LABELS:
            point["clocks"][clock] = estimate([clocks(record["counters"])[clock] for _, record in records], bootstrap_samples)
        for stratum in records[0][1]["metrics"]:
            accuracy = [record["metrics"][stratum]["accuracy"] for _, record in records]
            point["metrics"][stratum] = {"accuracy": estimate(accuracy, bootstrap_samples),
                                          "error": estimate([1 - value for value in accuracy], bootstrap_samples),
                                          "test_counts_by_seed": [record["metrics"][stratum]["count"] for _, record in records]}
        for parameter in records[0][1]["weight_probes"]:
            point["weight_probes"][parameter] = {name: estimate([record["weight_probes"][parameter][name] for _, record in records], bootstrap_samples)
                                                   for name in ("delta_norm", "relative_delta_norm")}
        result[arm].append(point)
    return result


def paired_contrasts(runs, bootstrap_samples):
    result = []
    for first, second in combinations(ARMS, 2):
        points = defaultdict(list)
        for run in runs:
            if first not in run["metrics"] or second not in run["metrics"]: continue
            for budget in run["metrics"][first].keys() & run["metrics"][second].keys():
                a, b = (run["metrics"][arm][budget] for arm in (first, second))
                for stratum in a["metrics"]:
                    difference = a["metrics"][stratum]["accuracy"] - b["metrics"][stratum]["accuracy"]
                    points[budget, stratum].append((run["seed"], difference))
        for (budget, stratum), values in sorted(points.items()):
            values.sort()
            result.append({"first": first, "second": second, "budget": budget, "stratum": stratum,
                           "seeds": [seed for seed, _ in values], "accuracy_difference": estimate([value for _, value in values], bootstrap_samples)})
    return result


def rollout_diagnostics(run):
    path = run["directory"] / "rollout_batches.jsonl"
    if not path.exists(): return {"status": "No rollout batches available."}
    problems = {problem["id"]: problem for problem in run["dataset"]["train"]}
    totals = {"prompts": 0, "accepted_prompts": 0, "candidates": 0, "successful_candidates": 0,
              "canonical_plans": 0, "canonical_texts": 0, "steps": 0, "shortest_steps": 0, "length_ratio_sum": 0.0}
    batches, all_candidates = [], []
    previous_budget = 0
    for row in read_jsonl(path):
        budget = row["end_budget"]
        selected = row["selected"]
        counts = {"prompts": budget - previous_budget, "accepted_prompts": len(selected), "candidates": len(row["candidates"]),
                  "successful_candidates": sum(candidate["verdict"]["correct"] for candidate in row["candidates"]),
                  "canonical_plans": 0, "canonical_texts": 0, "steps": 0, "shortest_steps": 0, "length_ratio_sum": 0.0}
        for accepted in selected:
            problem = problems[accepted["problem_id"]]
            verdict = verify_solution(problem, accepted["completion"])
            if not verdict["correct"]: raise ValueError("A retained rollout fails the independent task verifier.")
            counts["canonical_plans"] += tuple(verdict["plan"]) == shortest_plan(problem)
            counts["canonical_texts"] += accepted["completion"].strip() == teacher_solution(problem)
            counts["steps"] += verdict["steps"]
            counts["shortest_steps"] += problem["shortest_distance"]
            counts["length_ratio_sum"] += verdict["steps"] / problem["shortest_distance"]
        for field, value in counts.items(): totals[field] += value
        batches.append({"budget": budget, "counts": counts, **selection_rates(counts),
                        "candidate_mechanisms": candidate_mechanisms(row["candidates"], problems)})
        all_candidates.extend(row["candidates"])
        previous_budget = budget
    return {"status": "Observed rollout batches", "counts": totals, **selection_rates(totals), "batches": batches,
            "candidate_mechanisms": candidate_mechanisms(all_candidates, problems)}


def candidate_mechanisms(candidates, problems):
    """Diagnose concentration and arithmetic separately from strict acceptance.

    A visibly complete prefix remains inspectable when the final move is cut
    off or malformed. Parsing stops at the first malformed move; this never
    changes reward or acceptance. Correct-prefix length stops at the first
    wrong state transition, even if later claimed states happen to be right.
    """
    completions = defaultdict(set)
    first_operations = Counter({operation: 0 for operation in ("A", "B", "C", "unparsed")})
    termination = {name: {"samples": 0, "text_verifier_reasons": Counter(), "parsed_prefix_steps": Counter(),
                           "correct_prefix_steps": Counter()} for name in ("ended", "no_eos")}
    parsed_first, correct_first, text_valid_without_eos = 0, 0, 0
    for candidate in candidates:
        problem = problems[candidate["problem_id"]]
        text = candidate["text"].strip()
        completions[candidate["problem_id"]].add(text)
        plan, valid_prefix = visible_trace_prefix(problem, text)
        first_operations[plan[0][0] if plan else "unparsed"] += 1
        parsed_first += bool(plan)
        correct_prefix = len(valid_prefix)
        correct_first += correct_prefix > 0
        text_verdict = verify_solution(problem, text)
        text_valid_without_eos += not candidate["ended"] and text_verdict["correct"]
        group = termination["ended" if candidate["ended"] else "no_eos"]
        group["samples"] += 1
        group["text_verifier_reasons"][text_verdict["reason"]] += 1
        group["parsed_prefix_steps"][len(plan)] += 1
        group["correct_prefix_steps"][correct_prefix] += 1
    for group in termination.values():
        for key in ("parsed_prefix_steps", "correct_prefix_steps"):
            group[f"mean_{key}"] = sum(length * count for length, count in group[key].items()) / group["samples"] if group["samples"] else None
            group[key] = {str(length): count for length, count in sorted(group[key].items())}
        group["text_verifier_reasons"] = dict(group["text_verifier_reasons"])
    count = len(candidates)
    return {
        "definition": "Distinct completions ignore surrounding whitespace only. First-step accuracy over all candidates counts an unparseable first move as incorrect; conditional accuracy excludes those cases. Prefix histograms parse complete leading moves until malformed text, then count consecutive correct transitions from the start. These diagnostics never change strict trajectory acceptance. Text verification is repeated without the separate EOS requirement to expose termination-only failures.",
        "denominators": {"candidates": count, "distinct_prompts": len(completions), "parseable_first_moves": parsed_first},
        "mean_distinct_completions_per_prompt": sum(len(texts) for texts in completions.values()) / len(completions),
        "first_operation_counts": dict(first_operations),
        "first_operation_frequencies": {operation: frequency / count for operation, frequency in first_operations.items()},
        "first_step_correct_count": correct_first,
        "first_step_accuracy_all_candidates": correct_first / count,
        "first_step_accuracy_parseable_first_moves": correct_first / parsed_first if parsed_first else None,
        "text_valid_but_missing_eos_count": text_valid_without_eos,
        "by_termination": termination,
    }


def visible_trace_prefix(problem, text):
    """Return complete visible moves and their longest mathematically valid prefix."""
    plan = []
    for move in text.strip().split(";"):
        match = _MOVE.fullmatch(move.strip())
        if match is None: break
        operation, claimed = match.groups()
        plan.append((operation, int(claimed)))
    state, valid = problem["start"], []
    for operation, claimed in plan:
        state = transition(problem["modulus"], problem["add"], problem["subtract"], state, operation)
        if state != claimed: break
        valid.append((operation, claimed))
    return plan, valid


def trajectory_diagnostic(problem, prediction):
    """Diagnose opportunities to stop at the goal without changing task success."""
    parsed, valid = visible_trace_prefix(problem, prediction["text"])
    first_goal = next((index for index, (_, state) in enumerate(valid, start=1) if state == problem["target"]), None)
    within_horizon = first_goal is not None and first_goal <= problem["max_steps"]
    primary = prediction["verdict"]["correct"]
    text_verdict = verify_solution(problem, prediction["text"])
    if primary != (prediction["ended"] and text_verdict["correct"]):
        raise ValueError("Saved correctness disagrees with independent text verification plus EOS.")
    if primary:
        if not within_horizon: raise ValueError("A primary success has no valid goal-reaching prefix.")
        category = "primary_correct"
    elif within_horizon:
        category = "reached_goal_then_" + ("missing_eos" if not prediction["ended"] else text_verdict["reason"])
    elif len(valid) < min(len(parsed), problem["max_steps"]):
        category = "arithmetic_error_before_goal"
    elif len(valid) >= problem["max_steps"]:
        category = "no_goal_in_correct_horizon"
    elif text_verdict["reason"] == "format":
        category = "malformed_before_goal"
    elif not prediction["ended"]:
        category = "unfinished_before_goal"
    else:
        category = "stopped_before_goal"
    return {"primary_correct": primary, "valid_prefix_reaches_goal_within_horizon": within_horizon,
            "valid_prefix_reaches_goal_only_after_horizon": first_goal is not None and first_goal > problem["max_steps"],
            "reached_goal_but_failed_primary": within_horizon and not primary,
            "first_goal_step": first_goal, "parsed_prefix_steps": len(parsed), "correct_prefix_steps": len(valid),
            "first_operation": parsed[0][0] if parsed else "unparsed", "parsed_operation_counts": dict(Counter(operation for operation, _ in parsed)),
            "category": category, "saved_failure_reason": prediction["verdict"]["reason"],
            "text_verifier_reason_without_eos": text_verdict["reason"], "ended": prediction["ended"]}


def evaluation_diagnostics(run):
    problems = {problem["id"]: problem for group in run["dataset"]["eval"].values() for problem in group}
    expected_predictions = Counter((problem["id"], stratum) for stratum, group in run["dataset"]["eval"].items() for problem in group)
    baselines = {problem_id: constant_action_successes(problem) for problem_id, problem in problems.items()}
    checkpoints = []
    for arm, records in run["metrics"].items():
        for budget in sorted(records):
            predictions = read_json(run["directory"] / f"{arm}_n{budget}_predictions.json")
            if Counter((prediction["problem_id"], prediction["stratum"]) for prediction in predictions) != expected_predictions:
                raise ValueError("Saved predictions do not exactly cover the fixed held-out identities and strata.")
            grouped = defaultdict(list)
            for prediction in predictions:
                diagnostic = trajectory_diagnostic(problems[prediction["problem_id"]], prediction)
                diagnostic["constant_action_successes"] = baselines[prediction["problem_id"]]
                grouped[prediction["stratum"]].append(diagnostic)
            strata = {}
            for stratum, diagnostics in grouped.items():
                count = len(diagnostics)
                counts = {field: sum(row[field] for row in diagnostics) for field in
                          ("primary_correct", "valid_prefix_reaches_goal_within_horizon", "valid_prefix_reaches_goal_only_after_horizon", "reached_goal_but_failed_primary")}
                if counts["primary_correct"] / count != records[budget]["metrics"][stratum]["accuracy"] or count != records[budget]["metrics"][stratum]["count"]:
                    raise ValueError("Saved predictions and primary metrics disagree.")
                categories = Counter(row["category"] for row in diagnostics)
                first_operations = Counter(row["first_operation"] for row in diagnostics)
                all_operations = Counter()
                for row in diagnostics: all_operations.update(row["parsed_operation_counts"])
                failed = count - counts["primary_correct"]
                baseline_comparisons = {}
                for operation in ("A", "B", "C"):
                    baseline = [row["constant_action_successes"][operation] for row in diagnostics]
                    confusion = Counter((row["primary_correct"], success) for row, success in zip(diagnostics, baseline, strict=True))
                    baseline_comparisons[operation] = {"baseline_success_count": sum(baseline), "baseline_accuracy": sum(baseline) / count,
                                                       "exact_success_vector_agreement": (confusion[True, True] + confusion[False, False]) / count,
                                                       "both_correct": confusion[True, True], "both_fail": confusion[False, False],
                                                       "model_only_correct": confusion[True, False], "baseline_only_correct": confusion[False, True]}
                strata[stratum] = {"count": count, **{f"{field}_count": value for field, value in counts.items()},
                                   **{f"{field}_fraction": value / count for field, value in counts.items()},
                                   "category_counts": dict(categories), "failed_completion_count": failed,
                                   "correct_horizon_without_goal_count": categories["no_goal_in_correct_horizon"],
                                   "failed_completions_with_correct_horizon_without_goal_fraction": categories["no_goal_in_correct_horizon"] / failed if failed else None,
                                   "first_operation_counts": dict(first_operations), "all_parsed_operation_counts": dict(all_operations),
                                   "first_operation_fractions": {operation: frequency / count for operation, frequency in first_operations.items()},
                                   "all_parsed_operation_fractions": {operation: frequency / sum(all_operations.values()) for operation, frequency in all_operations.items()},
                                   "first_operation_empirical_entropy_bits": empirical_entropy(first_operations),
                                   "all_operations_empirical_entropy_bits": empirical_entropy(all_operations),
                                   "constant_action_comparisons": baseline_comparisons,
                                   "first_goal_step_counts": dict(Counter(str(row["first_goal_step"]) if row["first_goal_step"] is not None else "not_reached" for row in diagnostics)),
                                   "mean_correct_prefix_steps": float(np.mean([row["correct_prefix_steps"] for row in diagnostics])),
                                   "mean_parsed_prefix_steps": float(np.mean([row["parsed_prefix_steps"] for row in diagnostics]))}
            checkpoints.append({"arm": arm, "budget": budget, "strata": strata})
    return {"interpretation": "Diagnostic only; primary exact complete-trajectory accuracy is unchanged. A goal-reaching prefix must have every preceding transition correct and reach the goal within max_steps. A failed completion after this prefix had a valid stopping opportunity; it is not counted as a rescued success. Missing-EOS categories can also contain over-horizon or malformed continuation. Other categories separate arithmetic failure before the goal, exhaustion of a correct horizon, and early stopping/format failures. First arrival after the horizon is reported in the histogram but is not a within-horizon arrival. Operation entropy is the empirical marginal entropy of decoded actions, not conditional policy entropy. Constant-action oracles repeat A, B, or C with exact arithmetic and stop when the goal is reached within the allowed horizon; comparisons use the identical sampled held-out problems. These are exploratory mechanistic baselines, not rescored model accuracy.",
            "checkpoints": checkpoints}


def empirical_entropy(counts):
    """Descriptive marginal entropy across decoded operations, not policy entropy."""
    probabilities = np.array(list(counts.values()), dtype=float)
    if not len(probabilities): return None
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def constant_action_successes(problem):
    successes = {}
    for operation in ("A", "B", "C"):
        state = problem["start"]
        successes[operation] = False
        for _ in range(problem["max_steps"]):
            state = transition(problem["modulus"], problem["add"], problem["subtract"], state, operation)
            if state == problem["target"]:
                successes[operation] = True
                break
    return successes


def full_pool_constant_action_baselines(metadata):
    """Enumerate the same hashed held-out finite universe used by the task sampler."""
    strata = metadata["eval_strata"]
    counts = {name: 0 for name in strata}
    successes = {name: Counter({operation: 0 for operation in ("A", "B", "C")}) for name in strata}
    for modulus, add, subtract in product(metadata["moduli"], metadata["increments"], metadata["increments"]):
        for start in range(modulus):
            for target, path in _paths(modulus, add, subtract, start).items():
                distance = len(path)
                matching = [name for name, distances in strata.items() if distance in distances]
                if not matching: continue
                problem_id = identity(modulus, add, subtract, start, target)
                if split_for(problem_id) != "test": continue
                problem = {"modulus": modulus, "add": add, "subtract": subtract, "start": start,
                           "target": target, "max_steps": metadata["max_steps"]}
                name = matching[0]
                counts[name] += 1
                successes[name].update(constant_action_successes(problem))
    for name in strata:
        if counts[name] != metadata["candidate_counts"][name]:
            raise ValueError("Full finite-pool baseline does not match the dataset's held-out candidate count.")
    return {"interpretation": "Exact constant-action oracle success under a uniform draw from each full hashed held-out identity pool, conditioned on the same shortest-distance stratum. These are finite-pool theoretical baselines, not LLM measurements or unconditioned 8/(m−1) approximations.",
            "moduli": metadata["moduli"], "increments": metadata["increments"], "max_steps": metadata["max_steps"],
            "strata": {name: {"count": counts[name], "shortest_distances": distances,
                              "constant_action_success_counts": dict(successes[name]),
                              "constant_action_accuracies": {operation: count / counts[name] for operation, count in successes[name].items()}}
                       for name, distances in strata.items()}}


def selection_rates(counts):
    accepted = counts["accepted_prompts"]
    return {"accepted_prompt_fraction": counts["accepted_prompts"] / counts["prompts"],
            "successful_candidate_fraction": counts["successful_candidates"] / counts["candidates"],
            "canonical_plan_fraction": counts["canonical_plans"] / accepted if accepted else None,
            "exact_canonical_text_fraction": counts["canonical_texts"] / accepted if accepted else None,
            "mean_accepted_steps": counts["steps"] / accepted if accepted else None,
            "mean_shortest_steps": counts["shortest_steps"] / accepted if accepted else None,
            "mean_length_over_shortest": counts["length_ratio_sum"] / accepted if accepted else None}


def exponent_fits(curves):
    """Optional log-error summaries: no floor correction or asymptotic assertion."""
    fits = []
    for arm, points in curves.items():
        selected = [point for point in points if point["budget"] > 0]
        for stratum in points[0]["metrics"]:
            x = np.array([point["budget"] for point in selected], dtype=float)
            y = np.array([point["metrics"][stratum]["error"]["mean"] for point in selected])
            result = {"arm": arm, "stratum": stratum, "budgets": x.tolist(), "status": "unidentified", "beta": None}
            if len(x) >= 3 and np.all(y > 0):
                coefficients = np.polyfit(np.log(x), np.log(y), 1)
                residual = np.log(y) - np.polyval(coefficients, np.log(x))
                result.update(status="descriptive finite-range fit", beta=float(-coefficients[0]), amplitude=float(np.exp(coefficients[1])),
                              log_error_rmse=float(np.sqrt(np.mean(residual**2))), budget_span_factor=float(x[-1] / x[0]),
                              monotone_nonincreasing_error=bool(np.all(np.diff(y) <= 0)))
            fits.append(result)
    return {"interpretation": "Optional descriptive log(error) versus log(new prompts) slopes. No asymptotic law, error floor, or distinction from other curve families is established. Bumpy or nearly constant curves make exponent interpretations especially weak. No fit intervals are fabricated.", "fits": fits}


def weight_cosines(runs, probe_root=None):
    results, unavailable = [], []
    for run in tqdm(runs, desc="Weight alignment"):
        for first, second in combinations(run["metrics"], 2):
            for budget in sorted(run["metrics"][first].keys() & run["metrics"][second].keys()):
                if budget == 0: continue
                records = [run["metrics"][arm][budget] for arm in (first, second)]
                paths = [Path(record["weight_probe_path"]) if probe_root is None else probe_root / run["directory"].name / "weight_probes" / Path(record["weight_probe_path"]).name for record in records]
                missing = [str(path) for path in paths if not path.is_file()]
                if missing:
                    unavailable.append({"seed": run["seed"], "first": first, "second": second, "budget": budget, "missing_paths": missing})
                    continue
                with np.load(paths[0]) as first_archive, np.load(paths[1]) as second_archive:
                    for parameter in records[0]["weight_probes"]:
                        a, b = (archive[parameter].ravel().astype(np.float64) for archive in (first_archive, second_archive))
                        denominator = np.linalg.norm(a) * np.linalg.norm(b)
                        results.append({"seed": run["seed"], "first": first, "second": second, "budget": budget, "parameter": parameter,
                                        "cosine": float(a @ b / denominator) if denominator > 0 else None})
    return {"interpretation": "Cosines compare each arm's weight displacement from its shared warm-start; they are not overlap with a known true model. An explicit probe root resolves root/run_directory_name/weight_probes/original_filename, with no fallback to raw metric paths. Missing files are not fetched.",
            "probe_root_override": str(probe_root) if probe_root is not None else None, "results": results, "unavailable": unavailable}


def make_plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    curves = summary["curves"]
    strata = next(iter(curves.values()))[0]["metrics"]
    for clock, clock_label in CLOCK_LABELS.items():
        for stratum in strata:
            figure, axis = plt.subplots()
            for arm, points in curves.items():
                x = [point["clocks"][clock]["mean"] for point in points]
                values = [point["metrics"][stratum]["accuracy"] for point in points]
                line, = axis.plot(x, [value["mean"] for value in values], marker="o", label=LABELS[arm])
                available = [(position, value["ci95"]) for position, value in zip(x, values, strict=True) if value["ci95"] is not None]
                if available:
                    axis.fill_between([position for position, _ in available], [interval[0] for _, interval in available],
                                      [interval[1] for _, interval in available], color=line.get_color(), alpha=0.2)
            distances = summary["eval_strata"][stratum]
            title = f"{stratum.replace('_', ' ').capitalize()}: shortest distance {min(distances)}–{max(distances)}"
            axis.set(xlabel=clock_label, ylabel="Test accuracy", ylim=(0, 1), title=title)
            error_axis = axis.secondary_yaxis("right", functions=(lambda accuracy: 1 - accuracy, lambda error: 1 - error))
            error_axis.set_ylabel("Test error")
            axis.legend()
            figure.tight_layout()
            path = output / f"{stratum}_{clock}.png"
            figure.savefig(path)
            plt.close(figure)
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--fit-exponents", action="store_true")
    parser.add_argument("--weight-cosines", action="store_true")
    parser.add_argument("--weight-probe-root", type=Path)
    parser.add_argument("--full-pool-baselines", action="store_true")
    args = parser.parse_args()
    if args.weight_probe_root is not None and not args.weight_cosines: parser.error("--weight-probe-root requires --weight-cosines.")
    load_dotenv(Path.home() / ".env")
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "matplotlib")
    if args.bootstrap_samples < 100: raise ValueError("Use at least 100 bootstrap replicates.")
    directories = [path.resolve() for path in args.runs]
    output = args.output.resolve() if args.output is not None else directories[0] / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(directories)
    diagnostics = {str(run["seed"]): rollout_diagnostics(run) for run in runs}
    summary = {
        "model": runs[0]["config"]["model"], "seeds": [run["seed"] for run in runs], "runs": [str(path) for path in directories],
        "eval_strata": runs[0]["dataset"]["metadata"]["eval_strata"],
        "interpretation": "All curves begin at the shared teacher warm-start. Horizontal costs exclude evaluation and warm-start, which are reported separately. Reported tokens count unpadded training inputs and sampled rollout inputs/outputs, not FLOPs. Seconds measure training/rollout phase wall time on the GPU run, including CPU preparation, not isolated GPU kernel time. Yoked SFT excludes the rollout arm's acquisition compute. Teacher/verifier computation is not charged as LLM compute.",
        "uncertainty": "Each seed is one independent training experiment. With one seed, results are descriptive and no confidence interval is reported. With two or more seeds, percentile bootstrap intervals resample seeds; paired contrasts preserve arm pairing. Very few seeds give unstable intervals, and pointwise intervals do not adjust for multiple comparisons.",
        "curves": summarize_curves(runs, args.bootstrap_samples), "paired_contrasts": paired_contrasts(runs, args.bootstrap_samples),
        "warm_start": [{"seed": run["seed"], "counters": run["warm_counters"], "clocks": clocks(run["warm_counters"]),
                        "base_metrics": run["base"]["metrics"], "warm_metrics": run["warm"]["metrics"]} for run in runs],
        "rollout_diagnostics": diagnostics,
        "evaluation_diagnostics": {str(run["seed"]): evaluation_diagnostics(run) for run in runs},
        "availability": [{"seed": run["seed"], "arms": {arm: sorted(records) for arm, records in run["metrics"].items()},
                          "missing_arms": [arm for arm in ARMS if arm not in run["metrics"]]} for run in runs],
        "run_configurations": [run["config"] for run in runs],
    }
    if args.fit_exponents: summary["effective_exponents"] = exponent_fits(summary["curves"])
    if args.full_pool_baselines: summary["full_pool_constant_action_baselines"] = full_pool_constant_action_baselines(runs[0]["dataset"]["metadata"])
    if args.weight_cosines:
        alignments = weight_cosines(runs, args.weight_probe_root.resolve() if args.weight_probe_root is not None else None)
        write_json(output / "weight_cosines.json", alignments)
        summary["weight_cosines"] = alignments
    paths = make_plots(summary, output)
    summary["plots"] = [str(path) for path in paths]
    write_json(output / "summary.json", summary)
    for path in paths: print(path)


if __name__ == "__main__": main()
