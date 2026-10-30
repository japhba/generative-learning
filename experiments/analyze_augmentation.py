"""Audit and compare teacher-anchored rollout augmentation with matched teacher replay."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from analyze_training import CLOCK_LABELS, STRATUM_LABELS, clocks, estimate, evaluation_diagnostics, read_json, read_jsonl, rollout_diagnostics, weight_cosines, write_json
from training_task import teacher_solution, verify_solution


ARMS = ("teacher_rollout", "teacher_replay")
HISTORICAL = "historical_sft"
LABELS = {"teacher_rollout": "Teacher + verified rollout", "teacher_replay": "Teacher + matched replay", HISTORICAL: "Historical teacher SFT"}
CLOCKS = ("problems_presented", "training_examples", "reported_tokens", "phase_seconds")
PAIR_COUNTERS = ("problems_presented", "accepted_examples", "pending_examples", "training_examples", "optimizer_updates",
                 "teacher_examples", "extra_examples", "teacher_training_examples", "extra_training_examples")
CONFIG_FIELDS = ("model", "revision", "train_size", "warm_size", "warm_epochs", "eval_per_stratum", "batch_size", "query_batch_size",
                 "generation_batch_size", "rollouts", "max_new_tokens", "temperature", "learning_rate", "warm_learning_rate", "checkpoints")


def indexed_metrics(directory, arm, seed):
    records = read_jsonl(directory / f"{arm}_metrics.jsonl")
    budgets = [record["budget"] for record in records]
    if budgets != sorted(set(budgets)) or budgets[0] != 0: raise ValueError(f"Invalid checkpoint ordering for {arm} in {directory}.")
    if any(record["arm"] != arm or record["seed"] != seed or record["budget"] != record["counters"]["problems_presented"] for record in records):
        raise ValueError(f"Mislabeled metrics or prompt accounting for {arm} in {directory}.")
    return {record["budget"]: record for record in records}


def load_runs(directories):
    runs = []
    for directory in directories:
        config, dataset = read_json(directory / "config.json"), read_json(directory / "dataset.json")
        parent = Path(config["parent_run"])
        parent_config = read_json(parent / "config.json")
        if not parent.is_absolute(): raise ValueError("Parent run paths must be absolute.")
        if hashlib.sha256((parent / "config.json").read_bytes()).hexdigest() != config["parent_config_sha256"]:
            raise ValueError("Parent configuration hash changed.")
        if hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest() != config["dataset_sha256"]:
            raise ValueError("Copied dataset hash changed.")
        if config["arms"] != list(ARMS): raise ValueError("Unexpected augmentation arms.")
        if any(config[field] != parent_config[field] for field in CONFIG_FIELDS): raise ValueError("Augmentation and parent protocols differ.")
        if dataset != read_json(parent / "dataset.json"): raise ValueError("Augmentation changed the parent data or its ordering.")
        if config["seed"] != parent_config["seed"]: raise ValueError("Parent and augmentation seeds differ.")
        historical = indexed_metrics(parent, "sft", config["seed"])
        metrics = {arm: indexed_metrics(directory, arm, config["seed"]) for arm in ARMS if (directory / f"{arm}_metrics.jsonl").exists()}
        if not metrics: raise ValueError(f"No observed augmentation metrics in {directory}.")
        baseline = read_json(parent / "sft_n0_predictions.json")
        for arm, records in metrics.items():
            if records[0]["metrics"] != historical[0]["metrics"] or read_json(directory / f"{arm}_n0_predictions.json") != baseline:
                raise ValueError("Augmentation does not reproduce the parent warm-start predictions exactly.")
            if any(value["delta_norm"] != 0 for value in records[0]["weight_probes"].values()):
                raise ValueError("A new arm has nonzero displacement at its warm-start.")
        runs.append({"directory": directory, "parent": parent, "config": config, "seed": config["seed"], "dataset": dataset,
                     "metrics": metrics, "historical": historical, "warm_counters": read_json(parent / "warm_counters.json")})
    if len({run["seed"] for run in runs}) != len(runs): raise ValueError("Training seeds must be distinct.")
    for run in runs[1:]:
        if any(run["config"][field] != runs[0]["config"][field] for field in CONFIG_FIELDS):
            raise ValueError("Incompatible augmentation experiments cannot be pooled.")
        fields = ("task", "moduli", "increments", "train_distances", "eval_strata", "max_steps")
        if any(run["dataset"]["metadata"][field] != runs[0]["dataset"]["metadata"][field] for field in fields):
            raise ValueError("Task distributions differ between augmentation seeds.")
    return runs


def source_record(problem, completion, source):
    return {"problem_id": problem["id"], "completion": completion, "source": source}


def verify_saved_prediction(problem, prediction):
    verdict = verify_solution(problem, prediction["text"])
    if not prediction["ended"]: verdict = {**verdict, "correct": False, "reason": "truncated"}
    if prediction["verdict"] != json.loads(json.dumps(verdict)):
        raise ValueError("Saved rollout or evaluation verdict differs from independent verification plus EOS.")
    return verdict


def audited_evaluation_diagnostics(run):
    problems = {problem["id"]: problem for group in run["dataset"]["eval"].values() for problem in group}
    for arm, records in run["metrics"].items():
        for budget, record in records.items():
            failures = defaultdict(Counter)
            for prediction in read_json(run["directory"] / f"{arm}_n{budget}_predictions.json"):
                verdict = verify_saved_prediction(problems[prediction["problem_id"]], prediction)
                failures[prediction["stratum"]][verdict["reason"]] += 1
            if any(dict(failures[stratum]) != metric["failure_reasons"] for stratum, metric in record["metrics"].items()):
                raise ValueError("Evaluation failure-reason totals disagree with independently verified predictions.")
    return evaluation_diagnostics(run)


def audit_rollout_rows(run, rows):
    """Recompute first-success selection from every raw completed candidate."""
    config, dataset = run["config"], run["dataset"]
    selected_by_budget, previous, candidate_count, completion_tokens = {}, 0, 0, 0
    for row in rows:
        end = row["end_budget"]
        if end != min(previous + config["query_batch_size"], config["train_size"]):
            raise ValueError("Rollout query batches do not follow the fixed prompt stream.")
        problems = dataset["train"][previous:end]
        expected_ids = [problem["id"] for problem in problems for _ in range(config["rollouts"])]
        if [candidate["problem_id"] for candidate in row["candidates"]] != expected_ids:
            raise ValueError("Rollout candidate identities or multiplicities differ from the fixed acquisition batch.")
        expected_selected = []
        for index, problem in enumerate(problems):
            candidates = row["candidates"][index * config["rollouts"]:(index + 1) * config["rollouts"]]
            successful = []
            for candidate in candidates:
                verdict = verify_saved_prediction(problem, candidate)
                if verdict["correct"]: successful.append(candidate)
            if successful: expected_selected.append((problem["id"], successful[0]["text"]))
        actual_selected = [(item["problem_id"], item["completion"]) for item in row["selected"]]
        if actual_selected != expected_selected: raise ValueError("Selected extra target is not the first valid completed sampled candidate.")
        if row["usage"]["completion_tokens"] != sum(candidate["generated_tokens"] for candidate in row["candidates"]):
            raise ValueError("Rollout completion-token accounting differs from raw candidates.")
        selected_by_budget[end] = dict(expected_selected)
        candidate_count += len(row["candidates"])
        completion_tokens += row["usage"]["completion_tokens"]
        previous = end
    return selected_by_budget, {"query_batches": len(rows), "prompts_sampled": previous, "raw_candidates_verified": candidate_count,
                                "selected_targets": sum(len(selected) for selected in selected_by_budget.values()), "completion_tokens": completion_tokens}


def expected_incoming(problems, selected, arm):
    incoming = []
    for problem in problems:
        canonical = teacher_solution(problem)
        incoming.append(source_record(problem, canonical, "teacher"))
        if problem["id"] in selected:
            completion = selected[problem["id"]] if arm == "teacher_rollout" else canonical
            incoming.append(source_record(problem, completion, "generated" if arm == "teacher_rollout" else "teacher_replay"))
    return incoming


def audit_training_rows(run, arm, rows, selected_by_budget):
    """Reconstruct every teacher presentation and optimizer minibatch independently."""
    config, pending, previous = run["config"], [], 0
    expected = {key: 0 for key in PAIR_COUNTERS}
    boundaries = {}
    for row in rows:
        end = row["budget"]
        if end != min(previous + config["query_batch_size"], config["train_size"]):
            raise ValueError("Training query batches do not follow the fixed prompt stream.")
        problems = run["dataset"]["train"][previous:end]
        incoming = expected_incoming(problems, selected_by_budget[end], arm)
        if row["incoming"] != incoming: raise ValueError("Teacher coverage, target provenance, or incoming example order differs from the declared design.")
        combined = pending + incoming
        ready_count = len(combined) if end == config["train_size"] else len(combined) // config["batch_size"] * config["batch_size"]
        ready, pending = combined[:ready_count], combined[ready_count:]
        if row["trained"] != ready: raise ValueError("Trained examples do not match independent full-minibatch buffering.")
        if row["teacher_examples"] != len(problems) or row["extra_examples"] != len(selected_by_budget[end]):
            raise ValueError("Per-batch teacher/extra presentation counts disagree with the source stream.")
        if row["accepted_examples"] != len(incoming) or row["trained_examples"] != len(ready) or row["pending_examples"] != len(pending):
            raise ValueError("Per-batch accepted/trained/pending counts disagree with the source stream.")
        updates = math.ceil(len(ready) / config["batch_size"])
        if len(row["losses"]) != updates or not all(math.isfinite(value) for value in row["losses"]):
            raise ValueError("Loss records do not cover every finite optimizer update.")
        expected["problems_presented"] = end
        expected["teacher_examples"] += len(problems)
        expected["extra_examples"] += len(selected_by_budget[end])
        expected["accepted_examples"] += len(incoming)
        expected["training_examples"] += len(ready)
        expected["pending_examples"] = len(pending)
        expected["optimizer_updates"] += updates
        expected["teacher_training_examples"] += sum(item["source"] == "teacher" for item in ready)
        expected["extra_training_examples"] += sum(item["source"] != "teacher" for item in ready)
        if any(row["counters"][key] != value for key, value in expected.items()):
            raise ValueError("Cumulative source or optimizer accounting differs from independently reconstructed training.")
        if arm == "teacher_replay" and any(row["counters"][key] != 0 for key in ("rollout_candidates", "rollout_prompt_tokens", "rollout_completion_tokens", "rollout_seconds")):
            raise ValueError("Matched teacher replay claims its own rollout costs.")
        boundaries[end] = {"counters": {key: row["counters"][key] for key in PAIR_COUNTERS},
                           "incoming_ids": [item["problem_id"] for item in incoming], "trained_ids": [item["problem_id"] for item in ready]}
        previous = end
    return boundaries, {"query_batches": len(rows), "last_budget": previous, **expected}


def audit_run(run):
    root = run["directory"]
    source_hashes = {}
    for name, expected_hash in run["config"]["source_sha256"].items():
        path = root / "source" / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash: raise ValueError(f"Archived source hash mismatch for {path}.")
        source_hashes[name] = actual
    rollout_path = root / "rollout_batches.jsonl"
    rollout_rows = read_jsonl(rollout_path) if rollout_path.exists() else []
    selected_by_budget, rollout_audit = audit_rollout_rows(run, rollout_rows)
    rollout_prefixes, running = {}, {"rollout_candidates": 0, "rollout_prompt_tokens": 0, "rollout_completion_tokens": 0, "rollout_seconds": 0.0}
    for row in rollout_rows:
        running["rollout_candidates"] += len(row["candidates"])
        for counter, usage in (("rollout_prompt_tokens", "prompt_tokens"), ("rollout_completion_tokens", "completion_tokens"), ("rollout_seconds", "seconds")):
            running[counter] += row["usage"][usage]
        rollout_prefixes[row["end_budget"]] = dict(running)
    training, boundaries, availability = {}, {}, {}
    for arm in ARMS:
        path = root / f"{arm}_training.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        boundaries[arm], training[arm] = audit_training_rows(run, arm, rows, selected_by_budget)
        if arm == "teacher_rollout" and any(row["counters"][key] != value for row in rows for key, value in rollout_prefixes[row["budget"]].items()):
            raise ValueError("Cumulative rollout costs differ from recorded acquisition batches.")
        final_path = root / f"{arm}_final.json"
        checkpoints = sorted(run["metrics"][arm]) if arm in run["metrics"] else []
        for budget in checkpoints:
            if budget > 0 and run["metrics"][arm][budget]["counters"] != next(row["counters"] for row in rows if row["budget"] == budget):
                raise ValueError("Evaluation checkpoint counters disagree with training records.")
        finished = final_path.exists()
        if finished:
            final = read_json(final_path)
            if training[arm]["last_budget"] != run["config"]["train_size"] or checkpoints != run["config"]["checkpoints"]:
                raise ValueError("A final checkpoint is missing required training or evaluation budgets.")
            if final["counters"] != rows[-1]["counters"]: raise ValueError("Final checkpoint counters differ from final training record.")
        availability[arm] = {"final_available": finished, "metric_budgets": checkpoints, "last_training_budget": training[arm]["last_budget"],
                             "missing_metric_budgets": sorted(set(run["config"]["checkpoints"]) - set(checkpoints))}
    shared = sorted(boundaries[ARMS[0]].keys() & boundaries[ARMS[1]].keys())
    if any(boundaries[ARMS[0]][budget] != boundaries[ARMS[1]][budget] for budget in shared):
        raise ValueError("Augmentation and matched replay do not share source identities, optimizer boundaries, and counts.")
    return {"source_sha256": source_hashes, "rollout_provenance": rollout_audit, "training": training, "paired_batches_checked": len(shared),
            "teacher_coverage": "Exactly one canonical teacher target per new problem; extras occupy identical positions and minibatches in both arms.",
            "warm_start": "Every available arm reproduces parent SFT n0 full predictions and metrics, with zero initial weight displacement.",
            "availability": availability, "complete": all(value["final_available"] for value in availability.values())}


def metric_sets(run):
    return {**run["metrics"], HISTORICAL: run["historical"]}


def summarize_curves(runs, bootstrap_samples):
    indexed = defaultdict(list)
    for run in runs:
        for arm, records in metric_sets(run).items():
            for budget, record in records.items(): indexed[arm, budget].append((run["seed"], record))
    curves = {arm: [] for arm in (*ARMS, HISTORICAL) if any(key[0] == arm for key in indexed)}
    for (arm, budget), records in sorted(indexed.items()):
        records.sort(key=lambda item: item[0])
        point = {"budget": budget, "seeds": [seed for seed, _ in records], "clocks": {}, "metrics": {}, "weight_probes": {}}
        for clock in CLOCK_LABELS:
            point["clocks"][clock] = estimate([clocks(record["counters"])[clock] for _, record in records], bootstrap_samples)
        for stratum in records[0][1]["metrics"]:
            values = [record["metrics"][stratum]["accuracy"] for _, record in records]
            point["metrics"][stratum] = {"accuracy": estimate(values, bootstrap_samples), "error": estimate([1 - value for value in values], bootstrap_samples),
                                          "test_counts_by_seed": [record["metrics"][stratum]["count"] for _, record in records]}
        for parameter in records[0][1]["weight_probes"]:
            point["weight_probes"][parameter] = {name: estimate([record["weight_probes"][parameter][name] for _, record in records], bootstrap_samples)
                                                  for name in ("delta_norm", "relative_delta_norm")}
        curves[arm].append(point)
    return curves


def paired_contrasts(runs, bootstrap_samples):
    contrasts = []
    for first, second in ((ARMS[0], ARMS[1]), (ARMS[0], HISTORICAL), (ARMS[1], HISTORICAL)):
        points = defaultdict(list)
        for run in runs:
            metrics = metric_sets(run)
            if first not in metrics or second not in metrics: continue
            for budget in metrics[first].keys() & metrics[second].keys():
                for stratum in metrics[first][budget]["metrics"]:
                    delta = metrics[first][budget]["metrics"][stratum]["accuracy"] - metrics[second][budget]["metrics"][stratum]["accuracy"]
                    points[budget, stratum].append((run["seed"], delta))
        for (budget, stratum), values in sorted(points.items()):
            values.sort()
            contrasts.append({"first": first, "second": second, "budget": budget, "stratum": stratum,
                              "role": "paired new-arm comparison" if second != HISTORICAL else "historical same-prompt context; not independent confirmation",
                              "seeds": [seed for seed, _ in values], "accuracy_difference": estimate([value for _, value in values], bootstrap_samples)})
    return contrasts


def summarize_endpoints(runs, contrasts, bootstrap_samples):
    budget = runs[0]["config"]["train_size"]
    endpoints, costs = {}, {}
    for arm in (*ARMS, HISTORICAL):
        observed = [(run["seed"], metric_sets(run)[arm][budget]) for run in runs if arm in metric_sets(run) and budget in metric_sets(run)[arm]]
        if not observed: continue
        observed.sort(key=lambda item: item[0])
        endpoints[arm] = {stratum: estimate([record["metrics"][stratum]["accuracy"] for _, record in observed], bootstrap_samples)
                          for stratum in observed[0][1]["metrics"]}
        costs[arm] = {clock: estimate([clocks(record["counters"])[clock] for _, record in observed], bootstrap_samples) for clock in CLOCK_LABELS}
        costs[arm]["counters_by_seed"] = [{"seed": seed, "counters": record["counters"]} for seed, record in observed]
    endpoint_contrasts = [row for row in contrasts if row["budget"] == budget]
    return {"budget": budget, "accuracy": endpoints, "contrasts": endpoint_contrasts,
            "primary": [row for row in endpoint_contrasts if row["first"] == ARMS[0] and row["second"] == ARMS[1] and row["stratum"] == "in_distribution"],
            "secondary_long_horizon": [row for row in endpoint_contrasts if row["first"] == ARMS[0] and row["second"] == ARMS[1] and row["stratum"] == "long_horizon"],
            "costs": costs}


def make_plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for clock in CLOCKS:
        for stratum in summary["eval_strata"]:
            figure, axis = plt.subplots()
            for arm, points in summary["curves"].items():
                x = [point["clocks"][clock]["mean"] for point in points]
                values = [point["metrics"][stratum]["accuracy"] for point in points]
                line, = axis.plot(x, [value["mean"] for value in values], marker="o", linestyle="--" if arm == HISTORICAL else "-", label=LABELS[arm])
                bands = [(position, value["ci95"]) for position, value in zip(x, values, strict=True) if value["ci95"] is not None]
                if bands:
                    axis.fill_between([position for position, _ in bands], [interval[0] for _, interval in bands],
                                      [interval[1] for _, interval in bands], color=line.get_color(), alpha=0.2)
            axis.set(xlabel=CLOCK_LABELS[clock], ylabel="Exact test accuracy", ylim=(0, 1), title=STRATUM_LABELS[stratum])
            axis.legend()
            figure.tight_layout()
            path = output / f"{stratum}_{clock}.png"
            figure.savefig(path)
            plt.close(figure)
            paths.append(path)
    for field, label, name in (("mean_distinct_completions_per_prompt", "Distinct completions among four samples", "rollout_diversity"),
                               ("first_step_accuracy_all_candidates", "Fraction with a correct first transition", "rollout_first_step_accuracy")):
        figure, axis = plt.subplots()
        for seed, diagnostics in summary["rollout_diagnostics"].items():
            if diagnostics["status"] != "Observed rollout batches": continue
            batches = diagnostics["batches"]
            axis.plot([row["budget"] for row in batches], [row["candidate_mechanisms"][field] for row in batches], label=f"Seed {seed}")
        axis.set(xlabel="New prompts after warm-start", ylabel=label, title="Teacher-anchored sampled rollouts")
        axis.legend()
        figure.tight_layout()
        path = output / f"{name}.png"
        figure.savefig(path)
        plt.close(figure)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--weight-cosines", action="store_true")
    parser.add_argument("--weight-probe-root", type=Path)
    args = parser.parse_args()
    if args.weight_probe_root is not None and not args.weight_cosines: parser.error("--weight-probe-root requires --weight-cosines.")
    if args.bootstrap_samples < 100: raise ValueError("Use at least 100 bootstrap replicates.")
    load_dotenv(Path.home() / ".env")
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "matplotlib")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs([path.resolve() for path in args.runs])
    audits = {str(run["seed"]): audit_run(run) for run in tqdm(runs, desc="Independent training audit")}
    contrasts = paired_contrasts(runs, args.bootstrap_samples)
    summary = {"model": runs[0]["config"]["model"], "seeds": [run["seed"] for run in runs], "runs": [str(run["directory"]) for run in runs],
               "eval_strata": runs[0]["dataset"]["metadata"]["eval_strata"], "audits": audits, "complete": all(audit["complete"] for audit in audits.values()),
               "interpretation": "Both new arms receive one canonical teacher target for every new prompt. They add the same selected prompt identities at identical stream positions and optimizer minibatch boundaries. Teacher-rollout extras are verified current-model trajectories sampled before that query batch is taught; teacher-replay extras are canonical duplicates. Historical SFT uses the same parents, prompt streams, and warm checkpoints; it is context from an earlier experiment, not fresh independent confirmation. Primary comparison is paired new-arm in-distribution endpoint accuracy; long-horizon endpoint is secondary. No scaling exponent is fitted.",
               "cost_conventions": "Costs exclude common warm-start, evaluation, model loading, checkpointing, and external teacher construction. Tokens count unpadded training inputs including targets, plus rollout inputs/outputs; they are not FLOPs. Phase seconds combine recorded training and rollout wall time. Matched replay excludes acquisition costs for the generated curriculum. Equal example identities and optimizer updates do not imply equal token lengths or compute.",
               "uncertainty": "Bootstrap intervals resample independent training seeds and preserve arm pairing. Three seeds offer limited uncertainty resolution; pointwise intervals are descriptive and do not adjust for multiple comparisons. One seed has no interval. Partial runs and missing final checkpoints are explicit in audits and endpoint seed counts.",
               "curves": summarize_curves(runs, args.bootstrap_samples), "paired_contrasts": contrasts,
               "endpoints": summarize_endpoints(runs, contrasts, args.bootstrap_samples),
               "warm_start": [{"seed": run["seed"], "parent_run": str(run["parent"]), "counters": run["warm_counters"], "metrics": run["historical"][0]["metrics"]} for run in runs],
               "rollout_diagnostics": {str(run["seed"]): rollout_diagnostics(run) for run in runs},
               "evaluation_diagnostics": {str(run["seed"]): audited_evaluation_diagnostics(run) for run in tqdm(runs, desc="Independent evaluation audit")},
               "historical_evaluation_diagnostics": {str(run["seed"]): audited_evaluation_diagnostics({**run, "directory": run["parent"], "metrics": {"sft": run["historical"]}}) for run in runs},
               "run_configurations": [run["config"] for run in runs]}
    if args.weight_cosines: summary["weight_cosines"] = weight_cosines(runs, args.weight_probe_root.resolve() if args.weight_probe_root is not None else None)
    paths = make_plots(summary, output)
    summary["plots"] = [str(path) for path in paths]
    write_json(output / "summary.json", summary)
    for path in paths: print(path)


if __name__ == "__main__": main()
