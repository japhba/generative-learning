"""Audit full feedback trajectories and compare all retained learning curves."""

import argparse
from collections import Counter, deque
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from analyze_training import estimate
from feedback_task import STRATA, Streams, group_id, group_split, state_id

ARMS = ("expert", "on_policy", "random", "frozen_policy")
LABELS = {"expert": "Expert demonstrations", "on_policy": "Current-model rollouts + correction", "random": "Random exploration + correction", "frozen_policy": "Frozen-model rollouts + correction"}


def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(line) for line in path.read_text().splitlines()]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def oracle(modulus, add, subtract, goal):
    successors = [[(state + add) % modulus, (state - subtract) % modulus, (2 * state) % modulus] for state in range(modulus)]
    incoming = [[] for _ in range(modulus)]
    for state, following in enumerate(successors):
        for target in following: incoming[target].append(state)
    depths, queue = {goal: 0}, deque([goal])
    while queue:
        state = queue.popleft()
        for predecessor in incoming[state]:
            if predecessor not in depths:
                depths[predecessor] = depths[state] + 1
                queue.append(predecessor)
    assert len(depths) == modulus
    actions = {state: [action for action, target in enumerate(row) if depths[target] == depths[state] - 1] for state, row in enumerate(successors)}
    return successors, depths, actions


def task_oracle(problem): return oracle(problem["modulus"], problem["add"], problem["subtract"], problem["target"])


def audit_rows(rows, streams, policy):
    lanes = len(streams.states)
    assert len(rows) % lanes == 0
    for offset in range(0, len(rows), lanes):
        actions = []
        for index, row in enumerate(rows[offset:offset + lanes]):
            p = row["problem"]
            assert group_split(p) == "train"
            assert p == streams.problems[index] and row["state"] == streams.states[index]
            assert row["episode_age"] == streams.ages[index]
            assert row["id"] == state_id(p, row["state"])
            following, depths, optimal = task_oracle(p)
            assert row["distance"] == depths[row["state"]] > 0
            assert row["optimal"] == optimal[row["state"]]
            assert row["following"] == following[row["state"]][row["action"]]
            assert row["action_optimal"] == (row["action"] in row["optimal"])
            if policy == "expert": assert row["action_optimal"]
            actions.append(row["action"])
        streams.advance(actions)


def grade_predictions(dataset, predictions):
    expected = [(stratum, problem) for stratum, problems in dataset["eval"].items() for problem in problems]
    assert len(predictions) == len(expected)
    result = {}
    for row, (stratum, problem) in zip(predictions, expected, strict=True):
        assert row["stratum"] == stratum and row["problem"] == problem and row["id"] == problem["id"]
        assert group_split(problem) == dataset["evaluation_split"]
        successors, depths, optimal = task_oracle(problem)
        assert row["states"][0] == problem["start"] and depths[problem["start"]] == problem["distance"]
        assert 1 <= len(row["actions"]) <= 8 and len(row["states"]) == len(row["actions"]) + 1
        assert len(row["optimal"]) == len(row["actions"])
        for index, action in enumerate(row["actions"]):
            state = row["states"][index]
            assert state != problem["target"]
            assert row["states"][index + 1] == successors[state][action]
            assert row["optimal"][index] == (action in optimal[state])
        assert row["success"] == (row["states"][-1] == problem["target"])
        assert row["success"] or len(row["actions"]) == 8
    for stratum in dataset["eval"]:
        rows = [row for row in predictions if row["stratum"] == stratum]
        result[stratum] = {"count": len(rows), "accuracy": float(np.mean([row["success"] for row in rows])),
                            "first_action_optimal": float(np.mean([row["optimal"][0] for row in rows])),
                            "all_action_optimal": float(np.mean([x for row in rows for x in row["optimal"]])),
                            "mean_moves": float(np.mean([len(row["actions"]) for row in rows]))}
    return result


def load_run(directory):
    config, dataset = read(directory / "config.json"), read(directory / "dataset.json")
    assert digest(directory / "dataset.json") == config["dataset_sha256"]
    assert {name: digest(directory / "source" / name) for name in config["source_sha256"]} == config["source_sha256"]
    warm = read(directory / "warm_training.json")
    warm_streams = Streams(dataset["train"], config["seed"], config["lanes"], config["horizon"])
    audit_rows(warm, warm_streams, "expert")
    records, audits, diagnostics = {}, {}, {}
    for arm in ARMS:
        final = read(directory / f"{arm}_final.json")
        logs = lines(directory / f"{arm}_training.jsonl")
        stream = Streams(dataset["train"], config["seed"] + 30000, config["lanes"], config["horizon"])
        history, seen, depths, groups, rounds = list(warm), set(), Counter(), set(), []
        for index, log in enumerate(logs):
            budget = (index + 1) * config["round_labels"]
            assert log["budget"] == budget and len(log["rows"]) == config["round_labels"]
            audit_rows(log["rows"], stream, arm)
            history.extend(log["rows"])
            assert len(log["replay_indices"]) == config["round_labels"]
            assert all(0 <= item < len(history) for item in log["replay_indices"])
            assert sorted(log["order"]) == list(range(2 * config["round_labels"]))
            count = log["counters"]
            assert count["labels_acquired"] == budget
            assert count["training_examples"] == count["supervised_tokens"] == 2 * budget
            assert count["optimizer_updates"] == 2 * budget // config["batch_size"]
            for row in log["rows"]:
                seen.add(row["id"]); groups.add(group_id(row["problem"])); depths[row["distance"]] += 1
            rounds.append({"budget": budget, "unique_states": len(seen), "unique_goal_contexts": len(groups),
                           "episodes_started": stream.episodes_started, "episodes_succeeded": stream.episodes_succeeded,
                           "distance_counts": dict(depths), "optimal_action_fraction": log["optimal_action_fraction"]})
        assert logs[-1]["budget"] == config["labels"] and final["counters"] == logs[-1]["counters"]
        if arm == "frozen_policy": assert read(directory / "frozen_acquisition.json")["rows"] == [row for log in logs for row in log["rows"]]
        metrics = lines(directory / f"{arm}_metrics.jsonl")
        assert [row["budget"] for row in metrics] == config["checkpoints"]
        for record in metrics:
            assert record["seed"] == config["seed"] and record["arm"] == arm
            predictions = read(directory / f"{arm}_n{record['budget']}_predictions.json")
            assert record["metrics"] == grade_predictions(dataset, predictions)
            if record["budget"]:
                assert record["counters"] == logs[record["budget"] // config["round_labels"] - 1]["counters"]
        records[arm] = metrics
        diagnostics[arm] = rounds
        audits[arm] = {"states_reverified": config["labels"], "evaluation_trajectories_reverified": len(metrics) * sum(len(v) for v in dataset["eval"].values()),
                        "all_checkpoints_present": True, "final_model_sha256": final["model_sha256"]}
    baseline = read(directory / "expert_n0_predictions.json")
    assert all(read(directory / f"{arm}_n0_predictions.json") == baseline for arm in ARMS)
    return {"config": config, "seed": config["seed"], "dataset": dataset, "records": records, "audits": audits, "diagnostics": diagnostics}


def analyze(runs, output):
    loaded = [load_run(run) for run in tqdm(runs, desc="Independent trajectory audits")]
    first = loaded[0]
    assert len({run["seed"] for run in loaded}) == len(loaded)
    for run in loaded[1:]:
        assert run["dataset"]["eval"] == first["dataset"]["eval"]
        for key in ("model", "revision", "labels", "warm_labels", "learning_rate", "checkpoints", "source_sha256"):
            assert run["config"][key] == first["config"][key]
    curves = {}
    for arm in ARMS:
        curves[arm] = []
        for index, budget in enumerate(first["config"]["checkpoints"]):
            rows = [run["records"][arm][index] for run in loaded]
            metrics = {stratum: {field: estimate([row["metrics"][stratum][field] for row in rows])
                                for field in ("accuracy", "first_action_optimal", "all_action_optimal", "mean_moves")} for stratum in STRATA}
            clocks = {"labels": estimate([budget] * len(rows)),
                      "training_examples": estimate([row["counters"]["training_examples"] for row in rows]),
                      "phase_seconds": estimate([sum(row["counters"][key] for key in ("training_seconds", "generation_seconds", "oracle_seconds")) for row in rows]),
                      "tokens": estimate([sum(row["counters"][key] for key in ("training_prompt_tokens", "supervised_tokens", "generation_prompt_tokens", "generated_actions")) for row in rows])}
            curves[arm].append({"budget": budget, "metrics": metrics, "clocks": clocks})
    contrasts = [{"first": "on_policy", "second": other, "stratum": stratum,
                  "accuracy_difference": estimate([run["records"]["on_policy"][-1]["metrics"][stratum]["accuracy"] - run["records"][other][-1]["metrics"][stratum]["accuracy"] for run in loaded])}
                 for other in ("expert", "random", "frozen_policy") for stratum in STRATA]
    summary = {"complete": True, "seeds": [run["seed"] for run in loaded], "model": first["config"]["model"], "budget": first["config"]["labels"],
               "evaluation_split": first["dataset"]["evaluation_split"], "curves": curves, "endpoint_contrasts": contrasts,
               "audits": {run["seed"]: run["audits"] for run in loaded}, "diagnostics": {run["seed"]: run["diagnostics"] for run in loaded},
               "uncertainty": "Paired training-seed bootstrap conditional on common test problems; no item-level uncertainty. Single-seed pilot has no confidence interval. No scaling exponent fitted.",
               "interpretation": "Constrained next-action LM with exact simulator transitions and oracle labels. This differs from free-text trajectory generation. Labels and updates match, acquired distributions and compute do not."}
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for stratum in STRATA:
        for clock in ("labels", "phase_seconds"):
            figure, axis = plt.subplots()
            for arm in ARMS:
                rows = curves[arm]
                x = [row["clocks"][clock]["mean"] for row in rows]
                y = [row["metrics"][stratum]["accuracy"]["mean"] for row in rows]
                axis.plot(x, y, marker="o", label=LABELS[arm])
                if len(loaded) > 1:
                    interval = np.array([row["metrics"][stratum]["accuracy"]["ci95"] for row in rows])
                    axis.fill_between(x, interval[:, 0], interval[:, 1], alpha=0.15)
            axis.set(xlabel="Acquired oracle state labels" if clock == "labels" else "Acquisition + training seconds", ylabel="Goal reached within 8 moves", ylim=(0, 1), title=stratum.replace("_", " "))
            axis.legend()
            figure.tight_layout()
            path = output / f"{stratum}_{clock}.png"
            figure.savefig(path)
            plt.close(figure)
            paths.append(str(path.resolve()))
    summary["plots"] = paths
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for path in paths: print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/matplotlib")
    analyze(args.runs, args.output)


if __name__ == "__main__": main()
