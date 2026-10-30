"""Teacher supervision plus verified rollouts, with a matched teacher-replay control."""

import argparse
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm

from train_planning import ROOT, append_json, completed_minibatches, configure_environment, evaluate, fresh_counters, generate, initialize, load_model, optimizer_for, probe_weights, save_json, train_examples
from training_task import teacher_solution, verify_solution

ARMS = ("teacher_rollout", "teacher_replay")
SOURCES = ("train_augmentation.py", "train_planning.py", "training_task.py")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def select_successes(problems, predictions, samples):
    assert len(predictions) == len(problems) * samples
    selected = []
    for index, problem in enumerate(problems):
        candidates = predictions[index * samples:(index + 1) * samples]
        assert all(candidate["problem_id"] == problem["id"] for candidate in candidates)
        valid = [candidate for candidate in candidates if candidate["verdict"]["correct"]]
        if valid:
            winner = valid[0]
            verdict = verify_solution(problem, winner["text"])
            assert winner["ended"] and verdict["correct"]
            selected.append({"problem_id": problem["id"], "completion": winner["text"], "steps": verdict["steps"], "teacher_steps": problem["shortest_distance"]})
    return selected


def augment_examples(problems, selected, arm):
    """Every problem gets its teacher target, followed by at most one supplemental target."""
    assert arm in ARMS
    indexed = {row["problem_id"]: row for row in selected}
    assert len(indexed) == len(selected)
    assert indexed.keys() <= {problem["id"] for problem in problems}
    incoming = []
    for problem in problems:
        canonical = teacher_solution(problem)
        incoming.append({"problem_id": problem["id"], "completion": canonical, "source": "teacher"})
        if problem["id"] in indexed:
            generated = arm == "teacher_rollout"
            incoming.append({"problem_id": problem["id"], "completion": indexed[problem["id"]]["completion"] if generated else canonical,
                             "source": "generated" if generated else "teacher_replay"})
    return incoming


def prepare(args):
    cache = configure_environment()
    parent, output = args.parent.resolve(), args.output.resolve()
    previous = json.loads((parent / "config.json").read_text())
    assert previous["seed"] in (6200, 6201, 6202) and previous["train_size"] == 4096
    for name, expected in previous["source_sha256"].items():
        assert sha256_file(ROOT / "experiments" / name) == expected, f"Parent implementation changed: {name}"
    warm = args.warm_checkpoint.resolve() if args.warm_checkpoint is not None else Path(previous["cache_directory"]) / "warm"
    assert (warm / "model.safetensors").is_file()
    config = {**previous, "arms": list(ARMS), "parent_run": str(parent), "parent_config_sha256": sha256_file(parent / "config.json"),
              "dataset_sha256": sha256_file(parent / "dataset.json"), "warm_checkpoint": str(warm),
              "warm_model_sha256": sha256_file(warm / "model.safetensors"), "cache_directory": str(cache / "training" / output.name),
              "training": "All original teacher examples plus accepted supplemental targets; per-problem teacher then supplemental order. Original full-weight token-mean completion CE, AdamW, learning rate and accepted-minibatch buffer retained.",
              "rollout_method": "Generate four current-model proposals before training on the current32 problems. Always retain all canonical teacher examples, then supplement each solved problem with its first verified completed proposal.",
              "control": "teacher_replay inherits exactly the teacher_rollout accepted prompt identities and supplemental positions, replacing only each supplemental generated target with the canonical teacher target. All teacher targets, stream positions and optimizer boundaries match.",
              "interpretation": "Exploratory follow-up using previously inspected test identities. Tests supplemental generated targets under continuing teacher access, including target-length/token-weight differences; not an isolated exploration or equal-compute effect.",
              "source_sha256": {name: sha256_file(ROOT / "experiments" / name) for name in SOURCES}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        assert json.loads((output / "config.json").read_text()) == config, "Configuration or code changed; use a fresh output directory."
    else:
        save_json(output / "config.json", config)
        (output / "dataset.json").write_bytes((parent / "dataset.json").read_bytes())
        source = output / "source"
        source.mkdir()
        for name in SOURCES: (source / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    assert sha256_file(output / "dataset.json") == config["dataset_sha256"], "Copied dataset changed."
    for name, expected in config["source_sha256"].items():
        assert sha256_file(output / "source" / name) == expected, f"Copied source changed: {name}"
    return output, config, json.loads((output / "dataset.json").read_text())


def run_arm(arm, output, config, dataset):
    if (output / f"{arm}_metrics.jsonl").exists(): raise FileExistsError("Arm outputs already exist; inspect before any rerun.")
    initialize(config["seed"])
    model, tokenizer = load_model(config["warm_checkpoint"])
    reference = probe_weights(model)
    optimizer = optimizer_for(model, config["learning_rate"])
    counters = {**fresh_counters(), "teacher_examples": 0, "extra_examples": 0, "teacher_training_examples": 0, "extra_training_examples": 0}
    baseline = evaluate(model, tokenizer, dataset, config, arm, 0, counters, output, reference)
    original_baseline = json.loads((Path(config["parent_run"]) / "sft_metrics.jsonl").read_text().splitlines()[0])
    assert baseline["metrics"] == original_baseline["metrics"], "Shared warm-start evaluation differs from the parent baseline."
    replay = {row["end_budget"]: row for row in map(json.loads, (output / "rollout_batches.jsonl").read_text().splitlines())} if arm == "teacher_replay" else {}
    pending, problems_by_id = [], {problem["id"]: problem for problem in dataset["train"]}
    for offset in tqdm(range(0, len(dataset["train"]), config["query_batch_size"]), desc=arm):
        problems = dataset["train"][offset:offset + config["query_batch_size"]]
        end = offset + len(problems)
        if arm == "teacher_rollout":
            predictions, usage = generate(model, tokenizer, problems, config["rollouts"], config["generation_batch_size"], config["max_new_tokens"], config["temperature"], config["seed"] * 10000 + end)
            selected = select_successes(problems, predictions, config["rollouts"])
            append_json(output / "rollout_batches.jsonl", {"end_budget": end, "selected": selected, "candidates": predictions, "usage": usage})
            counters["rollout_candidates"] += len(predictions)
            counters["rollout_prompt_tokens"] += usage["prompt_tokens"]
            counters["rollout_completion_tokens"] += usage["completion_tokens"]
            counters["rollout_seconds"] += usage["seconds"]
        else:
            selected = replay[end]["selected"]
        incoming = augment_examples(problems, selected, arm)
        ready, pending = completed_minibatches(pending, incoming, config["batch_size"], end == len(dataset["train"]))
        examples = [(problems_by_id[row["problem_id"]], row["completion"]) for row in ready]
        losses = train_examples(model, tokenizer, optimizer, examples, counters, config["batch_size"])
        counters["teacher_examples"] += len(problems)
        counters["extra_examples"] += len(selected)
        counters["teacher_training_examples"] += sum(row["source"] == "teacher" for row in ready)
        counters["extra_training_examples"] += sum(row["source"] != "teacher" for row in ready)
        counters["accepted_examples"] += len(incoming)
        counters["pending_examples"] = len(pending)
        counters["problems_presented"] = end
        assert counters["training_examples"] == counters["teacher_training_examples"] + counters["extra_training_examples"]
        append_json(output / f"{arm}_training.jsonl", {"budget": end, "teacher_examples": len(problems), "extra_examples": len(selected),
                    "accepted_examples": len(incoming), "trained_examples": len(ready), "pending_examples": len(pending),
                    "incoming": incoming, "trained": ready, "losses": losses, "counters": dict(counters)})
        if end in config["checkpoints"]: evaluate(model, tokenizer, dataset, config, arm, end, counters, output, reference)
    assert counters["teacher_training_examples"] == len(dataset["train"])
    assert counters["extra_training_examples"] == counters["extra_examples"] and counters["pending_examples"] == 0
    destination = Path(config["cache_directory"]) / arm
    destination.mkdir(parents=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    save_json(output / f"{arm}_final.json", {"checkpoint": str(destination), "counters": counters})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", *ARMS))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path)
    args = parser.parse_args()
    output, config, dataset = prepare(args)
    if args.mode != "prepare": run_arm(args.mode, output, config, dataset)


if __name__ == "__main__": main()
