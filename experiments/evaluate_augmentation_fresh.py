"""Freeze unseen test identities, then evaluate nine fixed final checkpoints without training."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from random import Random

from dotenv import load_dotenv
from tqdm.auto import tqdm

from analyze_training import estimate, read_json, write_json
from training_task import SPLIT_VERSION, _paths, identity, split_for, verify_solution

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("teacher_rollout", "teacher_replay", "historical_sft")
SEEDS = (6200, 6201, 6202)
SAMPLE_SEED = 93201
PER_STRATUM = 256
SOURCES = ("evaluate_augmentation_fresh.py", "training_task.py", "train_planning.py")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def sample_fresh(metadata, excluded_ids, sample_seed, per_stratum):
    pools = {stratum: [] for stratum in metadata["eval_strata"]}
    full_counts = Counter()
    for modulus, add, subtract in product(metadata["moduli"], metadata["increments"], metadata["increments"]):
        for start in range(modulus):
            for target, path in _paths(modulus, add, subtract, start).items():
                distance = len(path)
                strata = [name for name, distances in metadata["eval_strata"].items() if distance in distances]
                if not strata: continue
                problem_id = identity(modulus, add, subtract, start, target)
                if split_for(problem_id) != "test": continue
                stratum = strata[0]
                full_counts[stratum] += 1
                if problem_id in excluded_ids: continue
                pools[stratum].append({"id": problem_id, "split": "test", "modulus": modulus, "add": add, "subtract": subtract,
                                       "start": start, "target": target, "max_steps": metadata["max_steps"], "shortest_distance": distance})
    if any(len(pool) < per_stratum for pool in pools.values()): raise ValueError("Insufficient unseen identities for the predeclared sample size.")
    rng = Random(sample_seed)
    selected = {stratum: rng.sample(pool, per_stratum) for stratum, pool in pools.items()}
    identities = [problem["id"] for pool in selected.values() for problem in pool]
    if len(identities) != len(set(identities)) or set(identities) & excluded_ids:
        raise ValueError("Fresh evaluation contains duplicates or previously evaluated identities.")
    counts = {name: {"full_heldout_pool": full_counts[name], "previously_evaluated": full_counts[name] - len(pool),
                     "eligible_unseen_pool": len(pool), "selected": len(selected[name])} for name, pool in pools.items()}
    return selected, counts


def prepare(output):
    if output.exists(): raise FileExistsError("Fresh evaluation output already exists; never resample after inspecting outcomes.")
    metadata = read_json(ROOT / "results/training_main_seed6200/dataset.json")["metadata"]
    excluded, sources = set(), []
    for path in sorted((ROOT / "results").glob("training*/dataset.json")):
        dataset = read_json(path)
        if dataset["metadata"]["task"] != SPLIT_VERSION: continue
        identities = {problem["id"] for group in dataset["eval"].values() for problem in group}
        excluded.update(identities)
        sources.append({"path": str(path.resolve()), "sha256": sha256(path), "evaluated_identity_count": len(identities)})
    selected, pool_counts = sample_fresh(metadata, excluded, SAMPLE_SEED, PER_STRATUM)
    frozen_metadata = {**metadata, "seed": SAMPLE_SEED, "purpose": "Supplemental fresh-identity endpoint validation; no training examples.",
                       "eval_per_stratum": PER_STRATUM, "candidate_counts": {name: counts["eligible_unseen_pool"] for name, counts in pool_counts.items()}}
    dataset = {"eval": selected, "metadata": frozen_metadata}
    output.mkdir(parents=True)
    write_json(output / "dataset.json", dataset)
    source_directory = output / "selection_source"
    source_directory.mkdir()
    source_hashes = {}
    for name in SOURCES:
        path = ROOT / "experiments" / name
        (source_directory / name).write_bytes(path.read_bytes())
        source_hashes[name] = sha256(path)
    record = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "sample_seed": SAMPLE_SEED, "per_stratum": PER_STRATUM,
              "training_seeds": list(SEEDS), "arms": list(ARMS), "dataset_sha256": sha256(output / "dataset.json"),
              "selection_source_sha256": source_hashes, "excluded_dataset_sources": sources, "excluded_unique_identity_count": len(excluded),
              "excluded_identity_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest(), "pool_counts": pool_counts,
              "sampling": "Enumerate the fixed hashed test universe, exclude all prior evaluation identities, then uniform sample without replacement separately by shortest-distance stratum using Random(93201). All nine checkpoints use the identical 768 identities. Shortest distance is used only to stratify; generation prompts do not expose it.",
              "analysis_plan": "Primary original evaluation remains unchanged. This secondary validation compares final teacher_rollout and teacher_replay paired within training seed; historical SFT is context. Report strict full-trajectory accuracy plus EOS, three seed values and paired seed-level uncertainty. Reusing common test questions does not create independent training replications. No hyperparameter adjustment or checkpoint selection follows these results.",
              "decode": {"samples": 1, "temperature": 0.0, "generation_batch_size": 64, "max_new_tokens": 64},
              "checkpoint_rule": "All predeclared final checkpoints from seeds 6200, 6201, 6202; no intermediate checkpoint selection. Require completed final markers."}
    write_json(output / "selection_record.json", record)
    return record


def frozen_dataset(output):
    record, dataset = read_json(output / "selection_record.json"), read_json(output / "dataset.json")
    if sha256(output / "dataset.json") != record["dataset_sha256"]: raise ValueError("Frozen test dataset was modified.")
    for name, expected in record["selection_source_sha256"].items():
        if sha256(output / "selection_source" / name) != expected: raise ValueError("Frozen selection source was modified.")
    for source in record["excluded_dataset_sources"]:
        if sha256(Path(source["path"])) != source["sha256"]: raise ValueError("Previously evaluated dataset changed after exclusion was frozen.")
    return record, dataset


def checkpoint_run(seed, arm):
    prefix = "training_main" if arm == "historical_sft" else "training_augmentation"
    return ROOT / "results" / f"{prefix}_seed{seed}", "sft" if arm == "historical_sft" else arm


def freeze_evaluation_source(output, source_root=ROOT / "experiments"):
    """All checkpoints must use one archived evaluator version, frozen before prediction."""
    selection = read_json(output / "selection_record.json")
    current = {name: sha256(source_root / name) for name in SOURCES}
    for name in ("training_task.py", "train_planning.py"):
        if current[name] != selection["selection_source_sha256"][name]:
            raise ValueError("Evaluation helper or task differs from the frozen selection version.")
    archive = output / "evaluation_source"
    if archive.exists():
        manifest = read_json(archive / "manifest.json")
    else:
        archive.mkdir()
        for name in SOURCES: (archive / name).write_bytes((source_root / name).read_bytes())
        manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": current}
        write_json(archive / "manifest.json", manifest)
    if current != manifest["source_sha256"]: raise ValueError("Live evaluator sources changed after the first evaluation version was frozen.")
    if {name: sha256(archive / name) for name in SOURCES} != manifest["source_sha256"]:
        raise ValueError("Archived evaluation sources were modified.")
    return manifest


def verified_metrics(dataset, predictions):
    expected = Counter((problem["id"], stratum) for stratum, group in dataset["eval"].items() for problem in group)
    if Counter((row["problem_id"], row["stratum"]) for row in predictions) != expected:
        raise ValueError("Predictions do not exactly cover the frozen test identities and strata.")
    problems = {problem["id"]: problem for group in dataset["eval"].values() for problem in group}
    grouped = defaultdict(list)
    for prediction in predictions:
        verdict = verify_solution(problems[prediction["problem_id"]], prediction["text"])
        if not prediction["ended"]: verdict = {**verdict, "correct": False, "reason": "truncated"}
        if json.loads(json.dumps(prediction["verdict"])) != json.loads(json.dumps(verdict)):
            raise ValueError("Saved prediction differs from independent verification plus EOS.")
        grouped[prediction["stratum"]].append(verdict)
    return {stratum: {"count": len(rows), "accuracy": sum(row["correct"] for row in rows) / len(rows),
                      "correct": sum(row["correct"] for row in rows), "failure_reasons": dict(Counter(row["reason"] for row in rows))}
            for stratum, rows in grouped.items()}


def evaluate(output, seed, arm):
    from train_planning import configure_environment, generate, initialize, load_model
    import torch
    record, dataset = frozen_dataset(output)
    evaluation_manifest = freeze_evaluation_source(output)
    destination = output / f"seed{seed}_{arm}"
    if destination.exists(): raise FileExistsError("An evaluation for this checkpoint already exists; inspect it before any rerun.")
    run, saved_arm = checkpoint_run(seed, arm)
    config, final = read_json(run / "config.json"), read_json(run / f"{saved_arm}_final.json")
    for name in ("train_planning.py", "training_task.py"):
        if sha256(ROOT / "experiments" / name) != config["source_sha256"][name]:
            raise ValueError("Evaluation helper or task implementation changed from the trained checkpoint protocol.")
    if config["seed"] != seed or final["counters"]["problems_presented"] != config["train_size"]:
        raise ValueError("Checkpoint does not match the predeclared completed endpoint.")
    if config["generation_batch_size"] != 64 or config["max_new_tokens"] != 64: raise ValueError("Parent decoding settings differ from the frozen plan.")
    checkpoint = Path(final["checkpoint"])
    configure_environment()
    initialize(seed)
    model, tokenizer = load_model(str(checkpoint))
    model.requires_grad_(False)
    predictions, usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    prompt_tokens_independent = 0
    from training_task import prompt
    for stratum, problems in tqdm(dataset["eval"].items(), desc=f"Fresh evaluation seed {seed} {arm}"):
        generated, cost = generate(model, tokenizer, problems, 1, 64, 64, 0.0, seed + 900000)
        predictions.extend({**prediction, "stratum": stratum, "shortest_distance": problem["shortest_distance"]}
                           for prediction, problem in zip(generated, problems, strict=True))
        prompt_tokens_independent += sum(len(tokenizer.encode(prompt(problem), add_special_tokens=False)) for problem in problems)
        for field, value in cost.items(): usage[field] += value
    if usage["prompt_tokens"] != prompt_tokens_independent: raise ValueError("Prompt token accounting differs from independently tokenized frozen prompts.")
    if usage["completion_tokens"] != sum(row["generated_tokens"] for row in predictions): raise ValueError("Completion token accounting differs from saved candidates.")
    metrics = verified_metrics(dataset, predictions)
    destination.mkdir()
    write_json(destination / "predictions.json", predictions)
    result = {"seed": seed, "arm": arm, "parent_run": str(run), "checkpoint": str(checkpoint), "parent_config_sha256": sha256(run / "config.json"),
              "checkpoint_model_sha256": sha256(checkpoint / "model.safetensors"), "selection_record_sha256": sha256(output / "selection_record.json"),
              "dataset_sha256": record["dataset_sha256"], "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "decode": record["decode"],
              "metrics": metrics, "usage": usage, "independently_tokenized_prompt_tokens": prompt_tokens_independent,
              "weight_updates": 0, "parameters_require_grad": any(parameter.requires_grad for parameter in model.parameters()),
              "evaluation_source_sha256": evaluation_manifest["source_sha256"], "evaluation_source_directory": str(output / "evaluation_source"),
              "evaluation_source_manifest_sha256": sha256(output / "evaluation_source/manifest.json")}
    write_json(destination / "result.json", result)
    del model, tokenizer
    torch.cuda.empty_cache()
    return result


def summarize(output):
    record, dataset = frozen_dataset(output)
    evaluation_archive = output / "evaluation_source"
    evaluation_manifest = read_json(evaluation_archive / "manifest.json")
    if {name: sha256(evaluation_archive / name) for name in SOURCES} != evaluation_manifest["source_sha256"]:
        raise ValueError("Archived evaluation sources were modified.")
    results = {}
    for seed in SEEDS:
        for arm in ARMS:
            root = output / f"seed{seed}_{arm}"
            result = read_json(root / "result.json")
            if result["seed"] != seed or result["arm"] != arm or result["dataset_sha256"] != record["dataset_sha256"]:
                raise ValueError("Endpoint evaluation metadata disagrees with the frozen plan.")
            if result["selection_record_sha256"] != sha256(output / "selection_record.json") or result["decode"] != record["decode"]:
                raise ValueError("Frozen selection record or decoding plan changed after prediction.")
            if result["evaluation_source_sha256"] != evaluation_manifest["source_sha256"] or result["evaluation_source_manifest_sha256"] != sha256(evaluation_archive / "manifest.json"):
                raise ValueError("Endpoints were not evaluated with the identical archived evaluator version.")
            predictions = read_json(root / "predictions.json")
            if result["metrics"] != verified_metrics(dataset, predictions): raise ValueError("Saved metrics differ from independently regraded predictions.")
            if result["usage"]["completion_tokens"] != sum(row["generated_tokens"] for row in predictions): raise ValueError("Saved output-token accounting disagrees with predictions.")
            if result["usage"]["prompt_tokens"] != result["independently_tokenized_prompt_tokens"]: raise ValueError("Saved input-token accounting disagrees with independent tokenization.")
            if result["weight_updates"] != 0 or result["parameters_require_grad"]: raise ValueError("Fresh evaluation unexpectedly enabled learning.")
            results[seed, arm] = result
    endpoint = {arm: {stratum: estimate([results[seed, arm]["metrics"][stratum]["accuracy"] for seed in SEEDS])
                      for stratum in dataset["eval"]} for arm in ARMS}
    contrasts = []
    for first, second in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
        for stratum in dataset["eval"]:
            values = [results[seed, first]["metrics"][stratum]["accuracy"] - results[seed, second]["metrics"][stratum]["accuracy"] for seed in SEEDS]
            contrasts.append({"first": first, "second": second, "stratum": stratum, "seeds": list(SEEDS), "accuracy_difference": estimate(values),
                              "role": "secondary fresh-identity validation" if second == ARMS[1] else "historical same-checkpoint context"})
    summary = {"selection_record": record, "complete": True, "seeds": list(SEEDS), "per_stratum": PER_STRATUM, "accuracy": endpoint,
               "paired_contrasts": contrasts, "per_checkpoint": list(results.values()),
               "uncertainty": "Paired bootstrap resamples three training seeds only, conditional on this common frozen set of 768 test problems. Questions are reused across checkpoints and do not create nine independent replications. Three seeds provide limited uncertainty resolution. This excludes item-sampling uncertainty and is supplemental validation, not a new primary endpoint or new training replication.",
               "total_evaluation_usage": {field: sum(result["usage"][field] for result in results.values()) for field in ("prompt_tokens", "completion_tokens", "seconds")}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, sharey=True)
    for axis, stratum in zip(axes, dataset["eval"], strict=True):
        for index, arm in enumerate(ARMS):
            values = endpoint[arm][stratum]
            points = axis.scatter([index] * len(SEEDS), values["by_seed"])
            axis.plot(index, values["mean"], marker="_", color=points.get_facecolor()[0])
        axis.set(xticks=range(len(ARMS)), xticklabels=["Rollout\n+ teacher", "Matched\nteacher", "Historical\nSFT"], title=stratum.replace("_", " "), ylim=(0, 1))
    axes[0].set_ylabel("Exact accuracy on fresh identities")
    figure.tight_layout()
    path = output / "fresh_endpoint_comparison.png"
    figure.savefig(path)
    plt.close(figure)
    summary["plot"] = str(path)
    write_json(output / "summary.json", summary)
    print(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate", "summarize"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/training_augmentation_fresh_eval")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / "matplotlib")
    output = args.output.resolve()
    if args.mode == "prepare": prepare(output)
    elif args.mode == "evaluate":
        if args.seed is None or args.arm is None: parser.error("evaluate requires --seed and --arm.")
        evaluate(output, args.seed, args.arm)
    else: summarize(output)


if __name__ == "__main__": main()
