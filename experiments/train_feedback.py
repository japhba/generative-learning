"""Full-weight next-action learning from expert, random, frozen or live policy states."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from feedback_task import STRATA, Streams, distance, group_split, make_dataset, next_state, optimal_actions, prompt, state_id
from train_planning import MODEL, REVISION, PROBES, append_json, configure_environment, initialize, load_model, optimizer_for, probe_weights, save_json

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("expert", "on_policy", "random", "frozen_policy")
SOURCES = ("train_feedback.py", "feedback_task.py", "train_planning.py", "training_task.py")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def action_tokens(tokenizer):
    encoded = [tokenizer.encode(" " + action, add_special_tokens=False) for action in "ABC"]
    assert all(len(value) == 1 for value in encoded), encoded
    return [value[0] for value in encoded]


def counters():
    return {"labels_acquired": 0, "training_examples": 0, "optimizer_updates": 0, "training_prompt_tokens": 0,
            "supervised_tokens": 0, "generation_prompt_tokens": 0, "generated_actions": 0,
            "training_seconds": 0., "generation_seconds": 0., "oracle_seconds": 0.}


def encode(tokenizer, rows):
    tokenizer.padding_side = "left"
    return tokenizer([prompt(row["problem"], row["state"]) for row in rows], padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")


def predict(model, tokenizer, rows, tokens, counter, rng=None, temperature=0.8):
    import torch
    model.eval()
    batch = encode(tokenizer, rows)
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, tokens].float()
        probabilities = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
        greedy = logits.argmax(-1).cpu().tolist()
    torch.cuda.synchronize()
    counter["generation_seconds"] += time.monotonic() - start
    counter["generation_prompt_tokens"] += int(batch["attention_mask"].sum())
    counter["generated_actions"] += len(rows)
    actions = greedy if rng is None else [int(rng.choice(3, p=probability / probability.sum())) for probability in probabilities]
    return actions


def train(model, tokenizer, tokens, optimizer, rows, counter, batch_size):
    import torch
    assert len(rows) % batch_size == 0
    model.train()
    losses = []
    start = time.monotonic()
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        batch = encode(tokenizer, chunk)
        target = torch.zeros((len(chunk), 3), device="cuda")
        for index, row in enumerate(chunk): target[index, row["optimal"]] = 1 / len(row["optimal"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, tokens].float()
            loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
        assert torch.isfinite(loss), "Nonfinite action loss."
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        counter["training_examples"] += len(chunk)
        counter["optimizer_updates"] += 1
        counter["training_prompt_tokens"] += int(batch["attention_mask"].sum())
        counter["supervised_tokens"] += len(chunk)
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    counter["training_seconds"] += time.monotonic() - start
    return {"loss": float(np.mean(losses)), "last_gradient_norm": float(norm)}


def collect(streams, count, policy, rng, counter, model=None, tokenizer=None, tokens=None):
    assert count % len(streams.states) == 0
    rows = []
    for _ in range(count // len(streams.states)):
        begin = time.monotonic()
        current = [{"problem": p, "state": state, "id": state_id(p, state), "optimal": optimal_actions(p, state),
                    "distance": distance(p, state), "episode_age": age}
                   for p, state, age in zip(streams.problems, streams.states, streams.ages, strict=True)]
        assert all(group_split(row["problem"]) == "train" for row in current)
        counter["oracle_seconds"] += time.monotonic() - begin
        if policy == "expert": actions = [int(rng.choice(row["optimal"])) for row in current]
        elif policy == "random": actions = rng.integers(0, 3, len(current)).tolist()
        else: actions = predict(model, tokenizer, current, tokens, counter, rng)
        for row, action in zip(current, actions, strict=True):
            rows.append({**row, "action": action, "action_optimal": action in row["optimal"], "following": next_state(row["problem"], row["state"], action)})
        streams.advance(actions)
        counter["labels_acquired"] += len(current)
    return rows


def evaluate(model, tokenizer, tokens, dataset, counter, output, arm, budget, reference, cache):
    usage, predictions, metrics = counters(), [], {}
    for stratum, problems in dataset["eval"].items():
        trajectories = [{"problem": p, "states": [p["start"]], "actions": [], "optimal": [], "success": False} for p in problems]
        for step in range(8):
            active = [row for row in trajectories if not row["success"]]
            if not active: break
            states = [{"problem": row["problem"], "state": row["states"][-1]} for row in active]
            actions = []
            for offset in range(0, len(states), 128):
                actions.extend(predict(model, tokenizer, states[offset:offset + 128], tokens, usage))
            for row, action in zip(active, actions, strict=True):
                p, state = row["problem"], row["states"][-1]
                row["optimal"].append(action in optimal_actions(p, state))
                row["actions"].append(action)
                following = next_state(p, state, action)
                row["states"].append(following)
                row["success"] = following == p["target"]
        for row in trajectories:
            row["stratum"] = stratum
            row["id"] = row["problem"]["id"]
        metrics[stratum] = {"count": len(trajectories), "accuracy": float(np.mean([row["success"] for row in trajectories])),
                            "first_action_optimal": float(np.mean([row["optimal"][0] for row in trajectories])),
                            "all_action_optimal": float(np.mean([value for row in trajectories for value in row["optimal"]])),
                            "mean_moves": float(np.mean([len(row["actions"]) for row in trajectories]))}
        predictions.extend(trajectories)
    current = probe_weights(model)
    deltas = {name: current[name] - reference[name] for name in PROBES}
    probe_dir = cache / "weight_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / f"{arm}_n{budget}.npz"
    np.savez_compressed(probe_path, **deltas)
    record = {"seed": dataset["seed"], "arm": arm, "budget": budget, "metrics": metrics, "counters": dict(counter),
              "evaluation_usage": usage, "weight_probe_path": str(probe_path),
              "weight_norms": {name: float(np.linalg.norm(deltas[name])) for name in PROBES}}
    save_json(output / f"{arm}_n{budget}_predictions.json", predictions)
    append_json(output / f"{arm}_metrics.jsonl", record)
    print(arm, budget, {key: round(row["accuracy"], 4) for key, row in metrics.items()}, flush=True)


def prepare(args, cache):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert args.labels % 512 == 0
    dataset = make_dataset(args.seed, args.eval_seed, args.eval_per_stratum, args.evaluation_split)
    config = {"model": MODEL, "revision": REVISION, "seed": args.seed, "labels": args.labels, "warm_labels": 2048,
              "warm_epochs": 2, "batch_size": 64, "lanes": 128, "round_labels": 512, "horizon": 8,
              "learning_rate": 3e-5, "warm_learning_rate": 3e-5, "temperature": 0.8,
              "checkpoints": [n for n in (0, 512, 1024, 2048, 4096, 8192, 16384, 32768) if n <= args.labels],
              "arms": list(ARMS), "cache_directory": str(cache / "feedback" / output.name),
              "training": "All weights FP32, BF16 autocast, AdamW no decay, norm clip1. Conditional cross entropy on three legal action-token logits. Target uniform over all optimal actions. Every round trains512new+512uniform replay labels in shuffled order, batch64.",
              "acquisition": "Expert shortest trajectories, live sampled student trajectories, uniform random action trajectories, or fixed sampled warm-student trajectories. Every visited nonterminal state labeled, including mistakes. Reset on goal or8moves.",
              "comparisons": "Equal acquired labels, supervised tokens, examples and optimizer updates. Start episode counts, unique states, prompt tokens and generation compute can differ; all measured. Frozen acquisition charged up front, plus used-prefix counts retained.",
              "primary": "On-policy minus expert ID success at final budget; on-policy minus random and frozen-policy controls secondary. No asymptotic exponent assumed.",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": {name: sha256(ROOT / "experiments" / name) for name in SOURCES}}
    save_json(output / "dataset.json", dataset)
    config["dataset_sha256"] = sha256(output / "dataset.json")
    save_json(output / "config.json", config)
    source = output / "source"
    source.mkdir()
    for name in SOURCES: (source / name).write_bytes((ROOT / "experiments" / name).read_bytes())


def load_run(output):
    config = json.loads((output / "config.json").read_text())
    assert config["source_sha256"] == {name: sha256(ROOT / "experiments" / name) for name in SOURCES}
    assert sha256(output / "dataset.json") == config["dataset_sha256"]
    return config, json.loads((output / "dataset.json").read_text()), Path(config["cache_directory"])


def warm(output, config, dataset, cache):
    assert not (output / "warm_final.json").exists()
    initialize(config["seed"])
    model, tokenizer = load_model(MODEL)
    tokens = action_tokens(tokenizer)
    reference, count = probe_weights(model), counters()
    rng = np.random.default_rng(config["seed"])
    streams = Streams(dataset["train"], config["seed"], config["lanes"], config["horizon"])
    rows = collect(streams, config["warm_labels"], "expert", rng, count)
    save_json(output / "warm_training.json", rows)
    optimizer = optimizer_for(model, config["warm_learning_rate"])
    for epoch in tqdm(range(config["warm_epochs"]), desc="Feedback warm start"):
        order = rng.permutation(len(rows))
        train(model, tokenizer, tokens, optimizer, [rows[i] for i in order], count, config["batch_size"])
    destination = cache / "warm"
    destination.mkdir(parents=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    save_json(output / "warm_final.json", {"checkpoint": str(destination), "model_sha256": sha256(destination / "model.safetensors"), "counters": count, "action_tokens": tokens})


def run_arm(output, config, dataset, cache, arm):
    assert not (output / f"{arm}_metrics.jsonl").exists(), "Partial/existing run requires explicit inspection."
    initialize(config["seed"])
    model, tokenizer = load_model(str(cache / "warm"))
    tokens = action_tokens(tokenizer)
    reference, count = probe_weights(model), counters()
    rng = np.random.default_rng(config["seed"] + 10000)
    replay_rng = np.random.default_rng(config["seed"] + 20000)
    streams = Streams(dataset["train"], config["seed"] + 30000, config["lanes"], config["horizon"])
    warm_rows = json.loads((output / "warm_training.json").read_text())
    history = list(warm_rows)
    frozen = None
    if arm == "frozen_policy":
        frozen = collect(streams, config["labels"], arm, rng, count, model, tokenizer, tokens)
        save_json(output / "frozen_acquisition.json", {"rows": frozen, "counters": dict(count)})
        count["labels_acquired"] = 0
    optimizer = optimizer_for(model, config["learning_rate"])
    evaluate(model, tokenizer, tokens, dataset, count, output, arm, 0, reference, cache)
    for begin in tqdm(range(0, config["labels"], config["round_labels"]), desc=arm):
        if frozen is None:
            rows = collect(streams, config["round_labels"], arm, rng, count, model, tokenizer, tokens)
        else:
            rows = frozen[begin:begin + config["round_labels"]]
            count["labels_acquired"] += len(rows)
        history.extend(rows)
        indices = replay_rng.integers(0, len(history), len(rows))
        mixed = rows + [history[i] for i in indices]
        order = replay_rng.permutation(len(mixed))
        trained = [mixed[i] for i in order]
        stats = train(model, tokenizer, tokens, optimizer, trained, count, config["batch_size"])
        budget = begin + len(rows)
        append_json(output / f"{arm}_training.jsonl", {"budget": budget, "rows": rows, "replay_indices": indices.tolist(), "order": order.tolist(),
                    "counters": dict(count), "stats": stats, "episodes_started": streams.episodes_started,
                    "episodes_succeeded": streams.episodes_succeeded, "unique_states": len({row["id"] for row in history}),
                    "distance_counts": dict(Counter(row["distance"] for row in rows)), "optimal_action_fraction": float(np.mean([row["action_optimal"] for row in rows]))})
        if budget in config["checkpoints"]: evaluate(model, tokenizer, tokens, dataset, count, output, arm, budget, reference, cache)
    destination = cache / arm
    destination.mkdir(parents=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    save_json(output / f"{arm}_final.json", {"checkpoint": str(destination), "model_sha256": sha256(destination / "model.safetensors"), "counters": count})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "warm", *ARMS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7310)
    parser.add_argument("--eval-seed", type=int, default=94100)
    parser.add_argument("--labels", type=int, default=16384)
    parser.add_argument("--eval-per-stratum", type=int, default=256)
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    cache = configure_environment()
    if args.mode == "prepare": prepare(args, cache)
    else:
        config, dataset, directory = load_run(args.output.resolve())
        if args.mode == "warm": warm(args.output.resolve(), config, dataset, directory)
        else: run_arm(args.output.resolve(), config, dataset, directory, args.mode)


if __name__ == "__main__": main()
