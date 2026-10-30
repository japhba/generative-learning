"""SmolLM2 replication of the frozen Qwen tracking protocol; only model loading differs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from train_feedback import counters, sha256
from train_planning import MODEL, REVISION, PROBES, append_json, configure_environment, initialize, load_model, optimizer_for, probe_weights, save_json
from tracking_task import HORIZON, LETTERS, Streams, grade, make_evaluation, prompt

MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
ROOT = Path(__file__).resolve().parents[1]
ARMS = ("expert", "on_policy", "random", "frozen_policy")
SOURCES = ("train_tracking_smollm.py", "tracking_task.py", "train_feedback.py", "feedback_task.py", "train_planning.py", "training_task.py")


class Learner:
    def __init__(self, checkpoint):
        if checkpoint == MODEL:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=REVISION)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        else:
            self.model, self.tokenizer = load_model(checkpoint)
        values = [self.tokenizer.encode(" " + letter, add_special_tokens=False) for letter in LETTERS]
        assert all(len(value) == 1 for value in values)
        self.tokens = [value[0] for value in values]
        self.tokenizer.padding_side = "left"

    def encode(self, rows):
        return self.tokenizer([prompt(row) for row in rows], padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")

    def predict(self, rows, count, rng=None):
        import torch
        self.model.eval()
        batch = self.encode(rows)
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, self.tokens].float()
            probs = logits.softmax(-1).cpu().numpy()
            greedy = logits.argmax(-1).cpu().tolist()
        torch.cuda.synchronize()
        count["generation_seconds"] += time.monotonic() - start
        count["generation_prompt_tokens"] += int(batch["attention_mask"].sum())
        count["generated_actions"] += len(rows)
        if rng is None: return greedy
        mixture = 0.9 * probs + 0.1 / len(self.tokens)
        return [int(rng.choice(len(self.tokens), p=p / p.sum())) for p in mixture]

    def train(self, rows, optimizer, count):
        import torch
        assert len(rows) % 64 == 0
        self.model.train()
        start, losses = time.monotonic(), []
        for offset in range(0, len(rows), 64):
            chunk = rows[offset:offset + 64]
            batch = self.encode(chunk)
            targets = torch.tensor([row["target"] for row in chunk], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, self.tokens].float()
                loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.1)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            count["training_examples"] += len(chunk)
            count["supervised_tokens"] += len(chunk)
            count["optimizer_updates"] += 1
            count["training_prompt_tokens"] += int(batch["attention_mask"].sum())
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        count["training_seconds"] += time.monotonic() - start
        return float(np.mean(losses))

    def save(self, destination):
        destination.mkdir(parents=True)
        self.model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        return {"checkpoint": str(destination), "model_sha256": sha256(destination / "model.safetensors")}


def collect(stream, count_rows, policy, rng, count, learner):
    assert count_rows % stream.lanes == 0
    rows = []
    for _ in range(count_rows // stream.lanes):
        current = stream.current()
        if policy == "expert": actions = [row["target"] for row in current]
        elif policy == "random": actions = rng.integers(0, 7, len(current)).tolist()
        else: actions = learner.predict(current, count, rng)
        rows.extend({**row, "action": action} for row, action in zip(current, actions, strict=True))
        stream.advance(actions)
    count["labels_acquired"] += count_rows
    return rows


def evaluate(learner, dataset, output, arm, budget, count, reference, cache):
    usage, predictions, metrics = counters(), [], {}
    for stratum, episodes in dataset["eval"].items():
        errors = [episode["initial_error"] for episode in episodes]
        actions = [[] for _ in episodes]
        for step in range(HORIZON):
            rows = [{"u": episode["commands"][step], "e": error} for episode, error in zip(episodes, errors, strict=True)]
            decisions = []
            for begin in range(0, len(rows), 128): decisions.extend(learner.predict(rows[begin:begin + 128], usage))
            for index, (row, action) in enumerate(zip(rows, decisions, strict=True)):
                errors[index] = row["e"] + action - 3 - row["u"]
                actions[index].append(action)
        results = [{"episode": episode, "actions": action, "stratum": stratum, **grade(episode, action)}
                   for episode, action in zip(episodes, actions, strict=True)]
        metrics[stratum] = {field: float(np.mean([row[field] for row in results])) for field in ("mean_absolute_error", "stable_tracking", "final_error")}
        metrics[stratum]["action_accuracy"] = float(np.mean([value for row in results for value in row["optimal"]]))
        metrics[stratum]["count"] = len(results)
        predictions.extend(results)
    current = probe_weights(learner.model)
    deltas = {name: current[name] - reference[name] for name in PROBES}
    probe = cache / "weight_probes" / f"{arm}_n{budget}.npz"
    probe.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probe, **deltas)
    record = {"seed": dataset["seed"], "arm": arm, "budget": budget, "metrics": metrics, "counters": dict(count), "evaluation_usage": usage,
              "weight_probe_path": str(probe), "weight_norms": {name: float(np.linalg.norm(value)) for name, value in deltas.items()}}
    save_json(output / f"{arm}_n{budget}_predictions.json", predictions)
    append_json(output / f"{arm}_metrics.jsonl", record)
    print(arm, budget, {key: {k: round(value[k], 3) for k in ("mean_absolute_error", "stable_tracking")} for key, value in metrics.items()}, flush=True)


def prepare(args, cache):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert args.labels % 512 == 0
    dataset = {"seed": args.seed, "evaluation_seed": args.eval_seed, "eval": make_evaluation(args.eval_seed, args.eval_per_stratum)}
    save_json(output / "dataset.json", dataset)
    config = {"model": MODEL, "revision": REVISION, "seed": args.seed, "labels": args.labels, "warm_labels": 2048, "warm_epochs": 20,
              "warm_calibration": "Stop after two consecutive epochs with perfect greedy accuracy on the five nominal training inputs; fail if this is not reached in20epochs. No test feedback used.",
              "learning_rate": 1e-5, "label_smoothing": 0.1, "preserve_warm_optimizer": True, "batch_size": 64, "round_labels": 512, "lanes": 128, "horizon": HORIZON,
              "checkpoints": [n for n in (0, 512, 1024, 2048, 4096, 8192, 16384) if n <= args.labels],
              "arms": list(ARMS), "cache_directory": str(cache / "tracking" / output.name),
              "dataset_sha256": sha256(output / "dataset.json"), "source_sha256": {name: sha256(ROOT / "experiments" / name) for name in SOURCES},
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "policy": "Seven legal LM action logits. Training behavior .9*student softmax(temp1)+.1*uniform exploration. Greedy test policy. Teacher labels clip(u-e,-3,3).",
              "training": "Full FP32 weights and AdamW, BF16 autocast, LR1e-5, no decay, clip1, label smoothing0.1. Preserve warm optimizer moments. Each round512new labels+512uniform aggregate replay, shuffled; one action CE target per row.",
              "primary": "On-policy minus expert recovery mean absolute tracking error at final label budget (negative favors on-policy). Nominal performance and random/frozen controls reported.",
              "evaluation": "Fresh independent command sequences. Nominal e0=0, recovery e0 in+-1..3, large_shift e0 in+-4..8. Training episodes all start e0=0. This explicitly tests robustness to initial-state interventions, not only IID expert states."}
    save_json(output / "config.json", config)
    (output / "source").mkdir()
    for name in SOURCES: (output / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())


def warm(output, config, dataset, cache):
    import torch
    initialize(config["seed"])
    assert not (output / "warm_final.json").exists()
    learner, count = Learner(MODEL), counters()
    rng = np.random.default_rng(config["seed"])
    rows = collect(Streams(config["seed"]), config["warm_labels"], "expert", rng, count, learner)
    save_json(output / "warm_training.json", rows)
    optimizer = optimizer_for(learner.model, config["learning_rate"])
    nominal = [{"u": u, "e": 0, "target": u + 3} for u in range(-2, 3)]
    streak, calibration = 0, []
    for epoch in tqdm(range(config["warm_epochs"]), desc="Tracker warm start"):
        loss = learner.train([rows[i] for i in rng.permutation(len(rows))], optimizer, count)
        actions = learner.predict(nominal, count)
        correct = actions == [row["target"] for row in nominal]
        streak = streak + 1 if correct else 0
        calibration.append({"epoch": epoch + 1, "loss": loss, "actions": actions, "perfect_training_accuracy": correct})
        print('Nominal training calibration', calibration[-1], flush=True)
        if streak == 2: break
    assert streak == 2, "Nominal training task was not mastered within the declared warm budget."
    checkpoint = learner.save(cache / "warm")
    optimizer_path = cache / "warm" / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    save_json(output / "warm_final.json", {**checkpoint, "optimizer_sha256": sha256(optimizer_path), "counters": count, "calibration": calibration})


def run(output, config, dataset, cache, arm):
    import torch
    initialize(config["seed"])
    assert not (output / f"{arm}_metrics.jsonl").exists()
    learner, count = Learner(str(cache / "warm")), counters()
    reference = probe_weights(learner.model)
    rng, replay_rng = np.random.default_rng(config["seed"] + 10000), np.random.default_rng(config["seed"] + 20000)
    stream = Streams(config["seed"] + 30000)
    history = json.loads((output / "warm_training.json").read_text())
    frozen = None
    if arm == "frozen_policy":
        frozen = collect(stream, config["labels"], arm, rng, count, learner)
        save_json(output / "frozen_acquisition.json", {"rows": frozen, "counters": dict(count)})
        count["labels_acquired"] = 0
    optimizer = optimizer_for(learner.model, config["learning_rate"])
    warm_record = json.loads((output / "warm_final.json").read_text())
    assert sha256(cache / "warm" / "model.safetensors") == warm_record["model_sha256"]
    assert sha256(cache / "warm" / "optimizer.pt") == warm_record["optimizer_sha256"]
    optimizer.load_state_dict(torch.load(cache / "warm" / "optimizer.pt", map_location="cuda", weights_only=True))
    evaluate(learner, dataset, output, arm, 0, count, reference, cache)
    for begin in tqdm(range(0, config["labels"], 512), desc=arm):
        if frozen is None: rows = collect(stream, 512, arm, rng, count, learner)
        else:
            rows = frozen[begin:begin + 512]
            count["labels_acquired"] += len(rows)
        history.extend(rows)
        indices = replay_rng.integers(0, len(history), 512)
        mixed = rows + [history[i] for i in indices]
        order = replay_rng.permutation(len(mixed))
        loss = learner.train([mixed[i] for i in order], optimizer, count)
        budget = begin + len(rows)
        append_json(output / f"{arm}_training.jsonl", {"budget": budget, "rows": rows, "replay_indices": indices.tolist(), "order": order.tolist(), "loss": loss,
                    "counters": dict(count), "nonzero_error_fraction": float(np.mean([row["e"] != 0 for row in rows])),
                    "mean_squared_input_error": float(np.mean([row["e"] ** 2 for row in rows])), "distinct_inputs": len({(row["u"], row["e"]) for row in history})})
        if budget in config["checkpoints"]: evaluate(learner, dataset, output, arm, budget, count, reference, cache)
    save_json(output / f"{arm}_final.json", {**learner.save(cache / arm), "counters": count})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "warm", *ARMS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7510)
    parser.add_argument("--labels", type=int, default=8192)
    parser.add_argument("--eval-seed", type=int, default=94300)
    parser.add_argument("--eval-per-stratum", type=int, default=128)
    args = parser.parse_args()
    cache = configure_environment()
    if args.mode == "prepare": prepare(args, cache)
    else:
        output = args.output.resolve()
        config, dataset = json.loads((output / "config.json").read_text()), json.loads((output / "dataset.json").read_text())
        assert config["source_sha256"] == {name: sha256(ROOT / "experiments" / name) for name in SOURCES}
        assert config["dataset_sha256"] == sha256(output / "dataset.json")
        directory = Path(config["cache_directory"])
        if args.mode == "warm": warm(output, config, dataset, directory)
        else: run(output, config, dataset, directory, args.mode)


if __name__ == "__main__": main()
