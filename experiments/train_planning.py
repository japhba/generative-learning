"""Full-weight SFT and on-policy verified-rollout distillation on modular planning."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from training_task import make_dataset, prompt, teacher_solution, verify_solution

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
ARMS = ("sft", "rollout_distill", "yoked_sft")
PROBES = tuple(f"model.layers.{layer}.self_attn.q_proj.weight" for layer in (0, 14, 27))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def configure_environment():
    load_dotenv(Path.home() / ".env")
    cache = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning"
    os.environ["HF_HUB_CACHE"] = str(cache / "hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return cache


def initialize(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")


def load_model(source):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"revision": REVISION} if source == MODEL else {}
    tokenizer = AutoTokenizer.from_pretrained(source, fix_mistral_regex=False, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, attn_implementation="sdpa", **kwargs).to("cuda")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def optimizer_for(model, learning_rate):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, fused=True)


def train_examples(model, tokenizer, optimizer, examples, counters, batch_size):
    """One pass over real target text; prefix and padding positions have no loss."""
    import torch
    from torch.nn.utils.rnn import pad_sequence
    model.train()
    losses = []
    start = time.monotonic()
    for offset in range(0, len(examples), batch_size):
        rows = examples[offset:offset + batch_size]
        inputs, labels, masks = [], [], []
        for problem, completion in rows:
            prefix = tokenizer.encode(prompt(problem), add_special_tokens=False)
            target = tokenizer.encode(" " + completion.strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
            inputs.append(torch.tensor(prefix + target))
            labels.append(torch.tensor([-100] * len(prefix) + target))
            masks.append(torch.ones(len(prefix) + len(target), dtype=torch.long))
            counters["training_tokens"] += len(prefix) + len(target)
            counters["supervised_tokens"] += len(target)
        batch = {"input_ids": pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id).cuda(),
                 "attention_mask": pad_sequence(masks, batch_first=True, padding_value=0).cuda(),
                 "labels": pad_sequence(labels, batch_first=True, padding_value=-100).cuda()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        counters["optimizer_updates"] += 1
        counters["training_examples"] += len(rows)
        losses.append(float(loss.detach()))
        counters["last_gradient_norm"] = float(gradient_norm)
    torch.cuda.synchronize()
    counters["training_seconds"] += time.monotonic() - start
    return losses


def generate(model, tokenizer, problems, samples, batch_size, max_new_tokens, temperature, seed):
    import torch
    model.eval()
    tokenizer.padding_side = "left"
    expanded = [problem for problem in problems for _ in range(samples)]
    results, input_tokens, output_tokens = [], 0, 0
    start = time.monotonic()
    # fork_rng prevents evaluation checkpoints from changing later training rollouts.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for offset in range(0, len(expanded), batch_size):
            rows = expanded[offset:offset + batch_size]
            batch = tokenizer([prompt(problem) for problem in rows], padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            options = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens, "use_cache": True}
            if temperature > 0: options.update(temperature=temperature, top_p=0.95, top_k=0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sequences = model.generate(**batch, **options)
            generated = sequences[:, batch["input_ids"].shape[1]:].cpu().tolist()
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            input_tokens += int(batch["attention_mask"].sum())
            for problem, text, token_ids in zip(rows, texts, generated, strict=True):
                ended = tokenizer.eos_token_id in token_ids
                count = token_ids.index(tokenizer.eos_token_id) + 1 if ended else len(token_ids)
                output_tokens += count
                verdict = verify_solution(problem, text)
                # A length-limited prefix is not a completed accepted trajectory.
                if not ended: verdict = {**verdict, "correct": False, "reason": "truncated"}
                results.append({"problem_id": problem["id"], "text": text, "verdict": verdict, "generated_tokens": count, "ended": ended})
    torch.cuda.synchronize()
    return results, {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "seconds": time.monotonic() - start}


def completed_minibatches(pending, incoming, batch_size, final):
    """Accumulate accepted examples; a partial optimizer batch occurs only at run end."""
    combined = pending + incoming
    count = len(combined) if final else len(combined) // batch_size * batch_size
    return combined[:count], combined[count:]


def fresh_counters():
    return {"problems_presented": 0, "accepted_examples": 0, "pending_examples": 0, "training_examples": 0, "training_tokens": 0, "supervised_tokens": 0,
            "optimizer_updates": 0, "rollout_candidates": 0, "rollout_prompt_tokens": 0, "rollout_completion_tokens": 0,
            "training_seconds": 0.0, "rollout_seconds": 0.0, "last_gradient_norm": 0.0}


def probe_weights(model):
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().float().cpu().numpy().copy() for name in PROBES}


def evaluate(model, tokenizer, dataset, config, arm, budget, counters, output, reference):
    grouped, all_results = {}, []
    evaluation_cost = {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    for stratum, problems in dataset["eval"].items():
        predictions, usage = generate(model, tokenizer, problems, 1, config["generation_batch_size"], config["max_new_tokens"], 0., config["seed"] + 900000)
        results = [{**prediction, "stratum": stratum, "shortest_distance": problem["shortest_distance"]} for prediction, problem in zip(predictions, problems, strict=True)]
        all_results.extend(results)
        grouped[stratum] = {"accuracy": float(np.mean([row["verdict"]["correct"] for row in results])), "count": len(results),
                            "failure_reasons": dict(Counter(row["verdict"]["reason"] for row in results))}
        for field, amount in usage.items(): evaluation_cost[field] += amount
    current = probe_weights(model)
    deltas = {name: current[name] - reference[name] for name in PROBES}
    weight_norms = {name: {"delta_norm": float(np.linalg.norm(deltas[name])), "relative_delta_norm": float(np.linalg.norm(deltas[name]) / np.linalg.norm(reference[name]))} for name in PROBES}
    probe_directory = Path(config["cache_directory"]) / "weight_probes"
    probe_directory.mkdir(parents=True, exist_ok=True)
    probe_path = probe_directory / f"{arm}_n{budget}_weight_deltas.npz"
    np.savez_compressed(probe_path, **deltas)
    record = {"arm": arm, "seed": config["seed"], "budget": budget, "metrics": grouped, "counters": dict(counters), "evaluation_usage": evaluation_cost, "weight_probes": weight_norms, "weight_probe_path": str(probe_path)}
    append_json(output / f"{arm}_metrics.jsonl", record)
    save_json(output / f"{arm}_n{budget}_predictions.json", all_results)
    print(f"{arm} n={budget}: " + ", ".join(f"{group}={value['accuracy']:.1%}" for group, value in grouped.items()), flush=True)
    return record


def prepare(args, cache):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL, "revision": REVISION, "seed": args.seed, "train_size": args.train_size,
              "warm_size": args.warm_size, "warm_epochs": args.warm_epochs, "eval_per_stratum": args.eval_per_stratum,
              "batch_size": args.batch_size, "query_batch_size": args.query_batch_size, "generation_batch_size": args.generation_batch_size,
              "rollouts": args.rollouts, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "learning_rate": args.learning_rate, "warm_learning_rate": args.warm_learning_rate,
              "checkpoints": [n for n in (0, 64, 128, 256, 512, 1024, 2048, 4096) if n <= args.train_size],
              "cache_directory": str(cache / "training" / output.name), "arms": ARMS,
              "training": "All model weights float32; bfloat16 autocast; AdamW no weight decay; one pass over accepted examples; masked completion-only CE. Accumulate accepted examples into full optimizer minibatches; flush partial batch only at run end.",
              "rollout_method": "Fresh on-policy best-of-K rejection distillation: retain first valid completed trajectory per prompt, then train on that actual generated text. No teacher fallback.",
              "control": "Yoked SFT uses canonical teacher traces only for prompts accepted by rollout_distill, with matching minibatch boundaries. All arms start at same common teacher warm-start.",
              "evaluation": "Greedy fully decoded trajectories; exact format, every state transition, target, horizon, and EOS checked. Fixed hashed held-out identities; eval RNG isolated.",
              "budget_interpretation": "Post-warm-start new prompts presented; also record accepted training examples, updates, target/input tokens, generation tokens and timing. Warm-start cost reported separately.",
              "source_sha256": {name: hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest() for name in ("train_planning.py", "training_task.py")}}
    config = json.loads(json.dumps(config))
    if (output / "config.json").exists():
        assert json.loads((output / "config.json").read_text()) == config, "Configuration/code changed; choose a new output directory."
    else:
        dataset = make_dataset(args.seed, args.train_size + args.warm_size, args.eval_per_stratum)
        dataset["warm"] = dataset["train"][:args.warm_size]
        dataset["train"] = dataset["train"][args.warm_size:]
        save_json(output / "dataset.json", dataset)
        save_json(output / "config.json", config)
    return output, config, json.loads((output / "dataset.json").read_text())


def warmup(config, dataset, output):
    if (output / "base_metrics.jsonl").exists(): raise FileExistsError("Warm-start outputs exist; inspect before any rerun.")
    initialize(config["seed"])
    model, tokenizer = load_model(MODEL)
    reference = probe_weights(model)
    optimizer = optimizer_for(model, config["warm_learning_rate"])
    counters = fresh_counters()
    evaluate(model, tokenizer, dataset, config, "base", 0, counters, output, reference)
    for epoch in tqdm(range(config["warm_epochs"]), desc="Common warm-start"):
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(dataset["warm"]))
        examples = [(dataset["warm"][i], teacher_solution(dataset["warm"][i])) for i in order]
        losses = train_examples(model, tokenizer, optimizer, examples, counters, config["batch_size"])
        print(f"warm epoch={epoch+1}, loss={np.mean(losses):.4f}", flush=True)
    counters["problems_presented"] = len(dataset["warm"])
    counters["accepted_examples"] = counters["training_examples"]
    evaluate(model, tokenizer, dataset, config, "warm", len(dataset["warm"]), counters, output, reference)
    destination = Path(config["cache_directory"]) / "warm"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    save_json(output / "warm_counters.json", counters)
    print(f"Common checkpoint saved: {destination}", flush=True)


def run_arm(arm, config, dataset, output):
    if (output / f"{arm}_metrics.jsonl").exists(): raise FileExistsError("Arm outputs exist; inspect before any rerun.")
    initialize(config["seed"])
    model, tokenizer = load_model(str(Path(config["cache_directory"]) / "warm"))
    reference = probe_weights(model)
    optimizer = optimizer_for(model, config["learning_rate"])
    counters = fresh_counters()
    evaluate(model, tokenizer, dataset, config, arm, 0, counters, output, reference)
    replay = {row["end_budget"]: row for row in map(json.loads, (output / "rollout_batches.jsonl").read_text().splitlines())} if arm == "yoked_sft" else {}
    pending = []
    for offset in tqdm(range(0, len(dataset["train"]), config["query_batch_size"]), desc=arm):
        problems = dataset["train"][offset:offset + config["query_batch_size"]]
        end = offset + len(problems)
        if arm == "sft":
            examples = [(problem, teacher_solution(problem)) for problem in problems]
        elif arm == "rollout_distill":
            predictions, usage = generate(model, tokenizer, problems, config["rollouts"], config["generation_batch_size"], config["max_new_tokens"], config["temperature"], config["seed"] * 10000 + end)
            examples, selected = [], []
            for index, problem in enumerate(problems):
                candidates = predictions[index * config["rollouts"]:(index + 1) * config["rollouts"]]
                valid = [candidate for candidate in candidates if candidate["verdict"]["correct"]]
                if valid:
                    examples.append((problem, valid[0]["text"]))
                    selected.append({"problem_id": problem["id"], "completion": valid[0]["text"], "steps": valid[0]["verdict"]["steps"], "teacher_steps": problem["shortest_distance"]})
            append_json(output / "rollout_batches.jsonl", {"end_budget": end, "selected": selected, "candidates": predictions, "usage": usage})
            counters["rollout_candidates"] += len(predictions)
            counters["rollout_prompt_tokens"] += usage["prompt_tokens"]
            counters["rollout_completion_tokens"] += usage["completion_tokens"]
            counters["rollout_seconds"] += usage["seconds"]
        else:
            selected_ids = [row["problem_id"] for row in replay[end]["selected"]]
            indexed = {problem["id"]: problem for problem in problems}
            examples = [(indexed[key], teacher_solution(indexed[key])) for key in selected_ids]
        ready, pending = completed_minibatches(pending, examples, config["batch_size"], end == len(dataset["train"]))
        losses = train_examples(model, tokenizer, optimizer, ready, counters, config["batch_size"])
        counters["accepted_examples"] += len(examples)
        counters["pending_examples"] = len(pending)
        counters["problems_presented"] = end
        append_json(output / f"{arm}_training.jsonl", {"budget": end, "accepted_examples": len(examples), "trained_examples": len(ready), "pending_examples": len(pending), "losses": losses, "counters": dict(counters)})
        if end in config["checkpoints"]:
            evaluate(model, tokenizer, dataset, config, arm, end, counters, output, reference)
    destination = Path(config["cache_directory"]) / arm
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    save_json(output / f"{arm}_final.json", {"checkpoint": str(destination), "counters": counters})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "warmup", *ARMS))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "training_pilot")
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--warm-size", type=int, default=256)
    parser.add_argument("--warm-epochs", type=int, default=4)
    parser.add_argument("--eval-per-stratum", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warm-learning-rate", type=float, default=3e-5)
    args = parser.parse_args()
    assert args.train_size in (64, 128, 256, 512, 1024, 2048, 4096)
    assert args.query_batch_size in (16, 32, 64) and args.train_size % args.query_batch_size == 0
    assert min(args.warm_size, args.warm_epochs, args.eval_per_stratum, args.batch_size, args.generation_batch_size, args.rollouts) > 0
    cache = configure_environment()
    output, config, dataset = prepare(args, cache)
    if args.mode == "warmup": warmup(config, dataset, output)
    elif args.mode != "prepare": run_arm(args.mode, config, dataset, output)


if __name__ == "__main__": main()
