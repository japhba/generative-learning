"""Calibrate and run full-weight learning on the exact original Zipf dictionaries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from zipf_task import BUDGETS, LABELS, VOCABULARY, make_world
from zipf_finetuning_task import CURRICULA, MAIN_SEEDS, PILOT_SEEDS, acquisition, evaluate_probabilities, oracle, prompt, replay_indices, verify_original_worlds

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"),
          "smollm": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")}
SOURCES = ("train_zipf_finetuning.py", "zipf_finetuning_task.py", "zipf_task.py")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hashes():
    return {name: hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest() for name in SOURCES}


class Learner:
    def __init__(self, family, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("high")
        name, revision = MODELS[family]
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, fix_mistral_regex=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32, attn_implementation="sdpa").cuda()
        self.base = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        targets = [self.tokenizer.encode(" " + letter, add_special_tokens=False) for letter in LABELS]
        assert all(len(tokens) == 1 for tokens in targets)
        self.target_tokens = torch.tensor([tokens[0] for tokens in targets], device="cuda")
        encoded = self.tokenizer([prompt(key) for key in range(1, VOCABULARY + 1)], padding=True, add_special_tokens=False, return_tensors="pt")
        self.table = {key: value.cuda() for key, value in encoded.items()}
        self.lengths = encoded["attention_mask"].sum(dim=1).numpy()
        self.width = encoded["input_ids"].shape[1]
        self.trie = None

    def reset(self, seed, lr):
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model.load_state_dict(self.base)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0., fused=True)
        self.count = {"optimizer_updates": 0, "training_presentations": 0, "training_prompt_tokens": 0, "training_padded_tokens": 0,
                      "train_seconds": 0., "evaluation_seconds": 0., "evaluation_prompt_tokens": 0, "generation_seconds": 0.,
                      "generation_prompt_tokens": 0, "generation_output_tokens": 0, "guess_prompt_tokens": 0}

    def logits(self, batch):
        torch = self.torch
        hidden = self.model.model(**batch, use_cache=False).last_hidden_state[:, -1]
        # The ordinary LM output embeddings, restricted to the four legal labels.
        head = self.model.get_output_embeddings().weight.index_select(0, self.target_tokens)
        return torch.nn.functional.linear(hidden, head).float()

    def learn(self, queried, targets, indices):
        torch = self.torch
        self.model.train()
        keys = np.array(queried, dtype=np.int64)
        truths = np.array(targets, dtype=np.int64)
        start = time.monotonic()
        losses = []
        for selected in indices:
            chosen = keys[selected]
            ids = torch.as_tensor(chosen - 1, device="cuda")
            batch = {name: tensor.index_select(0, ids) for name, tensor in self.table.items()}
            gold = torch.as_tensor(truths[selected], device="cuda")
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(self.logits(batch), gold)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.count["optimizer_updates"] += 1
            self.count["training_presentations"] += len(selected)
            self.count["training_prompt_tokens"] += int(self.lengths[chosen - 1].sum())
            self.count["training_padded_tokens"] += len(selected) * self.width
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        self.count["train_seconds"] += time.monotonic() - start
        return float(np.mean(losses))

    def predict(self, keys, usage="evaluation"):
        torch = self.torch
        self.model.eval()
        results = []
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(0, len(keys), 128):
                current = keys[offset:offset + 128]
                ids = torch.tensor([key - 1 for key in current] + [0] * (128 - len(current)), device="cuda")
                probs = self.logits({name: value.index_select(0, ids) for name, value in self.table.items()}).softmax(-1)
                results.extend(probs[:len(current)].cpu().tolist())
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        self.count["evaluation_seconds" if usage == "evaluation" else "generation_seconds"] += elapsed
        self.count["evaluation_prompt_tokens" if usage == "evaluation" else "guess_prompt_tokens"] += int(self.lengths[np.array(keys) - 1].sum())
        return results

    def probe(self):
        torch = self.torch
        with torch.no_grad():
            squared = sum((p.float() - self.base[name].float()).square().sum() for name, p in self.model.named_parameters())
        return {"parameter_displacement_l2": float(squared.sqrt())}

    def make_trie(self):
        encodings = self.tokenizer([" " + str(key) for key in range(1, VOCABULARY + 1)], add_special_tokens=False)["input_ids"]
        trie = {}
        for key, tokens in enumerate(encodings, start=1):
            node = trie
            assert int(self.tokenizer.decode(tokens).strip()) == key
            for token in [*tokens, self.tokenizer.eos_token_id]:
                if token not in node: node[token] = {}
                node = node[token]
        self.trie = trie
        self.maximum_key_tokens = max(map(len, encodings)) + 1

    def choose(self, previous):
        torch = self.torch
        assert self.trie is not None
        text = ("You are selecting practice queries for a secret dictionary of keys 1 through 65536. "
                "Every key has an independent random label A, B, C, or D. Future test keys have probability "
                "proportional to key^(-1.5), so smaller numbered keys matter more. Each request costs one "
                "example, including repeats. Choose the most useful next key to minimize future test error.\n"
                f"Previously requested keys: {json.dumps(previous)}\nNext practice key:")
        batch = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        prefix = batch["input_ids"].shape[1]
        def allowed(batch_id, sequence):
            node = self.trie
            for token in sequence[prefix:].tolist(): node = node[token]
            return list(node)
        self.model.eval()
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.model.generate(**batch, do_sample=False, max_new_tokens=self.maximum_key_tokens,
                                         prefix_allowed_tokens_fn=allowed, eos_token_id=self.tokenizer.eos_token_id,
                                         pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        tokens = result[0, prefix:].cpu().tolist()
        assert tokens[-1] == self.tokenizer.eos_token_id
        key = int(self.tokenizer.decode(tokens[:-1]).strip())
        assert 1 <= key <= VOCABULARY
        torch.cuda.synchronize()
        self.count["generation_seconds"] += time.monotonic() - start
        self.count["generation_prompt_tokens"] += prefix
        self.count["generation_output_tokens"] += len(tokens)
        return key


def evaluate(learner, seed, queried, output, budget, include_test):
    keys = sorted(set(queried) | (set(make_world(seed)["test_keys"]) if include_test else set()))
    probabilities = learner.predict(keys)
    metrics = evaluate_probabilities(seed, queried, keys, probabilities, include_test)
    record = {"budget": budget, "queried": queried.copy(), "keys": keys, "probabilities": probabilities,
              "metrics": metrics, "counters": dict(learner.count), "probes": learner.probe()}
    save(output / f"n{budget}.json", record)
    return record


def run(learner, family, seed, curriculum, lr, repeats, output, include_test, cache, save_model=False, live=False):
    output.mkdir(parents=True, exist_ok=False)
    learner.reset(seed + 410000, lr)
    schedule = replay_indices(seed, repeats)
    np.save(output / "replay_indices.npy", schedule)
    fixed = acquisition(seed, curriculum) if not live else []
    queried, targets, checkpoints, generated = [], [], [], []
    start = time.monotonic()
    save(output / "config.json", {"family": family, "model": MODELS[family], "seed": seed, "curriculum": curriculum,
                                  "learning_rate": lr, "repeats": repeats, "batch_size": 32, "include_test": include_test,
                                  "live": live, "source_sha256": hashes(), "started_at": datetime.now(timezone.utc).isoformat()})
    if include_test: checkpoints.append(evaluate(learner, seed, queried, output, 0, include_test))
    for budget in tqdm(BUDGETS, desc=f"{family} {seed} {curriculum}", leave=False):
        begin = len(queried)
        if live:
            candidates = []
            for _ in range(budget - begin): candidates.append(learner.choose(queried + candidates))
            probabilities = learner.predict(candidates, usage="guess")
            guesses = np.argmax(probabilities, axis=1).tolist()
            generated.append({"budget": budget, "previous": queried.copy(), "queries": candidates, "guesses": guesses,
                              "guess_probabilities": probabilities, "feedback": [oracle(seed, key) for key in candidates]})
        else: candidates = fixed[begin:budget]
        for key in candidates:
            queried.append(key)
            targets.append(oracle(seed, key))
            loss = learner.learn(queried, targets, schedule[len(queried) - 1])
        if include_test or budget == 128: checkpoints.append(evaluate(learner, seed, queried, output, budget, include_test))
    assert learner.count["optimizer_updates"] == repeats * 128
    assert learner.count["training_presentations"] == 32 * repeats * 128
    save(output / "observations.json", {"queries": queried, "targets": targets, "generated": generated})
    final = {"family": family, "seed": seed, "curriculum": curriculum, "learning_rate": lr, "repeats": repeats,
             "metrics": checkpoints[-1]["metrics"], "counters": dict(learner.count), "wall_seconds": time.monotonic() - start,
             "parameter_count": learner.parameter_count, "complete": True, "source_sha256": hashes()}
    if save_model:
        destination = cache / family / f"{seed}_{curriculum}"
        destination.mkdir(parents=True, exist_ok=False)
        learner.model.save_pretrained(destination)
        learner.tokenizer.save_pretrained(destination)
        final["checkpoint"] = str(destination)
    save(output / "complete.json", final)
    print(json.dumps({"finished": str(output), **final["metrics"], "train_seconds": final["counters"]["train_seconds"]}), flush=True)
    return final


def calibrate(learner, family, output, cache):
    results = []
    for repeats in (1, 4, 16):
        for lr in (1e-5, 3e-5, 1e-4):
            passed = True
            for seed in PILOT_SEEDS:
                for curriculum in ("ranked", "iid"):
                    result = run(learner, family, seed, curriculum, lr, repeats,
                                 output / "calibration" / f"r{repeats}_lr{lr:g}" / f"{seed}_{curriculum}", False, cache)
                    results.append(result)
                    if result["metrics"]["seen_accuracy"] < .99:
                        passed = False
                        break
                if not passed: break
            if passed:
                selection = {"status": "passed", "family": family, "learning_rate": lr, "repeats": repeats,
                             "rule": "Smallest R, then smallest learning rate passing >=99% observed-key accuracy in every pilot dictionary and both curricula. Eliminate a configuration at its first failed pilot; no test evaluation in calibration.",
                             "selected_at": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes(), "pilot_results": results}
                save(output / "selection.json", selection)
                return selection
    save(output / "selection.json", {"status": "failed", "family": family, "pilot_results": results, "source_sha256": hashes()})
    raise RuntimeError("No declared configuration passed the observed-key retention threshold.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "pipeline", "calibrate", "main", "live"))
    parser.add_argument("--model", choices=MODELS, default="qwen")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/zipf_finetuning")
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    cache = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/zipf_finetuning"
    os.environ["HF_HUB_CACHE"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.mode == "prepare":
        verify_original_worlds()
        args.output.mkdir(parents=True, exist_ok=True)
        assert not (args.output / "protocol.json").exists()
        save(args.output / "protocol.json", {"created_at": datetime.now(timezone.utc).isoformat(), "models": MODELS,
             "pilot_seeds": PILOT_SEEDS, "main_seeds": MAIN_SEEDS, "curricula": CURRICULA, "live_worlds": MAIN_SEEDS,
             "budgets": [0, *BUDGETS], "grid": {"repeats": [1, 4, 16], "learning_rate": [1e-5, 3e-5, 1e-4]},
             "criterion": "Every pilot/curriculum achieves >=99% unweighted observed-key accuracy at 128 labels. Early eliminate failures; select smallest R then smallest LR. No test evaluation during calibration.",
             "training": "All weights FP32 with BF16 autocast, four legal LM output logits, AdamW fused, no weight decay or label smoothing, clip norm1. Each observation appends to multiset buffer, then R batches of current observation plus31 uniform draws from full acquired buffer.",
             "primary": "Weighted test error and paired ranked-minus-IID effective exponent at8–128, same32 original worlds; primary remains raw test predictions. Report4–64 and16–128 sensitivity and curve residuals.",
             "evaluation": "Same40 stratified keys and weights as ICL, independently predicted using only Dictionary key/Label prompt. No observation history at evaluation. Exact coverage plus exhaustive seen-key retention is a diagnostic marginalized over unobserved random labels.",
             "live": "Current model greedily emits grammar-constrained valid decimal keys, including repeats, at original block boundaries; bookkeeping context contains past key IDs but no labels. Guesses recorded before oracle feedback. Correct pairs receive identical SFT updates. No success filtering or automatic deduplication.",
             "archival": "Preselect world4000 final checkpoint for every family/curriculum; all worlds retain all predictions, observations, replay indices, counters and functional probes.",
             "spending": {"new_api_calls": 0, "vast_usd": 0, "slurm_job": 3529684}, "source_sha256": hashes()})
        source = args.output / "source"
        source.mkdir()
        for name in SOURCES: (source / name).write_bytes((ROOT / "experiments" / name).read_bytes())
        print(args.output / "protocol.json")
        return
    protocol = json.loads((args.output / "protocol.json").read_text())
    assert hashes() == protocol["source_sha256"]
    output = args.output / args.model
    output.mkdir(parents=True, exist_ok=True)
    learner = Learner(args.model, args.device)
    save(output / "environment.json", {"parameter_count": learner.parameter_count, "torch": learner.torch.__version__,
                                      "gpu": learner.torch.cuda.get_device_name(), "prompt_width": learner.width,
                                      "label_tokens": learner.target_tokens.tolist(), "source_sha256": hashes()})
    if args.mode in ("pipeline", "calibrate"): selection = calibrate(learner, args.model, output, cache)
    else: selection = json.loads((output / "selection.json").read_text())
    assert selection["status"] == "passed"
    if args.mode in ("pipeline", "main"):
        for seed in tqdm(MAIN_SEEDS, desc=f"{args.model} confirmation"):
            for curriculum in CURRICULA:
                run(learner, args.model, seed, curriculum, selection["learning_rate"], selection["repeats"],
                    output / "main" / f"{seed}_{curriculum}", True, cache, save_model=seed == 4000)
    if args.mode in ("pipeline", "live"):
        learner.make_trie()
        for seed in tqdm(MAIN_SEEDS, desc=f"{args.model} on-policy"):
            run(learner, args.model, seed, "on_policy", selection["learning_rate"], selection["repeats"],
                output / "live" / str(seed), True, cache, save_model=seed == 4000, live=True)


if __name__ == "__main__": main()
