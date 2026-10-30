"""Measure a same-model elicitation gap before attempting self-training.

Every response, including parse failures and truncations, is retained. The model
runner imports no grader; answers are assessed by analyze_elicitation_pilot.py.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from tqdm.auto import tqdm

from elicitation_task import DEPTHS, FAMILIES, make_problems, messages, parse_answer


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "math": {"name": "Qwen/Qwen2.5-Math-1.5B-Instruct", "revision": "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"},
    "general": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
}
CONDITIONS = {"direct_greedy": ("direct", 1, 0., 96), "reason_greedy": ("reason", 1, 0., 512),
              "reason_samples": ("reason", 4, 1., 512)}
SOURCES = ("elicitation_task.py", "run_elicitation_pilot.py", "analyze_elicitation_pilot.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def setup():
    load_dotenv(Path.home() / ".env")
    cache = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning"
    os.environ["HF_HUB_CACHE"] = str(cache / "hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return cache


class Learner:
    def __init__(self, model_key):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config = MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["name"], revision=self.config["revision"])
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config["name"], revision=self.config["revision"],
                                                       dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

    def generate(self, problems, mode, samples, temperature, maximum, batch_size, seed, destination, demonstrations=()):
        import torch
        expanded = [(problem, sample) for problem in problems for sample in range(samples)]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "padded_prompt_tokens": 0, "padded_completion_tokens": 0,
                 "responses": 0, "seconds": 0., "seed": seed, "batch_size": batch_size}
        start = time.monotonic()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with destination.open("x") as stream:
                for offset in tqdm(range(0, len(expanded), batch_size), desc=f"{mode} T={temperature:g}"):
                    rows = expanded[offset:offset + batch_size]
                    texts = [self.tokenizer.apply_chat_template(messages(row, mode, demonstrations), tokenize=False, add_generation_prompt=True)
                             for row, _ in rows]
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                    options = {"do_sample": temperature > 0, "max_new_tokens": maximum, "use_cache": True,
                               "top_k": 0, "top_p": 1., "repetition_penalty": 1.}
                    if temperature > 0: options["temperature"] = temperature
                    with torch.inference_mode(): output = self.model.generate(**inputs, **options)
                    ids = output[:, inputs["input_ids"].shape[1]:].cpu().tolist()
                    usage["prompt_tokens"] += int(inputs["attention_mask"].sum())
                    usage["padded_prompt_tokens"] += inputs["input_ids"].numel()
                    usage["padded_completion_tokens"] += sum(map(len, ids))
                    for (problem, sample), generated in zip(rows, ids, strict=True):
                        ended = self.tokenizer.eos_token_id in generated
                        size = generated.index(self.tokenizer.eos_token_id) + 1 if ended else len(generated)
                        generated = generated[:size]
                        text = self.tokenizer.decode(generated, skip_special_tokens=True)
                        record = {"id": problem["id"], "sample": sample, "mode": mode, "temperature": temperature,
                                  "ended": ended, "answer": parse_answer(text), "text": text,
                                  "completion_tokens": size, "token_ids": generated}
                        stream.write(json.dumps(record) + "\n")
                        usage["responses"] += 1
                        usage["completion_tokens"] += size
                    stream.flush()
                    os.fsync(stream.fileno())
        torch.cuda.synchronize()
        usage["seconds"] = time.monotonic() - start
        save(destination.with_suffix(".usage.json"), usage)
        return usage


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for name in SOURCES: (output / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    save(output / "calibration.json", make_problems(61000, 16))
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "elicitation_screen",
                "models_in_order": MODELS, "families_in_order": FAMILIES, "depths": DEPTHS,
                "conditions": CONDITIONS, "calibration_seed": 61000, "calibration_per_cell": 16,
                "confirmation_seed": 62000, "confirmation_per_depth": 48,
                "source_sha256": {name: digest(output / "source" / name) for name in SOURCES},
                "calibration_sha256": digest(output / "calibration.json"),
                "gate": "First model/family in declared order with plurality accuracy >=.70, plurality-minus-direct >=.15, and >=.95 completion rate among sampled reasoning. Model general is run only if math supplies no qualifying family.",
                "confirmation": "The chosen model/family is evaluated once on144fresh problems with the same three conditions. Report paired whole-problem bootstrap CI, all depths, formats and truncations. No exponent fit.",
                "information": "No weights updated. Exact answers used only by the offline analyzer. Consensus never inspects correctness. No external verifier or label is used to select a response.",
                "resources": "One institutional gpu_lowp H100; no paid API or Vast calls. All generation and padding tokens and seconds retained."}
    save(output / "protocol.json", protocol)


def run(output, model_key, stage, batch_size):
    import torch
    import transformers
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    problems = json.loads((output / f"{stage}.json").read_text())
    directory = output / stage / model_key
    directory.mkdir(parents=True, exist_ok=False)
    learner = Learner(model_key)
    save(directory / "environment.json", {"torch": torch.__version__, "transformers": transformers.__version__,
         "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "model": learner.config,
         "parameters": sum(p.numel() for p in learner.model.parameters()), "dtype": str(next(learner.model.parameters()).dtype),
         "job_id": os.environ["SLURM_JOB_ID"]})
    for index, (condition, (mode, samples, temperature, maximum)) in enumerate(CONDITIONS.items()):
        usage = learner.generate(problems, mode, samples, temperature, maximum, batch_size,
                                 (61000 if stage == "calibration" else 62000) + index * 100000,
                                 directory / f"{condition}.jsonl")
        print(condition, usage, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/elicitation_pilot")
    parser.add_argument("--model", choices=MODELS, default="math")
    parser.add_argument("--stage", choices=("calibration", "confirmation"), default="calibration")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    setup()
    if args.command == "prepare": prepare(args.output)
    else: run(args.output, args.model, args.stage, args.batch_size)


if __name__ == "__main__": main()
