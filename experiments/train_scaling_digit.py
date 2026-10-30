"""Resumable full-weight best-of-two with exact one-token evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from tqdm.auto import tqdm

from bestn_llm import dot
from bestn_precision import exact_matmul
from run_elicitation_pilot import MODELS, ROOT, digest, save, setup
from scaling_choice_model import ChoiceStudent
from scaling_conditional_gate import forecast
from scaling_digit_task import SYMBOLS, encode, messages, problems
from train_bestn import checkpoint, proposal, stamp, weight_overlap


SOURCES = ("train_scaling_digit.py", "scaling_choice_model.py", "scaling_choice_task.py", "scaling_digit_task.py", "scaling_conditional_gate.py", "train_bestn.py", "bestn_precision.py", "bestn_llm.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def prepare(output, seeds, steps):
    selected = json.loads((ROOT / "results/scaling_s4g/digit/shorter_count_calibration/complete.json").read_text())["selected"]
    assert selected is not None
    output.mkdir(parents=True, exist_ok=False); (output / "source").mkdir()
    for name in SOURCES: (output / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    checks = sorted({0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, steps} & set(range(steps + 1)))
    for seed in seeds:
        directory = output / str(seed); directory.mkdir()
        evaluation = list({r["id"]: r for r in problems(seed + 100000, 2048, selected)}.values())[:1024]
        assert len(evaluation) == 1024
        excluded = {r["id"] for r in evaluation}
        train = []
        for step in range(steps):
            batch = [r for r in problems(seed + 200000 + step, 128, selected) if r["id"] not in excluded][:32]
            assert len(batch) == 32
            train.extend(batch)
        observer = [r for r in problems(seed + 300000, 1024, selected) if r["id"] not in excluded][:512]
        assert len(observer) == 512
        for split, rows in (("train", train), ("evaluation", evaluation), ("observer", observer)): save(directory / f"{split}.json", rows)
    save(output / "protocol.json", {"created_at_uk": stamp(), "batch_id": "s4g", "model": MODELS["general"], "cell": selected,
         "seeds": seeds, "steps": steps, "batch": 32, "checks": checks, "target_initial_gain": .005,
         "checkpoint_checks": [0, 32, 128, 512, steps], "gradient_checks": [0, 32, 128, 512, steps],
         "source_sha256": {name: digest(output / "source" / name) for name in SOURCES},
         "policy": "Full vocabulary at temperature1; generate exactly one token after a fixed Answer prefix. All model weights train in FP32, TF32 off, plain SGD, no clipping/decay/head/vocabulary mask. Two IID tokens per question; uniform maximum-reward ties including two failures. Only selected sampled tokens enter cross-entropy.",
         "comparison": "Current versus frozen initial proposals. Same fresh question batches, RNG schedules, candidate and optimizer budgets; both students update. Prompt computation is shared for the two independent first-token draws. Gold labels enter only correctness selection and separately accounted diagnostic evaluation/gradients.",
         "data": "1024 held-out distinct questions, never used for training; fresh IID training draws conditional on excluding held-out IDs, retaining incidental repeats. The finite counting support and accumulated unique question count are reported. Observer cohort512 is independent of the train stream and disjoint from evaluation, but may overlap train support.",
         "calibration": "Before any sampled training, set eta=.005/(initial training/evaluation success-gradient overlap). Freeze a conditional one-weight Boltzmann forecast with per-question initial biases. Use exact initial probabilities and overlap only. Check a single population-gradient Taylor step, then restore the initial weights; this diagnostic never seeds training.",
         "forecast_assumptions": "A shared additive success-logit shift, stable gradient metric, and attainable correctness1. Input heterogeneity is retained through fixed initial logits. These are hypotheses about the full network, not architecture constraints.",
         "stationarity": "At64 or more updates, fit log exact held-out error and log mean sampled-target NLL against log candidates over the latest four positive checkpoints. Stop if either absolute slope<.02. Abort on nonfinite gradients/loss, valid-digit mass<.95, or accuracy more than.15 below baseline.",
         "resume": "Full weights, optimizer, scheduler, RNG states, step/order position and history; local first then asynchronous SHA256-verified persistent copies at0,32,128,512,1024 and every early stop. Every retry retains an attempt directory.",
         "analysis": "Exploratory pilot. Report all checkpoints and sources; compare predeclared whole-curve forecasts, gradient geometry, shifted powers and exponentials with floors. Separate fresh confirmation is required before a robust LLM exponent claim. This horizon1 task does not test multistep reasoning exploration.",
         "resources": "Institutional gpu_lowp H100; no paid API or rental."})


def initialize(output):
    import torch
    setup(); os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"; assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4); exact_matmul()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    student = ChoiceStudent(); student.answer_symbols = SYMBOLS
    return protocol, student


def calibrated_data(student, directory):
    return {split: encode(student, json.loads((directory / f"{split}.json").read_text())) for split in ("train", "evaluation", "observer")}


def calibrate(output, seed, allocation):
    import torch
    protocol, student = initialize(output)
    directory = output / str(seed); rows = calibrated_data(student, directory)
    initial = [p.detach().clone() for p in student.model.parameters()]
    train = student.evaluate(rows["observer"]); evaluation = student.evaluate(rows["evaluation"])
    _, train_gradient = student.success_gradient(rows["observer"])
    _, eval_gradient = student.success_gradient(rows["evaluation"])
    overlap = dot(train_gradient, eval_gradient); assert overlap > 0
    eta = protocol["target_initial_gain"] / overlap
    prediction = forecast([np.exp(r["correct_logprob"]) for r in train["records"]],
                          [np.exp(r["correct_logprob"]) for r in evaluation["records"]], eta * overlap, protocol["steps"])
    numeric = {"created_before_training_at_uk": stamp(), "allocation": allocation, "seed": seed, "eta": eta,
               "train": train, "evaluation": evaluation, "gradient_overlap": overlap,
               "train_gradient_norm_squared": dot(train_gradient, train_gradient), "eval_gradient_norm_squared": dot(eval_gradient, eval_gradient),
               "prediction": prediction}
    save(directory / "forecast.json", numeric)
    with torch.no_grad():
        for p, g in zip(student.model.parameters(), train_gradient, strict=True): p.add_(g, alpha=-eta)
    after = student.evaluate(rows["evaluation"])
    actual = after["accuracy"] - evaluation["accuracy"]
    save(directory / "taylor_audit.json", {"observed_at_uk": stamp(), "predicted_gain": eta * overlap, "population_step_gain": actual,
                                           "relative_difference": actual / (eta * overlap) - 1, "after": after})
    with torch.no_grad():
        for p, value in zip(student.model.parameters(), initial, strict=True): p.copy_(value)
    assert abs(actual / (eta * overlap) - 1) < .25, "Initial Taylor step is nonlinear; redesign rate before training."
    save(directory / "calibrated.json", {"finished_at_uk": stamp(), "forecast_sha256": digest(directory / "forecast.json"), "eta": eta})
    print(json.dumps({"seed": seed, "eta": eta, "baseline_accuracy": evaluation["accuracy"], "predicted_gain": eta * overlap, "observed_population_gain": actual}), flush=True)


def update(student, initial, source, batch, seed, optimizer):
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    target = torch.tensor([r["target_id"] for r in batch], device="cuda")
    with proposal(student, initial, source), torch.no_grad():
        policy = student.logits(batch).softmax(-1)
        draws = torch.multinomial(policy, 2, replacement=True)
        rewards = draws == target[:, None]
        index = torch.randint(2, (len(batch),), device="cuda")
        single = rewards.sum(-1) == 1
        index[single] = rewards[single].long().argmax(-1)
        chosen = draws.gather(1, index[:, None]).squeeze(1)
        source_success = float(policy.gather(1, target[:, None]).mean())
    loss_value = 0.
    for attempt, microbatch in enumerate((16, 8, 4, 2, 1)):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        try:
            for offset in range(0, len(batch), microbatch):
                loss = torch.nn.functional.cross_entropy(student.logits(batch[offset:offset + microbatch]), chosen[offset:offset + microbatch], reduction="sum") / len(batch)
                assert torch.isfinite(loss)
                loss.backward(); loss_value += float(loss.detach())
            break
        except torch.cuda.OutOfMemoryError:
            if microbatch == 1: raise
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(json.dumps({"transient_oom": "training", "retry_microbatch": microbatch // 2}), flush=True)
            time.sleep(min(2**attempt, 16))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.model.parameters())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    records = [{"id": row["id"], "ground_truth": row["label"], "candidate_ids": tokens,
                "candidate_texts": [student.tokenizer.decode([token]) for token in tokens], "correct": correct, "chosen_index": winner}
               for row, tokens, correct, winner in zip(batch, draws.cpu().tolist(), rewards.cpu().tolist(), index.cpu().tolist(), strict=True)]
    return {"loss": loss_value, "source_success": source_success, "sampled_source_success": float(rewards.float().mean()),
            "selected_success": float((chosen == target).float().mean()), "records": records}


def stopping(history):
    if history["step"] < 64: return False, {}
    records = history["evaluations"][-4:]
    x = np.log([r["step"] for r in records])
    slopes = {"log_error_slope": float(np.polyfit(x, np.log([r["error"] for r in records]), 1)[0]),
              "log_loss_slope": float(np.polyfit(x, np.log([r["block_loss"] for r in records]), 1)[0])}
    return any(abs(value) < .02 for value in slopes.values()), slopes


def run(output, seed, source, resume, allocation):
    import torch
    protocol, student = initialize(output)
    cohort = output / str(seed); numeric = json.loads((cohort / "forecast.json").read_text())
    assert json.loads((cohort / "calibrated.json").read_text())["forecast_sha256"] == digest(cohort / "forecast.json")
    rows = calibrated_data(student, cohort)
    root = cohort / source; root.mkdir(exist_ok=resume)
    relative = output.relative_to(ROOT / "results") / str(seed) / source
    local = setup() / relative; local.mkdir(parents=True, exist_ok=resume)
    persistent = Path(os.environ["CACHE_DIR"]) / "generative-learning" / relative
    attempt = len(list(root.glob("attempt*"))); directory = root / f"attempt{attempt}"; directory.mkdir()
    initial = [p.detach().clone() for p in student.model.parameters()]
    _, direction = student.success_gradient(rows["evaluation"])
    direction_norm = np.sqrt(dot(direction, direction)); assert direction_norm > 0
    for value in direction: value.mul_(-1 / direction_norm)
    optimizer = torch.optim.SGD(student.model.parameters(), lr=numeric["eta"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    if resume:
        marker = max((json.loads(p.read_text()) for p in root.glob("*.pt.local.json")), key=lambda r: (r["step"], r["attempt"]))
        path = Path(marker["local"])
        if not path.exists(): path = Path(marker["persistent"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        student.model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        history = state["history"]; del state
    else:
        history = {"step": 0, "seed": seed, "source": source, "eta": numeric["eta"], "started_at_uk": stamp(), "updates": [], "gradients": [],
                   "evaluations": [{"step": 0, **numeric["evaluation"]}], "candidates": 0, "presentations": 0,
                   "unique_questions": 0, "proposal_prompt_tokens": 0, "training_tokens": 0,
                   "diagnostic_prompt_tokens": sum(len(r["input_ids"]) for r in rows["evaluation"])}
    save(directory / "environment.json", {"started_at_uk": stamp(), "allocation": allocation, "device": torch.cuda.get_device_name(),
         "inherited_gpu_count": torch.cuda.device_count(), "torch": torch.__version__, "parameter_count": sum(p.numel() for p in student.model.parameters()), "resumed_step": history["step"]})
    halt, slopes = stopping(history)
    if halt:
        save(root / "complete.json", {**history, "stationarity": slopes, "stop_reason": "stationary_at_resume", "finished_at_uk": stamp()}); return
    seen = {r["id"] for r in rows["train"][:history["step"] * 32]}
    pending = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        if history["step"] == 0: pending = checkpoint(student, optimizer, scheduler, history, local, persistent, root, attempt, executor)
        for step in tqdm(range(history["step"], protocol["steps"]), desc=f"s4g digit {seed} {source}"):
            if step in protocol["gradient_checks"]:
                train_success, train_gradient = student.success_gradient(rows["observer"])
                eval_success, eval_gradient = student.success_gradient(rows["evaluation"])
                record = {"step": step, "written_before_update_at_uk": stamp(), "train_success": train_success, "eval_success": eval_success,
                          "current_success_gradient_overlap": dot(train_gradient, eval_gradient),
                          "train_gradient_norm_squared": dot(train_gradient, train_gradient), "eval_gradient_norm_squared": dot(eval_gradient, eval_gradient),
                          "initial_direction_projection": -dot(eval_gradient, direction), "weight_overlap": weight_overlap(student, initial, direction)}
                save(directory / f"gradient_{step:04d}.json", record); history["gradients"].append(record)
                history["diagnostic_prompt_tokens"] += sum(len(r["input_ids"]) for split in ("observer", "evaluation") for r in rows[split])
                del train_gradient, eval_gradient
            batch = rows["train"][step * 32:(step + 1) * 32]
            start = time.monotonic()
            record = update(student, initial, source, batch, seed + 400000 + step, optimizer)
            scheduler.step(); history["step"] = step + 1
            path = directory / f"train_{step:04d}.json"; save(path, {"step": step + 1, **record})
            seen.update(r["id"] for r in batch)
            tokens = sum(len(r["input_ids"]) for r in batch)
            history["candidates"] += 64; history["presentations"] += 32; history["unique_questions"] = len(seen)
            history["proposal_prompt_tokens"] += tokens; history["training_tokens"] += tokens + 32
            history["updates"].append({"step": step + 1, **{k: v for k, v in record.items() if k != "records"}, "path": str(path), "seconds": time.monotonic() - start})
            if history["step"] in protocol["checks"]:
                previous = history["evaluations"][-1]["step"]
                evaluation = {"step": history["step"], **student.evaluate(rows["evaluation"])}
                evaluation["block_loss"] = float(np.mean([r["loss"] for r in history["updates"] if previous < r["step"] <= history["step"]]))
                evaluation["weight_overlap"] = weight_overlap(student, initial, direction)
                history["evaluations"].append(evaluation)
                history["diagnostic_prompt_tokens"] += sum(len(r["input_ids"]) for r in rows["evaluation"])
                halt, slopes = stopping(history); history["stationarity"] = slopes
                save(root / "progress.json", history)
                print(json.dumps({"step": history["step"], "accuracy": evaluation["accuracy"], "greedy": evaluation["greedy_accuracy"], "loss": evaluation["block_loss"], "slopes": slopes}), flush=True)
                assert evaluation["accuracy"] >= numeric["evaluation"]["accuracy"] - .15 and evaluation["valid_mass"] >= .95, "Learner degraded or lost format; redesign required."
                if history["step"] in protocol["checkpoint_checks"] or halt:
                    if pending is not None: pending.result()
                    pending = checkpoint(student, optimizer, scheduler, history, local, persistent, root, attempt, executor)
                if halt:
                    history["stop_reason"] = "stationary"; break
        if pending is not None: pending.result()
    save(root / "complete.json", {**history, "finished_at_uk": stamp()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "calibrate", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/scaling_s4g/digit/pilot")
    parser.add_argument("--seed", type=int, default=99101); parser.add_argument("--seeds", type=int, nargs="+", default=[99101])
    parser.add_argument("--steps", type=int, default=1024); parser.add_argument("--source", choices=("current", "frozen"))
    parser.add_argument("--allocation"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); args.output = args.output.resolve()
    if args.command == "prepare": prepare(args.output, args.seeds, args.steps)
    elif args.command == "calibrate": calibrate(args.output, args.seed, args.allocation)
    else: run(args.output, args.seed, args.source, args.resume, args.allocation)
