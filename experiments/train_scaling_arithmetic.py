"""Fresh-task, resumable best-of-two arithmetic curves for the s4g study."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from tqdm.auto import tqdm

from bestn_llm import cohort, dot, forecast_weights, graded, selected
from bestn_precision import PreciseStudent, exact_matmul
from elicitation_task import messages
from run_elicitation_pilot import MODELS, ROOT, digest, save, setup
from train_bestn import checkpoint, proposal, stamp, weight_overlap
from train_elicitation import read_rows


SOURCES = ("train_scaling_arithmetic.py", "train_bestn.py", "bestn_precision.py", "bestn_llm.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def prepare(output, seeds, eta, steps):
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for name in SOURCES: (output / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    checks = sorted({0, 1, 2, 4, 8, 16, 32, 48, 64, *range(96, steps + 1, 32), steps})
    for seed in seeds:
        directory = output / str(seed)
        directory.mkdir()
        train = [p for step in range(steps) for p in cohort(seed + 1000000 + step, "high", 32)]
        evaluation = cohort(seed + 2000000, "high", 256)
        assert not {p["id"] for p in train} & {p["id"] for p in evaluation}
        save(directory / "train.json", train)
        save(directory / "evaluation.json", evaluation)
    save(output / "protocol.json", {"created_at_uk": stamp(), "batch_id": "s4g", "model": MODELS["general"], "seeds": seeds,
         "source_sha256": {name: digest(output / "source" / name) for name in SOURCES}, "eta": eta,
         "steps": steps, "batch": 32, "checks": checks, "evaluation_samples": 4,
         "forecast_checks": [0, 16, 64, 128], "checkpoint_checks": [0, 16, 64, 128, steps],
         "training": "Full FP32 plain SGD; summed completion NLL divided by128 and by32 selected targets. Temperature1 unrestricted full trajectories, max512tokens; uniform best-of-two correctness ties including all-failure groups. Fresh IID problem batches, retaining incidental repeats; no replay epochs.",
         "comparison": "Current versus frozen initial generator, same questions/RNG schedules/candidate budgets. Both students update. Token compute and diagnostic costs are separate.",
         "why_regime": "Previously calibrated four-operation arithmetic with small integer arguments; chosen before this pilot to seek progress beyond the hard task's early plateau, not by inspecting scaling fits.",
         "forecast": "At selected checkpoints estimate held-out success gradient from evaluation samples and source-selected gradient from an independent64-question eight-sample bank. Save their overlap and one-step linear forecast before further training. This estimates local geometry; it is not a whole-curve prediction by itself.",
         "stationarity": "From64updates onward, fit log(error) and log(mean selected-trace NLL) against log(candidates) at the last four positive checkpoints. Halt if either absolute slope is below.02, or if no evaluation errors remain. Abort if accuracy degrades more than.15 from baseline, completion below.95, or nonfinite loss/gradients.",
         "resume": "Full model, SGD and scheduler, RNG states, counters, position in the fresh-task order and history at updates0,16,64,128,256 (within maximum), and every early stop. Local first, asynchronously copied with SHA256 verification. Every invocation retains a new attempt directory.",
         "pilot_status": "Exploratory until separate confirmation protocol is frozen. Retain all outcomes, fit windows, and rejected scaling hypotheses."})


def generate(student, problems, samples, seed, directory, name):
    import torch
    for attempt, batch_size in enumerate((32, 16, 8, 4, 2, 1)):
        path = directory / f"{name}_try{attempt}.jsonl"
        try:
            usage = student.generate(problems, "reason", samples, 1., 512, batch_size, seed, path)
            return graded(read_rows(path), problems), usage, str(path)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1: raise
            student.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(json.dumps({"transient_oom": name, "retry_batch_size": batch_size // 2}), flush=True)
            time.sleep(min(2**attempt, 16))
    raise AssertionError("Generation OOM retry sequence exhausted")


def evaluate(student, problems, seed, directory, step):
    rows, usage, path = generate(student, problems, 4, seed + 300000 + step, directory, f"eval_{step:04d}")
    record = {"step": step, "candidates": 64 * step, "accuracy": float(np.mean([r["correct"] for r in rows])),
              "completion": float(np.mean([r["ended"] for r in rows])), "path": path, "usage": usage,
              "equation_probability": float(np.mean(["=" in r["text"] for r in rows])),
              "per_prompt": {p["id"]: float(np.mean([r["correct"] for r in rows if r["id"] == p["id"]])) for p in problems}}
    assert record["completion"] >= .95, "Generation completion collapsed; redesign required."
    return rows, record


def stopping(history):
    if history["step"] < 64: return False, {}
    records = history["evaluations"][-4:]
    errors = np.array([1 - row["accuracy"] for row in records])
    if np.any(errors == 0): return True, {"reason": "zero_observed_errors"}
    losses = [row["block_loss"] for row in records]
    x = np.log([row["candidates"] for row in records])
    slopes = {"log_error_slope": float(np.polyfit(x, np.log(errors), 1)[0]),
              "log_loss_slope": float(np.polyfit(x, np.log(losses), 1)[0])}
    return any(abs(value) < .02 for value in slopes.values()), slopes


def run(output, seed, source, resume, allocation):
    import torch
    relative = output.relative_to(ROOT / "results") / str(seed) / source
    local = setup() / relative
    persistent = Path(os.environ["CACHE_DIR"]) / "generative-learning" / relative
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    exact_matmul()
    protocol = json.loads((output / "protocol.json").read_text())
    assert seed in protocol["seeds"]
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    root = output / str(seed) / source
    root.mkdir(exist_ok=resume)
    local.mkdir(parents=True, exist_ok=resume)
    attempt = len(list(root.glob("attempt*")))
    directory = root / f"attempt{attempt}"
    directory.mkdir()
    train, evaluation = [json.loads((output / str(seed) / f"{split}.json").read_text()) for split in ("train", "evaluation")]
    student = PreciseStudent()
    initial = [p.detach().clone() for p in student.model.parameters()]
    optimizer = torch.optim.SGD(student.model.parameters(), lr=protocol["eta"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if resume:
        marker = max((json.loads(p.read_text()) for p in root.glob("*.pt.local.json")), key=lambda r: (r["step"], r["attempt"]))
        path = Path(marker["local"])
        if not path.exists(): path = Path(marker["persistent"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        history = state["history"]
        observer_rows = graded(read_rows(Path(history["observer_path"])), evaluation)
        _, direction = student.trace_gradient(observer_rows, evaluation, forecast_weights(observer_rows, evaluation, 2))
        student.model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        del state
        eval_rows = graded(read_rows(Path(history["evaluations"][-1]["path"])), evaluation)
    else:
        eval_rows, evaluation_record = evaluate(student, evaluation, seed, directory, 0)
        _, direction = student.trace_gradient(eval_rows, evaluation, forecast_weights(eval_rows, evaluation, 2))
        history = {"step": 0, "seed": seed, "source": source, "observer_path": evaluation_record["path"],
                   "evaluations": [evaluation_record], "updates": [], "forecasts": [], "rollout_tokens": 0,
                   "training_input_tokens": 0, "diagnostic_tokens": evaluation_record["usage"]["prompt_tokens"] + evaluation_record["usage"]["completion_tokens"],
                   "started_at_uk": stamp()}
    norm = np.sqrt(dot(direction, direction))
    assert norm > 0
    for vector in direction: vector.mul_(-1 / norm)
    save(directory / "environment.json", {"started_at_uk": stamp(), "allocation": allocation, "device": torch.cuda.get_device_name(),
         "gpu_count_inherited": torch.cuda.device_count(), "torch": torch.__version__, "resumed_step": history["step"]})
    halt, slopes = stopping(history)
    if halt:
        save(root / "complete.json", {**history, "stationarity": slopes, "stop_reason": "stationary_at_resume", "finished_at_uk": stamp()})
        return
    pending = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        if history["step"] == 0: pending = checkpoint(student, optimizer, scheduler, history, local, persistent, root, attempt, executor)
        progress = tqdm(range(history["step"], protocol["steps"]), desc=f"s4g {seed} {source}")
        for step in progress:
            if step in protocol["forecast_checks"]:
                _, eval_gradient = student.trace_gradient(eval_rows, evaluation, forecast_weights(eval_rows, evaluation, 2))
                forecast_tasks = cohort(seed + 500000 + step, "high", 64)
                with proposal(student, initial, source):
                    forecast_rows, forecast_usage, forecast_path = generate(student, forecast_tasks, 8, seed + 600000 + step, directory, f"forecast_{step:04d}")
                weights = forecast_weights(forecast_rows, forecast_tasks, 2)
                if source == "frozen": weights += 1 / len(forecast_rows)
                _, source_gradient = student.trace_gradient(forecast_rows, forecast_tasks, weights)
                overlap = dot(eval_gradient, source_gradient)
                forecast = {"step": step, "written_before_update_at_uk": stamp(), "gradient_overlap_nll_units": overlap,
                            "success_gradient_overlap": 128**2 * overlap,
                            "predicted_one_step_gain": 128 * protocol["eta"] * overlap,
                            "evaluation_accuracy": history["evaluations"][-1]["accuracy"],
                            "source_gradient_norm_squared": dot(source_gradient, source_gradient),
                            "evaluation_gradient_norm_squared": dot(eval_gradient, eval_gradient),
                            "path": forecast_path, "evaluation_path": history["evaluations"][-1]["path"],
                            "weight_overlap": weight_overlap(student, initial, direction)}
                save(directory / f"forecast_{step:04d}.json", forecast)
                history["forecasts"].append(forecast)
                history["diagnostic_tokens"] += forecast_usage["prompt_tokens"] + forecast_usage["completion_tokens"]
                del eval_gradient, source_gradient
            batch = train[step * 32:(step + 1) * 32]
            with proposal(student, initial, source):
                rows, usage, path = generate(student, batch, 2, seed + 700000 + step, directory, f"train_{step:04d}")
            chosen = selected(rows, batch, 2, seed + 800000 + step)
            loss, gradient = student.trace_gradient(chosen, batch, np.full(32, 1 / 32))
            for parameter, vector in zip(student.model.parameters(), gradient, strict=True): parameter.grad = vector
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            del gradient
            history["step"] = step + 1
            history["rollout_tokens"] += usage["prompt_tokens"] + usage["completion_tokens"]
            input_tokens = sum(len(student.tokenizer.apply_chat_template(messages(p, "reason"), tokenize=True, add_generation_prompt=True)) + r["completion_tokens"] for p, r in zip(batch, chosen, strict=True))
            history["training_input_tokens"] += input_tokens
            history["updates"].append({"step": step + 1, "loss": loss, "path": path, "usage": usage, "training_input_tokens": input_tokens,
                                       "preupdate_source_accuracy": float(np.mean([r["correct"] for r in rows])),
                                       "selected_success": float(np.mean([r["correct"] for r in chosen])),
                                       "selected": [{k: r[k] for k in ("id", "sample", "correct")} for r in chosen]})
            if history["step"] in protocol["checks"]:
                previous = history["evaluations"][-1]["step"]
                eval_rows, record = evaluate(student, evaluation, seed, directory, history["step"])
                record["block_loss"] = float(np.mean([u["loss"] for u in history["updates"] if previous < u["step"] <= history["step"]]))
                record["weight_overlap"] = weight_overlap(student, initial, direction)
                history["evaluations"].append(record)
                history["diagnostic_tokens"] += record["usage"]["prompt_tokens"] + record["usage"]["completion_tokens"]
                halt, slopes = stopping(history)
                history["stationarity"] = slopes
                save(root / "progress.json", history)
                print(json.dumps({"step": history["step"], "accuracy": record["accuracy"], "loss": record["block_loss"], "slopes": slopes}), flush=True)
                assert record["accuracy"] >= history["evaluations"][0]["accuracy"] - .15, "Learner degraded; redesign required."
                if history["step"] in protocol["checkpoint_checks"] or halt:
                    if pending is not None: pending.result()
                    pending = checkpoint(student, optimizer, scheduler, history, local, persistent, root, attempt, executor)
                if halt:
                    history["stop_reason"] = "stationary"
                    break
        if pending is not None: pending.result()
    save(root / "complete.json", {**history, "finished_at_uk": stamp()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/scaling_s4g/arithmetic/pilot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[95001])
    parser.add_argument("--eta", type=float, default=1.8291510011316367e-5)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=95001)
    parser.add_argument("--source", choices=("current", "frozen"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allocation", type=str)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.output, args.seeds, args.eta, args.steps)
    else: run(args.output, args.seed, args.source, args.resume, args.allocation)
