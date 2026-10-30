"""Resumable FP32 current-versus-frozen best-of-2, with prospective forecasts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from bestn_llm import cohort, dot, forecast_weights, graded, selected
from bestn_precision import PreciseStudent, exact_matmul
from probe_bestn_success import importance_change, log_probabilities
from run_elicitation_pilot import MODELS, ROOT, digest, save, setup
from train_elicitation import read_rows


OUTPUT = ROOT / "results/bestn_b2k/training"
SEEDS = (86001, 86002, 86003)
SOURCES = ("train_bestn.py", "bestn_precision.py", "bestn_llm.py", "probe_bestn_success.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def stamp(): return datetime.now(ZoneInfo("Europe/London")).isoformat()


def prepare(eta):
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "source").mkdir()
    for name in SOURCES: (OUTPUT / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    for seed in SEEDS:
        directory = OUTPUT / str(seed)
        directory.mkdir()
        train, evaluation = cohort(seed + 1000, "low", 64), cohort(seed + 2000, "low", 96)
        assert not {p["id"] for p in train} & {p["id"] for p in evaluation}
        save(directory / "train.json", train)
        save(directory / "evaluation.json", evaluation)
        rng = np.random.default_rng(seed)
        save(directory / "order.json", np.concatenate([rng.permutation(64) for _ in range(8)]).tolist())
    save(OUTPUT / "protocol.json", {"created_at_uk": stamp(), "batch_id": "b2k", "model": MODELS["general"], "seeds": SEEDS,
         "source_sha256": {name: digest(OUTPUT / "source" / name) for name in SOURCES}, "eta": eta,
         "precision": "Full FP32, TF32 disabled, no autocast; generation and fitting both at temperature1.",
         "training": "Plain SGD, batch16, no clipping or decay, one update per group; sum completion NLL divided by128 and batch size. Two IID full trajectories per prompt, uniformly select a maximum-correctness trace, retaining all-failure groups.",
         "contrast": "current versus initial frozen proposal; identical prompts, candidate counts, selection, update counts, and RNG schedules. First update should match exactly. Only subsequent proposal weights differ.",
         "train_unique": 64, "eval_unique": 96, "max_updates": 32, "presentations": 512, "candidates": 1024,
         "evaluation_every_updates": 4, "evaluation_samples_per_prompt": 4, "forecast_samples_per_training_prompt": 4,
         "forecast": "Before each four-update block, estimate the expected selected-score direction from an independent forecast bank and dot it with the current held-out success gradient. Predict the next block's success gain without fitting a rate. The current-source estimator subtracts its zero-mean ordinary score; frozen-source predictions retain the ordinary off-policy drift.",
         "measurement": "Fresh held-out sample accuracy at every checkpoint, plus paired unclipped importance-ratio estimates of each preceding block's gain. Forecast and actual updates use independent rollout banks. Evaluation/forecast generation and gradients are diagnostic costs and are reported separately from learner costs.",
         "weight_overlap": "Measure displacement from initial weights projected onto the initial held-out reward-gradient direction. This vector is diagnostic only and never changes training.",
         "primary": "Final current-minus-frozen held-out sampled accuracy at matched candidate budget, paired within three independent task/sampling seeds. Report all seeds and forecast errors; no exponent fit.",
         "stationarity": "At16 or more updates, stop if either recent log-error/log-candidate or log-training-NLL/log-candidate slope over four checkpoints has absolute value below.02. Abort on nonfinite gradients, accuracy below.01, completion below.90, or a drop over.15 from baseline.",
         "resume": "Checkpoint every four updates with full model, optimizer, LR scheduler, counters, RNG, data order position, evaluation/forecast history and observer provenance. Write local first and asynchronously copy to CACHE_DIR. Each retry uses a new attempt directory; interrupted outputs are retained.",
         "resources": "Institutional gpu_lowp H100; no paid services. Runs may exceed30 minutes, so full resume is mandatory."})


@contextmanager
def proposal(student, initial, source):
    import torch
    if source == "current":
        yield
        return
    assert source == "frozen"
    current = [p.detach().clone() for p in student.model.parameters()]
    with torch.no_grad():
        for p, original in zip(student.model.parameters(), initial, strict=True): p.copy_(original)
    try: yield
    finally:
        with torch.no_grad():
            for p, value in zip(student.model.parameters(), current, strict=True): p.copy_(value)


def generate(student, problems, samples, seed, directory, name):
    import torch
    for attempt, batch_size in enumerate((32, 16, 8, 4, 2, 1)):
        path = directory / f"{name}_try{attempt}.jsonl"
        try:
            usage = student.generate(problems, "reason", samples, 1., 1024, batch_size, seed, path)
            return graded(read_rows(path), problems), usage, str(path)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1: raise
            student.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(json.dumps({"transient_oom": name, "retry_batch_size": batch_size // 2}), flush=True)
            time.sleep(min(2**attempt, 16))
    raise AssertionError("Generation retry sequence exhausted")


def evaluate(student, problems, seed, directory, step):
    rows, usage, path = generate(student, problems, 4, seed + 100000 + step, directory, f"eval_{step:03d}")
    result = {"step": step, "candidates": step * 32, "accuracy": float(np.mean([r["correct"] for r in rows])),
              "completion": float(np.mean([r["ended"] for r in rows])), "path": path, "usage": usage,
              "per_prompt": {p["id"]: float(np.mean([r["correct"] for r in rows if r["id"] == p["id"]])) for p in problems}}
    assert result["accuracy"] >= .01, "Held-out learner is at chance; redesign required."
    assert result["completion"] >= .90, "Generation has collapsed; redesign required."
    return rows, result


def weight_overlap(student, initial, direction):
    import torch
    projected, squared = 0., 0.
    with torch.no_grad():
        for p, original, e in zip(student.model.parameters(), initial, direction, strict=True):
            delta = p - original
            projected += float(torch.sum(delta * e, dtype=torch.float64))
            squared += float(torch.sum(delta.square(), dtype=torch.float64))
    return {"success_direction_projection": projected, "displacement_squared_norm": squared,
            "cosine": projected / np.sqrt(squared) if squared else 0.}


def stationary(history):
    if history["step"] < 16: return False, {}
    checkpoints = history["evaluations"][-4:]
    x = np.log([r["candidates"] for r in checkpoints])
    error = np.log([1 - r["accuracy"] for r in checkpoints])
    losses = np.log([np.mean([u["loss"] for u in history["updates"] if r["step"] - 4 < u["step"] <= r["step"]]) for r in checkpoints])
    slopes = {"log_error_slope": float(np.polyfit(x, error, 1)[0]), "log_training_loss_slope": float(np.polyfit(x, losses, 1)[0])}
    return any(abs(v) < .02 for v in slopes.values()), slopes


def archive(local, remote, metadata):
    remote.parent.mkdir(parents=True, exist_ok=True)
    temporary = remote.with_suffix(".partial")
    shutil.copyfile(local, temporary)
    with local.open("rb") as stream: local_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with temporary.open("rb") as stream: remote_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert local_hash == remote_hash
    temporary.replace(remote)
    save(metadata, {"local": str(local), "persistent": str(remote), "sha256": local_hash, "bytes": remote.stat().st_size, "archived_at_uk": stamp()})


def checkpoint(student, optimizer, scheduler, history, local, persistent, root, attempt, executor):
    import torch
    name = f"step{history['step']:03d}_attempt{attempt}.pt"
    local_path, persistent_path = local / name, persistent / name
    state = {"model": {key: value.detach().cpu() for key, value in student.model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "saved_at_uk": stamp()}
    temporary = local_path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(local_path)
    save(root / f"{name}.local.json", {"step": history["step"], "attempt": attempt, "local": str(local_path), "persistent": str(persistent_path)})
    return executor.submit(archive, local_path, persistent_path, root / f"{name}.archive.json")


def run(seed, source, resume):
    import torch
    cache = setup() / "bestn_b2k/training" / str(seed) / source
    persistent = Path(os.environ["CACHE_DIR"]) / "generative-learning/bestn_b2k/training" / str(seed) / source
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    exact_matmul()
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    cohort_root = OUTPUT / str(seed)
    root = cohort_root / source
    root.mkdir(exist_ok=resume)
    cache.mkdir(parents=True, exist_ok=resume)
    attempt = len(list(root.glob("attempt*")))
    directory = root / f"attempt{attempt}"
    directory.mkdir()
    train, evaluation = [json.loads((cohort_root / f"{split}.json").read_text()) for split in ("train", "evaluation")]
    order = json.loads((cohort_root / "order.json").read_text())
    student = PreciseStudent()
    initial = [p.detach().clone() for p in student.model.parameters()]
    optimizer = torch.optim.SGD(student.model.parameters(), lr=protocol["eta"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    markers = [json.loads(p.read_text()) for p in root.glob("*.pt.local.json")]
    if resume and markers:
        marker = max(markers, key=lambda item: (item["step"], item["attempt"]))
        path = Path(marker["local"])
        if not path.is_file(): path = Path(marker["persistent"])
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
        history = {"step": 0, "seed": seed, "source": source, "observer_path": evaluation_record["path"], "order": order,
                   "evaluations": [evaluation_record], "updates": [], "forecasts": [], "rollout_tokens": 0, "training_tokens": 0,
                   "diagnostic_tokens": evaluation_record["usage"]["prompt_tokens"] + evaluation_record["usage"]["completion_tokens"], "started_at_uk": stamp()}
    norm = np.sqrt(dot(direction, direction))
    assert norm > 0
    for vector in direction: vector.mul_(-1 / norm)
    save(directory / "environment.json", {"started_at_uk": stamp(), "seed": seed, "source": source, "job_id": os.environ["SLURM_JOB_ID"],
         "device": torch.cuda.get_device_name(), "cuda_visible_devices_inherited": os.environ["CUDA_VISIBLE_DEVICES"], "resumed_step": history["step"]})
    halt, slopes = stationary(history)
    if halt:
        save(root / "complete.json", {**history, "stationarity": slopes, "stop_reason": "stationary_at_resume", "finished_at_uk": stamp()})
        return
    pending = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        if history["step"] == 0:
            pending = checkpoint(student, optimizer, scheduler, history, cache, persistent, root, attempt, executor)
        for block_start in tqdm(range(history["step"], protocol["max_updates"], 4), desc=f"{seed} {source} blocks"):
            baseline_logs = log_probabilities(student, eval_rows, evaluation)
            _, eval_gradient = student.trace_gradient(eval_rows, evaluation, forecast_weights(eval_rows, evaluation, 2))
            with proposal(student, initial, source):
                forecast_rows, forecast_usage, forecast_path = generate(student, train, 4, seed + 200000 + block_start, directory, f"forecast_{block_start:03d}")
            weights = forecast_weights(forecast_rows, train, 2)
            if source == "frozen": weights += 1 / len(forecast_rows)
            _, forecast_gradient = student.trace_gradient(forecast_rows, train, weights)
            predicted = 128 * protocol["eta"] * 4 * dot(eval_gradient, forecast_gradient)
            forecast = {"step": block_start, "end_step": block_start + 4, "written_before_updates_at_uk": stamp(), "predicted_success_gain": predicted,
                        "forecast_path": forecast_path, "evaluation_path": history["evaluations"][-1]["path"], "forecast_usage": forecast_usage,
                        "gradient_overlap": dot(eval_gradient, forecast_gradient), "expected_gradient_norm_squared": dot(forecast_gradient, forecast_gradient)}
            save(directory / f"forecast_{block_start:03d}.json", forecast)
            del eval_gradient, forecast_gradient
            history["diagnostic_tokens"] += forecast_usage["prompt_tokens"] + forecast_usage["completion_tokens"]
            for step in range(block_start, block_start + 4):
                batch = [train[i] for i in order[step * 16:(step + 1) * 16]]
                with proposal(student, initial, source):
                    rows, usage, path = generate(student, batch, 2, seed + 300000 + step, directory, f"train_{step:03d}")
                chosen = selected(rows, batch, 2, seed + 400000 + step)
                loss, gradient = student.trace_gradient(chosen, batch, np.full(len(chosen), 1 / len(chosen)))
                for p, g in zip(student.model.parameters(), gradient, strict=True): p.grad = g
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                del gradient
                history["step"] = step + 1
                history["rollout_tokens"] += usage["prompt_tokens"] + usage["completion_tokens"]
                history["training_tokens"] += sum(r["completion_tokens"] for r in chosen)
                history["updates"].append({"step": step + 1, "loss": loss, "path": path, "usage": usage,
                                           "selected_success": float(np.mean([r["correct"] for r in chosen])), "selected": [{k: r[k] for k in ("id", "sample", "correct")} for r in chosen]})
            current_logs = log_probabilities(student, eval_rows, evaluation)
            forecast["measured_importance_gain"] = importance_change(eval_rows, evaluation, current_logs - baseline_logs)
            forecast["weight_overlap"] = weight_overlap(student, initial, direction)
            history["forecasts"].append(forecast)
            eval_rows, record = evaluate(student, evaluation, seed, directory, history["step"])
            history["diagnostic_tokens"] += record["usage"]["prompt_tokens"] + record["usage"]["completion_tokens"]
            history["evaluations"].append(record)
            halt, slopes = stationary(history)
            history["stationarity"] = slopes
            save(root / "progress.json", history)
            print(json.dumps({"seed": seed, "source": source, "step": history["step"], "accuracy": record["accuracy"],
                              "forecast": predicted, "importance_gain": forecast["measured_importance_gain"]["delta"], "slopes": slopes}), flush=True)
            if pending is not None: pending.result()
            pending = checkpoint(student, optimizer, scheduler, history, cache, persistent, root, attempt, executor)
            assert record["accuracy"] >= history["evaluations"][0]["accuracy"] - .15, "Learner clearly degraded; stop and redesign."
            if halt:
                history["stop_reason"] = "stationary"
                break
        if pending is not None: pending.result()
    save(root / "complete.json", {**history, "finished_at_uk": stamp()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--eta", type=float)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--source", choices=("current", "frozen"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.eta)
    else: run(args.seed, args.source, args.resume)
