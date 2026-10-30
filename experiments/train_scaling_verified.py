"""Resumable accepted-mean self-training with current or frozen proposals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from tqdm.auto import tqdm

from bestn_precision import exact_matmul
from run_elicitation_pilot import MODELS, ROOT, digest, save, setup
from scaling_choice_model import ChoiceStudent
from scaling_digit_task import SYMBOLS, encode, problems
from train_bestn import proposal, stamp


CELL = {"name": "count_4_max_2", "family": "count", "length": 4, "max_count": 2}
RATES = (1.4e-6, 2.8e-6, 5.6e-6)
SOURCES = ("train_scaling_verified.py", "scaling_choice_model.py", "scaling_choice_task.py", "scaling_digit_task.py", "train_bestn.py", "bestn_precision.py", "bestn_llm.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py", "VERIFIED_SCALING_PROTOCOL.md")


def make_cohort(seed, per_class, excluded):
    groups = {str(k): {} for k in range(3)}
    for row in problems(seed, 200000, CELL):
        group = groups[row["label"]]
        if row["id"] not in excluded and len(group) < per_class: group[row["id"]] = row
    assert all(len(group) == per_class for group in groups.values())
    return [row for group in groups.values() for row in group.values()]


def make_stream(seed, count, excluded):
    rng = random.Random(seed)
    groups = {str(k): [] for k in range(3)}
    for row in problems(seed + 1000000, count * 4, CELL):
        if row["id"] not in excluded: groups[row["label"]].append(row)
    positions = {key: 0 for key in groups}
    rows = []
    for _ in range(count):
        key = str(rng.randrange(3)); rows.append(groups[key][positions[key]]); positions[key] += 1
    return rows


def prepare(output, stage, seeds, steps, rates):
    output.mkdir(parents=True, exist_ok=False); (output / "source").mkdir()
    for name in SOURCES: (output / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    split_seed = 195501 if stage == "pilot" else 196501
    monitor = make_cohort(split_seed, 341, set())
    test = make_cohort(split_seed + 1, 1365, {r["id"] for r in monitor})
    excluded = {r["id"] for r in monitor + test}
    assert len(excluded) == 5118
    save(output / "monitor.json", monitor); save(output / "test.json", test)
    cells = []
    for seed in tqdm(seeds, desc="Prepare independent training streams"):
        directory = output / str(seed); directory.mkdir()
        rows = make_stream(seed, steps * 32, excluded); save(directory / "train.json", rows)
        assert not {r["id"] for r in rows} & excluded
        for row in rows + monitor + test: assert int(row["label"]) == row["values"].count(row["target"])
        for source in ("current", "frozen"):
            for rate in rates[source]: cells.append({"seed": seed, "source": source, "eta": rate, "name": f"{source}_{rate:.1e}"})
    checks = sorted({0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, steps} & set(range(steps + 1)))
    save(output / "protocol.json", {"created_at_uk": stamp(), "batch_id": "r5s", "stage": stage, "model": MODELS["general"], "cell": CELL,
         "seeds": seeds, "steps": steps, "batch": 32, "checks": checks, "cells": cells,
         "checkpoint_checks": sorted({0, 128, 512, 1024, 2048, steps} & set(range(steps + 1))),
         "source_sha256": {name: digest(output / "source" / name) for name in SOURCES},
         "monitor_count": len(monitor), "test_count": len(test), "normalization": "mean over accepted rows; zero loss on rejected rows",
         "training": "Two unrestricted T1 first-token samples; choose uniformly among correct samples; reject all-failure groups; all weights FP32 plain SGD; same batch32 forward/backward computation for both sources.",
         "data": "Class-uniform IID stream conditional on excluding both exact-input held-out cohorts. Incidental training repeats retained. Final-test labels/inputs never enter learning, rate selection, or stopping.",
         "selection": "Pilot: lowest trapezoidal mean log monitoring-error over log steps128..1024 independently for each source among complete trajectories. All rates retained.",
         "analysis": "Matched candidate error ratios at512,1024,2048,4096 and crossing budgets for errors .1,.05,.02,.01,.005,.002. Paired seed/question bootstrap. Report accepted-example axis, rate sensitivity, greedy error and local exponents.",
         "stationarity": "From64 updates, latest4 positive checks: stop if either absolute log-error or log-accepted-NLL slope < .02; abort degradation>.15 or valid mass<.95.",
         "resources": "Institutional gpu_lowp; no paid API or rental; full resume and async durable checkpoint moves."})
    print(json.dumps({"output": str(output), "cells": cells, "steps": steps}), flush=True)


def initialize(output):
    import torch
    setup(); os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"; assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4); exact_matmul()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    student = ChoiceStudent(); student.answer_symbols = SYMBOLS
    assert all(p.dtype == torch.float32 for p in student.model.parameters())
    return protocol, student


def choose(draws, target):
    import torch
    rewards = draws == target[:, None]
    index = torch.randint(2, (len(draws),), device=draws.device)
    single = rewards.sum(-1) == 1
    index[single] = rewards[single].long().argmax(-1)
    return draws.gather(1, index[:, None]).squeeze(1), rewards.any(-1), index, rewards


def update(student, initial, source, batch, seed, optimizer):
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    target = torch.tensor([r["target_id"] for r in batch], device="cuda")
    with proposal(student, initial, source), torch.no_grad():
        policy = student.logits(batch).softmax(-1)
        draws = torch.multinomial(policy, 2, replacement=True)
        chosen, accepted, index, rewards = choose(draws, target)
        source_success = float(policy.gather(1, target[:, None]).mean())
    count = int(accepted.sum())
    loss_value = 0.
    optimizer.zero_grad(set_to_none=True)
    if count:
        for attempt, microbatch in enumerate((16, 8, 4, 2, 1)):
            optimizer.zero_grad(set_to_none=True); loss_value = 0.
            try:
                for offset in range(0, len(batch), microbatch):
                    losses = torch.nn.functional.cross_entropy(student.logits(batch[offset:offset + microbatch]), chosen[offset:offset + microbatch], reduction="none")
                    loss = (losses * accepted[offset:offset + microbatch]).sum() / count
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
                "candidate_texts": [student.tokenizer.decode([token]) for token in tokens], "correct": correct,
                "chosen_index": winner, "accepted": keep}
               for row, tokens, correct, winner, keep in zip(batch, draws.cpu().tolist(), rewards.cpu().tolist(), index.cpu().tolist(), accepted.cpu().tolist(), strict=True)]
    return {"loss": loss_value, "accepted": count, "source_success": source_success, "source_sampled_success": float(rewards.float().mean()), "records": records}


def stopping(history):
    if history["step"] < 64: return False, {}
    records = history["evaluations"][-4:]
    x = np.log([r["step"] for r in records])
    slopes = {"log_error_slope": float(np.polyfit(x, np.log([r["error"] for r in records]), 1)[0]),
              "log_loss_slope": float(np.polyfit(x, np.log([r["block_loss"] for r in records]), 1)[0])}
    return any(abs(value) < .02 for value in slopes.values()), slopes


def archive_move(local, remote, metadata):
    remote.parent.mkdir(parents=True, exist_ok=True)
    with local.open("rb") as stream: before = hashlib.file_digest(stream, "sha256").hexdigest()
    temporary = remote.with_suffix(".moving")
    shutil.move(local, temporary)
    with temporary.open("rb") as stream: after = hashlib.file_digest(stream, "sha256").hexdigest()
    assert before == after
    temporary.replace(remote); local.symlink_to(remote)
    save(metadata, {"local": str(local), "persistent": str(remote), "sha256": before, "bytes": remote.stat().st_size, "moved_at_uk": stamp()})


def checkpoint(student, optimizer, scheduler, history, local, persistent, root, attempt, executor):
    import torch
    name = f"step{history['step']:04d}_attempt{attempt}.pt"
    local_path, persistent_path = local / name, persistent / name
    state = {"model": {key: value.detach().cpu() for key, value in student.model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "saved_at_uk": stamp()}
    temporary = local_path.with_suffix(".partial"); torch.save(state, temporary); temporary.replace(local_path)
    save(root / f"{name}.local.json", {"step": history["step"], "attempt": attempt, "local": str(local_path), "persistent": str(persistent_path)})
    return executor.submit(archive_move, local_path, persistent_path, root / f"{name}.archive.json")


def run(output, cell_index, resume):
    import torch
    protocol, student = initialize(output)
    cell = protocol["cells"][cell_index]; seed, source, eta = cell["seed"], cell["source"], cell["eta"]
    cohort = output / str(seed)
    rows = encode(student, json.loads((cohort / "train.json").read_text()))
    monitor = encode(student, json.loads((output / "monitor.json").read_text()))
    root = cohort / cell["name"]; root.mkdir(exist_ok=resume)
    relative = output.relative_to(ROOT / "results") / str(seed) / cell["name"]
    local = setup() / relative; local.mkdir(parents=True, exist_ok=resume)
    persistent = Path(os.environ["CACHE_DIR"]) / "generative-learning" / relative
    attempt = len(list(root.glob("attempt*"))); directory = root / f"attempt{attempt}"; directory.mkdir()
    initial = [p.detach().clone() for p in student.model.parameters()]
    optimizer = torch.optim.SGD(student.model.parameters(), lr=eta)
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
        history = {"step": 0, **cell, "started_at_uk": stamp(), "updates": [], "evaluations": [{"step": 0, **student.evaluate(monitor)}],
                   "candidates": 0, "presentations": 0, "accepted": 0, "unique_questions": 0, "accepted_unique_questions": 0,
                   "proposal_prompt_tokens": 0, "training_tokens": 0, "accepted_training_tokens": 0,
                   "diagnostic_prompt_tokens": sum(len(r["input_ids"]) for r in monitor)}
    save(directory / "environment.json", {"started_at_uk": stamp(), "allocation": os.environ["SLURM_JOB_ID"], "device": torch.cuda.get_device_name(),
         "inherited_gpu_count": torch.cuda.device_count(), "torch": torch.__version__, "parameter_count": sum(p.numel() for p in student.model.parameters()), "resumed_step": history["step"]})
    halt, slopes = stopping(history)
    if halt:
        save(root / "complete.json", {**history, "stationarity": slopes, "stop_reason": "stationary_at_resume", "finished_at_uk": stamp()}); return
    seen = {r["id"] for r in rows[:history["step"] * 32]}
    accepted_seen = set(history["accepted_ids"]) if resume else set()
    pending = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        if history["step"] == 0: pending = checkpoint(student, optimizer, scheduler, {**history, "accepted_ids": []}, local, persistent, root, attempt, executor)
        for step in tqdm(range(history["step"], protocol["steps"]), desc=f"r5s {seed} {cell['name']}"):
            batch = rows[step * 32:(step + 1) * 32]; start = time.monotonic()
            record = update(student, initial, source, batch, seed + 400000 + step, optimizer)
            scheduler.step(); history["step"] = step + 1
            path = directory / f"train_{step:04d}.json"; save(path, {"step": step + 1, **record})
            seen.update(r["id"] for r in batch); accepted_seen.update(r["id"] for r in record["records"] if r["accepted"])
            tokens = sum(len(r["input_ids"]) for r in batch)
            history["candidates"] += 64; history["presentations"] += 32; history["accepted"] += record["accepted"]
            history["unique_questions"] = len(seen); history["accepted_unique_questions"] = len(accepted_seen)
            history["proposal_prompt_tokens"] += tokens; history["training_tokens"] += tokens + 32
            history["accepted_training_tokens"] += sum(len(r["input_ids"]) + 1 for r, selected in zip(batch, record["records"], strict=True) if selected["accepted"])
            history["updates"].append({"step": step + 1, **{k: v for k, v in record.items() if k != "records"}, "path": str(path), "seconds": time.monotonic() - start})
            if history["step"] in protocol["checks"]:
                previous = history["evaluations"][-1]["step"]
                evaluation = {"step": history["step"], **student.evaluate(monitor), "accepted": history["accepted"], "training_tokens": history["training_tokens"]}
                block = [r for r in history["updates"] if previous < r["step"] <= history["step"]]
                evaluation["block_loss"] = sum(r["loss"] * r["accepted"] for r in block) / sum(r["accepted"] for r in block)
                evaluation["acceptance"] = sum(r["accepted"] for r in block) / (32 * len(block))
                history["evaluations"].append(evaluation); history["diagnostic_prompt_tokens"] += sum(len(r["input_ids"]) for r in monitor)
                halt, slopes = stopping(history); history["stationarity"] = slopes; save(root / "progress.json", history)
                print(json.dumps({"step": history["step"], "accuracy": evaluation["accuracy"], "greedy": evaluation["greedy_accuracy"], "loss": evaluation["block_loss"], "acceptance": evaluation["acceptance"], "slopes": slopes}), flush=True)
                assert evaluation["accuracy"] >= history["evaluations"][0]["accuracy"] - .15 and evaluation["valid_mass"] >= .95, "Learner degraded or lost format; redesign required."
                if history["step"] in protocol["checkpoint_checks"] or halt:
                    if pending is not None: pending.result()
                    pending = checkpoint(student, optimizer, scheduler, {**history, "accepted_ids": sorted(accepted_seen)}, local, persistent, root, attempt, executor)
                if halt: history["stop_reason"] = "stationary"; break
        if pending is not None: pending.result()
    save(root / "complete.json", {**history, "finished_at_uk": stamp()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/scaling_r5s/pilot")
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="pilot"); parser.add_argument("--seeds", type=int, nargs="+", default=[99501])
    parser.add_argument("--steps", type=int, default=1024); parser.add_argument("--cell-index", type=int); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); args.output = args.output.resolve()
    if args.command == "prepare": prepare(args.output, args.stage, args.seeds, args.steps, {source: RATES for source in ("current", "frozen")})
    else: run(args.output, args.cell_index, args.resume)
