"""Untouched-task evaluation at a fixed, shared training checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from bestn_llm import cohort
from bestn_precision import PreciseStudent, exact_matmul
from run_elicitation_pilot import ROOT, digest, save, setup
from train_bestn import generate, stamp


OUTPUT = ROOT / "results/bestn_b2k/final_evaluation"
SOURCES = ("confirm_bestn_training.py", "train_bestn.py", "bestn_precision.py", "bestn_llm.py", "probe_bestn_success.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def prepare():
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "source").mkdir()
    for name in SOURCES: (OUTPUT / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    problems = cohort(890001, "low", 128)
    previous = set()
    for seed in (86001, 86002, 86003):
        for split in ("train", "evaluation"):
            previous.update(p["id"] for p in json.loads((ROOT / f"results/bestn_b2k/training/{seed}/{split}.json").read_text()))
    for regime in ("low", "high"):
        for split in ("train", "evaluation"):
            previous.update(p["id"] for p in json.loads((ROOT / f"results/bestn_b2k/success_probe/{regime}/{split}.json").read_text()))
    assert not previous & {p["id"] for p in problems}
    save(OUTPUT / "problems.json", problems)
    conditions = [{"index": 0, "seed": 0, "source": "baseline", "step": 0}] + [
        {"index": index, "seed": seed, "source": source, "step": 16}
        for index, (seed, source) in enumerate(((s, p) for s in (86001, 86002, 86003) for p in ("current", "frozen")), 1)]
    save(OUTPUT / "protocol.json", {"created_at_uk": stamp(), "batch_id": "b2k", "conditions": conditions,
         "source_sha256": {name: digest(OUTPUT / "source" / name) for name in SOURCES}, "problems_sha256": digest(OUTPUT / "problems.json"),
         "task_seed": 890001, "generation_seed": 890101, "tasks": 128, "samples_per_task": 8,
         "checkpoint": "Exactly update16:512 training candidates,256 presentations. This is the first checkpoint eligible for the fixed stationarity rule, hence available in all six planned runs. No best-checkpoint selection.",
         "why_now": "The recurring evaluation set was used to stop training and to estimate diagnostic gradient overlaps. A separate untouched cohort now tests generalization. Chosen after observing the benefit on prior cohorts, before generating any output on this set; not an original preregistered endpoint.",
         "assessment": "Primary is mean current-minus-frozen accuracy, paired by question and training seed. Report all seeds, whole-question and hierarchical seed/question bootstrap intervals. Baseline and equation-mode diagnostics are secondary. No task, rate, checkpoint, or model selection follows this evaluation.",
         "cost": "Inference only on institutional gpu_lowp. No weight updates and no paid calls."})


def run(index):
    import torch
    setup()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    exact_matmul()
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    assert protocol["problems_sha256"] == digest(OUTPUT / "problems.json")
    condition = next(r for r in protocol["conditions"] if r["index"] == index)
    directory = OUTPUT / f"{condition['seed']}_{condition['source']}"
    directory.mkdir()
    student = PreciseStudent()
    if condition["step"]:
        root = ROOT / f"results/bestn_b2k/training/{condition['seed']}/{condition['source']}"
        marker_path = sorted(root.glob("step016_attempt*.pt.archive.json"))[-1]
        marker = json.loads(marker_path.read_text())
        checkpoint = Path(marker["local"])
        if not checkpoint.exists(): checkpoint = Path(marker["persistent"])
        with checkpoint.open("rb") as stream: assert hashlib.file_digest(stream, "sha256").hexdigest() == marker["sha256"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["history"]["step"] == 16
        student.model.load_state_dict(state["model"], strict=True)
        del state
        checkpoint_provenance = {"path": str(checkpoint), "sha256": marker["sha256"]}
    else: checkpoint_provenance = None
    save(directory / "model.json", {**condition, "started_at_uk": stamp(), "checkpoint": checkpoint_provenance, "job_id": os.environ["SLURM_JOB_ID"]})
    problems = json.loads((OUTPUT / "problems.json").read_text())
    rows, usage, path = generate(student, problems, 8, 890101, directory, "evaluation")
    assert len(rows) == 1024 and np.mean([r["ended"] for r in rows]) >= .9
    per_prompt = {}
    for problem in problems:
        group = [r for r in rows if r["id"] == problem["id"]]
        assert len(group) == 8
        per_prompt[problem["id"]] = {"accuracy": float(np.mean([r["correct"] for r in group])),
            "equation_count": sum("=" in r["text"] for r in group), "equation_correct": sum(r["correct"] and "=" in r["text"] for r in group),
            "other_correct": sum(r["correct"] and "=" not in r["text"] for r in group)}
    result = {**condition, "finished_at_uk": stamp(), "accuracy": float(np.mean([r["correct"] for r in rows])),
              "equation_probability": float(np.mean(["=" in r["text"] for r in rows])), "usage": usage, "path": path, "per_prompt": per_prompt}
    save(directory / "complete.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_prompt"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    else: run(args.index)
