"""Calibrate task difficulty before selecting a best-of-N training protocol."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from random import Random
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

from elicitation_task import identifier, question, states
from elicitation_answer import parse_answer
from run_elicitation_pilot import ROOT, MODELS, digest, save, setup
from train_elicitation import Student


OUTPUT = ROOT / "results/bestn_b2k"
SOURCES = ("calibrate_bestn.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def stamp(): return datetime.now(ZoneInfo("Europe/London")).isoformat()


def problems(seed, per_cell):
    rng, rows = Random(seed), []
    for depth in (4, 12):
        for bound in (29, 9999):
            cell = f"depth{depth}_bound{bound}"
            for index in range(per_cell):
                operations, multiplications = [], 0
                for _ in range(depth):
                    operation = rng.choice(("add", "subtract", "multiply") if multiplications < 3 else ("add", "subtract"))
                    value = rng.randint(2, 5) if operation == "multiply" else rng.randint(2, bound)
                    operations.append([operation, value])
                    multiplications += operation == "multiply"
                p = {"family": "arithmetic", "depth": depth, "start": rng.randint(2, 39), "operations": operations}
                rows.append({**p, "id": identifier(p), "cell": cell, "cell_index": index})
    assert len({question(row) for row in rows}) == len(rows)
    return rows


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = OUTPUT / "calibration_source"
    source.mkdir()
    rows = problems(81001, 8)
    save(OUTPUT / "calibration.json", rows)
    for name in SOURCES: (source / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    save(OUTPUT / "calibration_protocol.json", {"created_at_uk": stamp(), "batch_id": "b2k", "model": MODELS["general"],
         "task_seed": 81001, "problems_per_cell": 8, "cells": sorted({row["cell"] for row in rows}), "samples_per_problem": 8,
         "sampling_temperature": 1., "top_p": 1., "top_k": 0, "max_new_tokens": 1024, "batch_size": 32,
         "source_sha256": {name: digest(source / name) for name in SOURCES}, "data_sha256": digest(OUTPUT / "calibration.json"),
         "purpose": "Calibrate completion and success rates before freezing any training comparison. No updates or choice of a reported exponent.",
         "desired_regimes": "Prefer a cell with success <=0.30 and one >=0.55, both with >=0.95 completion. Rates are estimated by cell and prompt; future predictions must account for heterogeneity and gradient overlap.",
         "feedback": "Offline exact arithmetic correctness only; all full sampled responses, including failures and truncations, are retained.",
         "resume": "Completed cells are reused; an interrupted cell is regenerated into a separate attempt file without deleting prior outputs.",
         "cost": "One institutional gpu_lowp H100. No paid API or rented compute."})


def run(resume):
    import torch
    setup()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    protocol = json.loads((OUTPUT / "calibration_protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    rows = json.loads((OUTPUT / "calibration.json").read_text())
    directory = OUTPUT / "calibration"
    directory.mkdir(exist_ok=resume)
    student = Student()
    save(directory / "environment.json", {"started_at_uk": stamp(), "model": MODELS["general"], "device": torch.cuda.get_device_name(),
         "job_id": os.environ["SLURM_JOB_ID"], "torch": torch.__version__, "parameter_count": sum(p.numel() for p in student.model.parameters()),
         "cuda_visible_devices_inherited": os.environ["CUDA_VISIBLE_DEVICES"], "hf_authenticated": True, "hf_xet_high_performance": os.environ["HF_XET_HIGH_PERFORMANCE"]})
    for index, cell in enumerate(tqdm(protocol["cells"], desc="Difficulty cells")):
        marker = directory / f"{cell}.complete.json"
        if resume and marker.exists(): continue
        cohort = [row for row in rows if row["cell"] == cell]
        attempt = len(list(directory.glob(f"{cell}.attempt*.jsonl")))
        destination = directory / f"{cell}.attempt{attempt}.jsonl"
        usage = student.generate(cohort, "reason", 8, 1., 1024, 32, 181001 + index, destination)
        predictions = [json.loads(line) for line in destination.read_text().splitlines()]
        truth = {p["id"]: states(p)[-1] for p in cohort}
        for row in predictions:
            row["parsed_answer"] = parse_answer(row["text"])
            row["correct"] = row["ended"] and row["parsed_answer"] == truth[row["id"]]
        result = {"cell": cell, "completed_at_uk": stamp(), "path": str(destination), "sha256": digest(destination), "usage": usage,
                  "sample_success": sum(row["correct"] for row in predictions)/len(predictions),
                  "completion_rate": sum(row["ended"] for row in predictions)/len(predictions),
                  "per_prompt": {p["id"]: {"successes": sum(row["correct"] for row in predictions if row["id"] == p["id"]), "samples": 8, "truth": truth[p["id"]]} for p in cohort}}
        save(marker, result)
        print(json.dumps(result), flush=True)
    save(directory / "complete.json", {"completed_at_uk": stamp(), "cells": [json.loads((directory / f"{cell}.complete.json").read_text()) for cell in protocol["cells"]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    else: run(args.resume)
