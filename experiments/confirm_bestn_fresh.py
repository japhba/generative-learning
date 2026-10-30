"""Independent fresh-rollout checks of the frozen FP32 one-step contrasts."""

import argparse
from datetime import datetime
import json
import os
from zoneinfo import ZoneInfo

import numpy as np

from bestn_llm import graded, selected
from bestn_precision import PreciseStudent, exact_matmul
from run_elicitation_pilot import ROOT, digest, save, setup
from train_elicitation import read_rows


OUTPUT = ROOT / "results/bestn_b2k/fresh_confirmation"
SOURCES = ("confirm_bestn_fresh.py", "bestn_precision.py", "bestn_llm.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def stamp(): return datetime.now(ZoneInfo("Europe/London")).isoformat()


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in SOURCES: (OUTPUT / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    save(OUTPUT / "protocol.json", {"created_at_uk": stamp(), "source_sha256": {name: digest(OUTPUT / name) for name in SOURCES},
         "reason": "The original full-precision probe's fresh generations were less favorable than its importance estimates. Increase independent generation samples and add the missing unmodified-model evaluation at the same original fresh seed.",
         "models": "Exactly reconstruct the already-frozen single SGD updates for N1,N2,N4 from the saved training bank, positive target weights and tie seed. No learning-rate or target-rule change.",
         "regimes": ["low", "high"], "n": [0, 1, 2, 4], "fresh_seed": 859001, "samples_per_prompt": 16,
         "paired_original_baseline": "Only the unchanged model also runs4samples/prompt at855001, the original fresh-evaluation seed.",
         "scope": "Independent rollout draws on the same64probe questions; this is a sampling confirmation, not a fresh-task training replication.",
         "uncertainty": "Compare models paired by whole problem within difficulty; bootstrap64questions. Retain the original8-sample baseline and all original outcomes.",
         "compute": "One short job per model/difficulty; expected under10minutes. Updates are deterministically reconstructed and no long trajectory is trained. No paid calls."})


def run(regime, n):
    import torch
    setup()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    exact_matmul()
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    source = ROOT / f"results/bestn_b2k/success_probe/{regime}"
    train, evaluation = [json.loads((source / f"{split}.json").read_text()) for split in ("train", "evaluation")]
    directory = OUTPUT / f"{regime}_n{n}"
    directory.mkdir()
    student = PreciseStudent()
    if n:
        candidates = graded(read_rows(source / "run/training_bank.jsonl"), train)
        targets = selected(candidates, train, n, 854001 + n)
        _, gradient = student.trace_gradient(targets, train, np.full(len(targets), 1 / len(targets)))
        eta = json.loads((source / "run/forecast.json").read_text())["base_eta"] * 16 / n
        with torch.no_grad():
            for parameter, direction in zip(student.model.parameters(), gradient, strict=True): parameter.add_(direction, alpha=-eta)
        del gradient
    else: eta = 0.
    save(directory / "model.json", {"created_at_uk": stamp(), "regime": regime, "n": n, "effective_eta": eta, "job_id": os.environ["SLURM_JOB_ID"]})
    conditions = [("independent", 16, 859001)]
    if n == 0: conditions.append(("original_fresh_seed", 4, 855001))
    records = []
    for name, samples, seed in conditions:
        path = directory / f"{name}.jsonl"
        usage = student.generate(evaluation, "reason", samples, 1., 1024, 32, seed, path)
        rows = graded(read_rows(path), evaluation)
        record = {"condition": name, "samples": samples, "seed": seed, "success": float(np.mean([r["correct"] for r in rows])),
                  "completion": float(np.mean([r["ended"] for r in rows])), "usage": usage,
                  "per_prompt": {p["id"]: float(np.mean([r["correct"] for r in rows if r["id"] == p["id"]])) for p in evaluation}}
        records.append(record)
        save(directory / f"{name}.json", record)
        print(json.dumps(record), flush=True)
    save(directory / "complete.json", {"finished_at_uk": stamp(), "regime": regime, "n": n, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--regime", choices=("low", "high"))
    parser.add_argument("--n", type=int, choices=(0, 1, 2, 4))
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    else: run(args.regime, args.n)
