"""Calibration-only check of prospective gradient-overlap learning predictions."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from bestn_llm import BestNStudent, cohort, dot, forecast_weights, graded, selected
from run_elicitation_pilot import MODELS, ROOT, digest, save, setup
from train_elicitation import read_rows


OUTPUT = ROOT / "results/bestn_b2k/probe"
SOURCES = ("probe_bestn.py", "bestn_llm.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def stamp(): return datetime.now(ZoneInfo("Europe/London")).isoformat()


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "source").mkdir()
    for name in SOURCES: (OUTPUT / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    for index, regime in enumerate(("low", "high")):
        directory = OUTPUT / regime
        directory.mkdir()
        train, evaluation = cohort(830001 + index, regime, 32), cohort(830101 + index, regime, 64)
        assert not {p["id"] for p in train} & {p["id"] for p in evaluation}
        save(directory / "train.json", train)
        save(directory / "evaluation.json", evaluation)
    save(OUTPUT / "protocol.json", {"created_at_uk": stamp(), "batch_id": "b2k", "model": MODELS["general"],
         "source_sha256": {name: digest(OUTPUT / "source" / name) for name in SOURCES},
         "purpose": "Calibration of a prospective mechanistic learning forecast; no confirmation claims from these tasks.",
         "train_unique": 32, "eval_unique": 64, "forecast_samples_per_prompt": 16, "independent_training_samples_per_prompt": 16,
         "evaluation_samples_per_prompt": 8, "temperature": 1, "maximum": 1024, "candidates": [1, 2, 4],
         "metric": "Mean held-out canonical-correct-trace NLL divided by fixed128; actual fresh-rollout correctness is measured separately.",
         "forecast": "Before any update, dot the exact held-out reference-loss gradient with an unbiased best-N-minus-ordinary score estimate from independent proposal bank A. N1 predicts zero expected drift. Bank B supplies actual positive-weight training targets. Uniform tie selection includes all-failure groups.",
         "optimizer": "One plain SGD update of the average selected trajectory loss, scaled by16/N to represent equal candidate budgets at first order. No clipping, decay, length normalization, Adam, or temperature change.",
         "step_size_rule": "Choose base eta so the largest positive forecast NLL decrease across N2/N4 is .02. Also test eta/4. This is calibration-only; freeze a fixed rate before later confirmation.",
         "test_separation": "Future learning-curve confirmation uses disjoint cohorts and RNG seeds. Selection uses exact correctness; canonical evaluation traces never enter updates.",
         "runtime": "Expected below15 minutes per regime on one H100. Each completed measurement is durable; no long training trajectory in this probe."})


def evaluate(student, problems, directory, name):
    tokens = student.reference_tokens(problems)
    loss = student.objective(problems, tokens, np.full(len(problems), 1 / len(problems)), False)
    path = directory / f"{name}.jsonl"
    usage = student.generate(problems, "reason", 8, 1., 1024, 32, 831001, path)
    rows = graded(read_rows(path), problems)
    return {"reference_nll_per128": loss, "accuracy": np.mean([r["correct"] for r in rows]).item(),
            "completion": np.mean([r["ended"] for r in rows]).item(), "usage": usage, "path": str(path),
            "per_prompt": {p["id"]: sum(r["correct"] for r in rows if r["id"] == p["id"]) / 8 for p in problems}}


def run(regime):
    import torch
    setup()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    root = OUTPUT / regime
    directory = root / "run"
    directory.mkdir()
    train = json.loads((root / "train.json").read_text())
    evaluation = json.loads((root / "evaluation.json").read_text())
    student = BestNStudent()
    save(directory / "environment.json", {"started_at_uk": stamp(), "job_id": os.environ["SLURM_JOB_ID"],
         "device": torch.cuda.get_device_name(), "torch": torch.__version__, "parameters": sum(p.numel() for p in student.model.parameters()),
         "cuda_visible_devices_inherited": os.environ["CUDA_VISIBLE_DEVICES"]})
    save(directory / "objective_check.json", student.verify_objective(train[0]))
    baseline = evaluate(student, evaluation, directory, "baseline")
    save(directory / "baseline.json", baseline)
    initial = [p.detach().clone() for p in student.model.parameters()]
    _, evaluation_gradient = student.objective(evaluation, student.reference_tokens(evaluation), np.full(len(evaluation), 1 / len(evaluation)), True)
    banks = []
    for index, name in enumerate(("forecast_bank", "training_bank")):
        path = directory / f"{name}.jsonl"
        student.generate(train, "reason", 16, 1., 1024, 32, 832001 + index, path)
        banks.append(graded(read_rows(path), train))
    predictions = {1: 0.}
    for n in (2, 4):
        _, gradient = student.trace_gradient(banks[0], train, forecast_weights(banks[0], train, n))
        predictions[n] = dot(evaluation_gradient, gradient) * 16 / n
        del gradient
    assert max(predictions.values()) > 0, "Expected selected-score direction does not help held-out correct-trace likelihood; redesign required."
    eta = .02 / max(predictions.values())
    forecast = {"written_before_updates_at_uk": stamp(), "reference_baseline": baseline["reference_nll_per128"],
                "loss_decrease_per_eta": predictions, "base_eta": eta, "reference_gradient_squared_norm": dot(evaluation_gradient, evaluation_gradient),
                "bank_sample_success": [float(np.mean([r["correct"] for r in bank])) for bank in banks]}
    save(directory / "forecast.json", forecast)
    print(json.dumps({"regime": regime, "forecast": forecast}), flush=True)
    measurements = []
    for n in (1, 2, 4):
        targets = selected(banks[1], train, n, 833001 + n)
        _, gradient = student.trace_gradient(targets, train, np.full(len(targets), 1 / len(targets)))
        realized_overlap = dot(evaluation_gradient, gradient) * 16 / n
        for fraction in (.25, 1.):
            with torch.no_grad():
                for parameter, original, direction in zip(student.model.parameters(), initial, gradient, strict=True):
                    parameter.copy_(original).add_(direction, alpha=-eta * fraction * 16 / n)
            result = evaluate(student, evaluation, directory, f"n{n}_fraction{fraction:g}")
            record = {"n": n, "fraction": fraction, "effective_sgd_eta": eta * fraction * 16 / n,
                      "forecast_loss_decrease": eta * fraction * predictions[n], "realized_direction_loss_decrease": eta * fraction * realized_overlap,
                      "actual_loss_decrease": baseline["reference_nll_per128"] - result["reference_nll_per128"],
                      "selected_success": float(np.mean([r["correct"] for r in targets])), "result": result, "completed_at_uk": stamp()}
            measurements.append(record)
            save(directory / "measurements.json", measurements)
            print(json.dumps({key: value for key, value in record.items() if key != "result"}), flush=True)
            with torch.no_grad():
                for parameter, original in zip(student.model.parameters(), initial, strict=True): parameter.copy_(original)
        del gradient
    save(directory / "complete.json", {"finished_at_uk": stamp(), "forecast": forecast, "measurements": measurements})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--regime", choices=("low", "high"))
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    else: run(args.regime)
