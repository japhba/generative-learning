"""Diagnostic: separate numerical rounding from the small-step approximation."""

from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from bestn_llm import BestNStudent, dot, graded, selected
from bestn_precision import PreciseStudent, exact_matmul
from run_elicitation_pilot import ROOT, digest, save, setup
from train_elicitation import read_rows


def stamp(): return datetime.now(ZoneInfo("Europe/London")).isoformat()


def run():
    import torch
    setup()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    exact_matmul()
    output = ROOT / "results/bestn_b2k/precision_audit"
    output.mkdir()
    for name in ("audit_bestn_precision.py", "bestn_precision.py", "bestn_llm.py"):
        (output / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    source = ROOT / "results/bestn_b2k/probe/high"
    train = json.loads((source / "train.json").read_text())
    evaluation = json.loads((source / "evaluation.json").read_text())
    train = [p for i, p in enumerate(train) if i < 8]
    evaluation = [p for i, p in enumerate(evaluation) if i < 8]
    candidates = [r for r in read_rows(source / "run/training_bank.jsonl") if r["id"] in {p["id"] for p in train}]
    targets = selected(graded(candidates, train), train, 2, 834001)
    save(output / "protocol.json", {"created_at_uk": stamp(), "purpose": "Post-hoc numerical diagnosis; reused calibration traces, no claim of independent replication.",
         "prediction": "Use the actual update gradient, so any mismatch at sufficiently small steps is numerical or implementation error, not forecast-bank sampling noise.",
         "learning_rates": [1e-6, 1e-5, 1e-4, 1e-3, .003], "modes": ["bf16_forward", "fp32_forward"],
         "tf32": False, "training_traces": len(targets), "evaluation_prompts": len(evaluation)})
    learner = PreciseStudent()
    initial = [p.detach().clone() for p in learner.model.parameters()]
    output_rows = []
    for mode, cls in (("bf16_forward", BestNStudent), ("fp32_forward", PreciseStudent)):
        student = object.__new__(cls)
        student.model, student.tokenizer, student.config = learner.model, learner.tokenizer, learner.config
        baseline, test_gradient = student.objective(evaluation, student.reference_tokens(evaluation), np.full(len(evaluation), 1 / len(evaluation)), True)
        _, update_gradient = student.trace_gradient(targets, train, np.full(len(targets), 1 / len(targets)))
        overlap = dot(test_gradient, update_gradient)
        save(output / f"{mode}_forecast.json", {"written_before_updates_at_uk": stamp(), "baseline": baseline, "overlap": overlap,
             "test_gradient_norm_squared": dot(test_gradient, test_gradient), "update_gradient_norm_squared": dot(update_gradient, update_gradient)})
        for eta in (1e-6, 1e-5, 1e-4, 1e-3, .003):
            with torch.no_grad():
                for p, original, g in zip(student.model.parameters(), initial, update_gradient, strict=True): p.copy_(original).add_(g, alpha=-eta)
            current = student.objective(evaluation, student.reference_tokens(evaluation), np.full(len(evaluation), 1 / len(evaluation)), False)
            row = {"mode": mode, "eta": eta, "predicted": eta * overlap, "actual": baseline - current}
            output_rows.append(row)
            save(output / "measurements.json", output_rows)
            print(json.dumps(row), flush=True)
            with torch.no_grad():
                for p, original in zip(student.model.parameters(), initial, strict=True): p.copy_(original)
        del test_gradient, update_gradient
    save(output / "complete.json", {"finished_at_uk": stamp(), "rows": output_rows})


if __name__ == "__main__": run()
