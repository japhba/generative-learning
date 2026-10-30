"""Freeze whole-curve forecasts using only initial success and gradient overlap."""

import json
from pathlib import Path

import numpy as np

from bestn_theory import sigmoid
from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


def logit(p): return np.log(p)-np.log1p(-p)


def curves(p_train, p_eval, coefficient, maximum=256):
    result = {}
    for source in ("current", "frozen"):
        z, rows = 0., []
        for step in range(maximum+1):
            pt, pe = sigmoid(logit(p_train)+z), sigmoid(logit(p_eval)+z)
            rows.append({"step": step, "candidates": 64*step, "accuracy": float(pe), "train_success": float(pt), "z": float(z)})
            target = 1-(1-pt)**2 if source == "current" else 1-(1-p_train)**2
            z += coefficient*(target-pt)
        result[source] = rows
    return result


def main():
    root = ROOT/"results/scaling_s4g/arithmetic/pilot/95001"
    forecast_path = root/"current/attempt0/forecast_0000.json"
    forecast = json.loads(forecast_path.read_text())
    from bestn_llm import cohort, graded
    from train_elicitation import read_rows
    tasks = cohort(595001, "high", 64)
    rows = graded(read_rows(Path(forecast["path"])), tasks)
    pt = float(np.mean([r["correct"] for r in rows]))
    pe = forecast["evaluation_accuracy"]
    coefficient = forecast["predicted_one_step_gain"]/(pt*(1-pt)*pe*(1-pe))
    cutoffs = {}
    for source in ("current", "frozen"):
        completed = []
        for path in (root/source).glob("attempt*/eval_*_try*.usage.json"):
            completed.append(int(path.name.split("_")[1]))
        cutoffs[source] = max(completed)
    output = ROOT/"results/scaling_s4g/gate_forecast"
    output.mkdir(parents=True, exist_ok=False)
    result = {"created_at_uk": stamp(), "cutoffs_completed_evaluation": cutoffs,
              "parameters": {"train_success0": pt, "evaluation_success0": pe, "coefficient": coefficient},
              "initial_forecast_sha256": digest(forecast_path), "source_sha256": digest(ROOT/"experiments/freeze_scaling_gate.py"),
              "model": "One normalized Bernoulli success gate, with different initial train/test logits and a shared logit displacement. Calibrate its fixed weight-space metric only from the pre-update gradient overlap; no learning-curve rate is fitted.",
              "assumption": "Success-logit gradient direction and metric stay approximately constant, and a correct policy is attainable. Full-trajectory conditional distributions are collapsed in this approximation. A theoretical ceiling of1 is a testable modeling assumption, not a measured LLM ceiling.",
              "prospective_scope": "Only checkpoints after the recorded cutoff are prospective. Earlier values are not independent evidence even though the numeric calibration uses only pre-update measurements.",
              "population": curves(pt, pe, coefficient)}
    save(output/"prospective.json", result)
    print(json.dumps({"cutoffs": cutoffs, "parameters": result["parameters"],
                      "predictions": {source: [r for r in records if r['step'] in (16,32,64,128,256)] for source,records in result["population"].items()}}, indent=2))


if __name__ == "__main__": main()
