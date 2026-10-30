"""Grade every elicitation attempt and make the prespecified calibration choice."""

import argparse
import json
from pathlib import Path

import numpy as np

from elicitation_task import DEPTHS, FAMILIES, consensus, make_problems, parse_answer, states
from run_elicitation_pilot import CONDITIONS, MODELS, digest, save


def analyze(output, model, stage):
    problems = json.loads((output / f"{stage}.json").read_text())
    directory = output / stage / model
    records = {name: [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()] for name in CONDITIONS}
    for name, rows in records.items():
        samples = CONDITIONS[name][1]
        assert [(r["id"], r["sample"]) for r in rows] == [(p["id"], i) for p in problems for i in range(samples)]
        assert all(row["answer"] == parse_answer(row["text"]) for row in rows)
    scored = []
    for i, problem in enumerate(problems):
        truth = states(problem)[-1]
        samples = records["reason_samples"][i * 4:(i + 1) * 4]
        vote = consensus(samples)
        row = {"id": problem["id"], "family": problem["family"], "depth": problem["depth"], "truth": truth,
               "consensus": int(vote["answer"] == truth), "consensus_answer": vote["answer"], "vote_support": vote["support"],
               "reason_sample_mean": float(np.mean([r["ended"] and r["answer"] == truth for r in samples])),
               "reason_sample_completion": float(np.mean([r["ended"] for r in samples])),
               "reason_sample_parse": float(np.mean([r["ended"] and r["answer"] is not None for r in samples])),
               "oracle_pass4_diagnostic": int(any(r["ended"] and r["answer"] == truth for r in samples))}
        for condition in ("direct_greedy", "reason_greedy"):
            response = records[condition][i]
            row[condition] = int(response["ended"] and response["answer"] == truth)
            row[condition + "_completion"] = int(response["ended"])
        scored.append(row)
    metrics = ("direct_greedy", "reason_greedy", "consensus", "reason_sample_mean", "reason_sample_completion",
               "reason_sample_parse", "oracle_pass4_diagnostic", "direct_greedy_completion", "reason_greedy_completion")
    summary, rng = {}, np.random.default_rng(911000)
    for family in FAMILIES:
        selected = [row for row in scored if row["family"] == family]
        if not selected: continue
        values = {key: float(np.mean([row[key] for row in selected])) for key in metrics}
        differences = np.array([row["consensus"] - row["direct_greedy"] for row in selected])
        draws = rng.integers(0, len(selected), (5000, len(selected)))
        values.update({"n": len(selected), "gain": float(differences.mean()),
                       "gain_ci95": np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist(),
                       "by_depth": {str(depth): {key: float(np.mean([row[key] for row in selected if row["depth"] == depth])) for key in metrics} for depth in DEPTHS}})
        values["passes_calibration"] = values["consensus"] >= .70 and values["gain"] >= .15 and values["reason_sample_completion"] >= .95
        summary[family] = values
    result = {"stage": stage, "model": model, "summary": summary, "scored": scored,
              "input_sha256": {name: digest(directory / f"{name}.jsonl") for name in CONDITIONS}}
    save(directory / "analysis.json", result)
    print(json.dumps({"stage": stage, "model": model, "summary": summary}, indent=2))
    return result


def select(output):
    analyses = {model: json.loads((output / "calibration" / model / "analysis.json").read_text())
                for model in MODELS if (output / "calibration" / model / "analysis.json").exists()}
    candidates = [(model, family) for model in MODELS for family in FAMILIES
                  if model in analyses and analyses[model]["summary"][family]["passes_calibration"]]
    if not candidates:
        assert set(analyses) == set(MODELS), "Run the next prespecified model before closing calibration."
        save(output / "selection.json", {"passed": False, "reason": "Neither prespecified model/family passed acquisition-free elicitation calibration."})
        return
    model, family = candidates[0]
    path = output / "confirmation.json"
    assert not path.exists()
    confirmation = make_problems(62000, 48, (family,))
    calibration = json.loads((output / "calibration.json").read_text())
    assert not {row["id"] for row in confirmation} & {row["id"] for row in calibration}
    save(path, confirmation)
    save(output / "selection.json", {"passed": True, "model": model, "family": family,
         "calibration": analyses[model]["summary"][family], "confirmation_sha256": digest(path),
         "calibration_analysis_sha256": digest(output / "calibration" / model / "analysis.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze", "select"))
    parser.add_argument("--output", type=Path, default=Path("results/elicitation_pilot"))
    parser.add_argument("--model", choices=MODELS, default="math")
    parser.add_argument("--stage", choices=("calibration", "confirmation"), default="calibration")
    args = parser.parse_args()
    if args.command == "select": select(args.output)
    else: analyze(args.output, args.model, args.stage)


if __name__ == "__main__": main()
