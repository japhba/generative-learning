"""Independent explicit-final-answer analysis; never overwrites strict scores."""

import argparse
import json
from pathlib import Path

import numpy as np

from elicitation_answer import parse_answer
from elicitation_task import DEPTHS, consensus, states
from run_elicitation_pilot import CONDITIONS, digest, save


def analyze(output, model, stage):
    problems = json.loads((output / f"{stage}.json").read_text())
    directory = output / stage / model
    records = {name: [{**(row := json.loads(line)), "answer": parse_answer(row["text"])}
                     for line in (directory / f"{name}.jsonl").read_text().splitlines()] for name in CONDITIONS}
    rows = []
    for i, problem in enumerate(problems):
        truth = states(problem)[-1]
        sample_rows = records["reason_samples"][4*i:4*(i+1)]
        assert [r["id"] for r in sample_rows] == [problem["id"]]*4
        vote = consensus(sample_rows)
        row = {"id": problem["id"], "depth": problem["depth"], "family": problem["family"], "truth": truth,
               "consensus": int(vote["answer"] == truth), "consensus_answer": vote["answer"], "support": vote["support"],
               "reason_sample_mean": float(np.mean([r["ended"] and r["answer"] == truth for r in sample_rows])),
               "reason_sample_completion": float(np.mean([r["ended"] for r in sample_rows])),
               "reason_sample_parse": float(np.mean([r["ended"] and r["answer"] is not None for r in sample_rows])),
               "oracle_pass4_diagnostic": int(any(r["ended"] and r["answer"] == truth for r in sample_rows))}
        for condition in ("direct_greedy", "reason_greedy"):
            response = records[condition][i]
            assert response["id"] == problem["id"]
            row[condition] = int(response["ended"] and response["answer"] == truth)
            row[condition + "_completion"] = int(response["ended"])
        rows.append(row)
    metrics = ("direct_greedy", "reason_greedy", "consensus", "reason_sample_mean", "reason_sample_completion", "reason_sample_parse",
               "oracle_pass4_diagnostic", "direct_greedy_completion", "reason_greedy_completion")
    summary, rng = {}, np.random.default_rng(931000)
    for family in sorted({r["family"] for r in rows}):
        chosen = [r for r in rows if r["family"] == family]
        diffs = np.array([r["consensus"] - r["direct_greedy"] for r in chosen])
        bootstrap = diffs[rng.integers(0, len(chosen), (5000, len(chosen)))].mean(axis=1)
        result = {key: float(np.mean([r[key] for r in chosen])) for key in metrics}
        result.update({"n": len(chosen), "gain": float(diffs.mean()), "gain_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                       "by_depth": {str(d): {key: float(np.mean([r[key] for r in chosen if r["depth"] == d])) for key in metrics} for d in DEPTHS}})
        result["passes_calibration"] = result["consensus"] >= .7 and result["gain"] >= .15 and min(result["direct_greedy_completion"], result["reason_sample_completion"]) >= .95
        summary[family] = result
    result = {"model": model, "stage": stage, "summary": summary, "scored": rows,
              "parser": "Explicit boxed integer, standalone integer, or unique integer after an explicit final-answer marker.",
              "source_sha256": {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("elicitation_answer.py")]},
              "raw_sha256": {name: digest(directory / f"{name}.jsonl") for name in CONDITIONS}}
    save(directory / "explicit_answer_analysis.json", result)
    print(json.dumps({"model": model, "stage": stage, "summary": summary}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/elicitation_pilot"))
    parser.add_argument("--model", default="general")
    parser.add_argument("--stage", default="calibration")
    args = parser.parse_args()
    analyze(args.output, args.model, args.stage)
