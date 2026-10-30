"""Audit every selected token and measure the changing accepted input mix."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from analyze_scaling_verified import crossing, read_runs, THRESHOLDS
from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def audit_run(output, run):
    cell = run["cell"]; history = run["history"]
    rows = json.loads((output / str(cell["seed"]) / "train.json").read_text())
    presented = np.zeros((history["step"], 3), dtype=int); accepted = np.zeros_like(presented)
    accepted_ids = set()
    for i, update in enumerate(tqdm(history["updates"], desc=f"Audit {cell['seed']} {cell['name']}", leave=False)):
        raw = json.loads(Path(update["path"]).read_text())
        batch = rows[32 * i:32 * (i + 1)]
        assert [r["id"] for r in raw["records"]] == [r["id"] for r in batch]
        assert raw["accepted"] == update["accepted"] == sum(r["accepted"] for r in raw["records"])
        for record, question in zip(raw["records"], batch, strict=True):
            assert record["ground_truth"] == question["label"]
            actual = [text == record["ground_truth"] for text in record["candidate_texts"]]
            assert actual == record["correct"] and any(actual) == record["accepted"]
            key = int(record["ground_truth"]); presented[i, key] += 1
            if record["accepted"]:
                assert record["correct"][record["chosen_index"]]
                accepted[i, key] += 1; accepted_ids.add(record["id"])
    assert int(accepted.sum()) == history["accepted"]
    assert len(accepted_ids) == history["accepted_unique_questions"]
    blocks = []
    previous = 0
    for evaluation in history["evaluations"][1:]:
        step = evaluation["step"]; counts = accepted[previous:step].sum(0); shown = presented[previous:step].sum(0)
        blocks.append({"step": step, "accepted_counts": counts.tolist(), "accepted_shares": (counts / counts.sum()).tolist(),
                       "class_acceptance": (counts / shown).tolist(), "cumulative_accepted": int(accepted[:step].sum())})
        previous = step
    monitor = json.loads((output / "monitor.json").read_text()); labels = {r["id"]: r["label"] for r in monitor}
    class_curves = [{"step": evaluation["step"], "errors": [float(np.mean([r["error"] for r in evaluation["records"] if labels[r["id"]] == str(k)])) for k in range(3)]} for evaluation in history["evaluations"]]
    evaluations = [r for r in history["evaluations"] if r["step"] > 0]
    x = np.array([r["accepted"] for r in evaluations]); y = np.array([r["error"] for r in evaluations])
    return {"cell": cell, "steps": history["step"], "complete": run["complete"], "accepted": history["accepted"], "presented": history["presentations"],
            "all_selected_tokens_verified": True, "accepted_class_counts": accepted.sum(0).tolist(), "blocks": blocks, "class_curves": class_curves,
            "accepted_crossings": {str(t): crossing(x, y, t) for t in THRESHOLDS}, "learner_seconds": sum(r["seconds"] for r in history["updates"]),
            "actual_token_work_proxy": history["proposal_prompt_tokens"] + 3 * history["training_tokens"],
            "skip_rejected_token_work_proxy": history["proposal_prompt_tokens"] + 3 * history["accepted_training_tokens"]}


def main(output):
    protocol, runs = read_runs(output)
    audits = [audit_run(output, run) for run in runs if run["complete"]]
    result = {"updated_at_uk": stamp(), "stage": protocol["stage"], "runs": audits,
              "scope": "Accepted-example and hypothetical skipped-rejection token-work axes are secondary; matching accepted examples also changes optimizer update count."}
    save(output / "audit.json", result)
    if protocol["stage"] == "confirmation":
        fig, ax = plt.subplots()
        for source, style in (("current", "-"), ("frozen", "--")):
            selected = [r for r in audits if r["cell"]["source"] == source]
            common = sorted(set.intersection(*[{b["step"] for b in r["blocks"]} for r in selected]))
            shares = np.array([[next(b["accepted_shares"] for b in r["blocks"] if b["step"] == step) for step in common] for r in selected]).mean(0)
            for key in range(3): ax.semilogx(np.array(common) * 64, shares[:, key], linestyle=style, color=f"C{key}", label=f"{source}, count {key}")
        ax.axhline(1 / 3, color="gray", linestyle=":")
        ax.set(xlabel="Generated candidate tokens", ylabel="Share of accepted training examples", title="Refreshing changes which questions supply correct examples")
        ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-accepted-mix.png"); plt.close(fig)
    print(json.dumps({"audited_runs": len(audits), "audited_updates": sum(r["steps"] for r in audits), "all_selected_tokens_verified": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=ROOT / "results/scaling_r5s/confirmation")
    args = parser.parse_args(); main(args.output)
