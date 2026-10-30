"""Evaluate the frozen gate forecasts without fitting their rate or ceiling."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bestn_llm import graded


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/bestn_b2k"
REPORTS = ROOT / "reports"


def analyze_seed(seed):
    cohort = RESULTS / "training" / str(seed)
    problems = json.loads((cohort / "evaluation.json").read_text())
    records = []
    for source in ("current", "frozen"):
        path = cohort / source / "progress.json"
        if not path.exists(): continue
        history = json.loads(path.read_text())
        for evaluation in history["evaluations"]:
            rows = graded([json.loads(line) for line in Path(evaluation["path"]).read_text().splitlines()], problems)
            equation = np.array(["=" in row["text"] for row in rows])
            correctness = np.array([row["correct"] for row in rows])
            record = {"seed": seed, "source": source, "step": evaluation["step"], "candidates": evaluation["candidates"],
                      "accuracy": float(correctness.mean()), "equation_probability": float(equation.mean()),
                      "conditional_equation_accuracy": float(correctness[equation].mean()) if equation.any() else None,
                      "conditional_other_accuracy": float(correctness[~equation].mean()) if (~equation).any() else None,
                      "equation_count": int(equation.sum()), "other_count": int((~equation).sum()), "responses": len(rows),
                      "per_prompt": {p["id"]: float(np.mean([r["correct"] for r in rows if r["id"] == p["id"]])) for p in problems}}
            if evaluation["step"]:
                forecast = next(f for f in history["forecasts"] if f["end_step"] == evaluation["step"])
                record["weight_projection"] = forecast["weight_overlap"]["success_direction_projection"]
                record["weight_cosine"] = forecast["weight_overlap"]["cosine"]
            else: record["weight_projection"], record["weight_cosine"] = 0., 0.
            records.append(record)
    return records


def main():
    prediction = json.loads((RESULTS / "gate_prediction/prospective.json").read_text())
    records = [record for seed in (86001, 86002, 86003) for record in analyze_seed(seed)]
    for seed, label in ((86001, "Discovery"), (86003, "Confirmation")):
        for metric, expected, ylabel, suffix in (("accuracy", "accuracy", "Sampled test accuracy", "accuracy"),
                                               ("equation_probability", "mode_probability", "Probability of an equation response", "mode"),
                                               ("weight_projection", "weight_projection", "Weight displacement along success direction", "weights")):
            fig, ax = plt.subplots(layout="constrained")
            for index, source in enumerate(("current", "frozen")):
                theory = [p for p in prediction["rows"] if p["source"] == source]
                actual = [r for r in records if r["seed"] == seed and r["source"] == source]
                ax.plot([p["candidates"] for p in theory], [p[expected] for p in theory], color=f"C{index}", linestyle="--", label=source.capitalize() + ": gate prediction")
                if metric == "accuracy":
                    ax.fill_between([p["candidates"] for p in theory], [p["finite_interval95"][0] for p in theory],
                                    [p["finite_interval95"][1] for p in theory], color=f"C{index}", alpha=.15)
                if actual:
                    ax.plot([r["candidates"] for r in actual], [r[metric] for r in actual], "o-", color=f"C{index}", label=source.capitalize() + ": LLM")
            if seed == prediction["primary_confirmation_seed"]:
                ax.axvspan(0, 32 * prediction["completed_checkpoints_at_freeze"]["current"], color="gray", alpha=.1)
                ax.set_title(f"Seed {seed}: current checkpoints after shading are prospective")
            else: ax.set_title(f"{label} seed {seed}: one Boltzmann gate")
            ax.set_xlabel("Training candidate rollouts")
            ax.set_ylabel(ylabel)
            ax.legend()
            path = REPORTS / f"bestn-gate-{seed}-{suffix}.png"
            fig.savefig(path)
            plt.close(fig)
            print(path)
    fig, ax = plt.subplots(layout="constrained")
    for seed in (86001, 86002, 86003):
        rows = [r for r in records if r["seed"] == seed and r["source"] == "current"]
        if rows: ax.plot([r["candidates"] for r in rows], [r["conditional_equation_accuracy"] for r in rows], "o-", label=f"Seed {seed}")
    ax.set_xlabel("Training candidate rollouts")
    ax.set_ylabel("Correctness conditional on an equation response")
    ax.set_title("Does competence within the output mode change?")
    ax.legend()
    path = REPORTS / "bestn-gate-conditional.png"
    fig.savefig(path)
    plt.close(fig)
    print(path)
    assessments = []
    for source in ("current", "frozen"):
        observed = [r for r in records if r["seed"] == prediction["primary_confirmation_seed"] and r["source"] == source
                    and r["step"] > prediction["completed_checkpoints_at_freeze"][source]]
        if not observed: continue
        theory = {r["step"]: r for r in prediction["rows"] if r["source"] == source}
        for metric, expected in (("accuracy", "accuracy"), ("equation_probability", "mode_probability"), ("weight_projection", "weight_projection")):
            errors = np.array([r[metric] - theory[r["step"]][expected] for r in observed])
            assessments.append({"source": source, "metric": metric, "steps": [r["step"] for r in observed],
                                "mean_observed_minus_predicted": float(errors.mean()), "mae": float(np.abs(errors).mean()),
                                "rmse": float(np.sqrt(np.mean(errors**2)))})
    decompositions = []
    for seed in (86001, 86002, 86003):
        for source in ("current", "frozen"):
            observed = [r for r in records if r["seed"] == seed and r["source"] == source]
            if len(observed) < 2: continue
            initial, final = observed[0], observed[-1]
            l0, l1 = initial["equation_probability"], final["equation_probability"]
            a0, a1 = initial["conditional_equation_accuracy"], final["conditional_equation_accuracy"]
            b0, b1 = initial["conditional_other_accuracy"], final["conditional_other_accuracy"]
            if None in (a0, a1, b0, b1): continue
            gate_part = (l1 - l0) * (a0 + a1 - b0 - b1) / 2
            expert_part = (l0 + l1) / 2 * (a1 - a0) + (1 - (l0 + l1) / 2) * (b1 - b0)
            total = final["accuracy"] - initial["accuracy"]
            np.testing.assert_allclose(gate_part + expert_part, total, atol=1e-15)
            decompositions.append({"seed": seed, "source": source, "end_step": final["step"], "total_gain": total,
                                   "gate_frequency_component": gate_part, "conditional_accuracy_component": expert_part})
    (RESULTS / "gate_prediction/analysis.json").write_text(json.dumps({"predictions_frozen_at_uk": prediction["created_at_uk"],
        "prospective_cutoffs": prediction["completed_checkpoints_at_freeze"], "records": records,
        "prospective_assessment": assessments, "descriptive_decomposition": decompositions,
        "decomposition_scope": "Symmetric decomposition of aggregate probabilities, not a causal intervention. Conditional accuracy may change through prompt composition as well as changes within either output mode."}, indent=2) + "\n")


if __name__ == "__main__": main()
