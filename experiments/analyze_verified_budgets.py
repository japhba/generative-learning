"""Secondary budget axes for the same fixed trained trajectories."""

import json

import matplotlib.pyplot as plt
import numpy as np

from analyze_scaling_verified import crossing, THRESHOLDS
from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def main():
    output = ROOT / "results/scaling_r5s/confirmation"
    protocol = json.loads((output / "protocol.json").read_text())
    allowed = {r["id"] for r in json.loads((output / "strict_test.json").read_text())}
    labels = {r["id"]: r["label"] for r in json.loads((output / "strict_test.json").read_text())}
    data = {source: [] for source in ("current", "frozen")}
    for cell in protocol["cells"]:
        root = output / str(cell["seed"]) / cell["name"]
        rows = json.loads((root / "test.json").read_text())["evaluations"]
        curve = []
        for row in rows:
            records = [r for r in row["records"] if r["id"] in allowed]
            error = float(np.mean([np.mean([r["error"] for r in records if labels[r["id"]] == str(k)]) for k in range(3)]))
            curve.append({"step": row["step"], "error": error, "candidates": row["candidates"], "accepted": row["accepted"],
                          "actual_work": row["proposal_prompt_tokens"] + 3 * row["training_tokens"],
                          "skip_rejected_work": row["proposal_prompt_tokens"] + 3 * row["accepted_training_tokens"]})
        data[cell["source"]].append(curve)
    output_data = {"updated_at_uk": stamp(), "axes": {}, "interpretation": "Candidate and actual-work budgets match exactly by design. The skipped-rejection proxy assumes a hypothetical optimized filter implementation. Accepted-count matching also changes optimizer step count. All secondary crossing ratios below are descriptive, without a new confidence interval."}
    for axis in ("candidates", "accepted", "actual_work", "skip_rejected_work"):
        source_curves = {}
        fig, ax = plt.subplots()
        for source, curves in data.items():
            common = sorted(set.intersection(*[{r["step"] for r in c if r["step"] > 0} for c in curves]))
            rows = [[next(r for r in curve if r["step"] == step) for step in common] for curve in curves]
            x = np.mean([[r[axis] for r in curve] for curve in rows], axis=0); y = np.mean([[r["error"] for r in curve] for curve in rows], axis=0)
            source_curves[source] = {"x": x.tolist(), "error": y.tolist(), "crossings": {str(t): crossing(x, y, t) for t in THRESHOLDS}}
            ax.loglog(x, y, "o-", label=source)
        labels_axis = {"candidates": "Generated candidate tokens", "accepted": "Accepted training examples", "actual_work": "Token-work proxy: proposal + 3 × all training", "skip_rejected_work": "Token-work proxy: proposal + 3 × accepted training"}
        ax.set(xlabel=labels_axis[axis], ylabel="Strict test error probability", title="The same learning curves under different budget accounting")
        ax.legend(); fig.tight_layout(); fig.savefig(ROOT / f"reports/scaling-verified-budget-{axis}.png"); plt.close(fig)
        gains = []
        for threshold in THRESHOLDS:
            a, b = [source_curves[s]["crossings"][str(threshold)] for s in ("current", "frozen")]
            row = {"error": threshold, "current": a, "frozen": b}
            if a["status"] == b["status"] == "interpolated": row["speedup"] = b["budget"] / a["budget"]
            gains.append(row)
        output_data["axes"][axis] = {"curves": source_curves, "gains": gains}
    save(output / "budget_analysis.json", output_data)
    print(json.dumps({axis: v["gains"] for axis, v in output_data["axes"].items()}, indent=2))


if __name__ == "__main__": main()
