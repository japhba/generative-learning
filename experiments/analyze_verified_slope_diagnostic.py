"""Post-hoc examination of the r5s exponent gap using existing evaluations only."""

import json

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


def main():
    root = ROOT / "results/scaling_r5s/confirmation"
    analyses = {split: json.loads((root / f"analysis_{split}.json").read_text()) for split in ("strict", "monitor", "test")}
    shared_steps = sorted(set.intersection(*[{r["step"] for r in data["pointwise"] if r["step"] > 0} for data in analyses.values()]))
    summaries = {}
    fig, ax = plt.subplots()
    for split, data in analyses.items():
        points = [r for r in data["pointwise"] if r["step"] in shared_steps]
        steps = np.array([r["step"] for r in points]); candidates = 64 * steps
        errors = np.array([[r[f"{source}_error"] for r in points] for source in ("current", "frozen")])
        ratios = errors[1] / errors[0]
        slopes = -np.diff(np.log(errors), axis=1) / np.diff(np.log(steps))
        windows = []
        for start, end in ((128, 1024), (512, 4096), (1024, 4096)):
            keep = (steps >= start) & (steps <= end)
            beta = -np.polyfit(np.log(steps[keep]), np.log(errors[:, keep]).T, 1)[0]
            windows.append({"start": start, "end": end, "current_beta": float(beta[0]), "frozen_beta": float(beta[1]), "beta_difference": float(beta[0] - beta[1])})
        summaries[split] = {"source_sha256": digest(root / f"analysis_{split}.json"), "common_checkpoint_windows": windows,
                            "adjacent_intervals": [{"start": int(a), "end": int(b), "current_beta": float(slopes[0, i]), "frozen_beta": float(slopes[1, i]),
                                                    "beta_difference": float(slopes[0, i] - slopes[1, i])} for i, (a, b) in enumerate(zip(steps[:-1], steps[1:], strict=True))],
                            "candidate_budgets": candidates.tolist(), "error_ratios": ratios.tolist()}
        if split == "test": continue
        bounds = np.array([r["ratio_ci95"] for r in points])
        line, = ax.loglog(candidates, ratios, "o-", label="Strict final test" if split == "strict" else "Monitoring cohort")
        ax.fill_between(candidates, bounds[:, 0], bounds[:, 1], color=line.get_color(), alpha=.2)
    ax.set(xlabel="Generated candidate tokens", ylabel="Frozen error / current error", title="Does the error gap grow with training budget?")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-error-ratio.png"); plt.close(fig)
    result = {"recorded_at_uk": stamp(), "status": "Post-hoc descriptive diagnostic; no new training, test examples, model selection or bootstrap draws.",
              "interpretation": "For two exact power laws, the log-ratio slope equals current exponent minus frozen exponent. An increasing ratio supports a finite-range exponent difference. It does not distinguish a persistent exponent gap from a transient. The confidence bands are the previously computed pointwise paired bootstrap intervals, not a simultaneous test of constant ratios.",
              "shared_updates": shared_steps, "cohorts": summaries, "analysis_sha256": digest(ROOT / "experiments/analyze_verified_slope_diagnostic.py")}
    save(root / "slope_diagnostic.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
