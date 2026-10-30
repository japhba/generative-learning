"""Check whether the pilot curves agree when measured in learning-rate time."""

import json

import matplotlib.pyplot as plt
import numpy as np

from analyze_scaling_verified import read_runs
from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def main():
    root = ROOT / "results/scaling_r5s/pilot"; protocol, runs = read_runs(root)
    assert len(runs) == 6 and all(r["complete"] for r in runs)
    base = min(r["cell"]["eta"] for r in runs); result = {"updated_at_uk": stamp(), "sources": {}}
    fig, ax = plt.subplots()
    for source in ("current", "frozen"):
        selected = sorted([r for r in runs if r["cell"]["source"] == source], key=lambda r: r["cell"]["eta"])
        curves = []
        for index, run in enumerate(selected):
            cell = run["cell"]; rows = [r for r in run["history"]["evaluations"] if r["step"] > 0]
            eta = cell["eta"]; scaled = {round(eta / base * r["step"]): r["error"] for r in rows}; curves.append(scaled)
            ax.loglog([eta * r["step"] for r in rows], [r["error"] for r in rows], linestyle=("-", "--", ":")[index], color="C0" if source == "current" else "C1", label=f"{source}, rate {eta:.1e}")
        shared = sorted(set.intersection(*[set(c) for c in curves]))
        checks = []
        for step in shared:
            errors = np.array([c[step] for c in curves])
            checks.append({"base_rate_equivalent_updates": step, "sgd_time": step * base, "errors_by_rate": errors.tolist(),
                           "max_over_min_error": float(errors.max() / errors.min()), "log_error_std": float(np.std(np.log(errors)))})
        result["sources"][source] = {"checks": checks, "median_max_over_min_for_128_to_1024": float(np.median([r["max_over_min_error"] for r in checks if 128 <= r["base_rate_equivalent_updates"] <= 1024]))}
    ax.set(xlabel="Learning rate × optimizer updates", ylabel="Monitoring error probability", title="Within each method, the rate grid approximately collapses")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-rate-collapse.png"); plt.close(fig)
    save(root / "rate_collapse.json", result)
    print(json.dumps({s: v["median_max_over_min_for_128_to_1024"] for s, v in result["sources"].items()}))


if __name__ == "__main__": main()
