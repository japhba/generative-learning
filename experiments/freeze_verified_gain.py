"""Secondary initial-difficulty prediction, with explicit exposure and assumptions."""

import json

import numpy as np

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


def main():
    root = ROOT / "results/scaling_r5s/confirmation"
    protocol = json.loads((root / "protocol.json").read_text())
    baseline = json.loads((root / "99601/current_5.6e-06/progress.json").read_text())["evaluations"][0]
    p = np.exp([r["correct_logprob"] for r in baseline["records"]]); q = p * (2 - p)
    labels = np.array([r["label"] for r in json.loads((root / "monitor.json").read_text())]); strata = [np.flatnonzero(labels == k) for k in ("0", "1", "2")]
    rng = np.random.default_rng(198782); estimates = []
    for _ in range(5000):
        indices = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata]); sampled = q[indices]
        estimates.append(sampled.mean() * np.mean(1 / sampled))
    exposure = []
    for cell in protocol["cells"]:
        path = root / str(cell["seed"]) / cell["name"] / "progress.json"
        if path.exists(): exposure.append({**cell, "latest_existing_monitor_step": json.loads(path.read_text())["step"]})
    prediction = {"frozen_at_uk": stamp(), "status": "Secondary mechanism hypothesis, formed after the pilot and early confirmation monitoring; only initial probabilities enter the numeric coefficient.",
          "predicted_asymptotic_candidate_speedup": float(q.mean() * np.mean(1 / q)), "initial_question_bootstrap_ci95": np.quantile(estimates, [.025, .975]).tolist(),
          "initial_mean_acceptance": float(q.mean()), "initial_mean_inverse_acceptance": float(np.mean(1 / q)), "initial_probability_quantiles": np.quantile(p, [0, .01, .1, .5, 1]).tolist(),
          "formula": "mean(q0) mean(1/q0), q0=p0(2-p0)", "exposure": exposure,
          "future_comparison": "Report error ratios at2048 and4096 and threshold crossing ratios alongside this eventual prefactor. Finite-time deviations or different local exponents do not validate an asymptotic match. No coefficient is refit after those endpoints.",
          "assumptions": "Balanced independent latent learning modes with identical logit metrics, representatively distributed like the initial monitoring probabilities. A mode may contain many literal inputs. Treating each held-out literal input as a separate one-hot feature would not generalize across the split; this is not that model.",
          "limits": "This approximation ignores LLM weight sharing and mode interactions. The bootstrap measures initial-input uncertainty under the formula, not model misspecification or uncertainty in the real LLM speedup.",
          "protocol_sha256": digest(root / "protocol.json")}
    save(root / "gain_prediction.json", prediction)
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__": main()
