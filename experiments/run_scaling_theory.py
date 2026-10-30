"""Confirm predeclared population exponents without fitting a time constant."""

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT, digest, save
from scaling_theory import asymptotic_constant, gate_time, rbm_population


def main():
    output = ROOT / "results/scaling_s4g/theory"
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", {"created_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(),
         "source_sha256": {name: digest(ROOT / "experiments" / name) for name in ("scaling_theory.py", "run_scaling_theory.py")},
         "predicted_exponents": {"gaussian_rbm": .5, "homogeneous_gate": 1.},
         "rbm_initial_norm": 1., "rbm_initial_overlap": .3, "signal": 1.5,
         "assessment_window": [1e6, 1e8], "timescale": "Population gradient-flow time, not sampled-training compute."})
    times = np.concatenate([[0.], np.geomspace(.1, 1e8, 241)])
    rbm = rbm_population(times)
    probability = 1 - np.geomspace(.4, 1e-9, 400)
    gate_t, gate_error = gate_time(probability), 1 - probability
    tail = times >= 1e6
    rbm_error = np.array([r["excess_error"] for r in rbm])
    rbm_gradient = np.array([r["gradient_squared"] for r in rbm])
    gate_tail = (gate_t >= 1e6) & (gate_t <= 1e8)
    results = {"rbm_beta": float(-np.polyfit(np.log(times[tail]), np.log(rbm_error[tail]), 1)[0]),
               "rbm_gamma": float(np.polyfit(np.log(rbm_error[tail]), np.log(rbm_gradient[tail]), 1)[0]),
               "gate_beta": float(-np.polyfit(np.log(gate_t[gate_tail]), np.log(gate_error[gate_tail]), 1)[0]),
               "rbm_asymptotic_error_coefficient": float(np.sqrt(asymptotic_constant() / 8))}
    save(output / "results.json", {"summary": results, "rbm": rbm, "gate": {"time": gate_t.tolist(), "error": gate_error.tolist()}})
    fig, ax = plt.subplots()
    ax.loglog(times[1:], rbm_error[1:], label="Gaussian RBM population")
    ax.loglog(times[1:], np.sqrt(asymptotic_constant() / (8 * times[1:])), "--", label="RBM prediction: exponent 1/2")
    keep = gate_t > 0
    ax.loglog(gate_t[keep], gate_error[keep], label="One-logit Boltzmann gate")
    ax.loglog(gate_t[keep], 1 / gate_t[keep], "--", label="Gate prediction: exponent 1")
    ax.set(xlabel="Population gradient-flow time", ylabel="Error above the known ceiling", ylim=(1e-9, 1), xlim=(.1, 1e8))
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/scaling-theory.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.loglog(rbm_error[1:], rbm_gradient[1:], label="Gaussian RBM")
    ax.loglog(gate_error, (probability * gate_error)**2, label="One-logit gate")
    ax.set(xlabel="Error above the known ceiling", ylabel="Squared success-gradient norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/scaling-gradient-geometry.png")
    plt.close(fig)
    assert abs(results["rbm_beta"] - .5) < .02
    assert abs(results["rbm_gamma"] - 3) < .03
    assert abs(results["gate_beta"] - 1) < .002
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
