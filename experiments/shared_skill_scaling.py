"""Numerically check an analytic exponent separation in a shared logistic skill.

This integrates the exact population drift of the existing verified-only update.
It does not simulate the stated number of SGD steps or run any LLM training.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from tqdm.auto import tqdm

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


def quadrature(a, maximum, order, truncated):
    edges = np.arange(0., maximum + 2., 2.)
    assert edges[-1] == maximum
    nodes, weights = leggauss(order)
    difficulty = (edges[:-1, None] + 1. + nodes).ravel()
    measure = np.tile(weights, len(edges) - 1) * a * np.exp(-a * difficulty)
    if truncated: measure /= -np.expm1(-a * maximum)
    assert abs(measure.sum() - 1) < 1e-12
    return difficulty, measure


def integrate(a, cutoff, theta_step, order):
    difficulty, measure = quadrature(a, 180. if cutoff is None else cutoff, order, cutoff is not None)
    theta = np.arange(0., 80. + theta_step / 2, theta_step)
    initial = 1 / (1 + np.exp(difficulty)); q0 = initial * (2 - initial); Q0 = measure @ q0
    batch_probability0 = 1 - (1 - Q0)**32
    frozen_drift_constant = batch_probability0 * (measure @ (q0 * np.exp(difficulty))) / Q0
    errors = np.empty(len(theta)); nll = np.empty_like(errors); drift = np.empty((2, len(theta)))
    current_h = np.empty_like(errors)
    for i, weight in enumerate(theta):
        probability = 1 / (1 + np.exp(difficulty - weight))
        epsilon = 1 / (1 + np.exp(weight - difficulty))
        acceptance = probability * (2 - probability); Q = measure @ acceptance
        errors[i] = measure @ epsilon; nll[i] = measure @ np.logaddexp(0., difficulty - weight)
        current_h[i] = measure @ (acceptance * epsilon)
        drift[0, i] = (1 - (1 - Q)**32) * current_h[i] / Q
        drift[1, i] = batch_probability0 * (measure @ (q0 * epsilon)) / Q0
    assert np.isfinite(drift).all() and (drift > 0).all()
    assert abs(drift[0, 0] / drift[1, 0] - 1) < 1e-14
    time = np.zeros_like(drift)
    time[:, 1:] = np.cumsum(theta_step * (1 / drift[:, 1:] + 1 / drift[:, :-1]) / 2, axis=1)
    greedy = np.exp(-a * theta)
    if cutoff is not None: greedy = np.maximum(greedy - np.exp(-a * cutoff), 0.) / (-np.expm1(-a * cutoff))
    diagnostics = {"initial_acceptance": float(Q0), "frozen_drift_constant": float(frozen_drift_constant),
                   "endpoint_frozen_drift_ratio": float(drift[1, -1] * np.exp(theta[-1]) / frozen_drift_constant)}
    if a == .5 and cutoff is None:
        closed_error = np.arctan(np.exp(theta / 2)) * np.exp(-theta / 2)
        inverse = 1 / (1 + np.exp(theta))
        closed_h = 5 * closed_error / 8 - 3 * inverse / 8 - inverse**2 / 4
        diagnostics.update({"max_relative_error_vs_closed_form": float(np.max(abs(errors / closed_error - 1))),
                            "max_relative_drift_numerator_vs_closed_form": float(np.max(abs(current_h / closed_h - 1)))})
        assert diagnostics["max_relative_error_vs_closed_form"] < 1e-9
        assert diagnostics["max_relative_drift_numerator_vs_closed_form"] < 1e-9
    common_time = np.geomspace(1., 1e12, 601)
    curves = {}
    for i, source in enumerate(("current", "frozen")):
        assert time[i, -1] > common_time[-1]
        weights = np.interp(np.log(common_time), np.log(time[i, 1:]), theta[1:])
        error = np.exp(np.interp(np.log(common_time), np.log(time[i, 1:]), np.log(errors[1:])))
        greedy_error = np.exp(-a * weights)
        if cutoff is not None: greedy_error = np.maximum(greedy_error - np.exp(-a * cutoff), 0.) / (-np.expm1(-a * cutoff))
        prediction = 1. if source == "current" or cutoff is not None else min(a, 1.)
        keep = common_time >= 1e8
        beta = -np.polyfit(np.log(common_time[keep]), np.log(error[keep]), 1)[0]
        curves[source] = {"time": common_time.tolist(), "weight": weights.tolist(), "error": error.tolist(), "greedy_error": greedy_error.tolist(),
                          "local_beta": (-np.gradient(np.log(error), np.log(common_time))).tolist(), "predicted_beta": prediction,
                          "measured_beta_1e8_to_1e12": float(beta)}
        assert abs(beta - prediction) < .002
        if cutoff is None:
            greedy_beta = -np.polyfit(np.log(common_time[keep]), np.log(greedy_error[keep]), 1)[0]
            curves[source]["measured_greedy_beta_1e8_to_1e12"] = float(greedy_beta)
            assert abs(greedy_beta - (1. if source == "current" and a < 1 else a)) < .002
        if a < 1 and cutoff is None:
            amplitude = 2 / (a*a*(3-a)) if source == "current" else a*np.pi/np.sin(np.pi*a)*frozen_drift_constant**(-a)
            curves[source].update({"predicted_error_amplitude": float(amplitude), "endpoint_amplitude_ratio": float(error[-1] * common_time[-1]**prediction / amplitude)})
            assert abs(curves[source]["endpoint_amplitude_ratio"] - 1) < .002
    return {"a": a, "difficulty_cutoff": cutoff, "theta_step": theta_step, "quadrature_order": order, "diagnostics": diagnostics, "curves": curves}


def main():
    output = ROOT / "results/scaling_r5s/shared_skill_theory"; output.mkdir(exist_ok=False)
    cases = ((.5, None), (.75, None), (1.5, None), (.5, 12.))
    protocol = {"created_at_uk": stamp(), "status": "Constructive theory developed after the LLM results, not an LLM fit or a prospective LLM forecast.",
                "cases": cases, "batch_questions": 32, "candidates_per_question": 2, "initial_shared_weight": 0.,
                "numerics": "Composite Gauss-Legendre difficulty integrals, then dt/dtheta=1/drift. Repeat with doubled theta spacing and half quadrature order.",
                "population_time": "t = learning rate times updates; proportional to candidate budget at fixed batch and learning rate. Numerical integration does not execute t updates.",
                "source_sha256": digest(ROOT / "experiments/shared_skill_scaling.py")}
    save(output / "protocol.json", protocol)
    (output / "source.py").write_bytes((ROOT / "experiments/shared_skill_scaling.py").read_bytes())
    results = []
    for a, cutoff in tqdm(cases, desc="Check theory and boundary cases"):
        fine = integrate(a, cutoff, .01, 16); coarse = integrate(a, cutoff, .02, 8)
        fine["coarse_check"] = {source: {"max_relative_curve_difference": float(np.max(abs(np.array(fine['curves'][source]['error']) / np.array(coarse['curves'][source]['error']) - 1))),
                                          "beta_difference": float(fine['curves'][source]['measured_beta_1e8_to_1e12'] - coarse['curves'][source]['measured_beta_1e8_to_1e12'])}
                                for source in ("current", "frozen")}
        assert all(r["max_relative_curve_difference"] < .001 for r in fine["coarse_check"].values())
        results.append(fine)
    result = {"completed_at_uk": stamp(), "protocol": protocol, "cases": results}
    save(output / "results.json", result)
    main_case = results[0]
    fig, ax = plt.subplots()
    for source in ("current", "frozen"):
        curve = main_case["curves"][source]; time = np.array(curve["time"])
        line, = ax.loglog(time, curve["error"], label=f"{source} generator")
        late = time >= 1e4
        ax.loglog(time[late], curve["predicted_error_amplitude"] * time[late]**(-curve["predicted_beta"]), "--", color=line.get_color(), label=f"predicted exponent {curve['predicted_beta']:g}")
    ax.set(xlabel="Population training time (proportional to candidates)", ylabel="Expected answer error", title="A shared skill permits different exponents")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-shared-skill.png"); plt.close(fig)
    fig, ax = plt.subplots()
    for index, case in enumerate((results[0], results[-1])):
        for source in ("current", "frozen"):
            curve = case["curves"][source]
            ax.semilogx(curve["time"], curve["local_beta"], linestyle="-" if index == 0 else "--", color="C0" if source == "current" else "C1", label=f"{source}, {'unbounded difficulty' if index == 0 else 'difficulty capped at 12'}")
    ax.set(xlabel="Population training time", ylabel="Local error exponent", title="A finite difficulty cap eventually removes the exponent gap")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-shared-skill-cutoff.png"); plt.close(fig)
    print(json.dumps([{k: case[k] for k in ("a", "difficulty_cutoff", "diagnostics", "coarse_check")} | {"fits": {s: {k: v for k, v in c.items() if not isinstance(v, list)} for s, c in case["curves"].items()}} for case in results], indent=2))


if __name__ == "__main__": main()
