"""Summarize completed source-refresh experiments and publish their local report."""

import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from self_contrastive_spectral import ARMS, LABELS, spectrum


def main():
    root = Path("results/self_contrastive_spectral")
    protocol = json.loads((root / "protocol.json").read_text())
    assert hashlib.sha256((root / "self_contrastive_spectral.py").read_bytes()).hexdigest() == protocol["source_sha256"]
    population = json.loads((root / "population.json").read_text())
    reference = json.loads((root / "refinement.json").read_text())
    finite = json.loads((root / "finite_training.json").read_text())
    primary, = [r for r in population if r["case"] == {"beta": 0.5, "signal": 2, "shape": "power", "initial_gain": 1}]
    numerical = {}
    for arm in ARMS:
        times = np.array([r["time"] for r in primary["records"][1:]])
        loss = np.array([r["arms"][arm]["excess_nll"] for r in primary["records"][1:]])
        times_ref = np.array([r["time"] for r in reference["records"][1:]])
        loss_ref = np.array([r["arms"][arm]["excess_nll"] for r in reference["records"][1:]])
        aligned = np.exp(np.interp(np.log(times), np.log(times_ref), np.log(loss_ref)))
        numerical[arm] = {"max_relative_curve_change_under_joint_refinement": float(np.max(np.abs(loss / aligned - 1))), "late_slope_change": float(reference["slopes"][arm]["100000..1e+07"] - primary["slopes"][arm]["100000..1e+07"])}
        assert numerical[arm]["max_relative_curve_change_under_joint_refinement"] < 0.005

    # Independent solution for the shallow current-source gradient flow.
    lam, target = spectrum(0.5, 2, 1024, "power")
    analytic = np.array([np.sum(lam * target**2 * np.exp(-lam * t)) / 2 for t in times])
    observed = np.array([r["arms"]["current_shallow"]["excess_nll"] for r in primary["records"][1:]])
    exact_error = float(np.max(np.abs(observed / analytic - 1)))
    assert exact_error < 0.005

    # Integrate original factor weights rather than the invariant-reduced state.
    sample_row = min(primary["records"], key=lambda row: abs(row["time"] - 30))
    endpoint = sample_row["time"]
    steps = int(np.ceil(endpoint / 0.002))
    dt, a, v = endpoint / steps, 1.0, np.zeros(len(lam))

    def derivative(a, v):
        grad = 0.5 * lam * (target - a * v)
        return np.dot(v, grad), a * grad

    for _ in range(steps):
        k1a, k1v = derivative(a, v)
        k2a, k2v = derivative(a + dt * k1a / 2, v + dt * k1v / 2)
        k3a, k3v = derivative(a + dt * k2a / 2, v + dt * k2v / 2)
        k4a, k4v = derivative(a + dt * k3a, v + dt * k3v)
        a += dt * (k1a + 2*k2a + 2*k3a + k4a) / 6
        v += dt * (k1v + 2*k2v + 2*k3v + k4v) / 6
    rk_loss = np.sum(lam * (target - a*v)**2) / 2
    rk_error = float(abs(sample_row["arms"]["current_deep"]["excess_nll"] / rk_loss - 1))
    assert rk_error < 0.005
    assert abs(a*a - np.dot(v, v) - 1) < 1e-8

    # Orthogonal rotations preserve the Gaussian sampled updates.
    rng = np.random.default_rng(840929)
    x = rng.normal(size=(127, 12))
    target_small = rng.normal(size=12)
    y = x @ target_small + rng.normal(size=len(x))
    noise = rng.normal(size=len(x))
    rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    vector = rng.normal(size=12) / 4
    rotation_errors = {}
    for current in (False, True):
        def update(x, v):
            readout = 0.8
            mean = readout * (x @ v)
            source = mean if current else np.zeros(len(x))
            negative = source + noise
            positive_logit = ((y-source)**2 - (y-mean)**2) / 2
            negative_logit = ((negative-source)**2 - (negative-mean)**2) / 2
            residual = -np.exp(-np.logaddexp(0, positive_logit)) * (y-mean) + np.exp(-np.logaddexp(0, -negative_logit)) * (negative-mean)
            return v - 0.01 * readout * (x.T @ residual) / len(x), readout - 0.01 * np.mean(residual * (x @ v))
        v1, a1 = update(x, vector)
        v2, a2 = update(x @ rotation, rotation.T @ vector)
        error = float(max(np.max(np.abs(v1 - rotation @ v2)), abs(a1-a2)))
        assert error < 1e-12
        rotation_errors[str(current)] = error

    rng = np.random.default_rng(840930)
    last = finite["records"][-1]
    draws = rng.integers(0, finite["config"]["repetitions"], size=(10000, finite["config"]["repetitions"]))
    pairs = {}
    for architecture in ("shallow", "deep"):
        frozen = np.array(last["arms"][f"frozen_{architecture}"]["excess_nll"])
        current = np.array(last["arms"][f"current_{architecture}"]["excess_nll"])
        difference = frozen - current
        pairs[architecture] = {"frozen_mean": float(frozen.mean()), "current_mean": float(current.mean()), "frozen_over_current_error_ratio": float(frozen.mean()/current.mean()), "paired_difference": float(difference.mean()), "paired_difference_seed_bootstrap_ci95": np.quantile(difference[draws].mean(axis=1), [0.025, 0.975]).tolist(), "paired_seed_differences": difference.tolist()}
    analysis = {"created_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "conclusion": "Feature learning changes the population time exponent. Refreshing the negative source improves finite-budget error but yields no distinct late exponent within an architecture in this Gaussian family.", "primary_late_slopes": {arm: primary["slopes"][arm]["100000..1e+07"] for arm in ARMS}, "all_case_late_slopes": [{"case": row["case"], "slopes": {arm: row["slopes"][arm]["100000..1e+07"] for arm in ARMS}} for row in population], "numerical_refinement": numerical, "shallow_exact_flow_max_relative_error": exact_error, "deep_independent_factor_RK4_relative_error": rk_error, "rotation_update_errors": rotation_errors, "finite_final": pairs, "finite_budget": {"updates": last["step"], "real_samples_per_seed": last["real_samples_per_seed"], "generated_negatives_per_seed": last["generated_negatives_per_seed"], "seeds": finite["config"]["repetitions"]}, "fisher_bounds_primary": primary["frozen_fisher_bounds_relative_to_covariance"], "limits": ["Late exponents are population-flow fits, not finite-sample exponent estimates.", "Population spectrum is logarithmic quadrature down to lambda=1e-18, not 1024 ordinary equally weighted modes.", "The finite SGD experiment uses actual 1024-dimensional Gaussian observations and power-law eigenvalues; its discrete spectrum differs from the population quadrature.", "The common Gaussian likelihood has additive observation noise; excess NLL is computed analytically, avoiding test-set noise.", "Six-seed bootstrap intervals cover training randomness in this model, not model-family uncertainty.", "Both methods receive real observations and generated negatives; this is not pure self-imitation or a direct LLM experiment."]}
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    rows = "".join(f"<tr><td>{LABELS[arm]}</td><td>{analysis['primary_late_slopes'][arm]:.4f}</td></tr>" for arm in ARMS)
    robust_rows = "".join("<tr>" + f"<td>{r['case']['beta']}</td><td>{r['case']['signal']}</td><td>{r['case']['shape']}</td><td>{r['case']['initial_gain']}</td>" + "".join(f"<td>{r['slopes'][arm]['100000..1e+07']:.4f}</td>" for arm in ARMS) + "</tr>" for r in population)
    finite_rows = "".join(f"<tr><td>{name}</td><td>{row['frozen_mean']:.5f}</td><td>{row['current_mean']:.5f}</td><td>{row['frozen_over_current_error_ratio']:.3f}</td><td>[{row['paired_difference_seed_bootstrap_ci95'][0]:.5f}, {row['paired_difference_seed_bootstrap_ci95'][1]:.5f}]</td></tr>" for name, row in pairs.items())
    limits = "".join(f"<li>{html.escape(item)}</li>" for item in analysis["limits"])
    report = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><title>Self-generation and spectral learning rates</title><style>body{{font-family:system-ui,sans-serif;max-width:75ch;margin:auto;padding:2em;line-height:1.5;color:#18202a}}img{{width:100%;height:auto}}table{{border-collapse:collapse;display:block;overflow-x:auto}}td,th{{border-bottom:1px solid #ccd;padding:.5em;text-align:left}}pre{{white-space:pre-wrap;background:#f3f5f8;padding:1em}}a{{color:#174da0}}.finding{{background:#edf5ff;padding:1em}}details{{margin:1em 0}}</style></head><body>
<h1>Does refreshing generated negatives change the learning exponent?</h1><p>Updated {analysis['created_at_uk']}. Local CPU experiments; no paid API or rented compute.</p>
<p class="finding"><strong>Feature learning changes the exponent; refreshing generated negatives improves error at a finite budget.</strong> In the main Gaussian setting, shallow models approach exponent 0.50 and two-layer models approach 0.67. Both frozen and refreshed sources share the exponent within each architecture.</p>
<h2>The comparison</h2><pre>x ~ Normal(0, Lambda)
y | x ~ Normal(w* · x, 1)
student: p(y | x) = Normal(f(x), 1)
shallow: f(x) = v · x
two layers: f(x) = a (v · x), both a and v trained
frozen negatives: q0(y | x) = Normal(0, 1)
current negatives: q_t = detached copy of current student</pre>
<p>Both arms learn the same normalized Gaussian conditional density by classifying real versus generated observations. Every update receives one real observation and one generated negative per input. Negative samples are labeled by their origin; no answer checker is used. All initial functions are zero and the two architectures have the same initial tangent kernel. Both source arms train the same parameters within an architecture.</p>
<h2>Population scaling</h2><p>The spectral error mass obeys dA(lambda) proportional to lambda^(beta−1) d lambda. The main case uses beta=0.5 and signal standard deviation 2. Curves integrate expected updates, with a common zero excess-error optimum. They are not records of millions of sampled SGD steps.</p><img src="self-contrastive-population.png" alt="Population errors under frozen and current negative sources for shallow and two-layer learners"><table><tr><th>Method</th><th>Exponent fitted over t=100,000 to 10,000,000</th></tr>{rows}</table>
<img src="self-contrastive-slopes.png" alt="Local exponents converge near 0.5 for shallow models and two thirds for two-layer models">
<h2>Actual sampled weight learning</h2><p>24 runs: six paired seeds for each of four methods. Each has 1,024 input coordinates, batch size 32, 8,192 SGD updates, and learning rate 0.05. Each run sees 262,144 real observations and generates 262,144 negatives. Input eigenvalues are k^−2 and initial error contributions are proportional to k^−2. All inputs and Gaussian noises are paired across methods. Full linear weights are trained. Evaluations run throughout training; all runs completed their budgets.</p><img src="self-contrastive-sgd.png" alt="Finite-sample SGD errors versus generated negative count with seed uncertainty bands"><table><tr><th>Architecture</th><th>Frozen error</th><th>Current error</th><th>Error ratio</th><th>95% paired CI for frozen minus current</th></tr>{finite_rows}</table><p>Errors are exact excess Gaussian negative log likelihood, computed from weights and the known teacher. Bands show 1.96 standard errors across six training seeds. Bootstrap intervals use 10,000 paired seed resamples. These short noisy curves are not claimed to independently establish the population exponents.</p>
<h2>Why the source has not changed the exponent</h2><p>For Gaussian inputs, the frozen-source Fisher matrix can be written as Lambda^(1/2) [h_perp (I−uuᵀ) + h_parallel uuᵀ] Lambda^(1/2), where u is the unit whitened target direction. Both coefficients are strictly positive at finite signal variance. In the main case they bound the Fisher matrix between {analysis['fisher_bounds_primary'][0]:.4f} Lambda and {analysis['fisher_bounds_primary'][1]:.4f} Lambda. Refreshing gives 0.5 Lambda. The frozen source weakens learning, but its local geometry retains the same slow spectral order.</p><p>The two-layer learner instead develops a growing effective kernel through its factor weights. This mechanism is available under both source choices. The hypothesis of equal source exponents and the slope windows were recorded in protocol.json before the main runs.</p>
<details><summary>All nine population settings</summary><table><tr><th>beta</th><th>Signal SD</th><th>Spectrum</th><th>Initial a</th><th>Frozen shallow</th><th>Current shallow</th><th>Frozen deep</th><th>Current deep</th></tr>{robust_rows}</table><p>beta=1.5 is an easy-task control and has exponent approximately 1.5 in every arm. beta=0.8 approaches 0.8 versus approximately 0.89; at weak signal the deep crossover is still incomplete. We also varied initial factor scale and smoothly perturbed the spectral mass.</p></details>
<details><summary>Numerical verification and reproducibility</summary><p>Doubling time-grid resolution, spectral quadrature, and Gaussian quadrature changes the main curves by at most {max(row['max_relative_curve_change_under_joint_refinement'] for row in numerical.values()):.2%}. The shallow current curve agrees with its exact modal solution within {exact_error:.2%}. An independent explicit RK4 integration of the original factor weights agrees with the reduced deep solver within {rk_error:.2%}. A finite-sample orthogonal-rotation update check passes to machine precision.</p><p>Source: experiments/self_contrastive_spectral.py. Analysis: experiments/analyze_self_contrastive_spectral.py. Evidence: results/self_contrastive_spectral/, including protocol, all curves, final SGD weights, and copied generating code.</p></details>
<h2>Scope</h2><ul>{limits}</ul><p>This experiment supports a coefficient benefit from adaptive self-generation in a canonical Gaussian family. It does not establish a generation-specific exponent advantage. Producing that would require an additional mechanism that changes the slow spectral order or its evolution.</p>
<p>Background: <a href="https://arxiv.org/html/2409.17858v2">feature learning and scaling</a>; <a href="https://arxiv.org/html/1412.6515">self-contrastive estimation</a>. <a href="self-generation-scaling.html">Earlier LLM scaling report</a>.</p></body></html>'''
    Path("reports/self-contrastive-scaling.html").write_text(report)
    print(json.dumps({"primary_late_slopes": analysis["primary_late_slopes"], "finite_final": pairs, "max_refinement_relative_error": max(row["max_relative_curve_change_under_joint_refinement"] for row in numerical.values()), "exact_shallow_relative_error": exact_error, "independent_deep_RK4_relative_error": rk_error}, indent=2))


if __name__ == "__main__": main()

