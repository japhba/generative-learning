"""Gaussian conditional NCE: source refresh crossed with linear feature learning."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ARMS = ("frozen_shallow", "current_shallow", "frozen_deep", "current_deep")
LABELS = {"frozen_shallow": "Frozen source, shallow", "current_shallow": "Current source, shallow", "frozen_deep": "Frozen source, two layers", "current_deep": "Current source, two layers"}


def gaussian_rule(order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return np.sqrt(2) * nodes, np.outer(weights, weights) / np.pi


def frozen_coefficients(w, target, lam, rule):
    """The positive NCE gradient is Lambda @ (c target - d w)."""
    variance = np.dot(lam * target, target)
    u, v = np.dot(lam * w, w), np.dot(lam * w, target)
    if u == 0: return 0.5, 0.5 + variance / 4
    nodes, weights = rule
    m = np.sqrt(u) * nodes[:, None]
    residual_variance = 1 + variance - v**2 / u
    assert residual_variance > 1 - 1e-10
    yp = v / np.sqrt(u) * nodes[:, None] + np.sqrt(residual_variance) * nodes[None, :]
    ym = m + nodes[None, :]
    wp = np.exp(-np.logaddexp(0, m * yp - m**2 / 2))
    wm = np.exp(-np.logaddexp(0, m * ym - m**2 / 2))
    sp, sm = yp - m, nodes[None, :]
    # Re-express the negative-source expectation under the student density.
    # This is exact integration, not the sampling procedure used in SGD below.
    c = np.sum(weights * (wp - m * wp * (1 - wp) * sp))
    d = np.sum(weights * (wp * (1 - wp) * sp**2 + wp - wm * (1 - wm) * sm * (sm + m)))
    return float(c), float(d)


def spectrum(beta, signal, modes, shape):
    edges = np.geomspace(1e-18, 1, modes + 1)
    lam = np.sqrt(edges[:-1] * edges[1:])
    mass = np.diff(edges**beta)
    if shape == "perturbed": mass += 0.5 * beta / (beta + 0.3) * np.diff(edges**(beta + 0.3))
    mass *= signal**2 / mass.sum()
    return lam, np.sqrt(mass / lam)


def metrics(w, target, lam, deep, initial_gain):
    error = target - w
    s = (initial_gain**2 + np.sqrt(initial_gain**4 + 4 * np.dot(w, w))) / 2 if deep else initial_gain**2
    denominator = np.dot(lam * error, lam * error)
    gain = s + np.dot(w, lam * error)**2 / (s * denominator) if deep else s
    return {"excess_nll": float(np.dot(lam * error, error) / 2), "readout_squared": float(s), "error_direction_kernel_gain": float(gain), "normalized_overlap": float(np.dot(lam * w, target) / np.dot(lam * target, target))}


def population_case(case, steps, modes, quadrature):
    lam, target = spectrum(case["beta"], case["signal"], modes, case["shape"])
    rule = gaussian_rule(quadrature)
    times = np.concatenate(([0.0], np.geomspace(1e-3, 1e7, steps)))
    capture = set(np.unique(np.linspace(0, len(times) - 1, 201, dtype=int)))
    weights = {arm: np.zeros(modes) for arm in ARMS}
    records = []
    for step, time in enumerate(times):
        if step in capture:
            records.append({"time": float(time), "arms": {arm: metrics(weights[arm], target, lam, arm.endswith("deep"), case["initial_gain"]) for arm in ARMS}})
        if step == len(times) - 1: break
        dt = times[step + 1] - time
        for arm in ARMS:
            w = weights[arm]
            c, d = (0.5, 0.5) if arm.startswith("current") else frozen_coefficients(w, target, lam, rule)
            assert d > 0 and c > 0
            gradient = lam * (c * target - d * w)
            deep = arm.endswith("deep")
            s = (case["initial_gain"]**2 + np.sqrt(case["initial_gain"]**4 + 4 * np.dot(w, w))) / 2 if deep else case["initial_gain"]**2
            r = np.dot(w, gradient) / s if deep else 0.0
            # IMEX Euler: implicit diagonal decay, explicit evolving coefficients.
            weights[arm] = (w + dt * (s * c * lam * target + r * w)) / (1 + dt * s * d * lam)
            assert np.isfinite(weights[arm]).all()
    slopes = {}
    for arm in ARMS:
        slopes[arm] = {}
        for lower, upper in ((1e2, 1e4), (1e4, 1e6), (1e5, 1e7)):
            rows = [r for r in records if lower <= r["time"] <= upper]
            slopes[arm][f"{lower:g}..{upper:g}"] = float(-np.polyfit(np.log([r["time"] for r in rows]), np.log([r["arms"][arm]["excess_nll"] for r in rows]), 1)[0])
    # Fisher metric at the target: common to all dimensions, no eigendecomposition.
    nodes, grid = rule
    m = case["signal"] * nodes[:, None]
    y = m + nodes[None, :]
    weight = np.exp(-np.logaddexp(0, m * y - m**2 / 2))
    h0 = float(np.sum(grid * weight * nodes[None, :]**2))
    h_parallel = float(np.sum(grid * weight * nodes[None, :]**2 * nodes[:, None]**2))
    return {"case": case, "steps": steps, "modes": modes, "quadrature": quadrature, "records": records, "slopes": slopes, "frozen_fisher_bounds_relative_to_covariance": [min(h0, h_parallel), max(h0, h_parallel)]}


def plot_population(results, destination):
    result, = [r for r in results if r["case"] == {"beta": 0.5, "signal": 2, "shape": "power", "initial_gain": 1}]
    records = [r for r in result["records"] if r["time"] >= 1]
    t = np.array([r["time"] for r in records])
    fig, ax = plt.subplots(layout="constrained")
    for arm in ARMS:
        loss = np.array([r["arms"][arm]["excess_nll"] for r in records])
        color = "C0" if arm.endswith("shallow") else "C1"
        style = "--" if arm.startswith("frozen") else "-"
        slope = result["slopes"][arm]["100000..1e+07"]
        ax.loglog(t, loss, color=color, linestyle=style, label=f"{LABELS[arm]} ({slope:.3f})")
    ax.set(title="Population dynamics: source refresh and feature learning", xlabel="Population gradient time (not sampled SGD steps)", ylabel="Excess Gaussian NLL")
    ax.legend()
    fig.savefig(destination)
    plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    for arm in ARMS:
        loss = np.array([r["arms"][arm]["excess_nll"] for r in records])
        ax.semilogx(np.sqrt(t[:-1] * t[1:]), -np.diff(np.log(loss)) / np.diff(np.log(t)), color="C0" if arm.endswith("shallow") else "C1", linestyle="--" if arm.startswith("frozen") else "-", label=LABELS[arm])
    ax.axhline(0.5, color="C0", linestyle=":", label="Shallow prediction: 0.5")
    ax.axhline(2/3, color="C1", linestyle=":", label="Two-layer prediction: 2/3")
    ax.set(xlabel="Population gradient time", ylabel="Local error-decay exponent", title="Both sources approach the same exponent within each architecture", ylim=(0, 0.9))
    ax.legend()
    fig.savefig(destination.with_name("self-contrastive-slopes.png"))
    plt.close(fig)


def finite_training(config, output):
    rng = np.random.default_rng(config["seed"])
    dimension, repetitions, batch = config["dimension"], config["repetitions"], config["batch"]
    indices = np.arange(1, dimension + 1, dtype=float)
    lam = indices**(-config["alpha"])
    mass = indices**(-config["alpha"] * config["beta"] - 1)
    mass *= config["signal"]**2 / mass.sum()
    target = np.sqrt(mass / lam)
    vectors = {arm: np.zeros((repetitions, dimension)) for arm in ARMS}
    readouts = {arm: np.ones(repetitions) for arm in ARMS}
    checkpoints = set([0, 1, 2, 4, 8, 16, 32, 64, 128, 256] + list(range(512, config["updates"] + 1, 256)))
    records = []
    stop_reason = "budget_completed"
    for step in tqdm(range(config["updates"] + 1), desc="Finite paired Gaussian SGD"):
        if step in checkpoints:
            arms = {}
            for arm in ARMS:
                effective = readouts[arm][:, None] * vectors[arm]
                error = target - effective
                losses = np.sum(lam * error**2, axis=1) / 2
                assert np.isfinite(losses).all()
                assert losses.max() < config["signal"]**2
                arms[arm] = {"excess_nll": losses.tolist(), "readout_squared": (readouts[arm]**2).tolist(), "normalized_overlap": (np.sum(lam * effective * target, axis=1) / config["signal"]**2).tolist()}
            records.append({"step": step, "real_samples_per_seed": step * batch, "generated_negatives_per_seed": step * batch, "arms": arms})
            (output / "finite_training.json").write_text(json.dumps({"config": config, "records": records, "stop_reason": "running"}, indent=2) + "\n")
            if step >= 2048:
                recent = records[-5:]
                slopes = [np.polyfit(np.log([r["step"] for r in recent]), np.log([np.mean(r["arms"][arm]["excess_nll"]) for r in recent]), 1)[0] for arm in ARMS]
                if min(slopes) > -0.02:
                    stop_reason = "all_arms_stationary"
                    break
        if step == config["updates"]: break
        x = rng.normal(size=(repetitions, batch, dimension)) * np.sqrt(lam)
        real = np.einsum("rbd,d->rb", x, target) + rng.normal(size=(repetitions, batch))
        negative_noise = rng.normal(size=(repetitions, batch))
        for arm in ARMS:
            v, a = vectors[arm], readouts[arm]
            hidden = np.einsum("rbd,rd->rb", x, v)
            mean = a[:, None] * hidden
            source_mean = mean if arm.startswith("current") else np.zeros_like(mean)
            negative = source_mean + negative_noise
            logit_real = ((real - source_mean)**2 - (real - mean)**2) / 2
            logit_negative = ((negative - source_mean)**2 - (negative - mean)**2) / 2
            weight_real = np.exp(-np.logaddexp(0, logit_real))
            discriminator_negative = np.exp(-np.logaddexp(0, -logit_negative))
            residual = -weight_real * (real - mean) + discriminator_negative * (negative - mean)
            grad_v = a[:, None] * np.einsum("rb,rbd->rd", residual, x) / batch
            grad_a = np.mean(residual * hidden, axis=1)
            vectors[arm] -= config["learning_rate"] * grad_v
            if arm.endswith("deep"): readouts[arm] -= config["learning_rate"] * grad_a
    result = {"config": config, "records": records, "stop_reason": stop_reason}
    (output / "finite_training.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(output / "finite_final_weights.npz", **{f"{arm}_vector": vectors[arm] for arm in ARMS}, **{f"{arm}_readout": readouts[arm] for arm in ARMS}, eigenvalues=lam, target=target)
    return result


def plot_finite(result, destination):
    fig, ax = plt.subplots(layout="constrained")
    records = [r for r in result["records"] if r["step"] >= 16]
    t = np.array([r["generated_negatives_per_seed"] for r in records])
    for arm in ARMS:
        loss = np.array([r["arms"][arm]["excess_nll"] for r in records])
        color = "C0" if arm.endswith("shallow") else "C1"
        style = "--" if arm.startswith("frozen") else "-"
        ax.loglog(t, loss.mean(axis=1), color=color, linestyle=style, label=LABELS[arm])
        se = loss.std(axis=1, ddof=1) / np.sqrt(loss.shape[1])
        ax.fill_between(t, loss.mean(axis=1) - 1.96 * se, loss.mean(axis=1) + 1.96 * se, color=color, alpha=0.15)
    ax.set(xlabel="Generated negatives per seed (equal number of real samples)", ylabel="Exact excess Gaussian NLL", title="Actual SGD: source refresh versus feature learning")
    ax.legend()
    fig.savefig(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/self_contrastive_spectral"))
    parser.add_argument("--population-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [{"beta": beta, "signal": signal, "shape": "power", "initial_gain": 1} for beta in (0.5, 0.8, 1.5) for signal in (0.5, 2)]
    cases += [{"beta": 0.5, "signal": 2, "shape": shape, "initial_gain": gain} for shape, gain in (("perturbed", 1), ("power", 0.5), ("power", 2))]
    finite = {"seed": 840921, "dimension": 1024, "repetitions": 6, "batch": 32, "updates": 8192, "learning_rate": 0.05, "alpha": 2, "beta": 0.5, "signal": 2}
    protocol = {"created_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "population_cases": cases, "population_steps": 6000, "spectral_quadrature_modes": 1024, "gaussian_quadrature": 32, "finite": finite, "predictions": {"source_refresh": "Same asymptotic time exponents at finite signal variance; coefficients and transients can differ.", "shallow": "beta", "deep": "2 beta / (1+beta) for beta<1; beta for beta>1", "reason": "The frozen-source Fisher metric is bounded above and below by positive constants times the input covariance. The deep model's weight growth is available to both source arms."}, "slope_windows_fixed_before_run": [[100, 10000], [10000, 1000000], [100000, 10000000]], "population_axis": "Numerical integration of expected updates. NOT generated-sample counts or actual SGD steps.", "finite_axis": "One real conditional observation and one source negative per input. Same batch, learning rate, trainable architecture, and source refresh interval within each pair. Both data and negative source are sampled online.", "limits": "This tests adaptive negative generation in anchored density learning, not unfiltered self-imitation, latent wake-sleep, or LLM solution rollouts."}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    results = []
    for case in tqdm(cases, desc="Population spectral cases"):
        result = population_case(case, 6000, 1024, 32)
        results.append(result)
        (args.output / "population.json").write_text(json.dumps(results, indent=2) + "\n")
        tqdm.write(json.dumps({"case": case, "late_slopes": {arm: result["slopes"][arm]["100000..1e+07"] for arm in ARMS}}))
    reference = population_case(cases[1], 12000, 2048, 64)
    (args.output / "refinement.json").write_text(json.dumps(reference, indent=2) + "\n")
    plot_population(results, Path("reports/self-contrastive-population.png"))
    if not args.population_only:
        finite_result = finite_training(finite, args.output)
        plot_finite(finite_result, Path("reports/self-contrastive-sgd.png"))
    print(str(args.output))


if __name__ == "__main__": main()
