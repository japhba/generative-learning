"""Aligned Curie--Weiss proposal-coverage control with actual block Gibbs trials.

Only a finite-data pair correlation is available during learning.  The known
magnetization reduction supplies exact diagnostics and an explicit exact control.
It is not used to replace the own-generator transition.  Fixed proposal sampling
is deliberately optimistic: its magnetizations are independent exact draws.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


TRUE_GAIN = 1.25
INITIAL_GAIN = 1.0
DIMENSIONS = [16, 64, 256, 1024]
CONFIG = {"dimensions": DIMENSIONS, "examples": 256, "own_chains": 128, "gibbs_sweeps": 4, "burn_in_sweeps": 256, "fixed_candidates": 512, "learning_rate": 0.05, "minimum_gain": 0.2, "maximum_gain": 2.0, "maximum_updates": 400, "evaluate_every": 5, "stationarity_minimum_updates": 100, "stationarity_evaluations": 12, "stationarity_log_slope": -0.1, "exact_moment_tolerance": 1e-12, "true_gain": TRUE_GAIN, "initial_gain": INITIAL_GAIN}
METHODS = ["own_gibbs", "exact_moment", "uniform_snis", "frozen_initial_snis", "data_two_mode_snis"]


def count_model(d):
    plus = np.arange(d + 1)
    magnetization = 2 * plus - d
    log_multiplicity = np.array([math.lgamma(d + 1) - math.lgamma(int(k) + 1) - math.lgamma(d - int(k) + 1) for k in plus])
    correlation = (magnetization ** 2 - d) / (d * (d - 1))
    return magnetization, log_multiplicity, correlation


def distribution(gain, d, magnetization, log_multiplicity):
    logits = log_multiplicity[None] + gain[:, None] ** 2 * magnetization[None] ** 2 / (2 * d)
    maximum = logits.max(axis=1, keepdims=True)
    logp = logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    return logp, np.exp(logp)


def conditional_probability(hidden, gain, d):
    logits = 2 * gain[:, None] * hidden / np.sqrt(d)
    return (1 + np.tanh(logits / 2)) / 2


def gibbs(gain, magnetization, d, rng, sweeps):
    for _ in range(sweeps):
        hidden = gain[:, None] * magnetization / np.sqrt(d) + rng.normal(size=magnetization.shape)
        magnetization = 2 * rng.binomial(d, conditional_probability(hidden, gain, d)) - d
    return magnetization


def trial_moment(gain, trial_magnetization, d, reference_gain):
    log_weights = (gain[:, None] ** 2 - reference_gain ** 2) * trial_magnetization ** 2 / (2 * d)
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    sample_correlation = (trial_magnetization ** 2 - d) / (d * (d - 1))
    return np.sum(weights * sample_correlation, axis=1), 1 / np.sum(weights ** 2, axis=1)


def fitted_gain(observed, d, magnetization, log_multiplicity, correlation):
    lower = np.full_like(observed, CONFIG["minimum_gain"])
    upper = np.full_like(observed, CONFIG["maximum_gain"])
    for _ in range(80):
        middle = (lower + upper) / 2
        model_correlation = distribution(middle, d, magnetization, log_multiplicity)[1] @ correlation
        lower = np.where(model_correlation < observed, middle, lower)
        upper = np.where(model_correlation >= observed, middle, upper)
    return (lower + upper) / 2


def run_dimension(d, seeds, seed_base):
    rng = np.random.default_rng(seed_base + 10000 + d)
    magnetization, log_multiplicity, correlation = count_model(d)
    true_logp, true_prob = distribution(np.array([TRUE_GAIN]), d, magnetization, log_multiplicity)
    initial_logp, initial_prob = distribution(np.array([INITIAL_GAIN]), d, magnetization, log_multiplicity)
    observed_indices = np.stack([np.random.default_rng(seed_base + 2000 * d + seed).choice(d + 1, CONFIG["examples"], p=true_prob[0]) for seed in range(seeds)])
    observed = correlation[observed_indices].mean(axis=1)
    proposal_probability = (1 + np.sqrt(np.maximum(observed, 0))) / 2
    mle = fitted_gain(observed, d, magnetization, log_multiplicity, correlation)
    assert np.min(mle) > CONFIG["minimum_gain"] and np.max(mle) < CONFIG["maximum_gain"]
    gain = {name: np.full(seeds, INITIAL_GAIN) for name in METHODS}
    alive = {name: np.ones(seeds, dtype=bool) for name in METHODS}
    cap_step = {name: np.full(seeds, -1, dtype=int) for name in METHODS}
    stopping = {name: {"step": CONFIG["maximum_updates"], "reason": "budget"} for name in METHODS}
    trial_max = {name: np.zeros(seeds) for name in METHODS}
    ordered_draws = {name: np.zeros(seeds, dtype=int) for name in METHODS}
    ess = {name: np.zeros(seeds) for name in METHODS}
    initial_cdf = np.cumsum(initial_prob[0])
    initial_cdf[-1] = 1.0
    own_states = 2 * rng.binomial(d, 0.5, size=(seeds, CONFIG["own_chains"])) - d
    own_states = gibbs(gain["own_gibbs"], own_states, d, rng, CONFIG["burn_in_sweeps"])
    history = {name: [] for name in METHODS}
    parameters = {name: [] for name in METHODS}
    steps = []
    for step in range(CONFIG["maximum_updates"] + 1):
        if step % CONFIG["evaluate_every"] == 0:
            steps.append(step)
            for name in METHODS:
                logp, prob = distribution(gain[name], d, magnetization, log_multiplicity)
                model_c = prob @ correlation
                true_kl = np.sum(true_prob * (true_logp - logp), axis=1)
                true_c = true_prob[0] @ correlation
                empirical_excess = true_kl + 0.5 * (d - 1) * (gain[name] ** 2 - TRUE_GAIN ** 2) * (true_c - observed)
                moment_error = (model_c - observed) ** 2
                history[name].append(np.stack([true_kl, model_c, moment_error, ess[name], trial_max[name], ordered_draws[name], empirical_excess], axis=1))
                parameters[name].append(gain[name].copy())
                assert np.all(np.isfinite(history[name][-1]))
                if np.any(alive[name]) and name == "exact_moment" and np.max(moment_error) < CONFIG["exact_moment_tolerance"]:
                    alive[name][:] = False
                    stopping[name] = {"step": step, "reason": "empirical_moment_converged"}
                if np.any(alive[name]) and name != "exact_moment" and step >= CONFIG["stationarity_minimum_updates"]:
                    length = CONFIG["stationarity_evaluations"]
                    recent = np.array(history[name])[-length:, alive[name], 2].mean(axis=1)
                    slope = np.polyfit(np.log(np.array(steps[-length:])), np.log(recent), 1)[0]
                    if slope >= CONFIG["stationarity_log_slope"]:
                        alive[name][:] = False
                        stopping[name] = {"step": step, "reason": "empirical_moment_error_stationary", "log_log_slope": float(slope)}
            if not any(np.any(mask) for mask in alive.values()):
                break
        if step == CONFIG["maximum_updates"]:
            break
        for name in METHODS:
            if not np.any(alive[name]):
                continue
            a = gain[name]
            if name == "own_gibbs":
                own_states = gibbs(a, own_states, d, rng, CONFIG["gibbs_sweeps"])
                trial = own_states
                estimated = ((trial ** 2 - d) / (d * (d - 1))).mean(axis=1)
                ess[name][:] = CONFIG["own_chains"]
            elif name == "exact_moment":
                estimated = distribution(a, d, magnetization, log_multiplicity)[1] @ correlation
                trial = np.zeros((seeds, 1))
            elif name == "uniform_snis":
                trial = 2 * rng.binomial(d, 0.5, size=(seeds, CONFIG["fixed_candidates"])) - d
                estimated, current_ess = trial_moment(a, trial, d, 0.0)
                ess[name] = np.where(alive[name], current_ess, ess[name])
            elif name == "frozen_initial_snis":
                trial = magnetization[np.searchsorted(initial_cdf, rng.random((seeds, CONFIG["fixed_candidates"]))) ]
                estimated, current_ess = trial_moment(a, trial, d, INITIAL_GAIN)
                ess[name] = np.where(alive[name], current_ess, ess[name])
            else:
                sign = rng.choice([-1, 1], size=(seeds, CONFIG["fixed_candidates"]))
                trial = sign * (2 * rng.binomial(d, proposal_probability[:, None], size=sign.shape) - d)
                positive_count = (trial + d) / 2
                log_plus = positive_count * np.log(proposal_probability[:, None]) + (d - positive_count) * np.log1p(-proposal_probability[:, None])
                log_minus = (d - positive_count) * np.log(proposal_probability[:, None]) + positive_count * np.log1p(-proposal_probability[:, None])
                log_proposal = np.logaddexp(log_plus, log_minus) - np.log(2)
                log_weights = a[:, None] ** 2 * trial ** 2 / (2 * d) - log_proposal
                normalized = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
                normalized /= normalized.sum(axis=1, keepdims=True)
                estimated = np.sum(normalized * (trial ** 2 - d) / (d * (d - 1)), axis=1)
                ess[name] = np.where(alive[name], 1 / np.sum(normalized ** 2, axis=1), ess[name])
            if name != "exact_moment":
                trial_max[name] = np.where(alive[name], np.maximum(trial_max[name], np.max(np.abs(trial) / d, axis=1)), trial_max[name])
                ordered_draws[name] += alive[name] * np.sum(np.abs(trial) / d >= 0.8, axis=1)
            proposal = a + CONFIG["learning_rate"] * a * (observed - estimated)
            hit = alive[name] & ((proposal <= CONFIG["minimum_gain"]) | (proposal >= CONFIG["maximum_gain"]))
            gain[name] = np.where(alive[name], np.clip(proposal, CONFIG["minimum_gain"], CONFIG["maximum_gain"]), a)
            cap_step[name][hit] = step + 1
            alive[name][hit] = False
            if np.all(cap_step[name] >= 0):
                stopping[name] = {"step": step + 1, "reason": "all_runs_hit_predeclared_gain_boundary"}
    history = {name: np.array(values) for name, values in history.items()}
    parameters = {name: np.array(values) for name, values in parameters.items()}
    threshold = np.abs(magnetization) / d >= 0.8
    uniform_logp, uniform_prob = distribution(np.array([0.0]), d, magnetization, log_multiplicity)
    tail_uniform, tail_frozen = float(uniform_prob[0, threshold].sum()), float(initial_prob[0, threshold].sum())
    opportunity_budget = CONFIG["fixed_candidates"] * CONFIG["maximum_updates"]
    info = {"dimension": d, "true_correlation": float(true_prob[0] @ correlation), "finite_data_correlation_mean": float(observed.mean()), "mle_gain_mean": float(mle.mean()), "initial_true_kl": float(history["own_gibbs"][0, :, 0].mean()), "final_gain_mean": {name: float(values[-1].mean()) for name, values in parameters.items()}, "final_true_kl_mean": {name: float(values[-1, :, 0].mean()) for name, values in history.items()}, "final_true_kl_sem": {name: float(values[-1, :, 0].std(ddof=1) / np.sqrt(seeds)) for name, values in history.items()}, "gain_boundary_hits": {name: int(np.sum(cap_step[name] >= 0)) for name in METHODS}, "cap_steps": {name: value.tolist() for name, value in cap_step.items()}, "stopping": stopping, "ordered_region_abs_m_over_d_at_least": 0.8, "ordered_region_true_mass": float(true_prob[0, threshold].sum()), "ordered_region_uniform_mass": tail_uniform, "ordered_region_frozen_mass": tail_frozen, "ordered_region_uniform_budget_expected_draws": opportunity_budget * tail_uniform, "ordered_region_frozen_budget_expected_draws": opportunity_budget * tail_frozen, "final_mean_importance_ess": {name: float(history[name][-1, :, 3].mean()) for name in ["uniform_snis", "frozen_initial_snis", "data_two_mode_snis"]}, "final_mean_max_seen_abs_m_over_d": {name: float(history[name][-1, :, 4].mean()) for name in ["own_gibbs", "uniform_snis", "frozen_initial_snis", "data_two_mode_snis"]}, "final_mean_ordered_draws": {name: float(history[name][-1, :, 5].mean()) for name in ["own_gibbs", "uniform_snis", "frozen_initial_snis", "data_two_mode_snis"]}}
    info["own_final_positive_chain_fraction_mean"] = float(np.mean((own_states > 0) + 0.5 * (own_states == 0)))
    arrays = {"steps": np.array(steps), "data_magnetizations": magnetization[observed_indices], "observed_correlation": observed, "mle_gain": mle, "own_final_magnetizations": own_states, **{f"curve_{name}": value for name, value in history.items()}, **{f"gain_{name}": value for name, value in parameters.items()}}
    return info, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    root = Path(__file__).resolve().parents[1]
    mode = "pilot" if args.pilot else "confirmation"
    seeds, seed_base = (4, 71100) if args.pilot else (32, 72100)
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_loop_coverage" / mode
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, output / "source.py")
    summary = {"uk_created": datetime.now(ZoneInfo("Europe/London")).isoformat(), "mode": mode, "seeds": seeds, "seed_base": seed_base, "config": CONFIG, "metric_columns": ["true_kl", "exact_model_correlation", "squared_empirical_moment_error", "nominal_importance_ess", "maximum_abs_magnetization_fraction_seen", "cumulative_ordered_draws", "empirical_nll_relative_to_true_parameter"], "dimensions": [], "scope": "Aligned gain and proposal coverage only. Exact magnetization integration is available and beats sampling in this artificial symmetry-reduced toy. Own Gibbs uses128persistent chains times4sweeps; fixed controls use512independent candidates, not matched FLOPs. SNIS is biased; persistent chains are not exactly stationary under changing weights. True samples are used only at initial acquisition. Updates retain only finite-data correlation."}
    trajectories = {}
    for d in tqdm(DIMENSIONS, desc=f"Curie-Weiss {mode}"):
        info, arrays = run_dimension(d, seeds, seed_base)
        summary["dimensions"].append(info)
        trajectories[d] = arrays
        np.savez_compressed(output / f"dimension_{d}.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.pilot:
        labels = {"own_gibbs": "Adaptive own Gibbs", "exact_moment": "Exact model moment", "uniform_snis": "Uniform proposals + SNIS", "frozen_initial_snis": "Frozen initial proposals + SNIS", "data_two_mode_snis": "Data-derived two-mode proposals"}
        for metric, filename, ylabel in [(0, "practice-loop-coverage.png", "True-density KL"), (None, "practice-loop-coverage-gains.png", "Gain")]:
            fig, axes = plt.subplots(2, 2, layout="constrained")
            for ax, d, info in zip(axes.flat, DIMENSIONS, summary["dimensions"]):
                arrays = trajectories[d]
                for name in METHODS:
                    stop = info["stopping"][name]["step"]
                    keep = arrays["steps"] <= CONFIG["evaluate_every"] * math.ceil(stop / CONFIG["evaluate_every"])
                    values = arrays[f"gain_{name}"] if metric is None else arrays[f"curve_{name}"][:, :, metric]
                    mean = values.mean(axis=1)
                    line, = ax.plot(arrays["steps"][keep], mean[keep], label=labels[name])
                    lo, hi = np.quantile(values, [0.1, 0.9], axis=1)
                    ax.fill_between(arrays["steps"][keep], lo[keep], hi[keep], color=line.get_color(), alpha=0.15)
                ax.set(title=f"{d} atoms", xlabel="Update", ylabel=ylabel)
                if metric is None:
                    ax.axhline(TRUE_GAIN, color="black", linestyle=":")
                else:
                    ax.set_yscale("log")
            fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
            fig.savefig(root / "tex" / "archive" / "figures" / filename)
            plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
