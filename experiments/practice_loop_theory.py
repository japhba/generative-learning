"""Finite-data negative-phase learning after exact tied-autoencoder convergence.

Three binary atom signs and one Gaussian hidden variable.  The main learner
uses an actual persistent block-Gibbs chain, not categorical oracle sampling.
Exact enumeration is restricted to diagnostics and the declared control.
"""

import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from tqdm.auto import trange


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(Path.home() / ".env")
OUT = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_loop_theory"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "tex" / "archive" / "figures"
STATES = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
EDGES = np.array([(0, 1), (0, 2), (1, 2)])
FEATURES = STATES[:, EDGES[:, 0]] * STATES[:, EDGES[:, 1]]
TRUE_H = np.array([1.2, 0.7, -0.4])
CONFIG = {"seeds": 32, "examples": 512, "chains": 128, "gibbs_sweeps": 4, "burn_in": 100, "learning_rate": 0.1, "maximum_steps": 3000, "evaluation_every": 10, "stationarity_minimum_steps": 400, "stationarity_points": 20, "stationarity_log_slope": -0.1, "seed_base": 2026101900, "true_h": TRUE_H.tolist(), "atom_pair_coupling": 6.0}


def distribution(h):
    logits = 0.5 * (h @ STATES.T) ** 2
    logq = logits - logsumexp(logits, axis=-1, keepdims=True)
    return logq, np.exp(logq)


def covariance(prob):
    return np.einsum("sk,ki,kj->sij", prob, STATES, STATES)


TRUE_LOGP, TRUE_P = distribution(TRUE_H[None])
TRUE_C = covariance(TRUE_P)[0]


def empirical_nll(h, freq):
    return -np.sum(freq * distribution(h)[0], axis=1)


def screening(h, freq):
    b = np.einsum("si,sj->sij", h, h)
    b[:, np.arange(3), np.arange(3)] = 0.0
    margin = np.einsum("sij,kj->ski", b, STATES) * STATES[None]
    factor = np.exp(-margin)
    loss = np.sum(freq * factor.mean(axis=2), axis=1)
    gradient_b = np.zeros_like(b)
    for i, j in EDGES:
        g = -np.sum(freq * FEATURES[:, np.flatnonzero(np.all(EDGES == [i, j], axis=1))[0]][None] * (factor[:, :, i] + factor[:, :, j]), axis=1) / 3
        gradient_b[:, i, j] = gradient_b[:, j, i] = g
    return loss, np.einsum("sij,sj->si", gradient_b, h)


def gibbs(h, signs, rng, sweeps):
    for _ in range(sweeps):
        hidden = np.einsum("sbi,si->sb", signs, h) + rng.normal(size=signs.shape[:2])
        signs = np.where(rng.random(signs.shape) < expit(2 * hidden[:, :, None] * h[:, None]), 1.0, -1.0)
    return signs


def metrics(h, freq, cdata, mle_nll):
    logq, _ = distribution(h)
    norm = np.linalg.norm(h, axis=1)
    cosine = np.abs(h @ TRUE_H) / (norm * np.linalg.norm(TRUE_H))
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    hch = np.einsum("si,sij,sj->s", h, cdata, h)
    reconstruction = 3 + (norm ** 2 - 2) * hch
    cross = np.einsum("si,sj->sij", h, h)
    margin = CONFIG["atom_pair_coupling"] + 0.5 * np.einsum("sij,kj->ski", cross, STATES) * STATES[None] - 0.5 * h[:, None] ** 2
    fixed_point_error = np.sum(freq * np.mean(margin <= 0, axis=2), axis=1)
    visible_conditional_ce = np.sum(freq * np.logaddexp(0, -2 * margin).mean(axis=2), axis=1)
    return np.stack([np.sum(TRUE_P * (TRUE_LOGP - logq), axis=1), angle, norm, reconstruction, empirical_nll(h, freq) - mle_nll, fixed_point_error, visible_conditional_ce], axis=1)


def transition_matrix(h, order=100):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    z = STATES @ h[:, None] + np.sqrt(2) * nodes[None]
    logits = 2 * z[:, :, None] * h[None, None]
    log_pplus = -np.logaddexp(0, -logits)
    log_pminus = -np.logaddexp(0, logits)
    target_bits = (STATES + 1) / 2
    log_transition = np.einsum("ani,ki->ank", log_pplus, target_bits) + np.einsum("ani,ki->ank", log_pminus, 1 - target_bits)
    return np.einsum("ank,n->ak", np.exp(log_transition), weights / np.sqrt(np.pi))


def mixing_audit(h):
    kernel = transition_matrix(h, 100)
    higher_order = transition_matrix(h, 200)
    _, prob = distribution(h[None])
    p = prob[0]
    sym = np.sqrt(p)[:, None] * kernel / np.sqrt(p)[None]
    even = np.stack([np.eye(8)[i] + np.eye(8)[7 - i] for i in range(4)], axis=1) / np.sqrt(2)
    eigen = np.linalg.eigvalsh((sym + sym.T) / 2)
    even_eigen = np.linalg.eigvalsh(even.T @ ((sym + sym.T) / 2) @ even)
    return {"quadrature_discrepancy": float(np.max(np.abs(kernel - higher_order))), "invariance_error": float(np.max(np.abs(p @ kernel - p))), "detailed_balance_error": float(np.max(np.abs(p[:, None] * kernel - p[None] * kernel.T))), "full_second_eigenvalue": float(eigen[-2]), "even_second_eigenvalue": float(even_eigen[-2]), "full_four_sweep_contraction": float(eigen[-2] ** CONFIG["gibbs_sweeps"]), "even_four_sweep_contraction": float(even_eigen[-2] ** CONFIG["gibbs_sweeps"]), "kernel": kernel.tolist()}


def main():
    rng = np.random.default_rng(740915)
    samples = np.stack([np.random.default_rng(CONFIG["seed_base"] + seed).choice(8, CONFIG["examples"], p=TRUE_P[0]) for seed in range(CONFIG["seeds"])])
    freq = np.stack([np.bincount(row, minlength=8) for row in samples]) / CONFIG["examples"]
    cdata = covariance(freq)
    _, eigenvectors = np.linalg.eigh(cdata)
    initial = eigenvectors[:, :, -1]
    initial *= np.sign(initial @ TRUE_H)[:, None]
    fitted_b, fitted_h, fitted_nll = [], [], []
    for empirical in freq:
        moment = empirical @ FEATURES

        def likelihood(b):
            logits = FEATURES @ b
            partition = logsumexp(logits)
            return partition - moment @ b, FEATURES.T @ np.exp(logits - partition) - moment

        result = minimize(likelihood, np.zeros(3), jac=True, method="BFGS", options={"gtol": 1e-10})
        assert np.linalg.norm(likelihood(result.x)[1]) < 1e-7
        assert np.prod(result.x) > 0
        h0 = np.sqrt(result.x[0] * result.x[1] / result.x[2])
        fitted_h.append([h0, result.x[0] / h0, result.x[1] / h0])
        fitted_b.append(result.x)
        fitted_nll.append(result.fun)
    fitted_h, fitted_b, fitted_nll = np.array(fitted_h), np.array(fitted_b), np.array(fitted_nll)
    models = {"gibbs": initial.copy(), "exact_negative_phase": initial.copy(), "interaction_screening": initial.copy(), "continued_autoencoder": initial.copy()}
    signs = rng.choice([-1.0, 1.0], size=(CONFIG["seeds"], CONFIG["chains"], 3))
    signs = gibbs(models["gibbs"], signs, rng, CONFIG["burn_in"])
    histories = {name: [metrics(h, freq, cdata, fitted_nll)] for name, h in models.items()}
    weights = {name: [h.copy()] for name, h in models.items()}
    steps = [0]
    stopped = {name: "maximum_steps" for name in models}
    active = {name: True for name in models}
    active["continued_autoencoder"] = False
    stopped["continued_autoencoder"] = "exact_empirical_global_optimum_at_initialization"
    original_ae_gradient = (2 - np.sum(initial ** 2, axis=1))[:, None] * np.einsum("sij,sj->si", cdata, initial) - np.einsum("si,sij,sj->s", initial, cdata, initial)[:, None] * initial
    assert np.max(np.abs(original_ae_gradient)) < 1e-13
    for step in trange(1, CONFIG["maximum_steps"] + 1, desc="negative-phase and reconstruction controls"):
        if active["gibbs"]:
            h = models["gibbs"]
            signs = gibbs(h, signs, rng, CONFIG["gibbs_sweeps"])
            negative = np.einsum("sbi,sbj,sj->si", signs, signs, h) / CONFIG["chains"]
            models["gibbs"] += CONFIG["learning_rate"] * (np.einsum("sij,sj->si", cdata, h) - negative)
        if active["exact_negative_phase"]:
            h = models["exact_negative_phase"]
            models["exact_negative_phase"] += CONFIG["learning_rate"] * np.einsum("sij,sj->si", cdata - covariance(distribution(h)[1]), h)
        if active["interaction_screening"]:
            models["interaction_screening"] -= CONFIG["learning_rate"] * screening(models["interaction_screening"], freq)[1]
        if step % CONFIG["evaluation_every"] == 0:
            steps.append(step)
            for name, h in models.items():
                current = metrics(h, freq, cdata, fitted_nll)
                assert np.all(np.isfinite(current))
                assert np.max(current[:, 5]) == 0
                histories[name].append(current)
                weights[name].append(h.copy())
                if active[name] and name == "gibbs" and step >= CONFIG["stationarity_minimum_steps"]:
                    window = CONFIG["stationarity_points"]
                    errors = np.array(histories[name])[-window:, :, 4].mean(axis=1)
                    slope = np.polyfit(np.log(np.array(steps[-window:])), np.log(errors), 1)[0]
                    if slope >= CONFIG["stationarity_log_slope"]:
                        active[name] = False
                        stopped[name] = {"reason": "empirical_likelihood_excess_stationary", "step": step, "log_log_slope": float(slope)}
                if active[name] and name == "exact_negative_phase" and current[:, 4].max() < 1e-7:
                    active[name] = False
                    stopped[name] = {"reason": "empirical_likelihood_converged", "step": step}
                if active[name] and name == "interaction_screening" and np.linalg.norm(screening(h, freq)[1], axis=1).max() < 1e-6:
                    active[name] = False
                    stopped[name] = {"reason": "empirical_screening_gradient_converged", "step": step}
            if not any(active.values()):
                break
    histories = {name: np.array(value) for name, value in histories.items()}
    weights = {name: np.array(value) for name, value in weights.items()}
    population_eigen, population_vectors = np.linalg.eigh(TRUE_C)
    population_ae = population_vectors[:, -1]
    population_ae *= np.sign(population_ae @ TRUE_H)
    audit = {"config": CONFIG, "environment": {"python_executable": sys.executable, "python_version": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}, "metric_columns": ["true_kl", "true_latent_angle_degrees", "hidden_weight_norm", "empirical_upper_ae_mse", "empirical_likelihood_excess_over_mle", "lower_atom_fixed_point_error", "finite_atom_one_site_conditional_ce"], "stopping": stopped, "initial_ae_gradient_max": float(np.max(np.abs(original_ae_gradient))), "population_ae_h": population_ae.tolist(), "population_ae_true_kl": float(np.sum(TRUE_P * (TRUE_LOGP - distribution(population_ae[None])[0]))), "population_ae_angle_degrees": float(np.degrees(np.arccos(abs(population_ae @ TRUE_H) / np.linalg.norm(TRUE_H)))), "true_covariance_eigenvalues": population_eigen.tolist(), "direct_mle_mean_metrics": metrics(fitted_h, freq, cdata, fitted_nll).mean(axis=0).tolist(), "final_mean_metrics": {name: value[-1].mean(axis=0).tolist() for name, value in histories.items()}, "final_sem_metrics": {name: (value[-1].std(axis=0, ddof=1) / np.sqrt(CONFIG["seeds"])).tolist() for name, value in histories.items()}, "mixing_at_population_ae": mixing_audit(population_ae), "mixing_at_true_h": mixing_audit(TRUE_H), "mixing_at_largest_fitted_norm": mixing_audit(fitted_h[np.argmax(np.linalg.norm(fitted_h, axis=1))])}
    for tag in ["mixing_at_population_ae", "mixing_at_true_h", "mixing_at_largest_fitted_norm"]:
        assert audit[tag]["invariance_error"] < 1e-7
        assert audit[tag]["quadrature_discrepancy"] < 1e-7
    np.savez(OUT / "curves_and_data.npz", samples=samples, empirical_frequencies=freq, empirical_covariances=cdata, steps=steps, true_h=TRUE_H, mle_h=fitted_h, mle_b=fitted_b, **{f"metrics_{name}": value for name, value in histories.items()}, **{f"weights_{name}": value for name, value in weights.items()})
    (OUT / "results.json").write_text(json.dumps(audit, indent=2))
    shutil.copy2(__file__, OUT / "source.py")
    labels = {"gibbs": "Actual Gibbs practice", "exact_negative_phase": "Exact negative phase", "interaction_screening": "Data-only screening", "continued_autoencoder": "Continue optimal autoencoder"}
    fig, axes = plt.subplots(1, 2, layout="constrained")
    for name, value in histories.items():
        mean, se = value.mean(axis=1), value.std(axis=1, ddof=1) / np.sqrt(CONFIG["seeds"])
        visible_curve = np.array(steps) <= (stopped[name]["step"] if isinstance(stopped[name], dict) else steps[-1])
        for ax, index in zip(axes, [0, 3]):
            ax.plot(np.array(steps)[visible_curve], mean[visible_curve, index], label=labels[name])
            ax.fill_between(np.array(steps)[visible_curve], mean[visible_curve, index] - se[visible_curve, index], mean[visible_curve, index] + se[visible_curve, index], alpha=0.15)
    axes[0].axhline(audit["direct_mle_mean_metrics"][0], color="black", linestyle=":", label="Direct finite-data MLE")
    axes[0].set(xlabel="Update", ylabel="True-density KL", yscale="log")
    axes[1].set(xlabel="Update", ylabel="Observed upper-layer reconstruction MSE")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.savefig(FIG / "practice-loop-negative-phase.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, layout="constrained")
    for name, value in histories.items():
        visible_curve = np.array(steps) <= (stopped[name]["step"] if isinstance(stopped[name], dict) else steps[-1])
        for ax, index in zip(axes, [1, 2]):
            mean, se = value[:, :, index].mean(axis=1), value[:, :, index].std(axis=1, ddof=1) / np.sqrt(CONFIG["seeds"])
            ax.plot(np.array(steps)[visible_curve], mean[visible_curve], label=labels[name])
            ax.fill_between(np.array(steps)[visible_curve], (mean - se)[visible_curve], (mean + se)[visible_curve], alpha=0.15)
    axes[0].set(xlabel="Update", ylabel="Angle to true hidden direction (degrees)")
    axes[1].set(xlabel="Update", ylabel="Hidden weight norm")
    axes[1].axhline(np.linalg.norm(TRUE_H), color="black", linestyle=":", label="True norm")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.savefig(FIG / "practice-loop-modes.png")
    plt.close(fig)
    print(json.dumps({"final_mean_metrics": audit["final_mean_metrics"], "stopping": stopped, "data": str(OUT / "results.json")}, indent=2))


if __name__ == "__main__":
    main()
