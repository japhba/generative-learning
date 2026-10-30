"""Five-atom memory-restricted extension and exact reconstruction-access counterexample."""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import trange


STATES = np.array(list(product([-1.0, 1.0], repeat=5)))


def distribution(h):
    logits = 0.5 * (h @ STATES.T) ** 2
    maximum = logits.max(axis=1, keepdims=True)
    logp = logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    return logp, np.exp(logp)


def covariance(p):
    return np.einsum("sk,ki,kj->sij", p, STATES, STATES)


def masked_gradient(h, freq):
    j = h[:, :, None] * h[:, None, :]
    j[:, np.arange(5), np.arange(5)] = 0
    fields = np.einsum("ki,sij->skj", STATES, j)
    error = np.tanh(fields) - STATES[None]
    derivative = np.einsum("sk,ski,kj->sij", freq, error, STATES)
    derivative += derivative.transpose(0, 2, 1).copy()
    derivative[:, np.arange(5), np.arange(5)] = 0
    return np.einsum("sij,sj->si", derivative, h)


def gibbs(h, signs, rng, sweeps):
    for _ in range(sweeps):
        hidden = np.einsum("sbi,si->sb", signs, h) + rng.normal(size=signs.shape[:2])
        logits = 2 * hidden[:, :, None] * h[:, None]
        signs = np.where(rng.random(signs.shape) < 1 / (1 + np.exp(-logits)), 1.0, -1.0)
    return signs


def main():
    load_dotenv(Path.home() / ".env")
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_loop_memory"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out / "run_source.py")
    rng = np.random.default_rng(904381)
    target = np.array([1.0, 0.65, -0.4, 0.35, -0.55])
    seeds, n, steps, batch, sweeps, rate = 32, 512, 1500, 128, 4, 0.05
    true_logp, truth = distribution(target[None])
    samples = np.stack([np.random.default_rng(80500 + seed).choice(32, n, p=truth[0]) for seed in range(seeds)])
    freq = np.stack([np.bincount(row, minlength=32) for row in samples]) / n
    memory = covariance(freq)
    _, directions = np.linalg.eigh(memory)
    initial = directions[:, :, -1]
    initial *= np.sign(initial @ target)[:, None]
    names = ["gibbs_memory", "exact_memory", "masked_raw_replay", "uniform_proposal_memory", "linear_ae"]
    hs = {name: initial.copy() for name in names}
    signs = rng.choice([-1.0, 1.0], size=(seeds, batch, 5))
    signs = gibbs(hs["gibbs_memory"], signs, rng, 100)
    saved = np.arange(0, steps + 1, 10)
    curves = {name: np.zeros((len(saved), seeds, 5)) for name in names}
    weights = {name: np.zeros((len(saved), seeds, 5)) for name in names}
    active = {name: name != "linear_ae" for name in names}
    stopping = {name: {"step": steps, "reason": "budget"} for name in names}
    stopping["linear_ae"] = {"step": 0, "reason": "exact_reconstruction_optimum"}
    for step in trange(steps + 1, desc="Five-atom memory access"):
        if step % 10 == 0:
            ci = step // 10
            for name, h in hs.items():
                logp, _ = distribution(h)
                kl = np.sum(truth * (true_logp - logp), axis=1)
                logits = 0.5 * (h @ STATES.T) ** 2
                maximum = logits.max(axis=1)
                log_partition = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
                empirical = log_partition - 0.5 * np.einsum("si,sij,sj->s", h, memory, h)
                angle = np.degrees(np.arccos(np.clip(np.abs(h @ target) / np.linalg.norm(h, axis=1) / np.linalg.norm(target), 0, 1)))
                hc = np.einsum("si,sij,sj->s", h, memory, h)
                reconstruction = 5 + (np.sum(h ** 2, axis=1) - 2) * hc
                curves[name][ci] = np.stack([kl, angle, np.linalg.norm(h, axis=1), reconstruction, empirical], axis=1)
                weights[name][ci] = h
                assert np.all(np.isfinite(curves[name][ci]))
                if active[name] and ci >= 50:
                    window = curves[name][ci - 19:ci + 1, :, 4].mean(axis=1)
                    slope = np.polyfit(np.log(saved[ci - 19:ci + 1]), np.log(window), 1)[0]
                    if abs(slope) < 1e-5:
                        active[name] = False
                        stopping[name] = {"step": step, "reason": "empirical_nll_log_slope_stationary", "slope": float(slope)}
        if step == steps:
            break
        if active["gibbs_memory"]:
            h = hs["gibbs_memory"]
            signs = gibbs(h, signs, rng, sweeps)
            negative = np.einsum("sbi,sbj,sj->si", signs, signs, h) / batch
            h += rate * (np.einsum("sij,sj->si", memory, h) - negative)
        if active["exact_memory"]:
            h = hs["exact_memory"]
            h += rate * np.einsum("sij,sj->si", memory - covariance(distribution(h)[1]), h)
        if active["masked_raw_replay"]:
            hs["masked_raw_replay"] -= rate * masked_gradient(hs["masked_raw_replay"], freq)
        if active["uniform_proposal_memory"]:
            h = hs["uniform_proposal_memory"]
            proposal = rng.choice([-1.0, 1.0], size=(seeds, batch * sweeps, 5))
            logits = 0.5 * np.einsum("sbi,si->sb", proposal, h) ** 2
            importance = np.exp(logits - logits.max(axis=1, keepdims=True))
            importance /= importance.sum(axis=1, keepdims=True)
            negative = np.einsum("sb,sbi,sbj,sj->si", importance, proposal, proposal, h)
            h += rate * (np.einsum("sij,sj->si", memory, h) - negative)
        for h in hs.values():
            assert np.max(np.linalg.norm(h, axis=1)) < 10
    chi = np.prod(STATES[:, :4], axis=1)
    paired_laws = np.stack([(1 + 0.5 * chi) / 32, (1 - 0.5 * chi) / 32])
    assert np.allclose(paired_laws * 64, np.round(paired_laws * 64))
    paired_memory = covariance(paired_laws)
    pair_h = np.tile(target, (2, 1))
    reconstruction_gradients = masked_gradient(pair_h, paired_laws)
    _, candidate_prob = distribution(pair_h)
    likelihood_gradients = np.einsum("sij,sj->si", covariance(candidate_prob) - paired_memory, pair_h)
    assert np.max(np.abs(paired_memory[0] - paired_memory[1])) == 0
    assert np.max(np.abs(likelihood_gradients[0] - likelihood_gradients[1])) == 0
    assert np.linalg.norm(reconstruction_gradients[0] - reconstruction_gradients[1]) > 0.01
    np.savez_compressed(out / "trajectories.npz", steps=saved, samples=samples, memory=memory, paired_empirical_laws=paired_laws, **{f"curves_{name}": value for name, value in curves.items()}, **{f"weights_{name}": value for name, value in weights.items()})
    summary = {"uk_created": datetime.now(ZoneInfo("Europe/London")).isoformat(), "parameters": {"target": target.tolist(), "seeds": seeds, "first_seed": 80500, "n": n, "steps": steps, "chains": batch, "sweeps": sweeps, "uniform_proposals": batch * sweeps, "rate": rate}, "metrics": ["true_kl", "true_axis_angle_degrees", "weight_norm", "linear_reconstruction", "empirical_nll"], "initial_means": curves["linear_ae"][0].mean(axis=0).tolist(), "final_means": {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, "stopping": stopping, "memory_counterexample": {"dataset_size": 64, "correlations": paired_memory.tolist(), "masked_gradient_first": reconstruction_gradients[0].tolist(), "masked_gradient_second": reconstruction_gradients[1].tolist(), "masked_gradient_difference_norm": float(np.linalg.norm(reconstruction_gradients[0] - reconstruction_gradients[1])), "likelihood_gradient_difference_max": float(np.max(np.abs(likelihood_gradients[0] - likelihood_gradients[1])))}, "interpretation": "Additional explicit memory-access restriction: only the masked-reconstruction control reads the old empirical distribution after exposure. Own Gibbs sampling is still not necessary: fresh uniform proposals plus self-normalized energy weighting use the same correlation memory. Uniform proposals use 512 candidates versus 128 chains times four Gibbs sweeps; this counts candidates/sweeps, not matched FLOPs. Normalizer enumeration is used by diagnostics and the exact control, not by sampled updates."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots()
    labels = {"gibbs_memory": "Correlation memory + own Gibbs trials", "exact_memory": "Correlation memory + exact integration", "masked_raw_replay": "Masked reconstruction, original examples", "uniform_proposal_memory": "Correlation memory + uniform trials", "linear_ae": "Continue optimal linear reconstruction"}
    for name, values in curves.items():
        line, = ax.plot(saved, values[:, :, 0].mean(axis=1), label=labels[name])
        lo, hi = np.quantile(values[:, :, 0], [0.1, 0.9], axis=1)
        ax.fill_between(saved, lo, hi, color=line.get_color(), alpha=0.15)
    ax.set(xlabel="Updates", ylabel="True generation KL", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "tex" / "archive" / "figures" / "practice-loop-memory.png")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
