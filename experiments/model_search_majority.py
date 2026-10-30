"""A small binary wake--sleep search with fixed exact reconstructions."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy.special import expit, logit
from tqdm.auto import tqdm

load_dotenv(Path.home() / ".env")
CONFIGURATION = argparse.ArgumentParser(add_help=False)
CONFIGURATION.add_argument("--observed-stride", type=int, default=1)
STRIDE = CONFIGURATION.parse_known_args()[0].observed_stride
D = 12
VISIBLE_D = len(range(0, D, STRIDE))
FLOOR = 0.05
ETA = 0.5
Z = ((np.arange(2**D)[:, None] >> np.arange(D)) & 1).astype(float)
POWERS = 2 ** np.arange(D)


def render(z):
    x = z.copy()
    for _ in range(2): x = (x + np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) >= 2).astype(float)
    return x[:, ::STRIDE]


X = render(Z)
XIDS = (X @ 2 ** np.arange(VISIBLE_D)).astype(int)
CATEGORIES, INVERSE = np.unique(XIDS, return_inverse=True)
VISIBLE = ((CATEGORIES[:, None] >> np.arange(VISIBLE_D)) & 1).astype(float)
FEATURES = np.column_stack((np.ones(len(CATEGORIES)), 2 * VISIBLE - 1))
FIBERS = [np.flatnonzero(INVERSE == i) for i in range(len(CATEGORIES))]


def law(p):
    latent = np.exp(Z @ np.log(p) + (1 - Z) @ np.log1p(-p))
    observed = np.bincount(INVERSE, weights=latent, minlength=len(CATEGORIES))
    return latent, observed


def expected_counts(p, frequencies, included):
    latent, observed = law(p)
    kept = latent * included
    mass = np.bincount(INVERSE, weights=kept, minlength=len(CATEGORIES))
    covered = mass > 0
    weights = np.zeros_like(latent)
    weights[included] = kept[included] * frequencies[INVERSE[included]] / mass[INVERSE[included]]
    means = weights @ Z
    total = frequencies[covered].sum()
    bound = frequencies @ np.log(np.maximum(mass, FLOOR**D))
    return means, total, bound, observed


def local_step(p, means, mass):
    s = (p - FLOOR) / (1 - 2 * FLOOR)
    u = logit(s)
    gradient = (1 - 2 * FLOOR) * s * (1 - s) * (means / p - (mass - means) / (1 - p))
    new = FLOOR + (1 - 2 * FLOOR) * expit(u + ETA * gradient)
    old_q = means @ np.log(p) + (mass - means) @ np.log1p(-p)
    new_q = means @ np.log(new) + (mass - means) @ np.log1p(-new)
    assert new_q + 1e-12 >= old_q + ETA * (1 - ETA / 4) * (gradient @ gradient)
    return new


def train(seed, steps, mode, evaluation_every):
    rng = np.random.default_rng(seed)
    teacher = FLOOR + (1 - 2 * FLOOR) * expit(rng.normal(0, 1.5, D))
    true_latent, truth = law(teacher)
    data = rng.multinomial(128, truth)
    frequency = data / data.sum()
    uniform = np.full(D, 0.5)
    initial_means, _, _, _ = expected_counts(uniform, frequency, np.ones(len(Z), dtype=bool))
    p = np.clip(initial_means, FLOOR + 1e-6, 1 - FLOOR - 1e-6)
    initial_latent, initial_observed = law(p)
    initial_likelihood = frequency @ np.log(initial_observed)
    initial_draws = rng.choice(len(Z), size=1024, p=initial_latent)
    included = np.zeros(len(Z), dtype=bool)
    included[initial_draws] = True
    included &= data[INVERSE] > 0
    for category in np.flatnonzero(data): included[rng.choice(FIBERS[category])] = True
    weights = np.zeros((D, VISIBLE_D + 1))
    weights[:, 0] = logit(p)
    history = []
    initial_cache = included.copy()
    fixed_means, fixed_mass, _, _ = expected_counts(p, frequency, np.ones(len(Z), dtype=bool))
    direct_posterior = np.clip(fixed_means, FLOOR, 1 - FLOOR)
    linear_matrix = ((2 * X - 1).T @ (2 * Z - 1)) / len(Z)
    linear_p = np.clip((1 + np.linalg.pinv(linear_matrix) @ (frequency @ (2 * VISIBLE - 1))) / 2, FLOOR, 1 - FLOOR)
    direct = {"full_initial_posterior": law(direct_posterior)[1], "uniform_pseudoinverse": initial_observed, "linear_moments": law(linear_p)[1]}
    for a in [0.1, 1, 10, 100]: direct[f"histogram_{a}"] = (data + a / len(data)) / (data.sum() + a)
    old_bound = -np.inf
    stopping = False
    for t in range(steps + 1):
        means, mass, bound, predicted = expected_counts(p, frequency, included)
        assert bound >= old_bound - 1e-10
        old_bound = bound
        if t % evaluation_every == 0 or t == steps:
            recognition = expit(FEATURES @ weights.T)
            recognition = 0.02 + 0.96 * recognition
            score = Z * np.log(recognition[INVERSE]) + (1 - Z) * np.log1p(-recognition[INVERSE])
            inverse_ce = -(true_latent[:, None] * score).sum()
            internal_ce = -(law(p)[0][:, None] * score).sum()
            proposal_probability = np.exp(score.sum(axis=1))
            acceptance = np.bincount(INVERSE, weights=proposal_probability, minlength=len(CATEGORIES))
            record = {"step": t, "kl": float(truth @ np.log(truth / predicted)), "nll": float(-frequency @ np.log(predicted)), "bound": float(bound), "certificate": float(bound - initial_likelihood), "cause_ce": float(inverse_ce), "internal_ce": float(internal_ce), "valid_proposal": float(truth @ acceptance), "coverage": float(mass), "cached_causes": int(included.sum()), "reconstruction": 0.0}
            history.append(record)
            assert np.isfinite(np.array(list(record.values()), dtype=float)).all()
            if t >= 400 and len(history) >= 9:
                recent = history[-9:]
                abscissa = np.log([row["step"] for row in recent])
                slopes = [np.polyfit(abscissa, np.log([-row["bound"] for row in recent]), 1)[0], np.polyfit(abscissa, np.log([row["internal_ce"] for row in recent]), 1)[0]]
                stopping = bool(max(abs(slope) for slope in slopes) < 0.002)
        if t == steps or stopping: break
        p = local_step(p, means, mass)
        latent, _ = law(p)
        dreamed_ids = rng.choice(len(Z), size=256, p=latent)
        dreamed_z = Z[dreamed_ids]
        dreamed_phi = FEATURES[INVERSE[dreamed_ids]]
        if mode == "practice":
            logits_probability = expit(dreamed_phi @ weights.T)
            weights += 0.3 * (dreamed_z - logits_probability).T @ dreamed_phi / len(dreamed_ids)
        if mode != "fixed_cache":
            targets = rng.choice(np.flatnonzero(data), size=128 if mode == "practice" else 384)
            probability = 0.02 + 0.96 * expit(FEATURES[targets] @ weights.T)
            proposed = rng.random(probability.shape) < probability
            proposed_ids = (proposed @ POWERS).astype(int)
            accepted = INVERSE[proposed_ids] == targets
            included[proposed_ids[accepted]] = True
    exact_p = np.clip(initial_means, FLOOR + 1e-6, 1 - FLOOR - 1e-6)
    for _ in range(500):
        em_means, _, _, _ = expected_counts(exact_p, frequency, np.ones(len(Z), dtype=bool))
        exact_p = np.clip(em_means, FLOOR, 1 - FLOOR)
    direct["iterative_exact_em"] = law(exact_p)[1]
    summary = {"seed": seed, "mode": mode, "teacher": teacher.tolist(), "initial": initial_means.tolist(), "final": p.tolist(), "recognition_weights": weights.tolist(), "data": data.tolist(), "initial_cache": np.flatnonzero(initial_cache).tolist(), "final_cache": np.flatnonzero(included).tolist(), "initial_likelihood": float(initial_likelihood), "history": history, "stopped": stopping, "controls": {key: {"kl": float(truth @ np.log(truth / prediction)), "nll": float(-frequency @ np.log(prediction))} for key, prediction in direct.items()}}
    return summary


def main():
    parser = argparse.ArgumentParser(parents=[CONFIGURATION])
    parser.add_argument("--seed", type=int, default=20106000)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path(os.environ["FAST_CACHE_DIR"]) / "model_search_majority")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"architecture": f"12 independent Bernoulli causes, two known majority-neighbor layers, observed stride {STRIDE}, learned 12x{VISIBLE_D + 1} logistic inverse", "worlds": args.worlds, "seed": args.seed, "steps": args.steps, "generator_step": ETA, "recognition_step": 0.3, "sample_count": 128, "dreams_per_step": 256, "proposals_per_step": 128, "initial_dreams": 1024, "initial_coverage": "one uniform valid cause per distinct observed x, in addition to initial dream cache; exact reconstruction of every observation from start", "floor": FLOOR, "primary": "practice versus full-initial-posterior one-shot, frozen recognition, and fixed initial cache", "status": "exploratory mechanism test, no population-dependent stopping", "stopping": "log-log slopes of negative cache likelihood bound and current-model recognition cross entropy, absolute below .002 across last9 evaluations, after400 updates"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    results = []
    for offset in tqdm(range(args.worlds), desc="worlds"):
        for mode in tqdm(["practice", "frozen_recognition", "fixed_cache"], desc="arms", leave=False):
            result = train(args.seed + offset, args.steps, mode, 20)
            (args.output / f"world-{offset}-{mode}.json").write_text(json.dumps(result, indent=2))
            results.append(result)
    summary = {}
    for mode in ["practice", "frozen_recognition", "fixed_cache"]:
        subset = [result for result in results if result["mode"] == mode]
        summary[mode] = {key: float(np.mean([result["history"][-1][key] for result in subset])) for key in ["kl", "nll", "certificate", "cause_ce", "valid_proposal", "cached_causes"]}
    main_results = [result for result in results if result["mode"] == "practice"]
    for key in main_results[0]["controls"]: summary[key] = {"kl": float(np.mean([result["controls"][key]["kl"] for result in main_results]))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    grid = np.arange(0, args.steps + 1, 20)
    fig, axes = plt.subplots(2, 2)
    for mode in ["practice", "frozen_recognition", "fixed_cache"]:
        subset = [result for result in results if result["mode"] == mode]
        for ax, key in zip(axes.flat, ["kl", "certificate", "valid_proposal", "reconstruction"]):
            trajectories = np.array([np.interp(grid, [row["step"] for row in result["history"]], [row[key] for row in result["history"]]) for result in subset])
            ax.plot(grid, trajectories.mean(axis=0), label=mode.replace("_", " "))
            ax.set(xlabel="Offline updates", ylabel=key.replace("_", " "))
    axes[0, 0].axhline(summary["full_initial_posterior"]["kl"], label="one posterior fit", linestyle="--")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(args.output / "learning.png")
    print(json.dumps(summary, indent=2))
    print(args.output / "learning.png")


if __name__ == "__main__": main()
