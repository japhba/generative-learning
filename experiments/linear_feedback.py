"""Exact Gaussian posterior over two control weights, with actual closed-loop acquisition."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

POLICIES = ("expert", "on_policy", "random")
LABELS = {"expert": "Expert demonstrations", "on_policy": "Learner exploration + teacher feedback", "random": "Random exploration + teacher feedback"}
BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def simulate(worlds, seed, process_std=0.):
    rng = np.random.default_rng(seed)
    commands = rng.normal(size=(1024, worlds))
    observation_noise = rng.normal(scale=0.25, size=(1024, worlds))
    exploration = rng.normal(scale=0.5, size=(1024, worlds))
    process_noise = rng.normal(scale=process_std, size=(1024, worlds))
    true = np.array([1., -1.])
    states = {arm: np.zeros(worlds) for arm in POLICIES}
    means = {arm: np.zeros((worlds, 2)) for arm in POLICIES}
    covariance = {arm: np.tile(np.eye(2), (worlds, 1, 1)) for arm in POLICIES}
    grams = {arm: np.zeros((worlds, 2, 2)) for arm in POLICIES}
    sufficient = {arm: np.zeros((worlds, 2)) for arm in POLICIES}
    records = {arm: [] for arm in POLICIES}
    for step in tqdm(range(1024), desc="Linear feedback acquisition"):
        for arm in POLICIES:
            x = np.stack([commands[step], states[arm]], axis=1)
            # Execute the learner's pre-feedback prediction; Gaussian exploration
            # does not use the teacher's current label.
            if arm == "expert": action = commands[step] - states[arm]
            elif arm == "on_policy": action = np.sum(means[arm] * x, axis=1) + exploration[step]
            else: action = exploration[step]
            label = x @ true + observation_noise[step]
            projected = np.einsum('nij,nj->ni', covariance[arm], x)
            gain = projected / (0.25 ** 2 + np.sum(x * projected, axis=1))[:, None]
            means[arm] += gain * (label - np.sum(means[arm] * x, axis=1))[:, None]
            covariance[arm] -= np.einsum('ni,nj->nij', gain, projected)
            grams[arm] += np.einsum('ni,nj->nij', x, x)
            sufficient[arm] += x * label[:, None]
            states[arm] += action - commands[step] + process_noise[step]
            if (step + 1) % 16 == 0: states[arm][:] = 0
            if step + 1 in BUDGETS:
                precision = np.eye(2)[None, :, :] + grams[arm] / 0.25 ** 2
                exact_cov = np.linalg.inv(precision)
                exact_mean = np.einsum('nij,nj->ni', exact_cov, sufficient[arm]) / 0.25 ** 2
                np.testing.assert_allclose(covariance[arm], exact_cov, atol=1e-10, rtol=1e-9)
                np.testing.assert_allclose(means[arm], exact_mean, atol=1e-10, rtol=1e-9)
                values = {"weight_mse": np.sum((means[arm] - true) ** 2, axis=1),
                          "corrective_weight": means[arm][:, 1], "normalized_overlap": means[arm] @ true / 2,
                          "posterior_corrective_variance": covariance[arm][:, 1, 1],
                          "minimum_gram_eigenvalue": np.linalg.eigvalsh(grams[arm])[:, 0]}
                record = {"budget": step + 1}
                for metric, value in values.items():
                    record[metric] = {"mean": float(value.mean()), "standard_error": float(value.std(ddof=1) / np.sqrt(worlds)), "by_world": value.tolist()}
                records[arm].append(record)
    if process_std == 0:
        assert np.all(means["expert"][:, 1] == 0)
        assert np.all(covariance["expert"][:, 1, 1] == 1)
    return {"seed": seed, "worlds": worlds, "true_weights": true.tolist(), "prior": "N(0,I)", "label_noise_std": 0.25,
            "exploration_std": 0.5, "process_std": process_std, "episode_horizon": 16, "paired": "Same command, observation-noise, process-noise and exploration arrays across curricula.",
            "posterior_checks": worlds * len(BUDGETS) * len(POLICIES), "curves": records,
            "interpretation": "With zero process noise, expert data leave the corrective coefficient at its prior. Positive process noise weakly excites that direction even under expert control. Exploratory actions supply stronger excitation. Isotropic test prediction MSE equals squared weight error. This is not a universal generation effect or an RBM training result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=97100)
    parser.add_argument("--process-std", type=float, default=0.)
    args = parser.parse_args()
    load_dotenv(Path.home() / '.env')
    os.environ['MPLCONFIGDIR'] = str(Path(os.environ['FAST_CACHE_DIR']) / 'generative-learning/matplotlib')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = simulate(args.worlds, args.seed, args.process_std)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    for metric, ylabel, logscale in (("weight_mse", "Squared error to true control weights", True), ("corrective_weight", "Learned corrective coefficient (true value -1)", False),
                                    ("minimum_gram_eigenvalue", "Smallest eigenvalue of accumulated input Gram matrix", False)):
        figure, axis = plt.subplots()
        for arm in POLICIES:
            rows = summary['curves'][arm]
            x = [row['budget'] for row in rows]
            mean = np.array([row[metric]['mean'] for row in rows])
            se = np.array([row[metric]['standard_error'] for row in rows])
            axis.plot(x, mean, marker='o', label=LABELS[arm])
            axis.fill_between(x, mean-se, mean+se, alpha=0.15)
        axis.set(xscale='log', xlabel='Teacher labels acquired', ylabel=ylabel)
        if logscale: axis.set_yscale('log')
        if metric == 'corrective_weight': axis.axhline(-1, color='black', linestyle=':')
        axis.legend()
        figure.tight_layout()
        path = args.output / f'{metric}.png'
        figure.savefig(path)
        plt.close(figure)
        paths.append(str(path.resolve()))
    summary['plots'] = paths
    (args.output/'source').mkdir()
    (args.output/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    for path in paths: print(path)


if __name__ == '__main__': main()
