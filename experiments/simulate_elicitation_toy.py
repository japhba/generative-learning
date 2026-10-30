"""A linear Gaussian possibility proof for learning from internal generation.

The generator A is fixed. This tests closure of an inference gap, NOT an advantage
from on-policy refresh. Its knowledge is an explicit assumption. A wrong-sign
generator illustrates why self-consistency alone cannot guarantee truth.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    dimension, noise, eta, batch, repetitions = 8, .35, .08, 64, 32
    checkpoints = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    rng = np.random.default_rng(64000)
    left = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
    right = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
    A = left @ np.diag(np.geomspace(.3, 2., dimension)) @ right.T
    C = A @ A.T + noise**2 * np.eye(dimension)
    Bstar = np.linalg.solve(C, A).T
    posterior_covariance = np.eye(dimension) - Bstar @ A
    bayes_error = np.trace(posterior_covariance)
    np.testing.assert_allclose(Bstar @ C, A.T, atol=1e-12)
    assert eta * np.linalg.eigvalsh(C).max() < 1
    arms = ("sleep", "self_imitation", "wrong_generator")
    config = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "dimension": dimension,
              "noise_sd": noise, "learning_rate": eta, "batch_size": batch, "repetitions": repetitions,
              "checkpoints": checkpoints, "matrix_seed": 64000, "sampling_seed": 65000,
              "arms": arms, "A": A.tolist(), "C": C.tolist(), "Bstar": Bstar.tolist(),
              "posterior_covariance": posterior_covariance.tolist(), "bayes_mse": float(bayes_error),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "Fixed correct generator provides latent-labeled dreams to an imperfect linear inverse. Self-imitation targets the recognizer's own mean plus independent Gaussian noise. Wrong generator reverses latent semantics while preserving the input marginal. No on-policy refresh contrast or exponent fit."}
    (output / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    weights = {arm: np.zeros((repetitions, dimension, dimension)) for arm in arms}
    sample_rng, records = np.random.default_rng(65000), []
    for step in tqdm(range(checkpoints[-1] + 1), desc="Gaussian recognition"):
        if step in checkpoints:
            point = {"steps": step, "generated_pairs": step * batch, "arms": {}}
            for arm in arms:
                delta = weights[arm] - Bstar
                gaps = np.einsum("rij,jk,rik->r", delta, C, delta)
                overlap = np.einsum("rij,jk,ik->r", weights[arm], C, Bstar) / np.einsum("ij,jk,ik", Bstar, C, Bstar)
                direct_risk = dimension - 2 * np.einsum("rij,ji->r", weights[arm], A) + np.einsum("rij,jk,rik->r", weights[arm], C, weights[arm])
                np.testing.assert_allclose(direct_risk, bayes_error + gaps, atol=1e-11)
                point["arms"][arm] = {"excess_mse": gaps.tolist(), "mse": direct_risk.tolist(), "normalized_overlap": overlap.tolist(),
                                       "first_seed_weights": weights[arm][0].tolist()}
            population = Bstar @ (np.eye(dimension) - np.linalg.matrix_power(np.eye(dimension) - eta * C, step))
            delta = population - Bstar
            point["population_sleep_excess_mse"] = float(np.trace(delta @ C @ delta.T))
            records.append(point)
        if step == checkpoints[-1]: break
        z = sample_rng.normal(size=(repetitions, batch, dimension))
        x = z @ A.T + noise * sample_rng.normal(size=z.shape)
        for arm in arms:
            prediction = x @ weights[arm].transpose(0, 2, 1)
            if arm == "sleep": target = z
            elif arm == "wrong_generator": target = -z
            else: target = prediction + noise * sample_rng.normal(size=prediction.shape)
            gradient = (prediction - target).transpose(0, 2, 1) @ x / batch
            weights[arm] -= eta * gradient
    final = records[-1]["arms"]
    assert np.mean(final["sleep"]["excess_mse"]) < .05 * np.mean(records[0]["arms"]["sleep"]["excess_mse"])
    assert np.mean(final["wrong_generator"]["mse"]) > np.mean(records[0]["arms"]["wrong_generator"]["mse"])
    result = {"protocol": config, "records": records,
              "checks": {"posterior_normal_equations": True, "risk_decomposition_all_checkpoints": True,
                         "sleep_closes_gap": True, "wrong_generator_can_harm": True}}
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    names = {"sleep": "Own generator → recognition", "self_imitation": "Recognize → imitate own answer", "wrong_generator": "Incorrect internal generator"}
    x_values = np.array([row["generated_pairs"] for row in records])
    fig, ax = plt.subplots()
    for arm in arms:
        values = np.array([row["arms"][arm]["mse"] for row in records])
        line, = ax.plot(x_values[1:], values.mean(axis=1)[1:], label=names[arm])
        se = values.std(axis=1, ddof=1) / np.sqrt(repetitions)
        ax.fill_between(x_values[1:], (values.mean(axis=1) - 1.96 * se)[1:], (values.mean(axis=1) + 1.96 * se)[1:], color=line.get_color(), alpha=.2)
    ax.axhline(bayes_error, linestyle="--", color="black", label="Optimal inference from this generator")
    ax.set(xscale="log", xlabel="Internally generated training pairs", ylabel="Expected squared prediction error",
           title="Generation can improve access to existing knowledge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "linear-elicitation.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    values = np.array([row["arms"]["sleep"]["excess_mse"] for row in records])
    ax.plot(x_values[1:], values.mean(axis=1)[1:], label="Finite-batch SGD")
    ax.plot(x_values[1:], [row["population_sleep_excess_mse"] for row in records[1:]], linestyle="--", label="Exact population-gradient dynamics")
    ax.set(xscale="log", yscale="log", xlabel="Internally generated training pairs", ylabel="Excess over optimal prediction error",
           title="The inference gap shrinks; SGD noise leaves a floor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "linear-elicitation-gap.png")
    plt.close(fig)
    print(json.dumps({"bayes_mse": float(bayes_error), "initial_mse": dimension,
                      "final": {arm: {key: float(np.mean(final[arm][key])) for key in ("mse", "excess_mse", "normalized_overlap")} for arm in arms}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/elicitation_toy"))
    run(parser.parse_args().output)
