"""Check two population-gradient identities; this does not fit learning exponents."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm


def forward(weights, x):
    for w in weights: x = x @ w.T
    return x


def main():
    output = Path("results/generation_gradient_audit")
    output.mkdir(parents=True, exist_ok=False)
    previous_path = Path("results/elicitation_toy/protocol.json")
    previous = json.loads(previous_path.read_text())
    a, sigma = np.array(previous["A"]), previous["noise_sd"]
    d = len(a)
    h = np.eye(d) + a.T @ a / sigma**2
    delta = 0.8 / np.linalg.eigvalsh(h).max()
    r, g = np.eye(d) - delta * h, delta * a.T / sigma**2
    chol = np.linalg.cholesky(2 * delta * np.eye(d) - delta**2 * h)
    rng = np.random.default_rng(840913)
    records = []
    for depth in tqdm((1, 2, 3), desc="Gaussian gradient identities"):
        for seed in range(8):
            x = rng.normal(size=(257, d)) @ a.T + sigma * rng.normal(size=(257, d))
            weights = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(depth)]
            directions = [rng.normal(size=w.shape) for w in weights]
            directions = [v / np.linalg.norm(v) for v in directions]
            prediction = forward(weights, x)
            target = prediction @ r.T + x @ g.T
            eps, xi = rng.normal(size=x.shape), rng.normal(size=x.shape) @ chol.T
            antithetic_target = ((prediction + eps) @ r.T + x @ g.T + xi + (prediction - eps) @ r.T + x @ g.T - xi) / 2
            np.testing.assert_allclose(antithetic_target, target, atol=1e-13)

            def variational(ws):
                f = forward(ws, x)
                return 0.5 * np.mean(np.sum(f**2 + (x - f @ a.T)**2 / sigma**2, axis=1))

            def imitation(ws):
                # Freeze targets when differentiating the student loss.
                return 0.5 * np.mean(np.sum((forward(ws, x) - target)**2, axis=1))

            eps_fd = 1e-5
            plus = [w + eps_fd * v for w, v in zip(weights, directions, strict=True)]
            minus = [w - eps_fd * v for w, v in zip(weights, directions, strict=True)]
            fd_vi = (variational(plus) - variational(minus)) / (2 * eps_fd)
            fd_rollout = (imitation(plus) - imitation(minus)) / (2 * eps_fd)
            np.testing.assert_allclose(fd_rollout, delta * fd_vi, rtol=1e-7, atol=1e-9)
            records.append({"depth": depth, "initialization": seed, "rollout_derivative": float(fd_rollout), "delta_times_variational_derivative": float(delta * fd_vi), "absolute_difference": float(abs(fd_rollout - delta * fd_vi))})

    # Deterministic quadrature of the normalized Gaussian self-contrastive loss.
    nodes, quadrature = np.polynomial.hermite.hermgauss(100)
    quadrature /= np.sqrt(np.pi)
    sce_records = []
    for source_mean, data_mean in ((0.0, 1.0), (-2.0, 1.5), (3.0, -1.0)):
        xp = data_mean + np.sqrt(2) * nodes
        xq = source_mean + np.sqrt(2) * nodes

        def contrastive_loss(theta):
            logit_p = -0.5 * (xp - theta)**2 + 0.5 * (xp - source_mean)**2
            logit_q = -0.5 * (xq - theta)**2 + 0.5 * (xq - source_mean)**2
            return quadrature @ np.logaddexp(0, -logit_p) + quadrature @ np.logaddexp(0, logit_q)

        derivative = (contrastive_loss(source_mean + 1e-5) - contrastive_loss(source_mean - 1e-5)) / 2e-5
        expected = 0.5 * (source_mean - data_mean)
        np.testing.assert_allclose(derivative, expected, rtol=1e-8, atol=1e-9)
        sce_records.append({"source_mean": source_mean, "data_mean": data_mean, "negative_nce_derivative": float(derivative), "half_negative_loglikelihood_derivative": expected})

    result = {"created_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "original_toy_protocol_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(), "scope": "Numerical checks of analytic expected-gradient identities, not training runs or evidence for distinct exponents. Recognition covariance and generative parameters are fixed. Targets are detached during the student update.", "posterior_step": float(delta), "gaussian_checks": records, "self_contrastive_checks": sce_records}
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"gaussian_checks": len(records), "largest_absolute_gradient_difference": max(row["absolute_difference"] for row in records), "self_contrastive_checks": len(sce_records), "output": str(output / "analysis.json")}, indent=2))


if __name__ == "__main__": main()
