"""Equal-temperature self-sampling and controls with matched gradient variance."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import numpy as np
from numpy.polynomial.hermite import hermgauss
from tqdm.auto import tqdm


ARMS = ("self_sampled", "matched_gaussian", "isotropic_gaussian")
CHECKPOINTS = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
SIGNAL = 1.5
REPLICAS = 512


def initial_weight(dimension, overlap):
    w = np.zeros(dimension)
    w[:2] = [overlap, np.sqrt(1 - overlap**2)]
    return w


def normal_cdf(z):
    return np.fromiter((0.5 * math.erfc(-float(t) / np.sqrt(2)) for t in z.flat), dtype=float, count=z.size).reshape(z.shape)


def metrics(w, signal=SIGNAL):
    q = np.sum(w**2, axis=-1)
    rho = w[..., 0] / np.sqrt(q)
    return {"error": normal_cdf(-signal * rho), "overlap": rho, "norm_squared": q,
            "raw_overlap": signal * w[..., 0], "parameter_error": q - 2 * signal * w[..., 0] + signal**2}


def metric_hessians(w, signal=SIGNAL):
    r = np.linalg.norm(w)
    n = w / r
    e = np.eye(len(w))[0]
    rho = n[0]
    gradient = (e - rho * n) / r
    h_rho = (-np.outer(e, n) - np.outer(n, e) - rho * np.eye(len(w)) + 3 * rho * np.outer(n, n)) / r**2
    density = np.exp(-0.5 * (signal * rho)**2) / np.sqrt(2 * np.pi)
    h_error = signal**3 * rho * density * np.outer(gradient, gradient) - signal * density * h_rho
    return h_rho, h_error


def initial_fisher(w, nodes=80, signal=SIGNAL):
    """Exact two-dimensional integral, approximated by Gaussian quadrature.

    w has support in its first two coordinates. Mixture symmetry lets us
    integrate x ~ N(mu,I), and integrate the remaining coordinates analytically.
    """
    assert np.all(w[2:] == 0)
    z, mass = hermgauss(nodes)
    x0, x1 = np.meshgrid(np.sqrt(2) * z + signal, np.sqrt(2) * z, indexing="ij")
    mass = np.outer(mass, mass) / np.pi
    variance = 1 - np.tanh(w[0] * x0 + w[1] * x1)**2
    fisher = np.eye(len(w)) * np.sum(mass * variance)
    fisher[0, 0] = np.sum(mass * variance * x0**2)
    fisher[1, 1] = np.sum(mass * variance * x1**2)
    fisher[0, 1] = fisher[1, 0] = np.sum(mass * variance * x0 * x1)
    return fisher


def draw_input(rng, replicas, batch, dimension):
    x = rng.normal(size=(replicas, batch, dimension))
    x[..., 0] += SIGNAL * (2 * rng.integers(0, 2, size=(replicas, batch)) - 1)
    return x


def gradients(w, x, rng):
    """All controls use their own current weights and the same input batch.

    At any fixed (w,x), matched Gaussian noise has exactly the same conditional
    covariance as the Bernoulli score. Isotropic noise preserves its trace.
    """
    batch, dimension = x.shape[1:]
    mean = np.tanh(np.einsum("rbd,ard->arb", x, w))
    probability = (1 + mean) / 2
    variance = 4 * probability * (1 - probability)
    labels = 2 * (rng.random(size=mean[0].shape) < probability[0]) - 1
    actual = np.einsum("rbd,rb->rd", x, labels - mean[0]) / batch
    matched = np.einsum("rbd,rb->rd", x, np.sqrt(variance[1]) * rng.normal(size=mean[1].shape)) / batch
    trace = np.einsum("arb,rb->ar", variance, np.sum(x**2, axis=-1)) / batch**2
    isotropic = rng.normal(size=w[2].shape) * np.sqrt(trace[2, :, None] / dimension)
    return np.stack([actual, matched, isotropic]), trace


def run_case(case, raw_directory):
    start = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([20260907, case["index"]]))
    w0 = initial_weight(case["dimension"], case["initial_overlap"])
    w = np.broadcast_to(w0, (len(ARMS), REPLICAS, len(w0))).copy()
    quadratic_variation = np.zeros((len(ARMS), REPLICAS))
    states, variations = [w.copy()], [quadratic_variation.copy()]
    for step in range(1, int(CHECKPOINTS[-1]) + 1):
        x = draw_input(rng, REPLICAS, case["batch"], len(w0))
        gradient, trace = gradients(w, x, rng)
        w += case["eta"] * gradient
        quadratic_variation += case["eta"]**2 * trace
        if step in CHECKPOINTS:
            assert np.isfinite(w).all()
            states.append(w.copy())
            variations.append(quadratic_variation.copy())
    path = Path(raw_directory) / (case["name"] + ".npz")
    np.savez_compressed(path, weights=np.stack(states), quadratic_variation=np.stack(variations), steps=CHECKPOINTS)
    return {"case": case, "raw_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "wall_seconds": time.perf_counter() - start}


def scalar_sanity_control(raw_directory):
    """An explicit nonlinear positive control, with exact finite-step moments.

    p_theta(y)=N(theta^2,1). Its own score is 2*theta*epsilon. No truth is used
    in training. The evaluation means 0.1 and 1 refer to two initial models;
    both are evaluated against the fixed ground-truth mean 1.
    """
    rng = np.random.default_rng(2026090701)
    replicas, eta, batch, steps = 200_000, 0.1, 1, 24
    multiplier = np.ones(replicas)
    rows = []
    u = 4 * eta**2 / batch
    for step in range(steps + 1):
        for initial_mean in (0.1, 1.0):
            prediction = initial_mean * multiplier**2
            error = (prediction - 1)**2
            exact = initial_mean**2 * (1 + 6*u + 3*u**2)**step - 2*initial_mean*(1 + u)**step + 1
            rows.append({"step": step, "initial_mean": initial_mean, "exact_mean_error": exact,
                         "sample_mean_error": float(error.mean()), "sample_mean_se": float(error.std(ddof=1) / np.sqrt(replicas)),
                         "median_error": float(np.median(error)), "fraction_improved": float(np.mean(error < (initial_mean - 1)**2)),
                         "q05_error": float(np.quantile(error, .05)), "q95_error": float(np.quantile(error, .95))})
        if step != steps: multiplier *= 1 + 2 * eta * rng.normal(size=replicas) / np.sqrt(batch)
    path = Path(raw_directory) / "scalar_control_final.npz"
    np.savez_compressed(path, multiplier=multiplier)
    return {"model": "N(theta^2,1)", "true_mean": 1.0, "initial_means": [0.1, 1.0], "eta": eta, "batch": batch,
            "replicas": replicas, "steps": steps, "seed": 2026090701, "rows": rows,
            "raw_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "interpretation": "Separate existence/sanity control. Nonlinear parameterization can change expected loss with zero mean score. It does not model latent-structure straightening."}


def run(output, workers):
    load_dotenv(Path.home() / ".env")
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / output.name
    raw.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, (dimension, overlap, batch, eta) in enumerate(itertools.product((2, 8), (.3, .8), (1, 8, 64), (.025, .1, .4))):
        name = f"d{dimension}_rho{overlap:g}_b{batch}_eta{eta:g}"
        cases.append({"index": index, "name": name, "dimension": dimension, "initial_overlap": overlap, "batch": batch, "eta": eta})
    source = Path(__file__)
    protocol_path = source.with_name("FINITE_SAMPLE_PROTOCOL.md")
    tests_path = source.with_name("test_finite_sample.py")
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "arms": ARMS, "cases": cases,
                "replicas_per_case": REPLICAS, "checkpoints": CHECKPOINTS.tolist(), "signal": SIGNAL,
                "sampling_temperature": 1, "training_temperature": 1, "updates_per_batch": 1,
                "optimizer": "plain SGD, fixed learning rate, no clipping/decay/normalization", "raw_directory": str(raw),
                "replica_rng": "Independent array entries within case stream; same inputs paired across arms. SeedSequence([20260907,case_index]).",
                "primary_case": "d8_rho0.3_b8_eta0.1", "primary_checkpoint": 512,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "tests_sha256": hashlib.sha256(tests_path.read_bytes()).hexdigest(),
                "scalar_control": {"initial_means": [0.1, 1.0], "true_mean": 1.0, "eta": .1, "batch": 1, "steps": 24, "replicas": 200000},
                "no_new_paid_compute": True}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for path in (source, protocol_path, tests_path): (output / path.name).write_bytes(path.read_bytes())
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_case, case, str(raw)) for case in cases]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Finite-sample sweep"):
            record = future.result()
            records.append(record)
            (output / "completed.json").write_text(json.dumps(sorted(records, key=lambda item: item["case"]["index"]), indent=2) + "\n")
    control = scalar_sanity_control(raw)
    (output / "scalar_control.json").write_text(json.dumps(control, indent=2) + "\n")
    result = {"cases": sorted(records, key=lambda item: item["case"]["index"]), "wall_seconds": time.perf_counter() - start,
              "case_cpu_wall_seconds_sum": sum(record["wall_seconds"] for record in records), "workers": workers,
              "scalar_control": "scalar_control.json", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "raw_directory": str(raw), "cases": len(records), "wall_seconds": result["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/finite_sample_20260907"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.workers)
