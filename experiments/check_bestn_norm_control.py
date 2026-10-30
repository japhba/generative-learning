"""Exploratory fixed-norm, no-feedback control; freeze before finite sampling."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from numpy.polynomial.hermite import hermgauss
from tqdm.auto import tqdm

from simulate_finite_sample import ARMS, CHECKPOINTS, draw_input, gradients, initial_fisher, initial_weight, metrics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/bestn_b2k/norm_control"
NODES, MASS = hermgauss(160)
NODES, MASS = np.sqrt(2) * NODES, MASS / np.sqrt(np.pi)


def density_loss(weights):
    norm = np.linalg.norm(weights, axis=-1)
    margins = 1.5 * weights[..., 0, None] + norm[..., None] * NODES
    return norm**2 / 2 - (np.logaddexp(margins, -margins) - np.log(2)) @ MASS


def initial_prediction():
    w = initial_weight(8, .3)
    fisher = initial_fisher(w, nodes=160)
    z, mass = hermgauss(160)
    x0, x1 = np.meshgrid(np.sqrt(2) * z + 1.5, np.sqrt(2) * z, indexing="ij")
    mass = np.outer(mass, mass) / np.pi
    mean = np.tanh(w[0] * x0 + w[1] * x1)
    gradient = w.copy()
    gradient[:2] -= np.array([np.sum(mass * x0 * mean), np.sum(mass * x1 * mean)])
    hessian, projection = np.eye(8) - fisher, np.eye(8) - np.outer(w, w)
    records = []
    for source in ("self_sampled", "isotropic_gaussian"):
        covariance = fisher if source == "self_sampled" else np.trace(fisher) / 8 * np.eye(8)
        drift = -projection @ covariance @ w - .5 * np.trace(projection @ covariance) * w
        projected = projection @ covariance @ projection
        coefficient = float(gradient @ drift + .5 * np.trace(projected @ hessian))
        records.append({"source": source, "density_loss_coefficient_eta_squared_over_batch": coefficient})
    return records


def run_chunk(index, raw):
    rng = np.random.default_rng([20260907, 9100 + index])
    w = np.broadcast_to(initial_weight(8, .3), (3, 512, 8)).copy()
    trajectory = [w.copy()]
    for step in range(1, 513):
        x = draw_input(rng, 512, 8, 8)
        update, _ = gradients(w, x, rng)
        w += .1 * update
        w /= np.linalg.norm(w, axis=-1, keepdims=True)
        if step in CHECKPOINTS:
            np.testing.assert_allclose(np.linalg.norm(w, axis=-1), 1., atol=1e-14)
            trajectory.append(w.copy())
    path = Path(raw) / f"chunk{index}.npz"
    np.savez_compressed(path, weights=np.stack(trajectory), steps=CHECKPOINTS)
    return {"index": index, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    load_dotenv(Path.home() / ".env")
    raw = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/bestn_b2k/norm_control"
    OUTPUT.mkdir()
    raw.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "experiments/simulate_finite_sample.py"]
    for path in sources: (OUTPUT / path.name).write_bytes(path.read_bytes())
    protocol = {"frozen_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "scope": "Exploratory extension after the unconstrained no-feedback failure. Not an LLM result.",
                "dimension": 8, "signal": 1.5, "initial_overlap": .3, "fixed_norm": 1., "eta": .1, "batch": 8, "steps": 512,
                "replicas": 4096, "arms": ARMS, "random_seeds": "SeedSequence([20260907,9100+chunk]) with eight chunks.",
                "training": "Self-sampled conditional labels at temperature1, one ordinary score update, then Euclidean projection to unit norm. No reward, label truth, sample filtering or temperature change. Controls project matched and isotropic Gaussian score noise in the same way.",
                "primary": "Change in true marginal cross-entropy from initialization, and self-minus-isotropic endpoint difference. Retain classification error and signed overlap; no isolated-system claim because fresh inputs come from the real Gaussian mixture.",
                "local_prediction": initial_prediction(), "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_chunk, i, str(raw)) for i in range(8)]
        for future in tqdm(as_completed(futures), total=8, desc="Fixed-norm finite sampling"): records.append(future.result())
    records.sort(key=lambda r: r["index"])
    w = np.concatenate([np.load(r["path"])["weights"] for r in records], axis=2)
    loss, statistics = density_loss(w), metrics(w)
    summary = []
    for index, source in enumerate(ARMS):
        delta = loss[-1, index] - loss[0, index]
        summary.append({"source": source, "density_change": float(delta.mean()), "density_change_ci95":
                        (delta.mean() + np.array([-1, 1]) * 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))).tolist(),
                        "final_axis_overlap_squared": float(np.mean(statistics["overlap"][-1, index]**2)),
                        "final_classification_error": float(statistics["error"][-1, index].mean())})
    contrast = loss[-1, 0] - loss[-1, 2]
    complete = {"finished_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "wall_seconds": time.monotonic() - start,
                "files": records, "summary": summary, "self_minus_isotropic": float(contrast.mean()), "self_minus_isotropic_ci95":
                (contrast.mean() + np.array([-1, 1]) * 1.96 * contrast.std(ddof=1) / np.sqrt(len(contrast))).tolist(),
                "curves": [{"source": source, "step": int(step), "density_loss": float(loss[j, i].mean()),
                            "classification_error": float(statistics["error"][j, i].mean()), "squared_overlap": float(np.mean(statistics["overlap"][j, i]**2))}
                           for i, source in enumerate(ARMS) for j, step in enumerate(CHECKPOINTS)]}
    (OUTPUT / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps({"local_prediction": protocol["local_prediction"], "summary": summary, "self_minus_isotropic": complete["self_minus_isotropic"]}))


if __name__ == "__main__": main()
