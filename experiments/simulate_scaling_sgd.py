"""Actual sampled best-of-two updates, with a predeclared finite-noise prediction."""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from bestn_theory import sigmoid
from run_elicitation_pilot import ROOT, digest, save, setup
from scaling_theory import normal_cdf, phi, rbm_metrics
from train_bestn import stamp


CASES = [(1, 8, .1, "current"), (8, 8, .1, "current"), (8, 1, .2, "current"),
         (1, 8, .1, "frozen"), (8, 8, .1, "frozen"), (8, 1, .2, "frozen")]
STEPS, REPLICAS = 65536, 256
CHECKS = sorted({0, *np.unique(np.rint(np.geomspace(1, STEPS, 65)).astype(int)).tolist()})


def simulate(specification):
    index, dimension, batch, eta, source = [specification[k] for k in ("index", "dimension", "batch", "eta", "source")]
    output, raw, replicas, steps, checks = [specification[k] for k in ("output", "raw", "replicas", "steps", "checks")]
    rng = np.random.default_rng(specification["seed"])
    initial = np.zeros(dimension)
    initial[0] = 1. if dimension == 1 else .3
    if dimension > 1: initial[1] = np.sqrt(1-.3**2)
    weights = np.tile(initial, (replicas, 1))
    snapshots, records, loss_sum, last_check = [], [], 0., 0
    name = f"d{dimension}_b{batch}_eta{eta}_{source}"
    for step in tqdm(range(steps + 1), desc=name, position=index % 3, leave=False):
        if step in checks:
            norms = np.linalg.norm(weights, axis=1)
            overlaps = weights[:, 0] / norms
            metrics = [rbm_metrics(r, rho) for r, rho in zip(norms, overlaps, strict=True)]
            errors = np.array([r["error"] for r in metrics])
            records.append({"step": step, "time": eta*step, "candidates": 2*batch*step,
                            "error_mean": float(errors.mean()), "excess_mean": float(errors.mean()-normal_cdf(-1.5)),
                            "errors": errors.tolist(), "norm_mean": float(norms.mean()), "overlap_mean": float(overlaps.mean()),
                            "block_loss": loss_sum/(step-last_check) if step else None})
            snapshots.append(weights.copy())
            loss_sum, last_check = 0., step
            if step >= 8192:
                recent = records[-4:]
                x = np.log([r["step"] for r in recent])
                slopes = {"excess_error": float(np.polyfit(x, np.log([r["excess_mean"] for r in recent]), 1)[0]),
                          "loss": float(np.polyfit(x, np.log([r["block_loss"] for r in recent]), 1)[0])}
                if any(abs(s) < .02 for s in slopes.values()): break
        if step == steps: break
        x = rng.standard_normal((replicas, batch, dimension))
        x[:, :, 0] += 1.5
        margins = np.einsum("rbd,rd->rb", x, weights)
        proposal_margins = margins if source == "current" else x @ initial
        probability = sigmoid(2*proposal_margins)
        draws = rng.random((replicas, batch, 2)) < probability[:, :, None]
        chosen = 2*draws.any(axis=-1)-1
        residual = chosen - np.tanh(margins)
        loss_sum += float(np.logaddexp(0, -2*chosen*margins).mean())
        weights += eta*np.einsum("rb,rbd->rd", residual, x)/batch
    raw_path = Path(raw)/f"{name}.npz"
    np.savez_compressed(raw_path, weights=np.stack(snapshots), steps=[r["step"] for r in records])
    save(Path(output)/f"{name}.json", {"dimension": dimension, "batch": batch, "eta": eta, "source": source,
         "seed": specification["seed"], "replicas": replicas, "records": records, "raw": str(raw_path),
         "rng_final": rng.bit_generator.state, "finished_at_uk": stamp()})
    return {"case": name, "last_step": step, "final_excess": records[-1]["excess_mean"]}


def main():
    output = ROOT/"results/scaling_s4g/finite_sgd"
    output.mkdir(parents=True, exist_ok=False)
    raw = setup()/"scaling_s4g/finite_sgd"
    raw.mkdir(parents=True, exist_ok=False)
    noise_coefficient = 7*.2*float(phi(1.5))
    save(output/"protocol.json", {"created_at_uk": stamp(), "source_sha256": digest(Path(__file__)),
         "cases": CASES, "replicas": REPLICAS, "maximum_steps": STEPS, "checks": CHECKS,
         "population_prediction": "Aligned Gaussian RBM excess error has exponent1/2.",
         "finite_noise_prediction": "For d>1, eta/B>0 and sufficiently large norm, tangential score noise produces an Ito radial drift and an angular error. Under a quasi-stationary angular approximation, R^3~3*lambda*t and excess~lambda/(2R), lambda=(d-1)*eta*phi(m)/B. Hence exponent1/3 in the noise-dominated regime. This is a prospective asymptotic approximation, not yet validated by these samples.",
         "noise_case_coefficient": float(noise_coefficient**(2/3)/(2*3**(1/3))),
         "assessment": "Compare empirical exponents and absolute coefficients over the final factor8 in population time, with whole-replica bootstrap. Retain all regimes including a moderate-noise crossover. No exponent or time constant fitted in the prediction.",
         "stopping": "At8192steps or later, halt if either the last four log-excess-error or log-selected-NLL slopes has magnitude below.02. All replicas in a condition stop together.",
         "sampling": "Explicit two Bernoulli conditional draws per real input. Binary selection is their maximum; uniform ties do not change the selected hidden state. The signed input y_true*x is sampled as Normal(mu,I), an exact symmetry reduction of balanced Gaussian classes."})
    arguments = [{"index": i, "dimension": case[0], "batch": case[1], "eta": case[2], "source": case[3],
                  "output": str(output), "raw": str(raw), "replicas": REPLICAS, "steps": STEPS,
                  "checks": CHECKS, "seed": 96000+i%3} for i, case in enumerate(CASES)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(tqdm(pool.map(simulate, arguments), total=len(arguments), desc="Finite SGD cases"))
    save(output/"complete.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
