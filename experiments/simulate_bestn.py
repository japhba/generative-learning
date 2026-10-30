"""Frozen-protocol finite RBM experiment with prospective population forecasts."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from bestn_theory import drift, sigmoid, success


ROOT = Path(__file__).resolve().parents[1]
STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
ARMS = ("current", "frozen")


def stamp(): return datetime.now(ZoneInfo("Europe/London")).isoformat()


def save(path, value): path.write_text(json.dumps(value, indent=2) + "\n")


def run_case(case, raw):
    rng = np.random.default_rng([820001, case["index"]])
    replicas, batch, eta, signal = 256, 32, .05, 1.5
    w0 = np.zeros(8)
    w0[:2] = case["rho"], np.sqrt(1 - case["rho"]**2)
    w = np.broadcast_to(w0, (2, replicas, 8)).copy()
    trajectory = [w.copy()]
    for step in range(1, STEPS[-1] + 1):
        z = rng.normal(size=(replicas, batch, 8))
        z[..., 0] += signal
        uniforms = rng.random((replicas, batch, case["n"]))
        for arm, source in enumerate(ARMS):
            proposal = w[arm] if source == "current" else np.broadcast_to(w0, w[arm].shape)
            probability = sigmoid(2 * np.einsum("rbd,rd->rb", z, proposal))
            correct = (uniforms < probability[..., None]).any(axis=-1)
            residual = 2 * correct - 1 - np.tanh(np.einsum("rbd,rd->rb", z, w[arm]))
            w[arm] += eta * np.einsum("rbd,rb->rd", z, residual) / batch
        assert np.isfinite(w).all()
        if step in STEPS: trajectory.append(w.copy())
    path = raw / (case["name"] + ".npz")
    np.savez_compressed(path, weights=np.stack(trajectory), steps=STEPS)
    return {"case": case, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run():
    load_dotenv(Path.home() / ".env")
    output = ROOT / "results/bestn_b2k/toy"
    raw = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/bestn_b2k/toy"
    output.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    cases = [{"index": i, "name": f"rho{rho:g}_n{n}", "rho": rho, "n": n}
             for i, (rho, n) in enumerate((rho, n) for rho in (-.3, .3) for n in (1, 2, 4))]
    sources = ("simulate_bestn.py", "bestn_theory.py", "test_bestn_theory.py")
    for name in sources: (output / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    save(output / "protocol.json", {"created_at_uk": stamp(), "cases": cases, "signal": 1.5, "dimension": 8,
         "initial_norm": 1, "replicas": 256, "batch": 32, "eta": .05, "steps": STEPS, "seed": 820001,
         "arms": ARMS, "raw_directory": str(raw), "temperature": 1,
         "source_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in sources},
         "forecast": "Before simulation, integrate the analytically specified population update without fitted parameters. Current N>1 increases normalized overlap. Current is expected to continue improving after frozen source approaches its selected-target optimum. N=1 has zero population current drift.",
         "limits": "Discrete population steps are a small-noise approximation to nonlinear finite-SGD metrics. No equality of exponents to an LLM is assumed. Binary feedback reveals the true class; label-information complexity differs from arithmetic verification.",
         "primary": "rho=.3, N=2, current-minus-frozen classification error at1024 steps; paired across256 seeds; 95% normal CI."})
    predictions = []
    for case in cases:
        w0 = np.array([case["rho"], np.sqrt(1 - case["rho"]**2)])
        for arm in ARMS:
            w = w0.copy()
            for step in range(STEPS[-1] + 1):
                if step in STEPS:
                    rho = w[0] / np.linalg.norm(w)
                    predictions.append({"case": case["name"], "arm": arm, "step": step, "rho": float(rho),
                                        "error": .5 * math.erfc(1.5 * rho / np.sqrt(2)), "success": success(w, 1.5), "w": w.tolist()})
                w += .05 * drift(w, w0, 1.5, case["n"], arm)
    save(output / "predictions.json", {"written_before_simulation_at_uk": stamp(), "rows": predictions})
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_case, case, raw) for case in cases]
        for future in tqdm(as_completed(futures), total=len(futures), desc="RBM best-of-N"):
            results.append(future.result())
    save(output / "complete.json", {"finished_at_uk": stamp(), "seconds": time.monotonic() - start,
                                   "results": sorted(results, key=lambda row: row["case"]["index"])})
    print(str(output))


if __name__ == "__main__": run()
