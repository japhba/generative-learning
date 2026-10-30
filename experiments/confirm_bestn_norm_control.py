"""Fresh fixed-norm replication and an initially well-aligned limitation case."""

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
from tqdm.auto import tqdm

from check_bestn_norm_control import density_loss, initial_prediction
from simulate_finite_sample import ARMS, CHECKPOINTS, draw_input, gradients, initial_weight, metrics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/bestn_b2k/norm_confirmation"


def run_chunk(case, raw):
    rng = np.random.default_rng([20260907, case["seed"]])
    w = np.broadcast_to(initial_weight(8, case["rho"]), (3, case["replicas"], 8)).copy()
    trajectory = [w.copy()]
    for step in range(1, 513):
        update, _ = gradients(w, draw_input(rng, case["replicas"], 8, 8), rng)
        w += .1 * update
        w /= np.linalg.norm(w, axis=-1, keepdims=True)
        if step in CHECKPOINTS: trajectory.append(w.copy())
    path = Path(raw) / f"rho{case['rho']}_seed{case['seed']}.npz"
    np.savez_compressed(path, weights=np.stack(trajectory), steps=CHECKPOINTS)
    return {"case": case, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def taylor_check():
    rng, replicas, eta = np.random.default_rng(93001), 32768, .001
    w0 = initial_weight(8, .3)
    w = np.broadcast_to(w0, (3, replicas, 8)).copy()
    g, _ = gradients(w, draw_input(rng, replicas, 8, 8), rng)
    records = []
    for index in (0, 2):
        plus, minus = w[index] + eta * g[index], w[index] - eta * g[index]
        plus /= np.linalg.norm(plus, axis=-1, keepdims=True)
        minus /= np.linalg.norm(minus, axis=-1, keepdims=True)
        coefficients = (density_loss(plus) + density_loss(minus) - 2 * density_loss(w0)) * 8 / (2 * eta**2)
        prediction = next(r["density_loss_coefficient_eta_squared_over_batch"] for r in initial_prediction() if r["source"] == ARMS[index])
        se = coefficients.std(ddof=1) / np.sqrt(replicas)
        assert abs(coefficients.mean() - prediction) < 5 * se
        records.append({"source": ARMS[index], "analytic": prediction, "antithetic_finite_difference": float(coefficients.mean()), "monte_carlo_se": float(se)})
    return records


def main():
    load_dotenv(Path.home() / ".env")
    raw = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/bestn_b2k/norm_confirmation"
    OUTPUT.mkdir()
    raw.mkdir(parents=True)
    cases = [{"rho": rho, "replicas": 2048 if rho == .3 else 512, "seed": 9200 + group * 8 + chunk}
             for group, rho in enumerate((.3, .8)) for chunk in range(8)]
    sources = [Path(__file__), ROOT / "experiments/check_bestn_norm_control.py", ROOT / "experiments/simulate_finite_sample.py"]
    for p in sources: (OUTPUT / p.name).write_bytes(p.read_bytes())
    protocol = {"frozen_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "cases": cases,
                "primary": "Replicate the fixed-norm no-feedback marginal-density improvement at rho=.3 on16384 fresh replicas. Report self-minus-isotropic and self-minus-matched contrasts, classification and mode-sign limits.",
                "secondary": "4096 replicas starting at rho=.8: an initially good representation may deteriorate under persistent diffusion. This is a limitation test, not a parameter search.",
                "training": "Exactly the exploratory norm control: B8, eta.1,512 updates, projection to initial unit norm. Real unlabeled input samples, no correctness feedback or generation-temperature change.",
                "taylor_audit": "Independently check the analytic second-order density drift with antithetic finite differences at eta=.001 on32768 sampled gradients. This is a numerical audit, not a training arm.",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start, records = time.monotonic(), []
    audit = taylor_check()
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_chunk, case, str(raw)) for case in cases]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Confirm fixed-norm effect"): records.append(future.result())
    records.sort(key=lambda r: r["case"]["seed"])
    summaries, contrasts, curves = [], [], []
    for rho in (.3, .8):
        w = np.concatenate([np.load(r["path"])["weights"] for r in records if r["case"]["rho"] == rho], axis=2)
        statistics = metrics(w)
        loss = np.stack([density_loss(weights) for weights in w])
        for index, source in enumerate(ARMS):
            delta = loss[-1, index] - loss[0, index]
            summaries.append({"rho0": rho, "source": source, "replicas": len(delta), "density_change": float(delta.mean()),
                              "density_change_ci95": (delta.mean() + np.array([-1, 1]) * 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))).tolist(),
                              "final_squared_overlap": float(np.mean(statistics["overlap"][-1, index]**2)),
                              "final_classification_error": float(statistics["error"][-1, index].mean()),
                              "sign_reversed": float(np.mean(w[-1, index, :, 0] < 0))})
            for j, step in enumerate(CHECKPOINTS):
                change = loss[j, index] - loss[0, index]
                curves.append({"rho0": rho, "source": source, "step": int(step), "density_change": float(change.mean()),
                               "density_change_se": float(change.std(ddof=1) / np.sqrt(len(change))),
                               "classification_error": float(statistics["error"][j, index].mean())})
        for comparison in (1, 2):
            delta = loss[-1, 0] - loss[-1, comparison]
            contrasts.append({"rho0": rho, "comparison": ARMS[comparison], "self_minus_control": float(delta.mean()),
                              "ci95": (delta.mean() + np.array([-1, 1]) * 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))).tolist()})
    result = {"finished_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "seconds": time.monotonic() - start,
              "files": records, "taylor_audit": audit, "summaries": summaries, "contrasts": contrasts, "curves": curves}
    (OUTPUT / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"audit": audit, "summaries": summaries, "contrasts": contrasts}))


if __name__ == "__main__": main()
