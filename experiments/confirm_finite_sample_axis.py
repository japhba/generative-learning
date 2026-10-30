"""Fresh confirmation of the post-hoc signal-axis diagnostic."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from analyze_finite_sample import summary
from simulate_finite_sample import ARMS, initial_fisher, initial_weight, metric_hessians, metrics, run_case


def run():
    load_dotenv(Path.home() / ".env")
    output = Path("results/finite_sample_axis_20260907")
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / output.name
    raw.mkdir(parents=True, exist_ok=False)
    parameters = {"dimension": 8, "initial_overlap": .3, "batch": 8, "eta": .1}
    cases = [{**parameters, "name": f"axis_block{block:02}", "index": 2000 + block} for block in range(32)]
    w = initial_weight(8, .3)
    n = w / np.linalg.norm(w)
    gradient = (np.eye(8)[0] - n[0] * n) / np.linalg.norm(w)
    h_overlap, _ = metric_hessians(w)
    h_axis = 2 * np.outer(gradient, gradient) + 2 * n[0] * h_overlap
    fisher = initial_fisher(w)
    source = Path(__file__)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "The unsigned-axis diagnostic was discovered post hoc in earlier runs. This confirmation is specified before new, independent sampling.",
                "parameters": parameters, "replicas": 16384, "blocks": 32, "steps": 512,
                "seed_indices": [case["index"] for case in cases], "raw_directory": str(raw),
                "primary_axis_metric": "rho^2=(w.mu)^2/(||w||^2*||mu||^2), the squared cosine to the signal axis",
                "primary_axis_contrast": "Endpoint mean rho^2 in self-sampled minus isotropic control; two-sided 95% paired interval.",
                "other_metrics": ["signed overlap", "absolute overlap", "sign-flip fraction", "classification error"],
                "matched_equivalence_margin_rho_squared": .01,
                "expected_initial_axis_drift_coefficients": {"self_sampled": float(np.trace(fisher @ h_axis)/2), "matched_gaussian": float(np.trace(fisher @ h_axis)/2),
                                                             "isotropic_gaussian": float(np.trace(np.eye(8)*np.trace(fisher)/8 @ h_axis)/2)},
                "interpretation_limit": "Axis alignment ignores sign; improved axis recovery does not imply improved labeled predictions or calibrated generative density. Ground-truth axis is used only by evaluation. Inputs still contain the true mixture structure.",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "simulation_source_sha256": hashlib.sha256(source.with_name("simulate_finite_sample.py").read_bytes()).hexdigest()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / source.name).write_bytes(source.read_bytes())
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_case, case, str(raw)) for case in cases]
        for future in tqdm(as_completed(futures), total=len(cases), desc="Fresh signal-axis confirmation"): records.append(future.result())
    records.sort(key=lambda row: row["case"]["index"])
    arrays = [np.load(row["raw_path"]) for row in records]
    weights = np.concatenate([row["weights"] for row in arrays], axis=2)
    variation = np.concatenate([row["quadratic_variation"] for row in arrays], axis=2)
    combined = raw / "axis_combined.npz"
    np.savez_compressed(combined, weights=weights, quadratic_variation=variation, steps=arrays[0]["steps"])
    values = metrics(weights)
    values["axis_overlap"] = values["overlap"]**2
    values["absolute_overlap"] = np.abs(values["overlap"])
    values["sign_flip"] = (values["overlap"] < 0).astype(float)
    result = {"parameters": parameters, "replicas": weights.shape[2], "raw_path": str(combined), "sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
              "steps": arrays[0]["steps"].tolist(), "samples": (arrays[0]["steps"] * parameters["batch"]).tolist(), "curves": {}, "endpoint": {}}
    for index, arm in enumerate(ARMS):
        result["curves"][arm] = {key: [summary(checkpoint[index]) for checkpoint in matrix] for key, matrix in values.items()}
        result["endpoint"][arm] = {key: summary(matrix[-1,index]) for key, matrix in values.items()}
    result["paired_axis_self_minus_isotropic"] = summary(values["axis_overlap"][-1,0] - values["axis_overlap"][-1,2])
    result["paired_axis_self_minus_matched"] = summary(values["axis_overlap"][-1,0] - values["axis_overlap"][-1,1])
    ci = result["paired_axis_self_minus_matched"]["ci95"]
    result["matched_axis_within_margin"] = ci[0] > -.01 and ci[1] < .01
    result["records"] = records
    result["wall_seconds"] = time.perf_counter() - start
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"wall_seconds": result["wall_seconds"], "axis_contrast": result["paired_axis_self_minus_isotropic"],
                      "matched_contrast": result["paired_axis_self_minus_matched"], "matched_within_margin": result["matched_axis_within_margin"]}, indent=2))


if __name__ == "__main__": run()
