"""Independent precision check after the frozen finite-sample sweep."""

import argparse
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

from simulate_finite_sample import ARMS, run_case, metrics
from analyze_finite_sample import summary


def run(output, workers):
    load_dotenv(Path.home() / ".env")
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning" / output.name
    raw.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    simulation_source = source.with_name("simulate_finite_sample.py")
    groups = {"primary": {"dimension": 8, "initial_overlap": .3, "batch": 8, "eta": .1},
              "variance_audit": {"dimension": 2, "initial_overlap": .8, "batch": 8, "eta": .1}}
    cases = []
    for group_index, (group, parameters) in enumerate(groups.items()):
        for block in range(32):
            cases.append({**parameters, "name": f"{group}_block{block:02}", "index": 1000 + 32*group_index + block, "group": group})
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "Independent confirmation specified after inspecting the original sweep, before drawing these samples.",
                "reason": "Narrow the primary self-minus-Gaussian interval and check the largest norm-variance diagnostic from the original sweep.",
                "groups": groups, "blocks_per_group": 32, "replicas_per_block": 512, "replicas_per_group": 16384,
                "steps": 512, "primary_equivalence_margin_error": .01, "equivalence_rule": "The entire paired 95% interval must lie inside +/-1 percentage point; this is a practical tolerance, not exact equality.",
                "seed_indices": [case["index"] for case in cases], "raw_directory": str(raw),
                "original_outlier": {"case": "d2_rho0.8_b8_eta0.1", "arm": "isotropic_gaussian", "norm_martingale_z": -3.8656967416345696},
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "simulation_source_sha256": hashlib.sha256(simulation_source.read_bytes()).hexdigest()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / source.name).write_bytes(source.read_bytes())
    records = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_case, case, str(raw)) for case in cases]
        for future in tqdm(as_completed(futures), total=len(cases), desc="Independent finite-sample check"): records.append(future.result())
    groups_result = {}
    for group, parameters in groups.items():
        selected = sorted((row for row in records if row["case"]["group"] == group), key=lambda row: row["case"]["index"])
        data = [np.load(row["raw_path"]) for row in selected]
        weights = np.concatenate([row["weights"] for row in data], axis=2)
        variation = np.concatenate([row["quadratic_variation"] for row in data], axis=2)
        values = metrics(weights)
        path = raw / (group + "_combined.npz")
        np.savez_compressed(path, weights=weights, quadratic_variation=variation, steps=data[0]["steps"])
        group_result = {"parameters": parameters, "replicas": weights.shape[2], "raw_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "steps": data[0]["steps"].tolist(), "samples": (data[0]["steps"] * parameters["batch"]).tolist(), "curves": {}, "endpoint": {}, "audit": {}}
        for index, arm in enumerate(ARMS):
            group_result["curves"][arm] = {key: [summary(checkpoint[index]) for checkpoint in matrix] for key, matrix in values.items()}
            endpoint = {key: summary(matrix[-1,index]) for key, matrix in values.items()}
            delta = values["error"][-1,index] - values["error"][0,index]
            endpoint["error_change"] = summary(delta)
            endpoint["fraction_error_improved"] = float(np.mean(delta < 0))
            group_result["endpoint"][arm] = endpoint
            residual = values["norm_squared"][-1,index] - values["norm_squared"][0,index] - variation[-1,index]
            group_result["audit"][arm] = {"norm_martingale_residual": summary(residual),
                                          "raw_overlap_change": summary(values["raw_overlap"][-1,index] - values["raw_overlap"][0,index])}
        paired = summary(values["error"][-1,0] - values["error"][-1,1])
        group_result["paired_self_minus_matched"] = paired
        group_result["self_minus_matched_within_one_pp"] = paired["ci95"][0] > -.01 and paired["ci95"][1] < .01
        groups_result[group] = group_result
    result = {"groups": groups_result, "records": sorted(records, key=lambda row: row["case"]["index"]),
              "wall_seconds": time.perf_counter() - start, "workers": workers, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"wall_seconds": result["wall_seconds"], "groups": {key: {"endpoint": val["endpoint"], "paired": val["paired_self_minus_matched"], "within_one_pp": val["self_minus_matched_within_one_pp"], "audit": val["audit"]} for key,val in groups_result.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/finite_sample_confirmation_20260907"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.workers)
