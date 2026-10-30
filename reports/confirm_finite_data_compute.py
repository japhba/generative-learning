"""Replay fixed finite-data prefixes and the paired recognition-capacity control."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil

from tqdm.auto import tqdm

from finite_data_wake_sleep import run_recognizer


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-control", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = ROOT / "results/finite_data_wake_sleep"
    source = root / "factorized_confirmation"
    destination = root / ("paired_categorical" if args.capacity_control else "factorized_matched")
    destination.mkdir(parents=True, exist_ok=args.resume)
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["output"] = str(destination.relative_to(ROOT))
    protocol["data_sizes"] = [512, 2048, 8192]
    if args.capacity_control:
        protocol["config"]["recognition_family"] = "categorical"
        protocol["config"]["budget"] = 469
        protocol["config"]["report_budgets"] = [128, 256, 469]
        selected = json.loads((root / "selected.json").read_text())["selected"]
        protocol["rates"] = {a: [selected[a]] for a in ("wake", "sleep")}
        protocol["scope"] = "Paired capacity control on exactly the factorized confirmation world, data and fitted generators. Recognition budgets align executed leading FLOPs with the 512-unit factorized replay; no new tuning."
    else:
        protocol["config"]["budget"] = 512
        protocol["config"]["report_budgets"] = [128, 256, 512]
        protocol["rates"] = {a: protocol["rates"][a] for a in ("wake", "sleep", "forward")}
        protocol["scope"] = "Actual-compute prefix replay; all arms finish before stationarity can save compute. Same task, data, generator, rates, and random streams as factorized confirmation. No new tuning."
    if args.resume: assert json.loads((destination / "protocol.json").read_text()) == protocol
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for filename in ("finite_data_wake_sleep.py", "deep_wake_sleep.py"):
        target = destination / filename
        if args.resume: assert target.read_bytes() == (ROOT / "experiments" / filename).read_bytes()
        shutil.copyfile(ROOT / "experiments" / filename, target)
    for seed in protocol["seeds"]:
        for n in protocol["data_sizes"]:
            folder = destination / f"s{seed}_n{n}"
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ("generator.json", "generator.npz"):
                shutil.copyfile(source / folder.name / filename, folder / filename)
    tasks = [(protocol["config"], s, n, a, protocol["rates"][a][0], destination, args.resume) for s in protocol["seeds"] for n in protocol["data_sizes"] for a in protocol["rates"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in tqdm(as_completed([pool.submit(run_recognizer, *task) for task in tasks]), total=len(tasks), desc=destination.name):
            result = future.result()
            assert result["final"]["wake_equivalent"] >= protocol["config"]["budget"] - 2.2


if __name__ == "__main__": main()
