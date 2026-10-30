"""Persist small b2k observations with SHA256 verification; retain all sources."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/bestn_b2k"


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    with source.open("rb") as stream: original = hashlib.file_digest(stream, "sha256").hexdigest()
    with destination.open("rb") as stream: copied = hashlib.file_digest(stream, "sha256").hexdigest()
    assert original == copied
    return {"source": str(source), "persistent": str(destination), "bytes": destination.stat().st_size, "sha256": original}


def main(phase):
    load_dotenv(Path.home() / ".env")
    destination = Path(os.environ["CACHE_DIR"]) / "generative-learning/bestn_b2k/observations"
    toy = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/bestn_b2k/toy"
    files = [(p, destination / "toy_raw" / p.name) for p in sorted(toy.glob("*.npz"))]
    assert len(files) == 6
    checkpoints = []
    if phase == "all":
        analysis = json.loads((RESULTS / "analysis.json").read_text())
        assert analysis["training_analysis"]["all_six_complete"] and analysis["final_evaluation"]["complete"]
        for component in ("norm_control", "norm_confirmation"):
            completed = json.loads((RESULTS / component / "complete.json").read_text())
            files.extend((Path(r["path"]), destination / f"{component}_raw" / Path(r["path"]).name) for r in completed["files"])
        files.extend((p, destination / "results" / p.relative_to(RESULTS)) for p in sorted(RESULTS.rglob("*"))
                     if p.is_file() and p.name not in ("archive.json", "toy_archive.json") and "__pycache__" not in p.parts)
        sources = set((ROOT / "experiments").glob("*bestn*")) | set((ROOT / "experiments").glob("BESTN*"))
        sources.add(ROOT / "experiments/simulate_finite_sample.py")
        sources.update(ROOT / "experiments" / name for name in ("train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py"))
        files.extend((p, destination / "experiments" / p.name) for p in sorted(sources) if p.is_file())
        files.extend((p, destination / "reports" / p.name) for p in sorted((ROOT / "reports").glob("bestn-*")) if p.is_file())
        for marker in RESULTS.glob("training/*/*/*.pt.archive.json"):
            record = json.loads(marker.read_text())
            assert Path(record["persistent"]).stat().st_size == record["bytes"]
            checkpoints.append(record)
    records = [copy_verified(source, target) for source, target in tqdm(files, desc="Archive and verify")]
    result = {"created_at_uk": datetime.now(ZoneInfo("Europe/London")).isoformat(), "phase": phase,
              "destination": str(destination), "files": records, "bytes": sum(r["bytes"] for r in records),
              "checkpoints": checkpoints, "checkpoint_check": "Full SHA256 verification was performed during each asynchronous training archive; persistent existence and size rechecked here."}
    name = "archive.json" if phase == "all" else "toy_archive.json"
    (RESULTS / name).write_text(json.dumps(result, indent=2) + "\n")
    (destination / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"destination": str(destination), "files": len(records), "bytes": result["bytes"], "checkpoints": len(checkpoints)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("toy", "all"))
    main(parser.parse_args().phase)
