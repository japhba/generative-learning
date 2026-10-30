"""Verify and retain the complete r5s evidence, including every pilot rate."""

import json
import hashlib
import os
from pathlib import Path
import tarfile

from tqdm.auto import tqdm

from archive_bestn import copy_verified
from run_elicitation_pilot import ROOT, save, setup
from train_bestn import stamp


def main():
    setup(); results = ROOT / "results/scaling_r5s"
    destination = Path(os.environ["CACHE_DIR"]) / "generative-learning/scaling_r5s/observations"
    protocol = json.loads((results / "confirmation/protocol.json").read_text())
    assert all((results / "confirmation" / str(c["seed"]) / c["name"] / "test.json").exists() for c in protocol["cells"])
    files = [p for p in sorted(results.rglob("*")) if p.is_file() and p.name != "archive.json" and "__pycache__" not in p.parts]
    files.extend(p for p in sorted((ROOT / "experiments").glob("*")) if p.suffix in (".py", ".md"))
    files.extend(p for p in sorted((ROOT / "reports").glob("*scaling*")) if p.is_file())
    files.extend(ROOT / name for name in ("pyproject.toml", "uv.lock", "reports/index.html"))
    checkpoints = []
    for path in sorted(results.rglob("*.pt.archive.json")):
        record = json.loads(path.read_text()); assert Path(record["persistent"]).stat().st_size == record["bytes"]
        checkpoints.append(record)
    local = setup() / "scaling_r5s/observations.tar.gz"; local.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(local, "w:gz", compresslevel=3, dereference=True) as archive:
        for path in tqdm(files, desc="Pack all r5s evidence on local storage"):
            member = path.relative_to(ROOT).as_posix(); archive.add(path, arcname=member, recursive=False)
            with path.open("rb") as stream: sha = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"source": str(path), "member": member, "bytes": path.stat().st_size, "sha256": sha})
    with tarfile.open(local, "r:gz") as archive:
        members = archive.getmembers(); assert [m.name for m in members] == [r["member"] for r in records]
        for member, record in tqdm(zip(members, records, strict=True), total=len(records), desc="Verify every archived member"):
            with archive.extractfile(member) as stream: sha = hashlib.file_digest(stream, "sha256").hexdigest()
            assert sha == record["sha256"] and member.size == record["bytes"]
    package = copy_verified(local, destination / "observations.tar.gz")
    result = {"created_at_uk": stamp(), "destination": str(destination), "files": records, "bytes": sum(r["bytes"] for r in records),
              "package": package,
              "checkpoints": checkpoints, "checkpoint_verification": "Full hashes verified on asynchronous durable moves; persistent sizes and existence checked again here.",
              "preservation": "All rates, seeds, raw sampled tokens, unsuccessful groups, checkpoints, protocols, source snapshots, diagnostic probes and original/strict test results retained. No data deleted."}
    save(results / "archive.json", result); save(destination / "archive.json", result)
    (destination / "README.md").write_text("# r5s research archive\n\nExtract `observations.tar.gz` into the project root. It contains `results/scaling_r5s/`, `experiments/`, `reports/`, and the project lockfile. Every member was SHA256-verified, and the complete archive was verified again after copying to durable storage. The archive is packed to avoid tens of thousands of small network writes. Original data and the local archive remain retained.\n\nFull checkpoints remain at the durable paths in archive.json. Their node-local paths are symlinks to the durable files. Each run contains full resume state, and the per-stage source snapshots and protocols preserve its exact implementation.\n\nThe canonical report is reports/self-generation-scaling.html, built by experiments/build_verified_scaling_report.py. Its historical s4g section uses the separately archived s4g report; that original study and its checkpoint archive remain preserved. Use a node-local virtual environment and the workspace instructions.\n")
    print(json.dumps({"files": len(records), "bytes": result["bytes"], "compressed_bytes": package["bytes"], "checkpoints": len(checkpoints), "destination": str(destination)}))


if __name__ == "__main__": main()
