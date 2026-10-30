"""Persist completed finite-sample runs and verify every copied file."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from tqdm.auto import tqdm


def main():
    load_dotenv(Path.home() / ".env")
    roots = [Path("results") / name for name in ("finite_sample_20260907", "finite_sample_confirmation_20260907", "finite_sample_axis_20260907")]
    destination = Path(os.environ["CACHE_DIR"]) / "generative-learning" / "finite_sample_20260907_bundle"
    destination.mkdir(parents=True, exist_ok=False)
    pairs = []
    for root in roots:
        protocol = json.loads((root / "protocol.json").read_text())
        raw = Path(protocol["raw_directory"])
        for path in sorted(raw.glob("*.npz")): pairs.append((path, destination / root.name / "raw" / path.name))
        for path in sorted(root.iterdir()):
            assert path.is_file()
            pairs.append((path, destination / root.name / "metadata" / path.name))
    sources = [*Path("experiments").glob("*finite_sample*.py"), *Path("experiments").glob("FINITE_SAMPLE*.md"), Path("experiments/README.md")]
    for path in sources: pairs.append((path, destination / "source" / path.name))
    for path in sorted(Path("reports").glob("finite-sample-*")): pairs.append((path, destination / "reports" / path.name))
    records = []
    for source, target in tqdm(pairs, desc="Archive and verify finite samples"):
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == expected
        records.append({"source": str(source.resolve()), "persistent_path": str(target), "sha256": actual, "bytes": target.stat().st_size})
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(destination), "files": records,
                "bytes": sum(row["bytes"] for row in records), "verification": "SHA256 of each persistent copy equals its source."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for root in roots:
        (root / "archive.json").write_text(json.dumps({"persistent_root": str(destination), "manifest": str(destination / "manifest.json"),
                                                      "run_raw_directory": str(destination / root.name / "raw"), "total_bundle_bytes": manifest["bytes"], "all_hashes_verified": True}, indent=2) + "\n")
    print(json.dumps({"persistent_root": str(destination), "files": len(records), "bytes": manifest["bytes"], "verified": True}, indent=2))


if __name__ == "__main__": main()
