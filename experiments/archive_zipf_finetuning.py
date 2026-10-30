"""Archive the eight prospectively selected final models and verify every file."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from tqdm.auto import tqdm


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main():
    load_dotenv(Path.home() / ".env")
    root = Path(__file__).resolve().parents[1] / "results/zipf_finetuning"
    destination = Path(os.environ["CACHE_DIR"]) / "generative-learning/zipf_finetuning_20260906"
    destination.mkdir(parents=True, exist_ok=False)
    cases = []
    for family in ("qwen", "smollm"):
        for arm in ("iid", "ranked", "historical_qwen", "on_policy"):
            folder = root / family / ("live/4000" if arm == "on_policy" else f"main/4000_{arm}")
            complete = json.loads((folder / "complete.json").read_text())
            assert complete["complete"] and complete["seed"] == 4000
            cases.append((family, arm, complete))
    manifest = []
    for family, arm, complete in tqdm(cases, desc="Archiving final models"):
        source = Path(complete["checkpoint"])
        target = destination / family / source.name
        target.mkdir(parents=True, exist_ok=False)
        files = []
        for path in tqdm(sorted(source.iterdir()), desc=f"{family}/{arm}", leave=False):
            assert path.is_file()
            output = target / path.name
            shutil.copy2(path, output)
            digest = sha256(path)
            assert output.stat().st_size == path.stat().st_size and sha256(output) == digest
            files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
        manifest.append({"family": family, "arm": arm, "world": 4000, "source": str(source), "archive": str(target), "files": files,
                         "source_sha256": complete["source_sha256"]})
    result = {"complete": True, "completed_at": datetime.now(timezone.utc).isoformat(), "archive": str(destination),
              "total_bytes": sum(row["bytes"] for case in manifest for row in case["files"]), "models": manifest}
    (root / "archive.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"archive": str(destination), "models": len(manifest), "bytes": result["total_bytes"]}))


if __name__ == "__main__": main()
