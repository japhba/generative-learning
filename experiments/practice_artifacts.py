"""Build the main figure, hash manifest, and persistent copy of research outputs."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


def main():
    load_dotenv(Path.home() / ".env")
    root = Path(__file__).resolve().parents[1]
    fast = Path(os.environ["FAST_CACHE_DIR"])
    summary_dir = fast / "generative_practice_synthesis"
    archive_root = Path(os.environ["CACHE_DIR"]) / "generative-learning" / "practice_research"
    source = fast / "generative-learning" / "practice_density_sigmoid_warm100"
    data = np.load(source / "trajectories.npz")
    metadata = json.loads((source / "metadata.json").read_text())
    early_stopping = {"scope": "Descriptive minima of the seed-mean true-KL curve, selected using unavailable true-target knowledge; not admissible stopping rules.", "source": str(source), "methods": {}}
    for method in ["direct_mle", "joint_importance"]:
        values = data[method][:, :, 0].mean(axis=0)
        index = values.argmin()
        early_stopping["methods"][method] = {"minimum_mean_true_kl": float(values[index]), "update": int(data["steps"][index]), "final_mean_true_kl": float(values[-1])}
    (summary_dir / "mle_early_stopping_audit.json").write_text(json.dumps(early_stopping, indent=2) + "\n")
    fig, ax = plt.subplots()
    for method, label in [("joint_importance", "Local importance practice"), ("direct_mle", "Continue fitting real observations"), ("unfiltered", "Unfiltered self-imitation"), ("score", "Positive score weighting")]:
        values = data[method][:, :, 0]
        line, = ax.plot(data["steps"], values.mean(axis=0), label=label)
        low, high = np.quantile(values, [0.1, 0.9], axis=0)
        ax.fill_between(data["steps"], low, high, color=line.get_color(), alpha=0.15)
    ax.axhline(np.mean(metadata["teacher_true_kl"]), linestyle="--", color="black", label="Frozen learned density")
    ax.set(xlabel="Practice updates after real-data pretraining", ylabel="KL(true distribution || generator)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "tex" / "archive" / "figures" / "practice-density-main-learning.png")
    plt.close(fig)
    data_roots = [fast / name for name in ("generative_practice_atoms", "generative_practice_hidden", "generative_practice_context", "generative_practice_local_recognizer", "generative_practice_gradient_audit", "generative_practice_density_audit", "generative_practice_density_reproduction", "generative_learning_practice_theory_inverse", "generative_practice_spectral_audit", "generative_practice_depth_modes", "generative_practice_spectral_theory_audit")]
    data_roots += sorted((fast / "generative-learning").glob("practice_density*"))
    files = []
    for data_root in data_roots:
        for path in sorted(data_root.rglob("*")):
            if path.is_file():
                files.append((path, archive_root / "runs" / data_root.name / path.relative_to(data_root)))
    for path in sorted(root.glob("experiments/practice*.py")):
        if path.name != "practice_plot.py":
            files.append((path, archive_root / "code" / path.name))
    for path in sorted(root.glob("tex/**/practice*.tex")):
        files.append((path, archive_root / "documents" / path.name))
    for path in sorted(root.glob("tex/**/figures/practice*.png")):
        files.append((path, archive_root / "documents" / "figures" / path.name))
    for path in sorted(root.glob("reports/practice-theory-*.png")):
        files.append((path, archive_root / "figures" / path.name))
    for name in ("atom_summary.json", "manifest.json", "mle_early_stopping_audit.json"):
        path = summary_dir / name
        files.append((path, archive_root / "summaries" / name))
    files.append((fast / "generative-learning" / "practice_density_inventory.json", archive_root / "summaries" / "practice_density_inventory.json"))
    files.append((summary_dir / "build" / "verification.json", archive_root / "summaries" / "verification.json"))
    entries = []
    for source_path, target_path in tqdm(files, desc="Archive and verify research artifacts"):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        target_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
        assert source_hash == target_hash
        entries.append({"source": str(source_path), "persistent_copy": str(target_path), "bytes": source_path.stat().st_size, "sha256": source_hash})
    inputs = []
    for path in sorted((root / "tex" / "notes" / "pdf").glob("*.pdf")):
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"uk_created": datetime.now(ZoneInfo("Europe/London")).isoformat(), "research_start_uk": "2026-10-18T16:20:52+01:00", "main_document": str(root / "tex" / "archive" / "practice_model.tex"), "persistent_root": str(archive_root), "file_count": len(entries), "bytes": sum(e["bytes"] for e in entries), "input_papers": inputs, "reproduction": {"environment_python": "/var/tmp/jbauer/venvs/generative-learning/bin/python", "primary_script": str(root / "experiments" / "practice_density_sigmoid.py"), "primary_arguments": metadata["arguments"], "exact_snapshot_note": "Each density run stores run_source.py and, for the sigmoid model, shared_source.py. To rerun the exact version, place run_source.py under its original script filename and shared_source.py as practice_density.py in a fresh directory; use saved arguments with a fresh output directory.", "context_script": str(root / "experiments" / "practice_context_memory.py"), "independent_checks": [str(fast / "generative_practice_gradient_audit" / "checks.json"), str(fast / "generative_practice_density_audit" / "checks.json")]}, "files": entries}
    manifest["spectral_extension"] = {"document": str(root / "tex" / "archive" / "practice_spectral.tex"), "weight_audit": str(root / "experiments" / "practice_spectral_audit.py"), "depth_control": str(root / "experiments" / "practice_depth_modes.py"), "independent_theory_checks": str(root / "experiments" / "practice_spectral_theory_audit.py"), "interpretation": "Weight modes and nonlinear aligned population dynamics; depth factorization changes optimization while preserving the generator family."}
    path = summary_dir / "research_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(path, archive_root / "research_manifest.json")
    print(json.dumps({"manifest": str(path), "persistent_manifest": str(archive_root / "research_manifest.json"), "files": len(entries), "bytes": manifest["bytes"]}, indent=2))


if __name__ == "__main__":
    main()
