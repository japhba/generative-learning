"""Preserve s4g observations, code, plots, and interrupted checkpoint pieces."""

import argparse
import json
import os
from pathlib import Path

from tqdm.auto import tqdm

from archive_bestn import copy_verified
from run_elicitation_pilot import ROOT, save, setup
from train_bestn import stamp


RESULTS=ROOT/"results/scaling_s4g"


def main(partials):
    local=setup()/"scaling_s4g"
    destination=Path(os.environ["CACHE_DIR"])/"generative-learning/scaling_s4g/observations"
    if partials:
        files=[(p,destination/"interrupted_checkpoints"/p.relative_to(local)) for p in sorted(local.rglob("*.partial"))]
        assert files, "Run the partial-checkpoint archive on the H100 node that retained the interrupted files."
        records=[copy_verified(p,target) for p,target in tqdm(files,desc="Preserve interrupted checkpoint pieces")]
        save(RESULTS/"partial_checkpoint_archive.json",{"created_at_uk":stamp(),"files":records,"bytes":sum(r["bytes"] for r in records)})
        print(json.dumps({"files":len(records),"bytes":sum(r["bytes"] for r in records)}));return
    analysis=json.loads((RESULTS/"digit/analysis.json").read_text())
    assert len(analysis["confirmation"])==6 and all(r["complete"] for r in analysis["confirmation"].values())
    files=[(p,destination/"results"/p.relative_to(RESULTS)) for p in sorted(RESULTS.rglob("*"))
           if p.is_file() and p.name!="archive.json" and "__pycache__" not in p.parts]
    for stage in ("finite_sgd","finite_confirmation","gate_finite"):
        paths=list((local/stage).glob("*.npz"));assert paths
        files.extend((p,destination/"raw"/stage/p.name) for p in paths)
    sources=set((ROOT/"experiments").glob("*.py"))|set((ROOT/"experiments").glob("*.md"))
    files.extend((p,destination/"experiments"/p.name) for p in sorted(sources) if p.is_file())
    files.extend((p,destination/"reports"/p.name) for p in sorted((ROOT/"reports").glob("*scaling*")) if p.is_file())
    files.append((ROOT/"reports/index.html",destination/"reports/index.html"))
    files.extend((ROOT/name,destination/name) for name in ("pyproject.toml","uv.lock"))
    checkpoints=[]
    for path in sorted(RESULTS.rglob("*.pt.archive.json")):
        record=json.loads(path.read_text());assert Path(record["persistent"]).stat().st_size==record["bytes"]
        checkpoints.append(record)
    records=[copy_verified(p,target) for p,target in tqdm(files,desc="Archive and verify s4g")]
    result={"created_at_uk":stamp(),"destination":str(destination),"files":records,"bytes":sum(r["bytes"] for r in records),
            "checkpoints":checkpoints,"checkpoint_verification":"Full hashes verified during asynchronous archive and relocation; durable existence and size rechecked here.",
            "data_preservation":"No observations or checkpoints deleted. Scratch copies were moved to separately retained durable copies, with original paths preserved as symlinks. Interrupted pieces have their own archive."}
    save(RESULTS/"archive.json",result);save(destination/"archive.json",result)
    (destination/"README.md").write_text("# s4g research archive\n\nRestore `results/` into the project at `results/scaling_s4g/`, `experiments/` into `experiments/`, and `reports/` into `reports/`. The per-run source snapshots and protocols retain the exact training implementation. Restore each `raw/<stage>/` into `$FAST_CACHE_DIR/generative-learning/scaling_s4g/<stage>/` if replaying analyses from their original metadata paths.\n\nFull model checkpoints remain at the persistent paths in `archive.json`; those bytes were hashed on archive and the paths and sizes were rechecked here. Relocated scratch copies remain separately preserved and keep their original local paths via symlinks on the H100 node. Interrupted checkpoint pieces are stored in `interrupted_checkpoints/`. No data was deleted.\n\nUse the included project lockfile and create the virtual environment outside the NFS project, following the workspace instructions. The canonical local report is `reports/self-generation-scaling.html`.\n")
    print(json.dumps({"files":len(records),"bytes":result["bytes"],"checkpoints":len(checkpoints),"destination":str(destination)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--partial-checkpoints",action="store_true")
    main(parser.parse_args().partial_checkpoints)
