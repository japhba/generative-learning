"""Move complete local checkpoints to durable storage while preserving paths."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from tqdm.auto import tqdm

import train_bestn
from run_elicitation_pilot import ROOT, save, setup


def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def relocate(local, persistent, expected_hash, metadata):
    assert not local.is_symlink()
    assert sha(local) == expected_hash
    assert persistent.stat().st_size == local.stat().st_size
    # Keep both pre-existing data copies: move the scratch copy to its own
    # durable location, then preserve the old pathname as a symlink.
    moved = persistent.parent / "relocated_scratch" / local.name
    moved.parent.mkdir(parents=True, exist_ok=True)
    assert not moved.exists()
    shutil.move(local, moved)
    assert sha(moved) == expected_hash
    local.symlink_to(moved)
    save(metadata, {"moved_at_uk": train_bestn.stamp(), "original_path_now_symlink": str(local), "relocated_copy": str(moved),
                    "existing_archive": str(persistent), "sha256": expected_hash, "bytes": moved.stat().st_size,
                    "preservation": "Both data copies retained; original scratch pathname resolves to the relocated copy."})


def move_existing():
    setup()
    records = sorted((ROOT / "results/scaling_s4g").rglob("*.pt.archive.json"))
    for path in tqdm(records, desc="Move verified scratch checkpoints"):
        record = json.loads(path.read_text())
        local = Path(record["local"])
        if local.is_symlink(): continue
        relocate(local, Path(record["persistent"]), record["sha256"], path.with_name(path.name.replace(".archive.json", ".relocation.json")))
        print(json.dumps({"moved": str(local), "free_scratch_bytes": shutil.disk_usage(local.parent).free}), flush=True)


def run(seed, source, allocation):
    import train_scaling_digit
    original_archive = train_bestn.archive

    def archive_and_move(local, persistent, metadata):
        original_archive(local, persistent, metadata)
        record = json.loads(metadata.read_text())
        relocate(local, persistent, record["sha256"], metadata.with_name(metadata.name.replace(".archive.json", ".relocation.json")))

    train_bestn.archive = archive_and_move
    output = ROOT / "results/scaling_s4g/digit/confirmation"
    root = output / str(seed) / source
    resume = bool(list(root.glob("*.pt.local.json")))
    if not resume:
        # A run that failed before its first checkpoint has no recoverable
        # training state. Preserve its directory before initializing afresh.
        retained = root.with_name(source + "_failed_before_checkpoint")
        assert not retained.exists()
        root.rename(retained)
    train_scaling_digit.run(output, seed, source, resume, allocation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("move", "run"))
    parser.add_argument("--seed", type=int); parser.add_argument("--source", choices=("current", "frozen")); parser.add_argument("--allocation")
    args = parser.parse_args()
    if args.command == "move": move_existing()
    else: run(args.seed, args.source, args.allocation)
