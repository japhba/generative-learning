"""Run a locked queue of independent dictionaries using the frozen training engine."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket

from dotenv import load_dotenv

import train_zipf_finetuning as engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "work"))
    parser.add_argument("--family", choices=engine.MODELS, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--root", type=Path, default=engine.ROOT / "results/zipf_finetuning")
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    assert engine.hashes() == protocol["source_sha256"]
    output = args.root / args.family
    selection = json.loads((output / "selection.json").read_text())
    assert selection["status"] == "passed"
    queue_path = output / "queue.json"
    if args.mode == "prepare":
        assert not queue_path.exists()
        tasks = []
        for live in (False, True):
            for seed in engine.MAIN_SEEDS:
                for arm in (("on_policy",) if live else engine.CURRICULA):
                    folder = output / (f"live/{seed}" if live else f"main/{seed}_{arm}")
                    status = "pending"
                    if folder.exists():
                        complete = json.loads((folder / "complete.json").read_text())
                        assert complete["complete"] and complete["source_sha256"] == protocol["source_sha256"]
                        status = "complete"
                    tasks.append({"seed": seed, "arm": arm, "live": live, "folder": str(folder), "status": status})
        engine.save(queue_path, tasks)
        print(queue_path)
        return
    load_dotenv(Path.home() / ".env")
    os.environ["HF_HUB_CACHE"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cache = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/zipf_finetuning"
    learner = engine.Learner(args.family, args.device)
    learner.make_trie()
    while True:
        with (output / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            tasks = json.loads(queue_path.read_text())
            task = next((row for row in tasks if row["status"] == "pending"), None)
            if task is None: break
            task.update(status="running", pid=os.getpid(), host=socket.gethostname(), claimed_at=datetime.now(timezone.utc).isoformat())
            engine.save(queue_path, tasks)
            task = dict(task)
        engine.run(learner, args.family, task["seed"], task["arm"], selection["learning_rate"], selection["repeats"],
                   Path(task["folder"]), True, cache, save_model=task["seed"] == 4000, live=task["live"])
        with (output / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            tasks = json.loads(queue_path.read_text())
            row = next(row for row in tasks if row["folder"] == task["folder"])
            assert row["status"] == "running" and row["pid"] == os.getpid()
            row.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
            engine.save(queue_path, tasks)


if __name__ == "__main__": main()
