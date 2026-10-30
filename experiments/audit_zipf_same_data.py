"""Replay a preselected live world without generation and compare its actual weights."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

import train_zipf_finetuning as engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=engine.MODELS, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4000)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    os.environ["HF_HUB_CACHE"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = engine.ROOT / "results/zipf_finetuning"
    source = root / args.family / "live" / str(args.seed)
    complete = json.loads((source / "complete.json").read_text())
    observations = json.loads((source / "observations.json").read_text())
    assert complete["source_sha256"] == engine.hashes()
    destination = root / args.family / "same_data_audit" / str(args.seed)
    destination.mkdir(parents=True, exist_ok=False)
    learner = engine.Learner(args.family, args.device)
    learner.reset(args.seed + 410000, complete["learning_rate"])
    replay = np.load(source / "replay_indices.npy")
    queries, targets, checkpoints = [], [], []
    for budget in tqdm([0, *engine.BUDGETS], desc=f"Replay audit {args.family}"):
        for index in range(len(queries), budget):
            queries.append(observations["queries"][index])
            targets.append(observations["targets"][index])
            learner.learn(queries, targets, replay[index])
        result = engine.evaluate(learner, args.seed, queries, destination, budget, True)
        reference = json.loads((source / f"n{budget}.json").read_text())
        assert result["keys"] == reference["keys"] and result["queried"] == reference["queried"]
        difference = np.abs(np.array(result["probabilities"]) - np.array(reference["probabilities"]))
        checkpoints.append({"budget": budget, "max_probability_difference": float(difference.max()), "identical": bool(np.all(difference == 0))})
    from safetensors import safe_open
    state = learner.model.state_dict()
    compared, differences = set(), []
    files = sorted(Path(complete["checkpoint"]).glob("*.safetensors"))
    assert files
    for path in tqdm(files, desc="Comparing saved weights"):
        with safe_open(path, framework="pt", device="cpu") as saved:
            for name in tqdm(saved.keys(), desc=path.name, leave=False):
                original = saved.get_tensor(name)
                current = state[name].cpu()
                assert original.shape == current.shape and original.dtype == current.dtype
                difference = float((original - current).abs().max())
                differences.append({"name": name, "numel": current.numel(), "max_absolute_difference": difference})
                compared.add(name)
    aliases = {}
    for name in set(state) - compared:
        tensor = state[name]
        alias = next(key for key in compared if state[key].data_ptr() == tensor.data_ptr() and state[key].shape == tensor.shape and state[key].stride() == tensor.stride())
        aliases[name] = alias
    identical = all(row["identical"] for row in checkpoints) and all(row["max_absolute_difference"] == 0 for row in differences)
    result = {"passed": identical, "family": args.family, "seed": args.seed, "completed_at": datetime.now(timezone.utc).isoformat(),
              "interpretation": "Identical correct pairs, replay draws and updates, with the key-selection and guessing forwards omitted. This is a numerical same-data control, not an additional independent experimental world.",
              "source_sha256": engine.hashes(), "checkpoints": checkpoints, "weight_tensors": differences, "tied_aliases": aliases,
              "training_presentations": learner.count["training_presentations"], "optimizer_updates": learner.count["optimizer_updates"]}
    engine.save(destination / "audit.json", result)
    assert identical, "Saved audit records identify a numerical difference; investigate before claiming same-data identity."
    print(json.dumps({"passed": identical, "family": args.family, "seed": args.seed, "tensors_compared": len(compared), "path": str(destination / "audit.json")}))


if __name__ == "__main__": main()
