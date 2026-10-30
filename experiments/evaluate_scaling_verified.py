"""Evaluate untouched questions only after all confirmation trajectories are fixed."""

import argparse
import json
from pathlib import Path
import time

from tqdm.auto import tqdm

from run_elicitation_pilot import ROOT, digest, save
from scaling_digit_task import encode
from train_bestn import stamp
from train_scaling_verified import initialize


def main(output, cell_index):
    import torch
    protocol, student = initialize(output)
    assert protocol["stage"] == "confirmation"
    assert all((output / str(c["seed"]) / c["name"] / "complete.json").exists() for c in protocol["cells"])
    cell = protocol["cells"][cell_index]; root = output / str(cell["seed"]) / cell["name"]
    history = json.loads((root / "complete.json").read_text())
    rows = encode(student, json.loads((output / "test.json").read_text()))
    markers = [json.loads(p.read_text()) for p in root.glob("*.pt.local.json")]
    steps = sorted({m["step"] for m in markers})
    assert steps[0] == 0
    evaluations = []
    result = {"opened_final_test_at_uk": stamp(), "cell": cell, "training_finished_at_uk": history["finished_at_uk"],
              "protocol_sha256": digest(output / "protocol.json"), "test_sha256": digest(output / "test.json"),
              "evaluator_sha256": digest(ROOT / "experiments/evaluate_scaling_verified.py"),
              "initialization": "Step0 uses the same pretrained initialization already loaded by the shared trainer initializer, avoiding a redundant full checkpoint read. Later steps load exact durable resume checkpoints.", "evaluations": evaluations}
    for step in tqdm(steps, desc=f"Untouched test: {cell['seed']} {cell['name']}"):
        start = time.monotonic()
        marker = max((m for m in markers if m["step"] == step), key=lambda m: m["attempt"])
        budget_keys = ("accepted", "training_tokens", "accepted_training_tokens", "proposal_prompt_tokens", "candidates", "presentations")
        if step == 0:
            budget = {key: 0 for key in budget_keys}
        else:
            state = torch.load(marker["persistent"], map_location="cpu", weights_only=False)
            student.model.load_state_dict(state["model"], strict=True)
            budget = {key: state["history"][key] for key in budget_keys}
            del state
        record = {"step": step, **budget, **student.evaluate(rows), "seconds": time.monotonic() - start}
        evaluations.append(record); save(root / "test_progress.json", result)
        print(json.dumps({"step": step, "error": record["error"], "greedy_error": 1 - record["greedy_accuracy"]}), flush=True)
    save(root / "test.json", {**result, "finished_at_uk": stamp()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=ROOT / "results/scaling_r5s/confirmation"); parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args(); main(args.output, args.cell_index)
