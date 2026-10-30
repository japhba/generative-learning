"""Training-only acquisition calibration after diagnosing the EOS boundary bias."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

import train_zipf_finetuning as engine
from zipf_task import P, ideal_curves

RESULTS = engine.ROOT / "results/zipf_finetuning"
SEEDS = [52000, 52001, 52002, 52003]


def read(path): return json.loads(path.read_text())


def install_boundary(learner):
    tokens = learner.tokenizer.encode("\n", add_special_tokens=False)
    assert len(tokens) == 1
    newline = tokens[0]
    original = learner.model.generate
    records = []
    def generate(**kwargs):
        prefix = kwargs["input_ids"].shape[1]
        allowed = kwargs["prefix_allowed_tokens_fn"]
        def boundaries(batch_id, sequence):
            choices = allowed(batch_id, sequence)
            return [*choices, newline] if learner.tokenizer.eos_token_id in choices else choices
        kwargs["prefix_allowed_tokens_fn"] = boundaries
        kwargs["eos_token_id"] = [learner.tokenizer.eos_token_id, newline]
        result = original(**kwargs)
        records.append({"generated_tokens": result[0, prefix:].tolist(), "termination": learner.tokenizer.decode(result[0, -1:].tolist())})
        # Adapt the boundary to the frozen integer parser only after generation.
        result[0, -1] = learner.tokenizer.eos_token_id
        return result
    learner.model.generate = generate
    return records


def run(family, seed, device, protocol):
    assert seed in SEEDS and protocol["source_sha256"] == engine.hashes()
    selection = read(RESULTS / family / "selection.json")
    learner = engine.Learner(family, device)
    learner.make_trie()
    boundaries = install_boundary(learner)
    output = RESULTS / family / "newline_pilot" / str(seed)
    cache = Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/zipf_finetuning"
    engine.run(learner, family, seed, "on_policy_newline", selection["learning_rate"], selection["repeats"], output, False, cache, live=True)
    assert len(boundaries) == 128
    engine.save(output / "boundaries.json", boundaries)
    engine.save(output / "decoder.json", {"driver_sha256": protocol["driver_sha256"], "boundary_rule": "Allow newline or EOS at a valid integer boundary; retain original prompt, legal prefixes, greedy decoding, repeats, feedback and training rule."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "analyze"])
    parser.add_argument("--family", choices=[*engine.MODELS, "both"], default="both")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    os.environ["HF_HUB_CACHE"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path = RESULTS / "newline_pilot_protocol.json"
    if args.mode == "prepare":
        assert not path.exists()
        protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "pilot_seeds": SEEDS,
                    "reason": "After the original cohort, a query-only diagnostic found that EOS forced long keys; allowing newline yielded repeated key1 in both pretrained models. Check the corrected boundary with actual feedback before attributing acquisition failure to the learner.",
                    "design": "Four fresh dictionaries per family, selected retention hyperparameters unchanged. Greedy legal integer acquisition now permits newline as well as EOS. No automatic novelty constraint, ranking, success filtering, test predictions or exponent fitting.",
                    "decision": "Acquisition passes calibration only if every pilot covers more test probability mass at128 requests than the exact expected IID coverage. A failed acquisition calibration does not support estimating a beneficial scaling exponent for this policy.",
                    "expected_iid_known_mass_at128": 1 - ideal_curves([128])["iid_errors"][0] / .75,
                    "source_sha256": engine.hashes(), "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "diagnostics_sha256": {family: hashlib.sha256((RESULTS / family / "query_termination_diagnostic.json").read_bytes()).hexdigest() for family in engine.MODELS}}
        engine.save(path, protocol)
        return
    protocol = read(path)
    assert protocol["driver_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.mode == "run":
        for family in (engine.MODELS if args.family == "both" else [args.family]):
            run(family, args.seed, args.device, protocol)
            gc.collect()
        return
    results = {}
    for family in engine.MODELS:
        cases = []
        for seed in SEEDS:
            folder = RESULTS / family / "newline_pilot" / str(seed)
            completed = read(folder / "complete.json")
            config = read(folder / "config.json")
            assert completed["complete"] and not config["include_test"] and config["live"]
            data, record = read(folder / "observations.json"), read(folder / "n128.json")
            assert len(data["queries"]) == len(data["targets"]) == 128
            assert data["targets"] == [engine.oracle(seed, key) for key in data["queries"]]
            assert record["keys"] == sorted(set(data["queries"]))
            probs = np.array(record["probabilities"])
            accuracy = np.mean(probs.argmax(axis=1) == [engine.oracle(seed, key) for key in record["keys"]])
            mass = float(P[np.array(record["keys"]) - 1].sum())
            np.testing.assert_allclose([mass, accuracy], [record["metrics"]["known_mass"], record["metrics"]["seen_accuracy"]])
            boundaries = read(folder / "boundaries.json")
            assert len(boundaries) == 128
            cases.append({"seed": seed, "unique_keys": len(record["keys"]), "known_mass": mass, "seen_accuracy": float(accuracy),
                          "queries": data["queries"], "boundary_counts": {value: sum(row["termination"] == value for row in boundaries) for value in sorted(set(row["termination"] for row in boundaries))},
                          "counters": completed["counters"]})
        results[family] = {"passes_acquisition_calibration": all(row["known_mass"] > protocol["expected_iid_known_mass_at128"] for row in cases), "pilots": cases}
    engine.save(RESULTS / "newline_pilot_summary.json", {"complete": True, "protocol": protocol, "families": results})
    print(json.dumps({family: {"passed": row["passes_acquisition_calibration"], "coverage": [case["known_mass"] for case in row["pilots"]]} for family, row in results.items()}))


if __name__ == "__main__": main()
