"""Freeze an identity-only clean subset before any final-test model outputs."""

from collections import Counter
import json

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


def main():
    pilot = ROOT / "results/scaling_r5s/pilot"; output = ROOT / "results/scaling_r5s/confirmation"
    protocol = json.loads((output / "protocol.json").read_text())
    assert not any((output / str(c["seed"]) / c["name"] / "test_progress.json").exists() for c in protocol["cells"])
    excluded = {r["id"] for path in (pilot / "monitor.json", pilot / "99501/train.json") for r in json.loads(path.read_text())}
    original = json.loads((output / "test.json").read_text()); strict = [r for r in original if r["id"] not in excluded]
    assert len(strict) == 2444
    save(output / "strict_test.json", strict)
    save(output / "strict_test_protocol.json", {"frozen_at_uk": stamp(), "strict_test_sha256": digest(output / "strict_test.json"),
         "original_test_sha256": digest(output / "test.json"), "retained_by_class": dict(Counter(r["label"] for r in strict)),
         "rule": "Retain every original final-test question absent from all r5s pilot training and pilot rate-selection monitoring. Select by literal IDs only, without any final-test model outputs. Also disjoint from all confirmation training and monitoring by the original split.",
         "estimator": "Equal-weight mean of the three class-conditional means. Bootstrap common whole questions within each answer class, and paired training seeds. No downsampling or outcome-based exclusion.",
         "timing": "Added after the first two confirmation pairs reached about256 monitoring updates. No final-test model predictions existed or were read. Original4095-question pool and all original analyses remain retained alongside this stricter validation.",
         "scope": "This is a clean holdout within batch r5s, not a claim that similar counting inputs were absent from pretraining or earlier project explorations."})
    print(json.dumps({"strict_questions": len(strict), "counts": dict(Counter(r["label"] for r in strict))}))


if __name__ == "__main__": main()
