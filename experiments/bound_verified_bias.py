"""Certify errors that no global answer-logit bias can remove on pilot data."""

import json

import numpy as np

from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def minimum_pair_errors(logits, labels, a, b):
    keep = (labels == a) | (labels == b)
    score = logits[keep, a] - logits[keep, b]; target = labels[keep]
    # Start with every example assigned b, then move the threshold through
    # whole equal-score groups. These are every possible pairwise bias rule.
    errors = int(np.sum(target == a)); best = errors
    for value in np.unique(score)[::-1]:
        group = score == value
        errors += int(np.sum(target[group] == b)) - int(np.sum(target[group] == a))
        best = min(best, errors)
    return best


def main():
    root = ROOT / "results/scaling_r5s/bias_probe"
    rows = json.loads((root / "monitor.json").read_text())
    logits = np.array([r["digit_logits"] for r in rows]); labels = np.array([int(r["label"]) for r in rows])
    pairs = [{"classes": [a, b], "unavoidable_pair_errors": minimum_pair_errors(logits, labels, a, b)} for a, b in ((0, 1), (0, 2), (1, 2))]
    lower = max(max(r["unavoidable_pair_errors"] for r in pairs), int(np.ceil(sum(r["unavoidable_pair_errors"] for r in pairs) / 2)))
    result = {"checked_at_uk": stamp(), "questions": len(rows), "pairs": pairs, "unavoidable_multiclass_errors": lower,
              "global_bias_greedy_accuracy_upper_bound": 1 - lower / len(rows),
              "proof": "Any globally biased multiclass predictor induces one threshold in each pairwise logit difference. Pairwise minimum errors are necessary errors even if the third answer is ignored. Global errors are at least every pair's minimum, and at least half their sum because one wrong example belongs to only two pairs. Equal-score groups move together, so ties do not create spurious attainable thresholds.",
              "scope": "A deterministic bound on the already exposed pilot monitoring cohort, allowing arbitrary constant answer biases and an optimistic restriction to answers0/1/2. It excludes global-prior correction as a complete explanation; it does not prove that hidden features must change rather than an input-dependent readout."}
    fit = json.loads((root / "results.json").read_text())
    assert fit["after"]["monitor"]["greedy_accuracy"] <= result["global_bias_greedy_accuracy_upper_bound"]
    save(root / "bound.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
