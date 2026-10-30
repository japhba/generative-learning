"""Freeze independent confirmation from the completed symmetric rate search."""

import json

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp
from train_scaling_verified import prepare


def main():
    pilot = ROOT / "results/scaling_r5s/pilot"
    analysis = json.loads((pilot / "analysis.json").read_text())
    assert all(r["complete"] for r in analysis["runs"]) and len(analysis["runs"]) == 6
    selected = analysis["selected"]
    output = ROOT / "results/scaling_r5s/confirmation"
    prepare(output, "confirmation", [99601, 99602, 99603], 4096, {source: [rate] for source, rate in selected.items()})
    save(output / "selection.json", {"frozen_before_confirmation_at_uk": stamp(), "selected_rates": selected,
         "pilot_analysis_sha256": digest(pilot / "analysis.json"), "pilot_results": analysis,
         "selection_scope": "Best of the same three predeclared rates for each source, not a claim of global optimizer optimality.",
         "analytical_changes": "Bootstrap common whole held-out questions within the designed answer strata. No change to learner, task or test budget based on confirmation outcomes."})
    for name in ("prepare_verified_confirmation.py", "analyze_scaling_verified.py", "evaluate_scaling_verified.py", "VERIFIED_SCALING_THEORY.md"):
        (output / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    print(json.dumps({"output": str(output), "selected_rates": selected}), flush=True)


if __name__ == "__main__": main()
