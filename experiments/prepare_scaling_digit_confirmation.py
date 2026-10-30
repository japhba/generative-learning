"""Freeze fresh-seed tests of the pilot's cubic gradient/error hypothesis."""

import json

from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp
from train_scaling_digit import prepare


def main():
    output = ROOT / "results/scaling_s4g/digit/confirmation"
    prepare(output, [99201, 99202, 99203], 2048)
    path = output / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["checks"] = sorted(set(protocol["checks"]) | {1024, 1536})
    protocol["gradient_checks"] = [0, 32, 64, 128, 256, 512, 1024, 1536]
    protocol["checkpoint_checks"] = [0, 128, 512, 1024, 2048]
    protocol["analysis"] = {
        "status": "Fresh-seed confirmation protocol, frozen before any confirmation outputs.",
        "declared_at_uk": stamp(),
        "primary": "Current-policy error power1/2, linked to cubic training/evaluation success-gradient overlap versus error. Fixed gamma3 forecast anchored at update128 with that run's measured error, learning rate and gradient overlap. Predict updates256,512,1024,2048; no later coefficient fit.",
        "pilot_exposure": "Pilot current trajectory had reached384 updates when the gamma3 forecast was recorded. Gamma3 has an independent Gaussian-RBM derivation, but its applicability to this LLM was suggested by the already observed32..128 gradient/error relation. Initial shared-logit forecast was overoptimistic and remains a separate retained failed reduction.",
        "comparison": "Both current and frozen proposals, full FP32 students, same candidate and optimizer budgets until each stopping rule triggers. A frozen floor is not an alternate raw-error exponent.",
        "windows": "Report log(error)/log(candidate) slopes over128..1024 and256..2048 when completed, plus adjacent-window variation. Stop-triggered missing endpoints remain missing. Fit error against measured compute tokens too; horizon1 makes learner token budgets proportional to candidates.",
        "gradient_geometry": "Fit log positive train/eval gradient overlap versus log held-out error over checkpoints128..1536. Report local one-step predictions and coefficient drift. A positive cross-distribution overlap is not mathematically a squared norm; report both gradient norms and their cosine.",
        "alternatives": "Compare gamma2 and gamma4 anchored forecasts, shifted exponential+floor and power+floor fits, and zero-floor power fits. Flexible fit comparisons are descriptive, with floor/exponent sensitivity; they do not identify an asymptotic floor by fiat.",
        "uncertainty": "Three independent training/task seeds; bootstrap seeds and whole held-out questions, preserving checkpoint pairing. These finite-window intervals do not establish asymptotic exponents or transfer to other tasks/models/ICL.",
        "failure": "Retain every seed, every task screen, and all stopping outcomes. Do not restart stationary runs. Claim a mechanistic bridge only to the degree predictions and measured geometry agree. No claim of no-feedback learning or generation-versus-recognition isolation."
    }
    source = ROOT / "experiments/prepare_scaling_digit_confirmation.py"
    (output / "source" / source.name).write_bytes(source.read_bytes())
    protocol["confirmation_driver_sha256"] = digest(source)
    save(path, protocol)
    print(json.dumps({"output": str(output), "seeds": protocol["seeds"], "steps": protocol["steps"], "declared_at_uk": protocol["analysis"]["declared_at_uk"]}))


if __name__ == "__main__": main()
