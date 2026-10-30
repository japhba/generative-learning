"""At fixed LLM weights, compare the two accepted-data update directions."""

import json

import numpy as np
from tqdm.auto import tqdm

from bestn_llm import dot
from run_elicitation_pilot import ROOT, save
from scaling_digit_task import encode
from train_bestn import stamp
from train_scaling_verified import initialize


def weighted_gradient(student, rows, q):
    import torch
    student.model.zero_grad(set_to_none=True)
    coefficients = torch.tensor(q / q.sum(), device="cuda", dtype=torch.float32)
    loss_value = 0.
    for offset in tqdm(range(0, len(rows), 16), desc="Expected accepted-example gradient", leave=False):
        batch = rows[offset:offset + 16]
        target = torch.tensor([r["target_id"] for r in batch], device="cuda")
        loss = (torch.nn.functional.cross_entropy(student.logits(batch), target, reduction="none") * coefficients[offset:offset + 16]).sum()
        loss.backward(); loss_value += float(loss.detach())
    gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in student.model.parameters()]
    student.model.zero_grad(set_to_none=True)
    return loss_value, gradients


def main():
    import torch
    output = ROOT / "results/scaling_r5s/pilot"
    protocol, student = initialize(output)
    run = output / "99501/current_5.6e-06"
    destination = ROOT / "results/scaling_r5s/geometry"; destination.mkdir(parents=True, exist_ok=False)
    rows = encode(student, json.loads((output / "99501/train.json").read_text())[-1024:])
    evaluation = encode(student, json.loads((output / "monitor.json").read_text()))
    initial = student.evaluate(rows); p0 = np.array([np.exp(r["correct_logprob"]) for r in initial["records"]]); q0 = p0 * (2 - p0)
    results = {"started_at_uk": stamp(), "status": "Exploratory read-only diagnostic on pilot checkpoints; no final confirmation test data or learner updates used.",
               "probe_inputs": [r["id"] for r in rows], "initial_probe": initial, "checkpoints": [], "eta": 5.6e-6}
    for step in (0, 128, 512):
        markers = [json.loads(p.read_text()) for p in run.glob("*.pt.local.json")]
        marker = max((m for m in markers if m["step"] == step), key=lambda m: m["attempt"])
        state = torch.load(marker["persistent"], map_location="cpu", weights_only=False)
        student.model.load_state_dict(state["model"], strict=True); del state
        base = [p.detach().clone() for p in student.model.parameters()]
        observed = student.evaluate(rows); p = np.array([np.exp(r["correct_logprob"]) for r in observed["records"]])
        before, eval_gradient = student.success_gradient(evaluation)
        record = {"step": step, "probe": observed, "eval_success": before, "sources": {}}
        for source, q in (("current", p * (2 - p)), ("frozen", q0)):
            loss, gradient = weighted_gradient(student, rows, q)
            overlap = dot(gradient, eval_gradient); gn = dot(gradient, gradient); en = dot(eval_gradient, eval_gradient)
            prediction = 5.6e-6 * overlap
            with torch.no_grad():
                for parameter, g in zip(student.model.parameters(), gradient, strict=True): parameter.add_(g, alpha=-5.6e-6)
            after = student.evaluate(evaluation)
            with torch.no_grad():
                for parameter, value in zip(student.model.parameters(), base, strict=True): parameter.copy_(value)
            record["sources"][source] = {"expected_acceptance": float(q.mean()), "weighted_nll": loss, "gradient_norm_squared": gn,
                 "eval_gradient_norm_squared": en, "gradient_overlap": overlap, "gradient_cosine": overlap / np.sqrt(gn * en),
                 "predicted_population_gain": prediction, "observed_population_gain": after["accuracy"] - before,
                 "after_population_step": after}
            del gradient
        del base, eval_gradient
        results["checkpoints"].append(record); save(destination / "progress.json", results)
        print(json.dumps({"step": step, "sources": {s: {k: v for k, v in r.items() if k != "after_population_step"} for s, r in record["sources"].items()}}), flush=True)
    save(destination / "results.json", {**results, "finished_at_uk": stamp()})


if __name__ == "__main__": main()
