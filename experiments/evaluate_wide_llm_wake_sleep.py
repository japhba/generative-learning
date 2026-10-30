"""Evaluate actual saved weights under the larger common compute allowance."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from llm_wake_sleep import MODEL, ROOT, data, evaluate, initialize, save


directory = ROOT / "results/llm_wake_sleep"
baseline = json.loads((directory / "confirmation_elbo_s9610/run.json").read_text())
allowance = baseline["rows"][-1]["flops"] * 8
fresh = json.loads((directory / "fresh_three_example_cohort.json").read_text())
model = initialize(9610)
results = []
for seed, arm in tqdm([(s,a) for s in (9610,9611,9612) for a in ("elbo","sleep","forward")], desc="Wide-budget saved checkpoints"):
    output = directory / f"extended_{arm}_s{seed}"
    run = json.loads((output / "run.json").read_text())
    resume = json.loads((output / "resume.json").read_text())
    assert run["rows"][-1]["step"] == 4096 or run["stop_reason"] != "budget"
    assert resume["step"] == run["rows"][-1]["step"]
    unit = run["rows"][1]["flops"] / run["rows"][1]["step"]
    chosen_step = min(resume["step"], int(allowance / unit / 32) * 32)
    checkpoint = Path(resume["persistent"]).with_name(f"step_{chosen_step}.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.restore(state["adapters"]); model.usage = state["usage"]
    assert model.compute() <= allowance
    two, three = evaluate(model, data()[1]), evaluate(model, fresh)
    result = {"seed":seed, "arm":arm, "step":chosen_step, "actual_flops":model.compute(), "allocated_flops":allowance,
              "model":MODEL, "model_revision":model.model.config._commit_hash,
              "retained_early_stop":run["stop_reason"] != "budget", "training_stop_reason":run["stop_reason"], "checkpoint":str(checkpoint),
              "two_examples":two, "three_examples":three}
    save(directory / f"wide_matched_{arm}_s{seed}.json", result)
    results.append({**result, "two_examples":{k:v for k,v in two.items() if k!="records"}, "three_examples":{k:v for k,v in three.items() if k!="records"}})
save(directory / "wide_matched_evaluation.json", results)
print(json.dumps(results, indent=2))
