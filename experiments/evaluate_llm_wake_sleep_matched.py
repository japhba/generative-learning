"""Evaluate actual budget-matched checkpoints and a new three-example cohort."""

import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from llm_wake_sleep import ROOT, initialize, data, truth, evaluate, save


directory = ROOT / "results/llm_wake_sleep"
contexts = list(itertools.permutations(range(10), 3))
rng = np.random.default_rng(9670); rng.shuffle(contexts)
fresh = [{"inputs":list(c), "outputs":[truth(z,x) for x in c], "latent":z} for c in contexts[:80] for z in range(8)]
save(directory / "fresh_three_example_cohort.json", fresh)
results = []
for seed in tqdm((9610,9611,9612), desc="Exact budget checkpoints"):
    for arm in ("elbo","sleep","forward"):
        original = directory / f"confirmation_{arm}_s{seed}"
        resume = json.loads((original / "resume.json").read_text())
        if arm == "sleep":
            output = directory / f"matched_sleep_s{seed}"
            subprocess.run([sys.executable, str(Path(__file__).with_name("llm_wake_sleep.py")), "--output", str(output), "--arm", arm, "--seed",str(seed),
                            "--steps","459","--lr-q","0.0003","--lr-p","0.000001","--coupled","--resume",str(Path(resume["persistent"]).with_name("step_448.pt"))],check=True)
            resume = json.loads((output / "resume.json").read_text())
        model = initialize(seed)
        state = torch.load(resume["persistent"], map_location="cpu", weights_only=False)
        model.restore(state["adapters"]); model.usage = state["usage"]
        test = evaluate(model,data()[1]); transfer = evaluate(model,fresh)
        record = {"seed":seed,"arm":arm,"step":state["step"],"flops":model.compute(),"two_examples":test,"three_examples":transfer}
        save(directory / f"matched_evaluation_{arm}_s{seed}.json",record)
        results.append({"seed":seed,"arm":arm,"step":state["step"],"flops":model.compute(),
                        "two_examples":{k:v for k,v in test.items() if k!="records"},"three_examples":{k:v for k,v in transfer.items() if k!="records"}})
        del model, state; torch.cuda.empty_cache()
save(directory / "matched_evaluation.json",results)
print(json.dumps(results,indent=2))
