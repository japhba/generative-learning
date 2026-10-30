"""Check that b2k's full checkpoint reproduces the next SGD/RNG transition."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from train_bestn import checkpoint
from run_elicitation_pilot import ROOT, setup


def main():
    cache = setup() / "bestn_b2k/resume_check"
    root = ROOT / "results/bestn_b2k/resume_check"
    cache.mkdir(parents=True)
    root.mkdir()
    torch.manual_seed(86701)
    np.random.seed(86702)
    random.seed(86703)
    student = SimpleNamespace(model=torch.nn.Linear(3, 2))
    optimizer = torch.optim.SGD(student.model.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    history = {"step": 4, "order": [3, 1, 2, 0], "updates": [], "evaluations": [], "forecasts": []}
    with ThreadPoolExecutor(max_workers=1) as executor:
        checkpoint(student, optimizer, scheduler, history, cache, cache / "persistent_copy", root, 0, executor).result()
    def next_update():
        x = torch.randn(5, 3) * np.random.uniform() + random.random()
        optimizer.zero_grad(set_to_none=True)
        student.model(x).square().mean().backward()
        optimizer.step()
        scheduler.step()
        return [p.detach().clone() for p in student.model.parameters()]
    first = next_update()
    saved = torch.load(cache / "step004_attempt0.pt", weights_only=False)
    student.model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    second = next_update()
    assert saved["history"] == history
    for a, b in zip(first, second, strict=True): torch.testing.assert_close(a, b, rtol=0, atol=0)
    result = {"passed": True, "checks": ["full model", "optimizer", "LR scheduler", "data order and counters", "Python, NumPy, Torch and CUDA RNG", "persistent SHA256 verified", "bit-identical next update"]}
    (root / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
