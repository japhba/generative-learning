"""Check a simple global answer-prior correction on already exposed pilot data."""

import json

import numpy as np
from tqdm.auto import tqdm

from run_elicitation_pilot import ROOT, save
from scaling_digit_task import encode
from train_bestn import stamp
from train_scaling_verified import initialize


def probabilities(logits, bias):
    value = logits + bias; value -= value.max(-1, keepdims=True)
    p = np.exp(value); return p / p.sum(-1, keepdims=True)


def evaluate(logits, labels, bias):
    p = probabilities(logits, bias)
    return {"accuracy": float(p[np.arange(len(p)), labels].mean()), "greedy_accuracy": float(np.mean(p.argmax(-1) == labels)),
            "nll": float(-np.log(p[np.arange(len(p)), labels]).mean())}


def fit(logits, labels):
    bias = np.zeros(3); counts = np.bincount(labels, minlength=3) / len(labels); history = []
    for step in range(100):
        p = probabilities(logits, bias); gradient = p.mean(0)[:2] - counts[:2]
        history.append({"step": step, "bias": bias.tolist(), "gradient_norm": float(np.linalg.norm(gradient)), **evaluate(logits, labels, bias)})
        if np.linalg.norm(gradient) < 1e-11: return bias, history
        hessian = np.diag(p.mean(0)[:2]) - p[:, :2].T @ p[:, :2] / len(p)
        direction = np.linalg.solve(hessian, gradient); rate = 1.
        candidate = bias.copy(); candidate[:2] -= direction
        while evaluate(logits, labels, candidate)["nll"] > history[-1]["nll"]:
            rate /= 2; assert rate > 1e-10
            candidate = bias.copy(); candidate[:2] -= rate * direction
        bias = candidate
    raise AssertionError("Convex bias correction did not converge")


def main():
    import torch
    root = ROOT / "results/scaling_r5s/pilot"; _, student = initialize(root)
    output = ROOT / "results/scaling_r5s/bias_probe"; output.mkdir(parents=True, exist_ok=False)
    data = {"train": json.loads((root / "99501/train.json").read_text())[:8192], "monitor": json.loads((root / "monitor.json").read_text())}
    ids = [student.tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(3)]
    arrays = {}
    for split, rows in data.items():
        encoded = encode(student, rows); logits = []
        with torch.inference_mode():
            for offset in tqdm(range(0, len(rows), 32), desc=f"Fixed initial model: {split}"):
                logits.extend(student.logits(encoded[offset:offset + 32])[:, ids].cpu().tolist())
        arrays[split] = {"logits": np.array(logits), "labels": np.array([int(r["label"]) for r in rows])}
        save(output / f"{split}.json", [{"id": row["id"], "label": row["label"], "digit_logits": value} for row, value in zip(rows, logits, strict=True)])
    bias, history = fit(arrays["train"]["logits"], arrays["train"]["labels"])
    results = {"finished_at_uk": stamp(), "status": "Exploratory fixed-feature diagnostic using only already exposed pilot training/monitoring inputs. No confirmation final-test questions or model outputs read.",
               "method": "Fit two relative global answer-logit biases by convex cross-entropy Newton optimization; hold pretrained transformer fixed. Grant a vocabulary restricted to0/1/2. This is a calibration control, not an exhaustive maximum-greedy-accuracy bound or the actual full-weight learner.",
               "bias": bias.tolist(), "history": history,
               "before": {s: evaluate(v["logits"], v["labels"], np.zeros(3)) for s, v in arrays.items()},
               "after": {s: evaluate(v["logits"], v["labels"], bias) for s, v in arrays.items()}}
    save(output / "results.json", results); print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))


if __name__ == "__main__": main()
