"""Independent FP32 probe forecasting actual success from score overlaps."""

import argparse
from datetime import datetime
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

from bestn_llm import cohort, dot, forecast_weights, graded, selected
from bestn_precision import PreciseStudent, exact_matmul
from elicitation_task import messages
from run_elicitation_pilot import MODELS, ROOT, digest, save, setup
from train_elicitation import read_rows


OUTPUT = ROOT / "results/bestn_b2k/success_probe"
SOURCES = ("probe_bestn_success.py", "bestn_precision.py", "bestn_llm.py", "train_elicitation.py", "run_elicitation_pilot.py", "elicitation_task.py", "elicitation_answer.py")


def stamp(): return datetime.now(ZoneInfo("Europe/London")).isoformat()


def log_probabilities(student, rows, problems):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    lookup, result = {p["id"]: p for p in problems}, []
    with torch.inference_mode():
        for offset in tqdm(range(0, len(rows), 4), desc="Trajectory likelihood ratios", leave=False):
            inputs, labels = [], []
            for row in rows[offset:offset + 4]:
                prefix = student.tokenizer.apply_chat_template(messages(lookup[row["id"]], "reason"), tokenize=True, add_generation_prompt=True)
                inputs.append(torch.tensor(prefix + row["token_ids"], dtype=torch.long))
                labels.append(torch.tensor([-100] * len(prefix) + row["token_ids"], dtype=torch.long))
            ids = pad_sequence(inputs, batch_first=True, padding_value=student.tokenizer.pad_token_id).cuda()
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).cuda()[:, 1:]
            attention = pad_sequence([torch.ones(len(row), dtype=torch.long) for row in inputs], batch_first=True, padding_value=0).cuda()
            mask = labels != -100
            hidden = student.model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state[:, :-1]
            logits = student.model.lm_head(hidden[mask])
            assert logits.dtype == torch.float32
            losses = torch.nn.functional.cross_entropy(logits, labels[mask], reduction="none")
            batch_indices = torch.arange(len(inputs), device="cuda")[:, None].expand_as(labels)[mask]
            per_example = torch.zeros(len(inputs), device="cuda", dtype=torch.float64).scatter_add_(0, batch_indices, losses.double())
            result.extend((-per_example).cpu().tolist())
    return np.array(result)


def importance_change(rows, problems, log_ratio):
    weights = np.exp(log_ratio)
    assert np.isfinite(weights).all()
    changes, raw_changes, baselines = [], [], []
    for problem in problems:
        mask = np.array([r["id"] == problem["id"] for r in rows])
        rewards = np.array([r["correct"] for r in rows])[mask]
        baseline = (rewards.sum() - rewards) / (len(rewards) - 1)
        changes.append(float(np.mean((rewards - baseline) * (weights[mask] - 1))))
        raw_changes.append(float(np.mean(rewards * (weights[mask] - 1))))
        baselines.append(float(rewards.mean()))
    change, raw = np.array(changes), np.array(raw_changes)
    return {"delta": float(change.mean()), "per_prompt_delta": changes,
            "ci95": (change.mean() + np.array([-1, 1]) * 1.96 * change.std(ddof=1) / np.sqrt(len(change))).tolist(),
            "raw_delta": float(raw.mean()), "baseline_success": float(np.mean(baselines)),
            "mean_weight": float(weights.mean()), "ess_fraction": float(weights.sum()**2 / (len(weights) * np.square(weights).sum())),
            "max_weight": float(weights.max())}


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "source").mkdir()
    for name in SOURCES: (OUTPUT / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    for index, regime in enumerate(("low", "high")):
        directory = OUTPUT / regime
        directory.mkdir()
        train, evaluation = cohort(85001 + index, regime, 32), cohort(85101 + index, regime, 64)
        earlier = []
        for stage in ("probe",):
            for split in ("train", "evaluation"):
                earlier.extend(json.loads((ROOT / f"results/bestn_b2k/{stage}/{regime}/{split}.json").read_text()))
        assert not {p["id"] for p in train + evaluation} & {p["id"] for p in earlier}
        assert not {p["id"] for p in train} & {p["id"] for p in evaluation}
        save(directory / "train.json", train)
        save(directory / "evaluation.json", evaluation)
    save(OUTPUT / "protocol.json", {"created_at_uk": stamp(), "batch_id": "b2k", "model": MODELS["general"],
         "source_sha256": {name: digest(OUTPUT / "source" / name) for name in SOURCES},
         "precision": "FP32 parameters, logits, attention, gradients and generation; TF32 disabled; sampling and fitting at temperature1.",
         "hypothesis": "Binary selected-score identity plus held-out reward-gradient overlaps predicts finite-update changes in actual success. Predict N4/N2 ordering at equal candidates; account for heterogeneous prompts through measured overlaps.",
         "regimes": {"low": [12, 9999], "high": [4, 29]}, "train_unique": 32, "eval_unique": 64,
         "forecast_and_training_bank_samples": 16, "evaluation_bank_samples": 8,
         "fresh_evaluation_samples": 4, "maximum": 1024, "candidates": [1, 2, 4],
         "forecast": "Use independent training bank A to estimate each expected best-N score with the U-statistic control variate. Dot with a held-out unbiased reward-gradient estimate from a separate evaluation cohort. Save predictions before updating on targets from independent bank B.",
         "step_size_rule": "Base eta gives maximum predicted success increase .05 across N2/N4 at equal candidate count; also evaluate eta/4. This is a local mechanism probe, not a complete learning curve.",
         "training": "One plain SGD update per contrast, of positive-weight full-trace mean NLL/128, scaled by16/N. Uniform reward ties and all-failure groups included. Each contrast starts from the same original model.",
         "measurement": "Primary paired estimate uses unclipped trajectory importance ratios from the initial evaluation bank, with an independent leave-one-out per-prompt reward baseline as a zero-mean control variate. Report ESS, weight mean and max, raw estimate, and prompt bootstrap uncertainty. Fresh rollout accuracy checks each full step separately. Ratios are never used for training or target selection.",
         "limitations": "Forecast gradient and ratio estimate share the evaluation bank, while training-bank uncertainty is independent. Confidence intervals on ratio estimates are conditional on the actual update, not complete forecast intervals. Fresh rollout checks and later independent training runs remain necessary.",
         "resources": "One institutional H100 per difficulty. No paid service. Expected below25 minutes per probe; no long training trajectory."})


def run(regime):
    import torch
    setup()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    assert os.environ["HF_TOKEN"]
    torch.set_num_threads(4)
    exact_matmul()
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    root = OUTPUT / regime
    directory = root / "run"
    directory.mkdir()
    train, evaluation = [json.loads((root / f"{split}.json").read_text()) for split in ("train", "evaluation")]
    student = PreciseStudent()
    save(directory / "environment.json", {"started_at_uk": stamp(), "job_id": os.environ["SLURM_JOB_ID"], "device": torch.cuda.get_device_name(),
         "torch": torch.__version__, "cuda_visible_devices_inherited": os.environ["CUDA_VISIBLE_DEVICES"], "float32_matmul_precision": torch.get_float32_matmul_precision()})
    path = directory / "baseline.jsonl"
    student.generate(evaluation, "reason", 8, 1., 1024, 32, 852001, path)
    test_rows = graded(read_rows(path), evaluation)
    baseline_logs = log_probabilities(student, test_rows, evaluation)
    baseline = {"success": float(np.mean([r["correct"] for r in test_rows])), "per_prompt": {p["id"]: np.mean([r["correct"] for r in test_rows if r["id"] == p["id"]]).item() for p in evaluation}}
    save(directory / "baseline.json", baseline)
    np.save(directory / "baseline_log_probabilities.npy", baseline_logs)
    _, evaluation_gradient = student.trace_gradient(test_rows, evaluation, forecast_weights(test_rows, evaluation, 2))
    banks = []
    for index, name in enumerate(("forecast_bank", "training_bank")):
        path = directory / f"{name}.jsonl"
        student.generate(train, "reason", 16, 1., 1024, 32, 853001 + index, path)
        banks.append(graded(read_rows(path), train))
    predictions = {1: 0.}
    for n in (2, 4):
        _, gradient = student.trace_gradient(banks[0], train, forecast_weights(banks[0], train, n))
        predictions[n] = 128 * dot(evaluation_gradient, gradient) * 16 / n
        del gradient
    assert max(predictions.values()) > 0, "No positive held-out success-gradient overlap; redesign required."
    eta = .05 / max(predictions.values())
    forecast = {"written_before_updates_at_uk": stamp(), "success_gain_per_eta": predictions, "base_eta": eta,
                "baseline_success": baseline["success"], "reward_gradient_norm_squared": 128**2 * dot(evaluation_gradient, evaluation_gradient),
                "bank_sample_success": [float(np.mean([r["correct"] for r in bank])) for bank in banks]}
    save(directory / "forecast.json", forecast)
    print(json.dumps({"regime": regime, "forecast": forecast}), flush=True)
    initial = [p.detach().clone() for p in student.model.parameters()]
    measurements = []
    for n in (1, 2, 4):
        targets = selected(banks[1], train, n, 854001 + n)
        _, gradient = student.trace_gradient(targets, train, np.full(len(targets), 1 / len(targets)))
        realized_overlap = 128 * dot(evaluation_gradient, gradient) * 16 / n
        for fraction in (.25, 1.):
            with torch.no_grad():
                for parameter, original, direction in zip(student.model.parameters(), initial, gradient, strict=True): parameter.copy_(original).add_(direction, alpha=-eta * fraction * 16 / n)
            new_logs = log_probabilities(student, test_rows, evaluation)
            estimate = importance_change(test_rows, evaluation, new_logs - baseline_logs)
            np.save(directory / f"n{n}_fraction{fraction:g}_log_probabilities.npy", new_logs)
            record = {"n": n, "fraction": fraction, "effective_sgd_eta": eta * fraction * 16 / n,
                      "forecast_success_gain": eta * fraction * predictions[n], "realized_direction_success_gain": eta * fraction * realized_overlap,
                      "importance": estimate, "selected_success": float(np.mean([r["correct"] for r in targets])), "completed_at_uk": stamp()}
            if fraction == 1:
                path = directory / f"n{n}_fresh.jsonl"
                usage = student.generate(evaluation, "reason", 4, 1., 1024, 32, 855001, path)
                rows = graded(read_rows(path), evaluation)
                record["fresh"] = {"success": float(np.mean([r["correct"] for r in rows])), "completion": float(np.mean([r["ended"] for r in rows])),
                                   "usage": usage, "per_prompt": {p["id"]: np.mean([r["correct"] for r in rows if r["id"] == p["id"]]).item() for p in evaluation}}
            measurements.append(record)
            save(directory / "measurements.json", measurements)
            print(json.dumps({k: record[k] for k in ("n", "fraction", "forecast_success_gain", "realized_direction_success_gain", "importance")}), flush=True)
            with torch.no_grad():
                for parameter, original in zip(student.model.parameters(), initial, strict=True): parameter.copy_(original)
        del gradient
    save(directory / "complete.json", {"finished_at_uk": stamp(), "forecast": forecast, "measurements": measurements})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--regime", choices=("low", "high"))
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    else: run(args.regime)
