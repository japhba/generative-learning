"""Pilot consolidation of a confirmed same-model elicitation gap.

No exact answers or verifier scores enter target creation. Frozen-source and
current-source teachers use identical prompts, temperatures, sample counts and
RNG schedules. Only the weights that generate the next reasoning traces differ.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from elicitation_answer import parse_answer
from elicitation_task import consensus, make_problems, messages
from run_elicitation_pilot import Learner, MODELS, ROOT, digest, save, setup


ARMS = ("frozen_reasoning", "on_policy_reasoning", "direct_self_imitation")
SOURCES = ("train_elicitation.py", "elicitation_task.py", "elicitation_answer.py", "run_elicitation_pilot.py")
SEEDS = (68000, 68001, 68002)
CHECKPOINTS = (0, 16, 64, 192, 384)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Student(Learner):
    def __init__(self):
        super().__init__("general")
        self.model.float()

    def generate(self, *args, **kwargs):
        import torch
        self.model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16): return super().generate(*args, **kwargs)

    def verify_objective(self, problem):
        import torch
        self.model.eval()
        prefix = self.tokenizer.apply_chat_template(messages(problem, "direct"), tokenize=True, add_generation_prompt=True)
        target = self.tokenizer.encode("\\boxed{0}", add_special_tokens=False) + [self.tokenizer.eos_token_id]
        ids = torch.tensor([prefix + target], device="cuda")
        labels = torch.tensor([[-100] * len(prefix) + target], device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            standard = self.model(input_ids=ids, labels=labels, use_cache=False).loss.float() * len(target)
            hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
            mask = labels[:, 1:] != -100
            selected = self.model.lm_head(hidden[mask]).float()
            subset = torch.nn.functional.cross_entropy(selected, labels[:, 1:][mask], reduction="sum")
        torch.testing.assert_close(standard, subset, rtol=1e-3, atol=1e-2)
        return {"standard_summed_nll": float(standard), "selected_position_summed_nll": float(subset), "passed": True,
                "purpose": "Validate completion masking and the unchanged natural LM objective before any update. The dummy target0 is not feedback or training data."}

    def update(self, problems, targets, optimizer, repeats, counters):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        self.model.train()
        inputs, labels = [], []
        for problem, target in zip(problems, targets, strict=True):
            prefix = self.tokenizer.apply_chat_template(messages(problem, "direct"), tokenize=True, add_generation_prompt=True)
            inputs.append(torch.tensor(prefix + target, dtype=torch.long))
            labels.append(torch.tensor([-100] * len(prefix) + target, dtype=torch.long))
        ids = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id).cuda()
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).cuda()
        attention = pad_sequence([torch.ones(len(row), dtype=torch.long) for row in inputs], batch_first=True, padding_value=0).cuda()
        shifted = labels[:, 1:]
        supervised = shifted != -100
        target_values = shifted[supervised]
        start, losses = time.monotonic(), []
        for _ in range(repeats):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state[:, :-1]
                logits = self.model.lm_head(hidden[supervised]).float()
                # Fixed normalization preserves the score-identity null at the
                # sampling parameters; do not divide by random completion length.
                loss = torch.nn.functional.cross_entropy(logits, target_values, reduction="sum") / (8 * len(problems))
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            counters["optimizer_updates"] += 1
            counters["training_presentations"] += len(problems)
            counters["supervised_tokens"] += int(supervised.sum())
            counters["training_input_tokens"] += int(attention.sum())
            counters["padded_training_tokens"] += ids.numel()
        torch.cuda.synchronize()
        counters["training_seconds"] += time.monotonic() - start
        return losses


def counters():
    return {"practice_presentations": 0, "valid_targets": 0, "optimizer_updates": 0, "training_presentations": 0,
            "supervised_tokens": 0, "training_input_tokens": 0, "padded_training_tokens": 0, "training_seconds": 0.,
            "rollout_responses": 0, "rollout_prompt_tokens": 0, "rollout_completion_tokens": 0,
            "rollout_padded_prompt_tokens": 0, "rollout_padded_completion_tokens": 0, "rollout_seconds": 0.}


def add_generation(count, usage):
    for field in ("responses", "prompt_tokens", "completion_tokens", "padded_prompt_tokens", "padded_completion_tokens", "seconds"):
        count["rollout_" + field] += usage[field]


def choose_targets(student, candidates, problems, arm):
    targets, selected = [], []
    group_size = 1 if arm == "direct_self_imitation" else 4
    assert len(candidates) == group_size * len(problems)
    for index, problem in enumerate(problems):
        group = candidates[index * group_size:(index + 1) * group_size]
        assert [row["id"] for row in group] == [problem["id"]] * group_size
        group = [{**row, "answer": parse_answer(row["text"])} for row in group]
        vote = consensus(group)
        if arm == "direct_self_imitation":
            target = group[0]["token_ids"]  # exact sampled tokens, even if wrong or truncated
        elif vote["answer"] is None:
            target = []
        else:
            target = student.tokenizer.encode("\\boxed{" + str(vote["answer"]) + "}", add_special_tokens=False) + [student.tokenizer.eos_token_id]
        targets.append(target)
        selected.append({"id": problem["id"], "vote": vote, "target_ids": target,
                         "target_text": student.tokenizer.decode(target, skip_special_tokens=True)})
    return targets, selected


def evaluate(student, evaluation, practice, directory, budget, count, seed):
    usage = {}
    for condition, mode, maximum in (("direct", "direct", 96), ("reason", "reason", 512)):
        usage[condition] = student.generate(evaluation, mode, 1, 0., maximum, 32, seed + 990000,
                                            directory / f"eval_{condition}_n{budget}.jsonl")
    usage["practice"] = student.generate(practice, "direct", 1, 0., 96, 32, seed + 980000,
                                          directory / f"practice_direct_n{budget}.jsonl")
    append(directory / "checkpoints.jsonl", {"budget": budget, "counters": dict(count), "evaluation_usage": usage})


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    confirm_path = ROOT / "results/elicitation_pilot/general_confirmation/confirmation/general/explicit_answer_analysis.json"
    confirmed = json.loads(confirm_path.read_text())["summary"]["arithmetic"]
    assert confirmed["passes_calibration"] and confirmed["gain_ci95"][0] > 0
    train, evaluation = make_problems(66000, 32, ("arithmetic",)), make_problems(67000, 48, ("arithmetic",))
    assert not {row["id"] for row in train} & {row["id"] for row in evaluation}
    old = json.loads((ROOT / "results/elicitation_pilot/calibration.json").read_text())
    old += json.loads((ROOT / "results/elicitation_pilot/general_confirmation/confirmation.json").read_text())
    assert not {row["id"] for row in train + evaluation} & {row["id"] for row in old}
    save(output / "train.json", train)
    save(output / "evaluation.json", evaluation)
    (output / "source").mkdir()
    for name in SOURCES: (output / "source" / name).write_bytes((ROOT / "experiments" / name).read_bytes())
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODELS["general"], "seeds": SEEDS,
                "train_seed": 66000, "train_unique": 96, "epochs": 4, "evaluation_seed": 67000, "evaluation_size": 144,
                "arms": ARMS, "checkpoints": CHECKPOINTS, "batch_size": 16, "updates_per_batch": 4, "learning_rate": 1e-5,
                "source_sha256": {name: digest(output / "source" / name) for name in SOURCES},
                "train_sha256": digest(output / "train.json"), "evaluation_sha256": digest(output / "evaluation.json"),
                "elicitation_confirmation_sha256": digest(confirm_path),
                "training": "Full FP32 weights/AdamW, BF16 autocast, no decay, clip1. Completion-token summed NLL divided by fixed8*batch_size, including exact raw sampled tokens for direct self-imitation. Natural LM head, no added head.",
                "targets": "Reasoning arms sample4complete candidates at temperature1 with identical RNG schedules. Plurality over parsed final integers, first-occurrence tie break, no correctness/confidence filter. Teach the direct prompt the canonical boxed plurality answer. Entirely invalid groups contribute zero target tokens; all prompts and sampling costs still count.",
                "frozen_source": "A fresh independent group from the initial model at each scheduled practice presentation; all groups may be precomputed before updates. Targets are not reused from a fixed per-question bank. Only the corresponding causal prefix enters learning/cost attribution.",
                "direct_control": "Sample1current-model short completion at temperature1 and train on its actual tokens. No answer canonicalization or correctness selection. Multiple finite-batch updates can move it; zero expected initial score is not a claim of unchanged realized weights.",
                "primary": "On-policy-minus-frozen-source short-answer accuracy at384practice presentations, across3optimization seeds and144paired fresh questions. This is a consolidation pilot; no exponent selection or fitting.",
                "secondary": "Short and reasoning accuracy at all checkpoints; training-question accuracy; target correctness graded only after completion; all generation/training compute. Improved short accuracy alone does not establish improved information or an on-policy advantage.",
                "icl": "Initial model only: first4and16frozen-source practice examples, either selected answer text or the selected sampled reasoning trace, in context. Evaluate both direct and reasoning modes on the same held-out questions. Seed68000 only; exploratory, not an independent statistical replication.",
                "resources": "Institutional gpu_lowp only. No paid APIs. Save final weights for seed68000 only to configured fast cache; archive before releasing allocations."}
    save(output / "protocol.json", protocol)


def run_seed(output, seed):
    import torch
    assert seed in SEEDS
    cache = setup() / "elicitation_consolidation"
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_sha256"] == {name: digest(ROOT / "experiments" / name) for name in SOURCES}
    train = json.loads((output / "train.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    ordering = np.random.default_rng(seed + 100000)
    order = np.concatenate([ordering.permutation(len(train)) for _ in range(4)]).tolist()
    sequence = [train[i] for i in order]
    directory = output / str(seed)
    directory.mkdir(exist_ok=False)
    frozen = directory / "frozen_candidates"
    frozen.mkdir()
    student = Student()
    save(directory / "objective_check.json", student.verify_objective(train[0]))
    initial = {key: value.detach().cpu().clone() for key, value in student.model.state_dict().items()}
    save(directory / "environment.json", {"job_id": os.environ["SLURM_JOB_ID"], "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
         "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "order": order,
         "parameter_count": sum(p.numel() for p in student.model.parameters()), "dtype": str(next(student.model.parameters()).dtype)})
    for step, begin in enumerate(tqdm(range(0, len(sequence), 16), desc=f"{seed} frozen-source draws")):
        student.generate(sequence[begin:begin + 16], "reason", 4, 1., 512, 32, seed + 200000 + step,
                         frozen / f"round_{step}.jsonl")
    if seed == SEEDS[0]:
        icl = directory / "icl"
        icl.mkdir()
        candidates = read_rows(frozen / "round_0.jsonl")
        _, selected = choose_targets(student, candidates, sequence[:16], "frozen_reasoning")
        for budget in (4, 16):
            for representation in ("answers", "traces"):
                demos = []
                for index, row in enumerate(selected[:budget]):
                    if row["vote"]["answer"] is None: continue
                    text = row["target_text"] if representation == "answers" else candidates[4 * index + row["vote"]["selected_index"]]["text"]
                    demos.append((sequence[index], text))
                for mode, maximum in (("direct", 96), ("reason", 512)):
                    student.generate(evaluation, mode, 1, 0., maximum, 8, seed + 990000,
                                     icl / f"{representation}_{mode}_n{budget}.jsonl", demos)
    for arm in ARMS:
        student.model.load_state_dict(initial, strict=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        optimizer = torch.optim.AdamW(student.model.parameters(), lr=protocol["learning_rate"], weight_decay=0., fused=True)
        current = directory / arm
        current.mkdir()
        count = counters()
        evaluate(student, evaluation, train, current, 0, count, seed)
        for step, begin in enumerate(tqdm(range(0, len(sequence), 16), desc=f"{seed} {arm}")):
            problems = sequence[begin:begin + 16]
            if arm == "frozen_reasoning":
                path = frozen / f"round_{step}.jsonl"
                usage = json.loads(path.with_suffix(".usage.json").read_text())
            else:
                path = current / f"round_{step}.jsonl"
                mode, samples, maximum = ("direct", 1, 96) if arm == "direct_self_imitation" else ("reason", 4, 512)
                usage = student.generate(problems, mode, samples, 1., maximum, 32, seed + 200000 + step, path)
            candidates = read_rows(path)
            targets, selected = choose_targets(student, candidates, problems, arm)
            add_generation(count, usage)
            count["practice_presentations"] += len(problems)
            count["valid_targets"] += sum(bool(t) for t in targets)
            losses = student.update(problems, targets, optimizer, protocol["updates_per_batch"], count)
            append(current / "updates.jsonl", {"step": step, "budget": begin + len(problems), "selected": selected,
                                               "candidate_path": str(path), "candidate_sha256": digest(path),
                                               "losses": losses, "counters": dict(count)})
            if count["practice_presentations"] in CHECKPOINTS:
                evaluate(student, evaluation, train, current, count["practice_presentations"], count, seed)
        if seed == SEEDS[0]:
            destination = cache / str(seed) / arm
            destination.mkdir(parents=True, exist_ok=False)
            student.model.save_pretrained(destination)
            student.tokenizer.save_pretrained(destination)
            save(current / "saved_model.json", {"path": str(destination), "files": {p.name: digest(p) for p in destination.iterdir() if p.is_file()}})
        save(current / "complete.json", {"counters": count, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        del optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/elicitation_consolidation")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.output)
    else: run_seed(args.output, args.seed)
