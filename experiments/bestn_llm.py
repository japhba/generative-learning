"""Shared arithmetic, selection, and exact completion-loss machinery for b2k."""

import math
from random import Random

import numpy as np
from tqdm.auto import tqdm

from elicitation_answer import parse_answer
from elicitation_task import identifier, messages, states
from train_elicitation import Student


REGIMES = {"low": (12, 9999), "high": (4, 29)}


def cohort(seed, regime, count):
    rng, result = Random(seed), []
    depth, bound = REGIMES[regime]
    for _ in range(count):
        operations, multiplications = [], 0
        for _ in range(depth):
            operation = rng.choice(("add", "subtract", "multiply") if multiplications < 3 else ("add", "subtract"))
            operations.append([operation, rng.randint(2, 5) if operation == "multiply" else rng.randint(2, bound)])
            multiplications += operation == "multiply"
        row = {"family": "arithmetic", "depth": depth, "start": rng.randint(2, 39), "operations": operations}
        result.append({**row, "id": identifier(row), "regime": regime})
    assert len({row["id"] for row in result}) == len(result)
    return result


def reference(problem):
    values, lines = states(problem), [f"Starting value: {problem['start']}"]
    symbols = {"add": "+", "subtract": "-", "multiply": "*"}
    for index, ((operation, argument), previous, value) in enumerate(zip(problem["operations"], values[:-1], values[1:], strict=True), 1):
        lines.append(f"{index}. {previous} {symbols[operation]} {argument} = {value}")
    lines.append("Final answer: \\boxed{" + str(values[-1]) + "}")
    return "\n".join(lines)


def graded(rows, problems):
    lookup = {problem["id"]: states(problem)[-1] for problem in problems}
    return [{**row, "correct": bool(row["ended"] and parse_answer(row["text"]) == lookup[row["id"]])} for row in rows]


def selected(rows, problems, n, seed):
    """Disjoint groups, uniform reward ties, and no rejection of failed groups."""
    rng, result = np.random.default_rng(seed), []
    for problem in problems:
        pool = [row for row in rows if row["id"] == problem["id"]]
        assert len(pool) % n == 0
        for offset in range(0, len(pool), n):
            group = pool[offset:offset + n]
            best = max(row["correct"] for row in group)
            winners = [row for row in group if row["correct"] == best]
            result.append(winners[int(rng.integers(len(winners)))])
    return result


def forecast_weights(rows, problems, n):
    """Unbiased U-statistic: expected best-N score minus the mean ordinary score.

    Averaging without-replacement N-subsets of an IID pool is an unbiased
    estimate of the true best-N expectation. Subtracting the pool mean score
    is a zero-expectation control variate. This is used ONLY for forecasting;
    actual training fits uniformly selected full traces with positive weights.
    """
    weights = []
    for problem in problems:
        pool = [row for row in rows if row["id"] == problem["id"]]
        size, successes = len(pool), sum(row["correct"] for row in pool)
        assert size >= n
        if successes in (0, size):
            weights.extend([0.] * size)
            continue
        q = 1 - math.comb(size - successes, n) / math.comb(size, n)
        weights.extend([(q / successes if row["correct"] else (1 - q) / (size - successes)) - 1 / size for row in pool])
    assert [row["id"] for row in rows] == [p["id"] for p in problems for _ in range(sum(row["id"] == p["id"] for row in rows))]
    return np.array(weights) / len(problems)


class BestNStudent(Student):
    def reference_tokens(self, problems):
        return [self.tokenizer.encode(reference(p), add_special_tokens=False) + [self.tokenizer.eos_token_id] for p in problems]

    def objective(self, problems, targets, weights, backward, microbatch=4):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        self.model.eval()
        if backward: self.model.zero_grad(set_to_none=True)
        value = 0.
        for offset in tqdm(range(0, len(problems), microbatch), desc="Completion gradients" if backward else "Reference likelihood", leave=False):
            group, sampled = problems[offset:offset + microbatch], targets[offset:offset + microbatch]
            coefficients = torch.tensor(weights[offset:offset + microbatch], device="cuda", dtype=torch.float32)
            if backward and bool(torch.all(coefficients == 0)): continue
            inputs, labels = [], []
            for problem, tokens in zip(group, sampled, strict=True):
                prefix = self.tokenizer.apply_chat_template(messages(problem, "reason"), tokenize=True, add_generation_prompt=True)
                inputs.append(torch.tensor(prefix + tokens, dtype=torch.long))
                labels.append(torch.tensor([-100] * len(prefix) + tokens, dtype=torch.long))
            ids = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id).cuda()
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).cuda()[:, 1:]
            attention = pad_sequence([torch.ones(len(row), dtype=torch.long) for row in inputs], batch_first=True, padding_value=0).cuda()
            mask = labels != -100
            with torch.set_grad_enabled(backward), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state[:, :-1]
                logits = self.model.lm_head(hidden[mask]).float()
                token_loss = torch.nn.functional.cross_entropy(logits, labels[mask], reduction="none")
                loss = (token_loss * coefficients[:, None].expand_as(labels)[mask]).sum() / 128
            assert torch.isfinite(loss)
            if backward: loss.backward()
            value += float(loss.detach())
        if backward:
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
            gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
            self.model.zero_grad(set_to_none=True)
            return value, gradients
        return value

    def trace_gradient(self, rows, problems, weights):
        lookup = {p["id"]: p for p in problems}
        return self.objective([lookup[row["id"]] for row in rows], [row["token_ids"] for row in rows], weights, True)


def dot(left, right):
    import torch
    return float(sum(torch.sum(a * b, dtype=torch.float64) for a, b in zip(left, right, strict=True)))
