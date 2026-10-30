"""Budgeted, resumable in-context learning curves on random permutation worlds."""

import argparse
import asyncio
from collections import defaultdict
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import httpx
from dotenv import load_dotenv
from tqdm.auto import tqdm

from permutation_task import STATES, bayes_reference, make_world, question_text

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [0, 2, 4, 6, 8]
MODELS = {
    "qwen/qwen3-30b-a3b-instruct-2507": {"provider": "Nebius", "reasoning": False},
    "google/gemini-2.5-flash-lite": {"provider": "Google", "reasoning": True},
    "mistralai/mistral-small-3.2-24b-instruct": {"provider": "DeepInfra", "reasoning": False},
}
HARD_CAP = Decimal("0.80")
INPUT_PRICE_CAP = Decimal("0.11") / 1_000_000
OUTPUT_PRICE_CAP = Decimal("0.45") / 1_000_000
MAX_OUTPUT_TOKENS = 256
SYSTEM = """Learn a new hidden state machine from examples. Its four states are A, B, C, D.
Its two operators f and g are fixed unknown permutations: each operator maps the four
states one-to-one onto the four states. The mappings were randomly chosen for this
episode and do not change. An input such as 'B f g f' means start at B and apply f,
then g, then f, from left to right. Return the final state. Each question starts
independently. Infer the mappings from the supplied correct examples. If the examples
do not determine an answer, make your best guess. Your own previous guesses are not
evidence; corrections are authoritative. Return only a JSON object with an 'answers'
array, containing one state letter per question, in order. Do not explain."""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class SpendingLedger:
    """Reserve before dispatch; retain unresolved reservations after errors/crashes."""

    def __init__(self, directory, cap=HARD_CAP):
        if not Decimal("0") < cap <= HARD_CAP:
            raise ValueError("Spending cap must be positive and no greater than $0.80.")
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.cap = directory, cap
        self.lockfile = (directory / "writer.lock").open("a")
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = directory / "ledger.jsonl"
        self.path.touch(exist_ok=True)
        self.reserved, self.paid = {}, {}
        for line in self.path.read_text().splitlines():
            event = json.loads(line)
            target = self.reserved if event["event"] == "reserve" else self.paid
            target[event["call_id"]] = Decimal(event["usd"])

    @property
    def committed(self):
        return sum(self.paid.values(), Decimal("0")) + sum(
            (amount for key, amount in self.reserved.items() if key not in self.paid), Decimal("0")
        )

    def reserve(self, call_id, amount, model):
        if call_id in self.reserved:
            raise RuntimeError("Unresolved request: reconcile its reservation before retrying.")
        if amount < 0 or self.committed + amount > self.cap:
            raise RuntimeError("Experiment spending cap reached before dispatch.")
        append_json(self.path, {"event": "reserve", "call_id": call_id, "usd": str(amount), "model": model})
        self.reserved[call_id] = amount

    def settle(self, call_id, amount):
        if not Decimal("0") <= amount <= self.reserved[call_id]:
            raise RuntimeError("Charge exceeds the conservative reservation.")
        append_json(self.path, {"event": "settle", "call_id": call_id, "usd": str(amount)})
        self.paid[call_id] = amount

    def close(self):
        self.lockfile.close()


class OpenRouter:
    def __init__(self, client, ledger, concurrency):
        self.client, self.ledger = client, ledger
        self.semaphore = asyncio.Semaphore(concurrency)
        self.call_locks = defaultdict(asyncio.Lock)
        self.progress = tqdm(desc="API responses", unit="call", mininterval=5)

    async def predict(self, model, history, questions, seed):
        schema = {"type": "object", "properties": {"answers": {"type": "array", "items": {"type": "string", "enum": list(STATES)}, "minItems": len(questions), "maxItems": len(questions)}}, "required": ["answers"], "additionalProperties": False}
        result = await self.complete(model, history, schema, seed)
        answers = result["data"]["answers"]
        assert len(answers) == len(questions) and all(answer in STATES for answer in answers)
        return {**result, "answers": answers}

    async def complete(self, model, history, schema, seed, max_output_tokens=MAX_OUTPUT_TOKENS):
        if not 1 <= max_output_tokens <= 1024:
            raise ValueError("Completion limit must be between 1 and 1024 tokens.")
        payload = {
            "model": model, "messages": history, "temperature": 0, "seed": seed,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "state_answers", "strict": True, "schema": schema}},
            "provider": {"only": [MODELS[model]["provider"]], "allow_fallbacks": False, "require_parameters": True, "max_price": {"prompt": 0.11, "completion": 0.45, "request": 0}},
        }
        if MODELS[model]["reasoning"]:
            payload["reasoning"] = {"enabled": False}
        serialized = json.dumps(payload, sort_keys=True)
        call_id = hashlib.sha256(serialized.encode()).hexdigest()
        async with self.call_locks[call_id]:
            cache_path = self.ledger.directory / f"{call_id}.json"
            if cache_path.exists():
                record = json.loads(cache_path.read_text())
                assert record["http_status"] == 200, "Cached failed request requires inspection; no automatic paid retry."
                settlement_id = record["recovery_attempt_call_id"] if "recovery_attempt_call_id" in record else call_id
                if settlement_id not in self.ledger.paid:
                    self.ledger.settle(settlement_id, Decimal(str(record["response"]["usage"]["cost"])))
            else:
                # Every UTF-8 byte as one input token, plus ample provider/template overhead.
                input_bound = len(serialized.encode()) + 4096
                reservation = 2 * (input_bound * INPUT_PRICE_CAP + max_output_tokens * OUTPUT_PRICE_CAP)
                async with self.semaphore:
                    self.ledger.reserve(call_id, reservation, model)
                    start = time.monotonic()
                    response = await self.client.post("https://openrouter.ai/api/v1/chat/completions", json=payload)
                    body = response.json()
                    record = {"call_id": call_id, "request": payload, "response": body, "http_status": response.status_code, "elapsed_seconds": time.monotonic() - start, "input_token_bound": input_bound, "reserved_usd": str(reservation)}
                    write_json(cache_path, record)
                    response.raise_for_status()
                    usage = body["usage"]
                    self.ledger.settle(call_id, Decimal(str(usage["cost"])))
                    assert usage["prompt_tokens"] <= input_bound
                    assert usage["completion_tokens"] <= max_output_tokens
                    self.progress.update()
                    self.progress.set_postfix(usd=f"{sum(self.ledger.paid.values()):.4f}", refresh=False)
        body = record["response"]
        assert body["choices"][0]["finish_reason"] == "stop", "Truncated or failed output is preserved in the request cache."
        content = body["choices"][0]["message"]["content"]
        return {"data": json.loads(content), "content": content, "call_id": call_id, "usage": body["usage"], "provider": body["provider"]}


def question_message(questions, purpose):
    return {"role": "user", "content": purpose + "\n" + "\n".join(f"{i+1}. {question_text(query)} -> ?" for i, query in enumerate(questions))}


def study_message(questions):
    return {"role": "user", "content": "Correct worked examples:\n" + "\n".join(f"{question_text(query)} -> {query['answer']}" for query in questions)}


async def gather_settled(*awaitables):
    """Finish sibling requests and record their charges before propagating errors."""
    outcomes = await asyncio.gather(*awaitables, return_exceptions=True)
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if failures: raise BaseExceptionGroup("Concurrent evaluations failed after all siblings finished", failures)
    return outcomes


async def run_world(api, model, world, output):
    base = [{"role": "system", "content": SYSTEM}]
    study, generated = base.copy(), base.copy()
    practice_cost, practice_completion, practice_prompt = 0., 0, 0
    targets = [query["answer"] for query in world["tests"]]

    async def evaluate(history, budget):
        request = history + [question_message(world["tests"], "Independent test questions. No test feedback will be provided.")]
        return await api.predict(model, request, world["tests"], world["seed"] * 100 + budget)

    def result_row(prediction, condition, budget):
        correct = [answer == target for answer, target in zip(prediction["answers"], targets)]
        return {
            "model": model, "world_seed": world["seed"], "condition": condition, "budget": budget,
            "answers": prediction["answers"], "targets": targets, "correct": correct, "accuracy": sum(correct)/len(correct),
            "prompt_tokens": prediction["usage"]["prompt_tokens"], "completion_tokens": prediction["usage"]["completion_tokens"],
            "eval_cost_usd": prediction["usage"]["cost"], "practice_cost_usd": practice_cost if condition == "generate_feedback" else 0.,
            "practice_completion_tokens": practice_completion if condition == "generate_feedback" else 0,
            "practice_prompt_tokens": practice_prompt if condition == "generate_feedback" else 0,
            "call_id": prediction["call_id"], "provider": prediction["provider"],
        }

    zero = await evaluate(base, 0)
    rows = [result_row(zero, condition, 0) for condition in ("study", "generate_feedback")]
    for previous, budget in zip(BUDGETS[:-1], BUDGETS[1:]):
        questions = world["practice"][previous:budget]
        generated.append(question_message(questions, "Practice: attempt each answer before receiving the correct answers."))
        attempt = await api.predict(model, generated, questions, world["seed"] * 100 + budget + 50)
        generated.append({"role": "assistant", "content": attempt["content"]})
        feedback = study_message(questions)
        generated.append(feedback)
        study.append(feedback)
        practice_cost += attempt["usage"]["cost"]
        practice_completion += attempt["usage"]["completion_tokens"]
        practice_prompt += attempt["usage"]["prompt_tokens"]
        predictions = await gather_settled(evaluate(study, budget), evaluate(generated, budget))
        rows.extend(result_row(prediction, condition, budget) for prediction, condition in zip(predictions, ("study", "generate_feedback")))
    path = output / "world_results" / f"{model.replace('/', '__')}__{world['seed']}.json"
    write_json(path, rows)
    print(f"Completed {model} world {world['seed']}: " + ", ".join(f"{row['condition']}={row['accuracy']:.0%}" for row in rows[-2:]), flush=True)
    return rows


def prepare(output, worlds_count, seed, models):
    output.mkdir(parents=True, exist_ok=True)
    (output / "world_results").mkdir(exist_ok=True)
    worlds = [make_world(seed+i) for i in range(worlds_count)]
    manifest = {
        "experiment": "permutation_state_machine_v1", "seed_start": seed, "world_count": worlds_count,
        "models": models, "providers": {model: MODELS[model]["provider"] for model in models}, "budgets": BUDGETS,
        "conditions": ["study", "generate_feedback"], "tests_per_world": len(worlds[0]["tests"]),
        "temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS, "reasoning": "disabled; Qwen is a non-thinking instruct variant",
        "spending_cap_usd": str(HARD_CAP), "cap_scope": "all requests in results/openrouter_budget, including pilot and main",
        "system_prompt": SYSTEM,
        "interpretation": "Frozen-model ICL, not weight training. Same gold examples; generation adds attempted answers, transcript length, and computation. Test calls never enter practice history. Costs are token charges, not hardware FLOPs.",
    }
    if (output / "manifest.json").exists():
        assert json.loads((output / "manifest.json").read_text()) == manifest, "Existing run has different configuration."
        assert json.loads((output / "worlds.json").read_text()) == worlds
    else:
        write_json(output / "manifest.json", manifest)
        write_json(output / "worlds.json", worlds)
    reference = []
    for world in tqdm(worlds, desc="Bayesian reference", unit="world"):
        for budget in BUDGETS:
            value = bayes_reference(world, budget)
            targets = [query["answer"] for query in world["tests"]]
            correct = [answer == target for answer, target in zip(value["answers"], targets, strict=True)]
            reference.append({"model": "bayes", "world_seed": world["seed"], "condition": "bayes", "budget": budget, "targets": targets, "correct": correct, **value})
    write_json(output / "bayes_reference.json", reference)
    print(f"Prepared {worlds_count} worlds, {len(models)} models, {13*worlds_count*len(models)} maximum paid calls.", flush=True)
    return worlds, reference


async def run(args, worlds, reference):
    load_dotenv(Path.home()/".env")
    load_dotenv(ROOT/".env", override=True)
    headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "X-OpenRouter-Title": "Generative learning bounded pilot"}
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        # Authentication is checked before any paid request or reservation.
        key_response = await client.get("https://openrouter.ai/api/v1/key")
        assert key_response.status_code == 200, f"OpenRouter authentication failed: {key_response.text}"
        key = key_response.json()["data"]
        write_json(args.output/"key_usage_before.json", {k:key[k] for k in ("usage", "usage_daily", "limit_remaining")})
        ledger = SpendingLedger(ROOT/"results"/"openrouter_budget")
        api = OpenRouter(client, ledger, args.concurrency)
        tasks = [run_world(api, model, world, args.output) for world in worlds for model in args.models]
        # Return exceptions so in-flight requests finish and their charges are recorded.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        api.progress.close()
        spend = {"settled_usd": str(sum(ledger.paid.values(), Decimal("0"))), "committed_usd": str(ledger.committed), "cap_usd": str(ledger.cap), "settled_calls": len(ledger.paid), "reserved_calls": len(ledger.reserved)}
        write_json(args.output/"spending.json", spend)
        ledger.close()
        failures = [value for value in outcomes if isinstance(value, BaseException)]
        if failures:
            raise ExceptionGroup("Experiment stopped; successful responses cached and reservations retained", failures)
        rows = reference + [row for world_rows in outcomes for row in world_rows]
        (args.output/"results.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        after = await client.get("https://openrouter.ai/api/v1/key")
        after.raise_for_status()
        key = after.json()["data"]
        write_json(args.output/"key_usage_after.json", {k:key[k] for k in ("usage", "usage_daily", "limit_remaining")})
        print(json.dumps(spend), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"openrouter_pilot")
    parser.add_argument("--worlds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    assert args.worlds > 0 and 1 <= args.concurrency <= 8
    worlds, reference = prepare(args.output, args.worlds, args.seed, args.models)
    if args.mode == "run":
        asyncio.run(run(args, worlds, reference))


if __name__ == "__main__":
    main()
