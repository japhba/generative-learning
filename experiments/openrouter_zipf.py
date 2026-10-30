"""Compare passive exposure, active query generation, and yoked study under Zipf demand."""

import argparse
import asyncio
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import httpx

from openrouter_learning import HARD_CAP, MODELS, OpenRouter, ROOT, SpendingLedger, gather_settled, write_json
from zipf_task import ALPHA, BUDGETS, LABELS, P, VOCABULARY, coverage_error, ideal_curves, label, make_world

CONDITIONS = ("iid_study", "active_generate", "active_yoked")
SYSTEM = f"""You are learning a new secret dictionary. Keys are integers 1 through {VOCABULARY}.
Every key has one fixed label, A, B, C, or D. Labels were assigned independently and
uniformly at random: there is no relationship between different keys' labels and no
numeric rule to infer. Correct observations reveal exact labels; your own guesses do
not. Remember observed labels and use them to answer later questions. For an unseen
key, make your best guess among A, B, C, D.
Future test queries are drawn with probability proportional to key^(-{ALPHA}), so
smaller numbered keys are queried more often. Aim to minimize expected test error.
When asked to select practice queries, you can choose any keys. Each requested label
costs one practice example, including repetitions. Return only the requested JSON."""


def feedback(world, keys):
    return {"role": "user", "content": "Correct dictionary observations:\n" + "\n".join(f"{key}: {label(world, key)}" for key in keys)}


async def run_world(api, model, world, output):
    histories = {condition: [{"role": "system", "content": SYSTEM}] for condition in CONDITIONS}
    observed = {condition: [] for condition in CONDITIONS}
    practice_cost, practice_prompt, practice_completion = 0., 0, 0
    rows = []
    previous = 0
    for budget in BUDGETS:
        count = budget - previous
        histories["active_generate"].append({"role": "user", "content": f"Choose your next {count} practice key IDs and guess their labels before feedback. Return JSON with 'queries' (exactly {count} integer IDs) and 'guesses' (exactly {count} corresponding label letters)."})
        schema = {"type": "object", "properties": {
            "queries": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": VOCABULARY}, "minItems": count, "maxItems": count},
            "guesses": {"type": "array", "items": {"type": "string", "enum": list(LABELS)}, "minItems": count, "maxItems": count},
        }, "required": ["queries", "guesses"], "additionalProperties": False}
        selected = await api.complete(model, histories["active_generate"], schema, world["seed"] * 1000 + budget, max_output_tokens=768)
        keys, guesses = selected["data"]["queries"], selected["data"]["guesses"]
        assert len(keys) == count and len(guesses) == count
        assert all(type(key) is int and 1 <= key <= VOCABULARY for key in keys)
        assert all(guess in LABELS for guess in guesses)
        histories["active_generate"].append({"role": "assistant", "content": selected["content"]})
        for condition in ("active_generate", "active_yoked"):
            observed[condition].extend(keys)
            histories[condition].append(feedback(world, keys))
        passive_keys = world["passive_keys"][previous:budget]
        observed["iid_study"].extend(passive_keys)
        histories["iid_study"].append(feedback(world, passive_keys))
        practice_cost += float(selected["usage"]["cost"])
        practice_prompt += selected["usage"]["prompt_tokens"]
        practice_completion += selected["usage"]["completion_tokens"]
        test_message = {"role": "user", "content": "Independent test queries. Return JSON {'answers': [...]} with one label per listed key, in order. No test feedback will be provided.\nKeys: " + json.dumps(world["test_keys"])}
        predictions = await gather_settled(*[
            api.predict(model, histories[condition] + [test_message], world["test_keys"], world["seed"] * 1000 + 500 + budget)
            for condition in CONDITIONS
        ])
        for condition, prediction in zip(CONDITIONS, predictions, strict=True):
            correct = [answer == target for answer, target in zip(prediction["answers"], world["test_labels"], strict=True)]
            seen = [key in observed[condition] for key in world["test_keys"]]
            coverage = coverage_error(observed[condition])
            rows.append({
                "model": model, "world_seed": world["seed"], "budget": budget, "condition": condition,
                "keys": world["test_keys"], "test_weights": world["test_weights"], "targets": world["test_labels"],
                "answers": prediction["answers"], "correct": correct, "seen": seen,
                "error": sum(weight * (not hit) for weight, hit in zip(world["test_weights"], correct, strict=True)),
                "coverage_error": coverage, "known_mass": 1 - coverage / .75,
                "query_keys": observed[condition].copy(), "unique_queries": len(set(observed[condition])),
                "prompt_tokens": prediction["usage"]["prompt_tokens"], "completion_tokens": prediction["usage"]["completion_tokens"],
                "eval_cost_usd": prediction["usage"]["cost"], "call_id": prediction["call_id"], "provider": prediction["provider"],
                "practice_cost_usd": practice_cost if condition == "active_generate" else 0.,
                "practice_prompt_tokens": practice_prompt if condition == "active_generate" else 0,
                "practice_completion_tokens": practice_completion if condition == "active_generate" else 0,
            })
        previous = budget
    write_json(output / "world_results" / f"{model.replace('/', '__')}__{world['seed']}.json", rows)
    print(f"Completed {model} world {world['seed']}: " + ", ".join(f"{row['condition']} error={row['error']:.3f}, distinct={row['unique_queries']}" for row in rows[-3:]), flush=True)
    return rows


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "world_results").mkdir(exist_ok=True)
    worlds = [make_world(args.seed+i) for i in range(args.worlds)]
    manifest = {
        "experiment": "zipf_associations_v1", "world_count": args.worlds, "seed_start": args.seed,
        "models": args.models, "providers": {model: MODELS[model]["provider"] for model in args.models},
        "budgets": BUDGETS, "conditions": CONDITIONS, "alpha": ALPHA, "vocabulary": VOCABULARY,
        "labels": LABELS, "tests_per_world": len(worlds[0]["test_keys"]), "temperature": 0,
        "system_prompt": SYSTEM, "spending_cap_usd": str(HARD_CAP),
        "evaluation": "Importance-weighted stratified test queries; same test set across conditions and budgets. Test calls never enter training history.",
        "theory": "IID expected error .75 sum p(1-p)^n; ranked-unique ideal error .75 sum_{r>n}p_r. Infinite-vocabulary asymptotic exponents (alpha-1)/alpha and alpha-1.",
        "primary_fit_budgets": [8, 16, 32, 64, 128], "fit_sensitivity": [[4, 8, 16, 32, 64], [16, 32, 64, 128]],
        "interpretation": "Tests query selection/coverage in associative ICL. Not weight training, compositional transfer, or an intrinsic human generation effect. Yoked study receives identical chosen labels without generation transcript.",
        "code_sha256": {name: hashlib.sha256((ROOT/"experiments"/name).read_bytes()).hexdigest() for name in ("openrouter_zipf.py", "zipf_task.py", "openrouter_learning.py")},
    }
    # JSON round-trip canonicalizes tuples before comparison on resume.
    manifest = json.loads(json.dumps(manifest))
    if (args.output / "manifest.json").exists():
        assert json.loads((args.output / "manifest.json").read_text()) == manifest, "Run configuration/code changed; use a new directory."
        assert json.loads((args.output / "worlds.json").read_text()) == worlds
    else:
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "worlds.json", worlds)
    write_json(args.output / "theory.json", ideal_curves(BUDGETS))
    print(f"Prepared {args.worlds} worlds × {len(args.models)} models; at most {32*args.worlds*len(args.models)} paid requests.", flush=True)
    return worlds


async def run(args, worlds):
    load_dotenv(Path.home()/".env")
    load_dotenv(ROOT/".env", override=True)
    headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "X-OpenRouter-Title": "Generative learning bounded pilot"}
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        auth = await client.get("https://openrouter.ai/api/v1/key")
        auth.raise_for_status()
        write_json(args.output/"key_usage_before.json", {k:auth.json()["data"][k] for k in ("usage", "usage_daily", "limit_remaining")})
        ledger = SpendingLedger(ROOT/"results"/"openrouter_budget")
        api = OpenRouter(client, ledger, args.concurrency)
        outcomes = await asyncio.gather(*[run_world(api, model, world, args.output) for world in worlds for model in args.models], return_exceptions=True)
        api.progress.close()
        spend = {"settled_usd": str(sum(ledger.paid.values(), Decimal("0"))), "committed_usd": str(ledger.committed), "cap_usd": str(ledger.cap), "settled_calls": len(ledger.paid), "reserved_calls": len(ledger.reserved)}
        write_json(args.output/"spending.json", spend)
        ledger.close()
        failures = [value for value in outcomes if isinstance(value, BaseException)]
        if failures:
            raise ExceptionGroup("Experiment stopped; inspect cached responses before retrying", failures)
        rows = [row for world_rows in outcomes for row in world_rows]
        (args.output/"results.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        after = await client.get("https://openrouter.ai/api/v1/key")
        after.raise_for_status()
        write_json(args.output/"key_usage_after.json", {k:after.json()["data"][k] for k in ("usage", "usage_daily", "limit_remaining")})
        print(json.dumps(spend), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"zipf_pilot")
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    assert args.worlds > 0 and 1 <= args.concurrency <= 8
    worlds = prepare(args)
    if args.mode == "run": asyncio.run(run(args, worlds))


if __name__ == "__main__": main()
