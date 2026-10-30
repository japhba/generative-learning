"""Check whether forcing EOS biases integer acquisition toward long, rare keys."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

import train_zipf_finetuning as engine
from zipf_task import P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=engine.MODELS, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    os.environ["HF_HUB_CACHE"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    learner = engine.Learner(args.family, args.device)
    learner.make_trie()
    newline = learner.tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline) == 1
    newline = newline[0]
    original = learner.model.generate
    variants = {}
    for variant in ["original_eos", "newline_boundary"]:
        learner.reset(61000, 1e-5)
        calls = []
        def generate(**kwargs):
            if variant == "original_eos": return original(**kwargs)
            prefix = kwargs["input_ids"].shape[1]
            allowed = kwargs["prefix_allowed_tokens_fn"]
            def boundaries(batch_id, sequence):
                tokens = allowed(batch_id, sequence)
                return [*tokens, newline] if learner.tokenizer.eos_token_id in tokens else tokens
            kwargs["prefix_allowed_tokens_fn"] = boundaries
            kwargs["eos_token_id"] = [learner.tokenizer.eos_token_id, newline]
            result = original(**kwargs)
            calls.append({"termination": learner.tokenizer.decode(result[0, -1:].tolist()), "generated_tokens": result[0, prefix:].tolist()})
            # The original choose parser expects EOS; replacing only its boundary
            # after decoding leaves all chosen numerical tokens unchanged.
            result[0, -1] = learner.tokenizer.eos_token_id
            return result
        learner.model.generate = generate
        queries = []
        for _ in tqdm(range(16), desc=f"{args.family}/{variant}"):
            queries.append(learner.choose(queries))
        variants[variant] = {"queries": queries, "known_mass_after16": float(P[[key - 1 for key in sorted(set(queries))]].sum()), "boundary_records": calls}
    result = {"family": args.family, "at": datetime.now(timezone.utc).isoformat(), "scope": "Query-only diagnostic on the original pretrained model: no labels, feedback, weight updates or held-out test predictions. Same instruction prompt and valid numerical prefixes; the only alternative allows newline as well as EOS at an integer boundary.",
              "source_sha256": engine.hashes(), "variants": variants}
    destination = engine.ROOT / "results/zipf_finetuning" / args.family / "query_termination_diagnostic.json"
    engine.save(destination, result)
    print(json.dumps(result))


if __name__ == "__main__": main()
