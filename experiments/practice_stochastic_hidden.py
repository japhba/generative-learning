"""Feature learning from atom scores with local REINFORCE at both layers."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

from practice_atoms import atom_features, learn_atom, score_atoms, sigmoid, signs, true_trace


def evaluate(b, w, x, verifier, uniforms, outputs):
    xb = np.column_stack((x, np.ones(len(x))))
    ph = sigmoid(xb @ b.T)
    h = (uniforms < ph[:, None]).astype(float)
    hb = np.concatenate((h, np.ones(h.shape[:-1] + (1,))), axis=-1)
    py = sigmoid(hb @ w.T)
    truth = true_trace(x, w.shape[0])
    success = np.prod(np.where(truth[:, None] == 1, py, 1 - py), axis=-1).mean(axis=1)
    marginal = np.prod(np.where(outputs[None, None] == 1, py[:, :, None], 1 - py[:, :, None]), axis=-1).mean(axis=1)
    learned_scores = score_atoms(x[:, None], outputs[None] + np.zeros((len(x), 1, 1)), verifier).prod(axis=-1)
    entropy = -np.sum(marginal * np.log(marginal), axis=-1)
    return {"success": float(success.mean()), "score": float(np.mean(np.sum(marginal * learned_scores, axis=-1))), "entropy": float(entropy.mean()), "greedy_success": float(np.mean(np.all(outputs[marginal.argmax(axis=-1)] == truth, axis=-1)))}


def run(seed, method, args):
    rng = np.random.default_rng(seed)
    x = signs(args.dimension)
    order = rng.permutation(len(x))
    train, test = order[: len(x) // 2], order[len(x) // 2 :]
    verifier, atom_data = learn_atom(rng, 48, 0.08, False)
    xb = np.column_stack((x, np.ones(len(x))))
    b = rng.normal(0, args.init_scale, (args.hidden, args.dimension + 1))
    w = rng.normal(0, 0.1, (args.length, args.hidden + 1))
    initial = {"b": b.tolist(), "w": w.tolist()}
    mb, vb, mw, vw = np.zeros_like(b), np.zeros_like(b), np.zeros_like(w), np.zeros_like(w)
    eval_rng = np.random.default_rng(seed + 990000)
    uniforms = eval_rng.random((len(x), args.eval_samples, args.hidden))
    outputs = signs(args.length)
    curves, attention_errors = [], []
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            row = {"step": step, "candidates": step * args.batch * args.candidates}
            for split, ids in (("train", train), ("test", test)):
                row.update({split + "_" + key: value for key, value in evaluate(b, w, x[ids], verifier, uniforms[ids], outputs).items()})
            assert np.isfinite(list(row.values())).all()
            curves.append(row)
            if step >= args.min_steps and len(curves) >= 11:
                window = curves[-10:]
                log_steps = np.log([r["step"] for r in window])
                slopes = [np.polyfit(log_steps, np.log([1 - r["train_score"] for r in window]), 1)[0], np.polyfit(log_steps, np.log([r["train_entropy"] for r in window]), 1)[0]]
                if max(abs(s) for s in slopes) < 0.01:
                    break
        if step == args.steps:
            break
        ids = rng.choice(train, args.batch)
        ph = sigmoid(xb[ids] @ b.T)
        h = (rng.random((args.batch, args.candidates, args.hidden)) < ph[:, None]).astype(float)
        hb = np.concatenate((h, np.ones(h.shape[:-1] + (1,))), axis=-1)
        py = sigmoid(hb @ w.T)
        y = (rng.random(py.shape) < py).astype(float)
        score = score_atoms(x[ids, None], 2 * y - 1, verifier).prod(axis=-1)
        if method == "shuffled":
            score = rng.permutation(score.reshape(-1)).reshape(score.shape)
        advantage = (args.candidates * score - score.sum(axis=1, keepdims=True)) / (args.candidates - 1)
        output_values = advantage[..., None] * (y - py)
        hidden_values = advantage[..., None] * (h - ph[:, None])
        gw = np.einsum("bml,bmh->lh", output_values, hb) / (args.batch * args.candidates)
        gb = hidden_values.mean(axis=1).T @ xb[ids] / args.batch
        if method == "frozen_hidden":
            gb *= 0
        if args.optimizer == "adam":
            mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb**2
            mw, vw = 0.9 * mw + 0.1 * gw, 0.999 * vw + 0.001 * gw**2
            db = args.lr * mb / (1 - 0.9 ** (step + 1)) / (np.sqrt(vb / (1 - 0.999 ** (step + 1))) + 1e-8)
            dw = args.lr * mw / (1 - 0.9 ** (step + 1)) / (np.sqrt(vw / (1 - 0.999 ** (step + 1))) + 1e-8)
        else:
            db, dw = args.lr * gb, args.lr * gw
        if step == 0:
            # The identity is for the raw local gradient, not elementwise Adam.
            explicit = xb[test] @ (b + gb).T
            attention = xb[test] @ b.T + (xb[test] @ xb[ids].T) @ hidden_values.mean(axis=1) / args.batch
            attention_errors.append(float(np.max(np.abs(explicit - attention))))
            query_hidden = np.column_stack((np.ones((1, args.hidden)), np.ones(1)))
            explicit_output = query_hidden @ (w + gw).T
            attention_output = query_hidden @ w.T + np.einsum("qh,bmh,bml->ql", query_hidden, hb, output_values) / (args.batch * args.candidates)
            attention_errors.append(float(np.max(np.abs(explicit_output - attention_output))))
        b += db
        w += dw
    final_uniforms = eval_rng.random((len(test), args.eval_samples * 8, args.hidden))
    final = evaluate(b, w, x[test], verifier, final_uniforms, outputs)
    return {"seed": seed, "method": method, "initial": initial, "b": b.tolist(), "w": w.tolist(), "atom": atom_data, "train_ids": train.tolist(), "test_ids": test.tolist(), "curves": curves, "final_independent_eval": final, "gradient_attention_error": max(attention_errors), "seconds": time.perf_counter() - start}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pilot")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seed-start", type=int, default=185000)
    p.add_argument("--methods", nargs="+", default=["local", "frozen_hidden", "shuffled"])
    p.add_argument("--dimension", type=int, default=8)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--init-scale", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--min-steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--lr", type=float, default=0.01)
    args = p.parse_args()
    load_dotenv(Path.home() / ".env")
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_hidden" / args.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args) | {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    (output / "source.py").write_text(Path(__file__).read_text())
    records = []
    for seed in tqdm(range(args.seed_start, args.seed_start + args.seeds), desc="Hidden-network worlds"):
        for method in tqdm(args.methods, desc="Conditions", leave=False):
            record = run(seed, method, args)
            records.append(record)
            (output / f"{seed}_{method}.json").write_text(json.dumps(record))
    summary = {method: {metric: float(np.mean([r["final_independent_eval"][metric] for r in records if r["method"] == method])) for metric in records[0]["final_independent_eval"]} for method in args.methods}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
