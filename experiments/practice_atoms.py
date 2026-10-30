"""Learn a shallow rollout generator from an imperfect, frozen atomic recognizer.

All external correctness is confined to the finite atom demonstrations and eval.
Practice scores use only the learned truth table. CPU-only, no API calls.
"""

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def signs(d):
    return np.array(list(itertools.product((-1.0, 1.0), repeat=d)))


def features(x, kind, degree=3):
    subsets = [s for k in range(1, degree + 1) for s in itertools.combinations(range(x.shape[1]), k)]
    if kind == "linear":
        h = np.column_stack((np.ones(len(x)), x))
    elif kind == "monomial":
        h = np.column_stack([np.ones(len(x))] + [x[:, s].prod(axis=1) for s in subsets])
    elif kind == "conjunction":
        h = np.column_stack([np.ones(len(x))] + [np.maximum(x[:, s] @ pattern - len(s) + 1, 0) for s in subsets for pattern in signs(len(s))])
    else:
        raise ValueError(kind)
    return h / np.sqrt(np.sum(h[0] ** 2))


def atom_features(a, b):
    return np.stack((np.ones_like(a), a, b, a * b), axis=-1)


def learn_atom(rng, demos, corruption, blind_spot):
    a, b = rng.choice((-1.0, 1.0), (2, demos))
    target = a * b
    if blind_spot:
        target[(a == -1) & (b == -1)] *= -1
    target *= np.where(rng.random(demos) < corruption, -1, 1)
    h = atom_features(a, b)
    v = np.zeros(4)
    history = []
    for step in range(400):
        p = sigmoid(h @ v)
        v += 0.3 * (h.T @ ((target + 1) / 2 - p) / demos - 0.025 * v)
        history.append(float(np.mean(np.logaddexp(0, -target * (h @ v)))))
    return v, {"a": a.tolist(), "b": b.tolist(), "observed_target": target.tolist(), "weights": v.tolist(), "loss": history}


def score_atoms(x, y, v, mode="likelihood"):
    previous = np.concatenate((np.ones(y.shape[:-1] + (1,)), y[..., :-1]), axis=-1)
    current_input = np.broadcast_to(x[..., : y.shape[-1]], y.shape)
    logits = atom_features(previous, current_input) @ v
    if mode == "likelihood":
        return sigmoid(y * logits)
    match = y == np.where(logits >= 0, 1, -1)
    if mode == "hard":
        return match.astype(float)
    if mode == "constant_confidence":
        return np.where(match, 0.8, 0.2)
    raise ValueError(mode)


def true_trace(x, length):
    return np.cumprod(x[:, :length], axis=1)


def exact_eval(w, h, x, v, outputs, mode="likelihood"):
    p = sigmoid(h @ w.T)
    q = np.prod(np.where(outputs[None] == 1, p[:, None], 1 - p[:, None]), axis=-1)
    truth = true_trace(x, w.shape[0])
    correct = np.all(outputs[None] == truth[:, None], axis=-1)
    score = score_atoms(x[:, None], outputs[None] + np.zeros((len(x), 1, 1)), v, mode).prod(axis=-1)
    local_score = score_atoms(x[:, None], outputs[None] + np.zeros((len(x), 1, 1)), v, mode).mean(axis=-1)
    return {"success": float(np.mean(np.sum(q * correct, axis=1))), "greedy_success": float(np.mean(np.all(np.where(p >= 0.5, 1, -1) == truth, axis=1))), "score": float(np.mean(np.sum(q * score, axis=1))), "local_score": float(np.mean(np.sum(q * local_score, axis=1))), "entropy": float(np.mean(-np.sum(p * np.log(p) + (1 - p) * np.log1p(-p), axis=1))), "score_truth_covariance": float(np.mean(np.sum(q * score * correct, axis=1) - np.sum(q * score, axis=1) * np.sum(q * correct, axis=1))), "selected_precision": float(np.mean(np.sum(q * score * correct, axis=1) / np.sum(q * score, axis=1))), "verifier_mode_success": float(np.mean(np.all(outputs[score.argmax(axis=1)] == truth, axis=-1)))}


def executor_accuracy(x, v, length):
    y = np.ones(len(x))
    trace = []
    for step in range(length):
        y = np.where(atom_features(y, x[:, step]) @ v >= 0, 1, -1)
        trace.append(y)
    return float(np.mean(np.all(np.stack(trace, axis=-1) == true_trace(x, length), axis=-1)))


def train(seed, method, args):
    rng = np.random.default_rng(seed)
    x = signs(args.dimension)
    order = rng.permutation(len(x))
    train_ids, test_ids = order[: len(x) // 2], order[len(x) // 2 :]
    if args.split == "composition":
        heldout = np.all(x[:, : args.length] == -1, axis=1)
        train_ids = order[~heldout[order]][: len(x) // 2]
        test_ids = np.flatnonzero(heldout)
    v, atom = learn_atom(rng, args.atom_demos, args.corruption, method in ("blind_spot", "sparse_blind"))
    kind = "linear" if method == "linear" else args.features
    h = features(x, kind, args.length)
    w = rng.normal(0, 0.03, (args.length, h.shape[1]))
    w0 = w.copy()
    outputs = signs(args.length)
    curves, attention_errors = [], []
    started = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            row = {"step": step, "candidates": step * args.batch * args.candidates}
            for split, indices in (("train", train_ids), ("test", test_ids)):
                row.update({split + "_" + k: value for k, value in exact_eval(w, h[indices], x[indices], v, outputs, args.score_mode).items()})
            assert all(np.isfinite(list(row.values())))
            curves.append(row)
            if step >= args.min_steps and len(curves) >= 11:
                window = curves[-10:]
                log_steps = np.log([r["step"] for r in window])
                entropy_slope = np.polyfit(log_steps, np.log([r["train_entropy"] for r in window]), 1)[0]
                internal_metric = "train_local_score" if method == "local" else "train_score"
                loss_slope = np.polyfit(log_steps, np.log([1 - r[internal_metric] for r in window]), 1)[0]
                if abs(entropy_slope) < args.stationarity and abs(loss_slope) < args.stationarity:
                    break
        if step == args.steps:
            break
        indices = rng.choice(train_ids, args.batch)
        hb, xb = h[indices], x[indices]
        p = sigmoid(hb @ w.T)
        source = sigmoid(hb @ w0.T) if method == "frozen" else p
        y = np.where(rng.random((args.batch, args.candidates, args.length)) < source[:, None], 1.0, -1.0)
        atom_scores = score_atoms(xb[:, None], y, v, args.score_mode)
        score = atom_scores.prod(axis=-1)
        if method in ("ungated", "sparse_ungated"):
            advantage = np.ones_like(score)
        elif method == "shuffled":
            score = rng.permutation(score.reshape(-1)).reshape(score.shape)
            advantage = (args.candidates * score - score.sum(axis=1, keepdims=True)) / (args.candidates - 1)
        elif method in ("frozen", "uncentered"):
            advantage = score
        else:
            advantage = (args.candidates * score - score.sum(axis=1, keepdims=True)) / (args.candidates - 1)
        residual = (y + 1) / 2 - p[:, None]
        if method == "local":
            local = (atom_scores + np.concatenate((atom_scores[..., 1:], np.zeros(atom_scores.shape[:-1] + (1,))), axis=-1)) / args.length
            local_advantage = (args.candidates * local - local.sum(axis=1, keepdims=True)) / (args.candidates - 1)
            values = (local_advantage * residual).mean(axis=1)
        else:
            values = (advantage[..., None] * residual).mean(axis=1)
        delta = args.lr * values.T @ hb / args.batch
        decay = args.lr * args.penalty * np.sign(w) if method in ("sparse", "sparse_blind", "sparse_ungated") else np.zeros_like(w)
        if step % args.eval_every == 0:
            qh = h[test_ids]
            explicit = qh @ (w + delta - decay).T
            attention = qh @ (w - decay).T + args.lr * (qh @ hb.T) @ values / args.batch
            attention_errors.append(float(np.max(np.abs(explicit - attention))))
        w += delta - decay
    return {"seed": seed, "method": method, "atom": atom, "curves": curves, "weights_initial": w0.tolist(), "weights_final": w.tolist(), "train_ids": train_ids.tolist(), "test_ids": test_ids.tolist(), "feature_count": h.shape[1], "atom_greedy_accuracy": float(np.mean(np.where(atom_features(*signs(2).T) @ v >= 0, 1, -1) == signs(2).prod(axis=1))), "executor_test_accuracy": executor_accuracy(x[test_ids], v, args.length), "attention_max_error": max(attention_errors), "seconds": time.perf_counter() - started}


def plot(results, output):
    methods = list(dict.fromkeys(r["method"] for r in results))
    names = {"current": "Learned atom scoring", "frozen": "Frozen proposals", "ungated": "Unfiltered self-fit", "shuffled": "Shuffled scores", "linear": "Linear generator", "blind_spot": "Wrong learned atom", "uncentered": "Learned scoring, no baseline", "local": "Local process scores", "sparse": "Scoring + sparse weights", "sparse_blind": "Sparse + wrong atom", "sparse_ungated": "Sparse + unfiltered self-fit"}
    for metric, label, filename in (("test_success", "Held-out one-shot trace success", "practice-atoms-learning.png"), ("train_score", "Expected learned recognition score", "practice-atoms-score.png")):
        fig, ax = plt.subplots()
        for method in methods:
            rows = [r for r in results if r["method"] == method]
            grid = sorted({c["candidates"] for r in rows for c in r["curves"]})
            ys = np.array([np.interp(grid, [c["candidates"] for c in r["curves"]], [c[metric] for c in r["curves"]]) for r in rows])
            mean = ys.mean(axis=0)
            sem = ys.std(axis=0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else np.zeros_like(mean)
            line, = ax.plot(grid, mean, label=names[method])
            ax.fill_between(grid, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.15, color=line.get_color())
        ax.set(xlabel="Generated candidate traces", ylabel=label, ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pilot")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=181000)
    parser.add_argument("--methods", nargs="+", default=["current", "frozen", "ungated", "shuffled", "linear", "blind_spot"])
    parser.add_argument("--features", choices=["monomial", "conjunction"], default="monomial")
    parser.add_argument("--split", choices=["random", "composition"], default="random")
    parser.add_argument("--score-mode", choices=["likelihood", "hard", "constant_confidence"], default="likelihood")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--atom-demos", type=int, default=48)
    parser.add_argument("--corruption", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--min-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--stationarity", type=float, default=0.015)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--lr", type=float, default=20)
    parser.add_argument("--penalty", type=float, default=0.001)
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    output = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_atoms" / args.name
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "source.py").write_text(Path(__file__).read_text())
    results = []
    for seed in tqdm(range(args.seed_start, args.seed_start + args.seeds), desc="Worlds"):
        for method in tqdm(args.methods, desc="Conditions", leave=False):
            record = train(seed, method, args)
            results.append(record)
            (output / f"{seed}_{method}.json").write_text(json.dumps(record) + "\n")
    plot(results, output)
    summary = {method: {"test_success": float(np.mean([r["curves"][-1]["test_success"] for r in results if r["method"] == method])), "test_greedy": float(np.mean([r["curves"][-1]["test_greedy_success"] for r in results if r["method"] == method])), "executor": float(np.mean([r["executor_test_accuracy"] for r in results if r["method"] == method])), "max_attention_error": float(max(r["attention_max_error"] for r in results if r["method"] == method)), "mean_final_step": float(np.mean([r["curves"][-1]["step"] for r in results if r["method"] == method]))} for method in args.methods}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
