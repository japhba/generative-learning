"""Summarize the exploratory atom-practice experiments without rerunning them."""

import hashlib
import itertools
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(Path.home() / ".env")
DATA = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_atoms"
HIDDEN = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_hidden"
OUT = ROOT / "tex" / "archive" / "figures"
SUMMARY = Path(os.environ["FAST_CACHE_DIR"]) / "generative_practice_synthesis"


def records(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("[0-9]*_*.json"))]


def estimate(values):
    a = np.asarray(values)
    rng = np.random.default_rng(191001)
    bootstrap = rng.choice(a, (10000, len(a)), replace=True).mean(axis=1)
    return {"mean": float(a.mean()), "seed_sd": float(a.std(ddof=1)), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(), "n": len(a)}


def curve(ax, rows, field, label):
    grid = sorted({c["candidates"] for r in rows for c in r["curves"]})
    y = np.array([np.interp(grid, [c["candidates"] for c in r["curves"]], [c[field] for c in r["curves"]]) for r in rows])
    mean = y.mean(axis=0)
    sem = y.std(axis=0, ddof=1) / np.sqrt(len(y))
    line, = ax.plot(grid, mean, label=label)
    ax.fill_between(grid, mean - 1.96 * sem, mean + 1.96 * sem, color=line.get_color(), alpha=0.15)


def learning(name, methods, filename):
    rows = records(DATA / name)
    fig, ax = plt.subplots()
    for method, label in methods:
        curve(ax, [r for r in rows if r["method"] == method], "test_success", label)
    ax.axhline(0.125, linestyle=":", color="gray", label="Uniform generator")
    ax.set(xlabel="Generated candidate traces", ylabel="Held-out complete-trace success", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in sorted(DATA.iterdir()):
        rows = records(path)
        methods = sorted({r["method"] for r in rows})
        summary[path.name] = {}
        for method in methods:
            group = [r for r in rows if r["method"] == method]
            result = {key: estimate([r["curves"][-1][key] for r in group]) for key in ("test_success", "test_greedy_success", "train_success", "train_score", "test_entropy")}
            result["atom_greedy_accuracy"] = estimate([r["atom_greedy_accuracy"] for r in group])
            result["executor_accuracy"] = estimate([r["executor_test_accuracy"] for r in group])
            result["attention_max_error"] = max(r["attention_max_error"] for r in group)
            result["initial_success"] = estimate([r["curves"][0]["test_success"] for r in group])
            result["gain"] = estimate([r["curves"][-1]["test_success"] - r["curves"][0]["test_success"] for r in group])
            imperfect = [r for r in group if r["atom_greedy_accuracy"] < 1]
            result["imperfect_worlds"] = [{"seed": r["seed"], "atom_accuracy": r["atom_greedy_accuracy"], "before": r["curves"][0]["test_success"], "after": r["curves"][-1]["test_success"], "executor": r["executor_test_accuracy"]} for r in imperfect]
            summary[path.name][method] = result
    learning("confirmation", [("current", "Learned atom scoring"), ("frozen", "Fixed initial proposal"), ("ungated", "Unfiltered practice"), ("linear", "Linear features"), ("blind_spot", "Wrong learned atom")], "practice-atoms-heldout.png")
    learning("composition_confirmation", [("sparse", "Atom scoring + sparse weights"), ("current", "Atom scoring"), ("sparse_ungated", "Sparse + unfiltered practice"), ("sparse_blind", "Sparse + wrong atom")], "practice-atoms-composition.png")
    fig, ax = plt.subplots()
    for root, label in (("recognition_matched_n", "Likelihood-product score"), ("recognition_hard_n", "Learned modal-atom gate")):
        means, lower, upper = [], [], []
        for n in (4, 8, 16, 48):
            result = summary[f"{root}{n}"]["sparse"]["test_success"]
            means.append(result["mean"])
            lower.append(result["mean"] - result["ci95"][0])
            upper.append(result["ci95"][1] - result["mean"])
        ax.errorbar([4, 8, 16, 48], means, yerr=[lower, upper], marker="o", label=label)
    executor = [summary[f"recognition_hard_n{n}"]["sparse"]["executor_accuracy"]["mean"] for n in (4, 8, 16, 48)]
    ax.plot([4, 8, 16, 48], executor, linestyle="--", label="Direct execution of learned atoms")
    ax.axhline(0.125, linestyle=":", color="gray", label="Uniform generator")
    ax.set(xlabel="Noisy atomic demonstrations (15% label noise)", ylabel="Unseen-combination trace success", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "practice-atoms-imperfect.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for n in (4, 8, 16, 48):
        losses = np.array([r["atom"]["loss"] for r in records(DATA / f"recognition_n{n}")])
        ax.plot(np.arange(1, losses.shape[1] + 1), losses.mean(axis=0), label=f"{n} atom observations")
    ax.set(xlabel="Atomic recognizer update", ylabel="Training negative log likelihood")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "practice-atoms-recognizer-learning.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for method, label in (("local", "Train both layers"), ("frozen_hidden", "Fixed random hidden layer"), ("shuffled", "Shuffled recognition scores")):
        curve(ax, [r for r in records(HIDDEN / "pilot") if r["method"] == method], "test_success", label)
    ax.axhline(0.125, linestyle=":", color="gray", label="Uniform generator")
    ax.set(xlabel="Generated candidate traces", ylabel="Held-out complete-trace success", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "practice-hidden-learning.png")
    plt.close(fig)
    rows = [r for r in records(DATA / "composition_confirmation") if r["method"] == "sparse"]
    weights = np.array([r["weights_final"] for r in rows]).mean(axis=0)
    features = [()] + [s for k in range(1, 4) for s in itertools.combinations(range(8), k)]
    chosen = np.argsort(np.max(np.abs(weights), axis=0))[-10:]
    chosen = chosen[np.argsort(chosen)]
    labels = ["1" if not features[i] else " ".join(f"x{j+1}" for j in features[i]) for i in chosen]
    fig, ax = plt.subplots()
    im = ax.imshow(weights[:, chosen], aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(chosen)), labels, rotation=45, ha="right")
    ax.set_yticks([0, 1, 2], ["y1", "y2", "y3"])
    ax.set(xlabel="Ten features with largest mean absolute output weights", ylabel="Generated step")
    fig.colorbar(im, ax=ax, label="Mean final weight (32 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "practice-atoms-weights.png")
    plt.close(fig)
    (SUMMARY / "atom_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifests = []
    for root in (DATA, HIDDEN):
        for path in sorted(root.glob("*/config.json")):
            manifests.append({"path": str(path.parent), "config": json.loads(path.read_text()), "record_count": len(records(path.parent)), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (SUMMARY / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print(SUMMARY / "atom_summary.json")
    print(SUMMARY / "manifest.json")


if __name__ == "__main__":
    main()
