"""Normalized linear-energy Bernoulli modes for accepted-mean self-training."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def probability(w): return 1 / (1 + np.exp(-w))


def population(initial, source, times):
    w = np.log(initial / (1 - initial))
    frozen_q = initial * (2 - initial)

    def drift(value):
        p = probability(value)
        q = p * (2 - p) if source == "current" else frozen_q
        return q * (1 - p) / q.sum()

    records = [{"time": 0., "weights": w.tolist(), "error": float(np.mean(1 - initial))}]
    previous = 0.
    for t in times:
        h = t - previous
        a = drift(w); b = drift(w + h * a / 2); c = drift(w + h * b / 2); d = drift(w + h * c)
        w += h * (a + 2 * b + 2 * c + d) / 6
        records.append({"time": float(t), "weights": w.tolist(), "error": float(np.mean(1 - probability(w)))})
        previous = t
    return records


def finite(initial, source, seed, steps, replicas=128, batch=32, eta=.1):
    rng = np.random.default_rng(seed)
    weights = np.tile(np.log(initial / (1 - initial)), (replicas, 1))
    checks = sorted(set(np.geomspace(1, steps, 65).astype(int)) | {0, steps})
    records = [{"step": 0, "error": (1 - probability(weights)).mean(-1).tolist(), "weights": weights.tolist(), "accepted": np.zeros(replicas).tolist()}]
    total = np.zeros(replicas)
    for step in tqdm(range(1, steps + 1), desc=f"Bernoulli modes: {source}"):
        p = probability(weights)
        classes = rng.integers(len(initial), size=(replicas, batch))
        source_p = p[np.arange(replicas)[:, None], classes] if source == "current" else initial[classes]
        accept = ((rng.random((replicas, batch, 2)) < source_p[..., None]).any(-1))
        count = accept.sum(-1); total += count
        accepted_by_class = np.stack([(accept & (classes == k)).sum(-1) for k in range(len(initial))], axis=-1)
        # An entirely rejected batch has zero update, as in the LLM rule.
        weights += eta * accepted_by_class * (1 - p) / np.maximum(count[:, None], 1)
        if step in checks:
            records.append({"step": step, "error": (1 - probability(weights)).mean(-1).tolist(), "weights": weights.tolist(), "accepted": total.tolist()})
    return {"source": source, "seed": seed, "steps": steps, "replicas": replicas, "batch": batch, "eta": eta, "records": records}


def main(output, steps):
    output.mkdir(parents=True, exist_ok=False)
    initial = np.array([.2, .75, .95])
    q = initial * (2 - initial)
    prediction = {"created_at_uk": stamp(), "initial_success": initial.tolist(), "initial_yield": q.tolist(), "beta": 1.,
                  "current_amplitude": len(initial), "frozen_amplitude": float(q.mean() * np.sum(1 / q)),
                  "predicted_candidate_speedup": float(q.mean() * np.mean(1 / q)),
                  "assumptions": "Equal-frequency independent Bernoulli modes; accepted-mean SGD; same eta; large-time population limit. No LLM exponent or amplitude is fitted or asserted."}
    save(output / "prospective.json", prediction)
    times = np.geomspace(.001, 1e7, 20000)
    curves = {source: population(initial, source, times) for source in ("current", "frozen")}
    simulations = {source: finite(initial, source, 197000 + i, steps) for i, source in enumerate(("current", "frozen"))}
    summary = {}
    fig, ax = plt.subplots()
    for source, curve in curves.items():
        t = np.array([r["time"] for r in curve]); error = np.array([r["error"] for r in curve])
        line, = ax.loglog(t[1:], error[1:], label=f"{source}: population")
        simulation = simulations[source]
        x = simulation["eta"] * np.array([r["step"] for r in simulation["records"]])
        y = np.array([r["error"] for r in simulation["records"]]).mean(-1)
        ax.loglog(x[1:], y[1:], "o", color=line.get_color(), label=f"{source}: finite batches")
        late = t >= 1e5
        beta = float(-np.polyfit(np.log(t[late]), np.log(error[late]), 1)[0])
        amplitude = prediction[f"{source}_amplitude"]
        ax.loglog(t[late], amplitude / t[late], "--", color=line.get_color())
        summary[source] = {"population_beta": beta, "endpoint_amplitude_ratio": float(error[-1] * t[-1] / amplitude)}
    ax.set(xlabel="SGD time (learning rate × updates)", ylabel="Mean error probability", title="Same exponent; heterogeneous modes give a sample-efficiency gain")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-toy.png"); plt.close(fig)
    fig, ax = plt.subplots()
    for source, curve in curves.items():
        w = np.array([r["weights"] for r in curve]); t = np.array([r["time"] for r in curve])
        alignment = w.mean(-1) / np.sqrt(np.mean(w * w, axis=-1))
        ax.semilogx(t[1:], alignment[1:], label=source)
    ax.set(xlabel="Population SGD time", ylabel="Cosine with equal positive mode weights", title="Interpretable weights in the linear-energy Bernoulli model")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/scaling-verified-toy-weights.png"); plt.close(fig)
    save(output / "results.json", {"prediction": prediction, "summary": summary, "population": curves, "finite": simulations})
    print(json.dumps({"prediction": prediction, "summary": summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=ROOT / "results/scaling_r5s/toy"); parser.add_argument("--steps", type=int, default=65536)
    args = parser.parse_args(); main(args.output, args.steps)
