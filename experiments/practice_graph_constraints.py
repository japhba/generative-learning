"""Learn atomic color relations from demonstrations, then practise on unlabeled graphs."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

load_dotenv(Path.home()/".env")
ROOT = Path(__file__).resolve().parents[1]


def graphs(rng, batch, vertices, probability):
    upper = np.triu(rng.random((batch, vertices, vertices)) < probability, 1)
    return upper | upper.transpose(0, 2, 1)


class ExactColorings:
    """Used for initial exposure and held-out solvability only; absent from practice."""

    def __init__(self, vertices):
        self.edges = np.triu_indices(vertices, 1)
        self.assignments = np.c_[np.zeros(3**(vertices-1), dtype=np.uint8),
            ((np.arange(3**(vertices-1))[:, None]//(3**np.arange(vertices-2, -1, -1))) % 3).astype(np.uint8)]
        self.equal = self.assignments[:, self.edges[0]] == self.assignments[:, self.edges[1]]

    def valid(self, graph):
        return ~self.equal[:, graph[self.edges]].any(axis=1)

    def demonstrations(self, rng, number, probability):
        vertices = self.assignments.shape[1]
        inputs, outputs, attempts = [], [], 0
        progress = tqdm(total=number, desc="Finite successful demonstrations")
        while len(inputs) < number:
            graph = graphs(rng, 1, vertices, probability)[0]
            valid = np.flatnonzero(self.valid(graph))
            attempts += 1
            if valid.size:
                permutation = rng.permutation(3)
                inputs.append(graph)
                outputs.append(permutation[self.assignments[rng.choice(valid)]])
                progress.update(1)
        progress.close()
        return np.array(inputs), np.array(outputs), attempts


def learn_relation(inputs, outputs):
    counts = np.zeros((3, 3), dtype=int)
    for graph, colors in zip(inputs, outputs):
        left, right = np.nonzero(graph)
        np.add.at(counts, (colors[left], colors[right]), 1)
    return counts, counts > 0


def features(graph, colors, step, relation):
    batch, vertices, _ = graph.shape
    encoded = colors[:, :, None] == np.arange(3)
    neighbor_counts = np.einsum("bij,bjc->bic", graph.astype(int), encoded.astype(int))
    allowed = (neighbor_counts @ (~relation).T.astype(int)) == 0
    future = np.arange(vertices) > step
    adjacent = graph[:, :, step] & future
    invalidated = allowed.astype(int) @ (~relation).astype(int)
    remaining = allowed.sum(axis=-1)[:, :, None]-graph[:, :, step, None]*invalidated
    dead = np.sum(adjacent[:, :, None] & (remaining == 0), axis=1)
    forced = adjacent[:, :, None] & (remaining == 1)
    future_degree = np.sum(graph[:, :, future], axis=-1)
    pressure = np.sum(forced*future_degree[:, :, None], axis=1)/vertices
    reuse = encoded.sum(axis=1)/vertices
    phi = np.stack([dead, forced.sum(axis=1), pressure, reuse], axis=-1).astype(float)
    return phi, allowed[:, step]


def policy(phi, allowed, weights):
    logits = phi @ weights
    logits = np.where(allowed, logits, -np.inf)
    p = np.exp(logits-logits.max(axis=-1, keepdims=True))
    return p/p.sum(axis=-1, keepdims=True)


def expose_features(inputs, outputs, relation):
    batch, vertices, _ = inputs.shape
    colors = np.full((batch, vertices), -1)
    all_phi, all_allowed, all_action = [], [], []
    for step in range(vertices):
        phi, allowed = features(inputs, colors, step, relation)
        assert np.all(allowed[np.arange(batch), outputs[:, step]])
        all_phi.append(phi); all_allowed.append(allowed); all_action.append(outputs[:, step])
        colors[:, step] = outputs[:, step]
    return np.concatenate(all_phi), np.concatenate(all_allowed), np.concatenate(all_action)


def fit_exposure(phi, allowed, actions, ridge):
    n = len(actions)
    chosen = phi[np.arange(n), actions]

    def objective(weights, derivatives=False):
        probability = policy(phi, allowed, weights)
        nll = -np.log(probability[np.arange(n), actions]).mean()
        value = nll+ridge*np.dot(weights, weights)/2
        if not derivatives: return value
        mean_phi = np.einsum("bc,bcf->bf", probability, phi)
        gradient = (mean_phi-chosen).mean(axis=0)+ridge*weights
        centered = phi-mean_phi[:, None]
        hessian = np.einsum("bc,bci,bcj->ij", probability, centered, centered)/n+ridge*np.eye(phi.shape[-1])
        return value, nll, gradient, hessian

    weights = np.zeros(phi.shape[-1])
    history = []
    for step in range(100):
        value, nll, gradient, hessian = objective(weights, True)
        history.append([step, value, nll, np.linalg.norm(gradient)])
        if np.linalg.norm(gradient) < 1e-9: break
        direction = np.linalg.solve(hessian, gradient)
        rate = 1.0
        while objective(weights-rate*direction) > value-.1*rate*np.dot(direction, gradient):
            rate *= .5
        weights -= rate*direction
    assert history[-1][3] < 1e-9
    return weights, np.array(history)


def generate(inputs, weights, relation, rng, deterministic=False):
    batch, vertices, _ = inputs.shape
    colors = np.full((batch, vertices), -1)
    alive, score = np.ones(batch, dtype=bool), np.zeros((batch, len(weights)))
    for step in range(vertices):
        phi, allowed = features(inputs, colors, step, relation)
        alive &= allowed.any(axis=-1)
        # Dead ends are absorbing failures; dummy actions only keep vectorized shapes fixed.
        allowed[~alive] = True
        probability = policy(phi, allowed, weights)
        if deterministic: action = probability.argmax(axis=-1)
        else: action = (rng.random((batch, 1)) > probability.cumsum(axis=-1)).sum(axis=-1)
        score += alive[:, None]*(phi[np.arange(batch), action]-np.einsum("bc,bcf->bf", probability, phi))
        colors[:, step] = action
    accepted = relation[colors[:, :, None], colors[:, None, :]] | ~inputs
    reward = accepted.all(axis=(1, 2)).astype(float)
    assert np.array_equal(reward.astype(bool), alive)
    return reward, score, colors


def dsatur(graph):
    vertices = len(graph)
    colors = np.full(vertices, -1)
    for _ in range(vertices):
        uncolored = np.flatnonzero(colors < 0)
        saturation = np.array([len(set(colors[graph[node]])-{-1}) for node in uncolored])
        priorities = saturation*(vertices+1)+graph[uncolored].sum(axis=-1)
        node = uncolored[priorities.argmax()]
        unavailable = set(colors[graph[node]])
        available = [color for color in range(3) if color not in unavailable]
        if not available: return False
        colors[node] = available[0]
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20103450)
    parser.add_argument("--vertices", type=int, default=12)
    parser.add_argument("--edge-probability", type=float, default=.4)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--test-graphs", type=int, default=256)
    parser.add_argument("--test-attempts", type=int, default=16)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--rate", type=float, default=1.)
    parser.add_argument("--ridge", type=float, default=.001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    (args.output/"protocol.json").write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        status="Exploratory pilot; all defined controls retained",
        exposure="Finite successful examples from ER conditioned 3-colorable; a uniform exact coloring, followed by random palette permutation.",
        practice="Fresh unconditional ER graphs, learned atom checker only, no exact solver. Raw per-attempt success includes dead-end failure; no restarts.",
        initial_fit="Convex masked softmax conditional NLL plus fixed ridge; Newton optimization to gradient norm below 1e-9.",
        features=["new future-neighbor dead ends", "future forced neighbors", "future-degree-weighted forced neighbors", "previous color reuse"],
        controls="Uniform legal, lexicographic legal, DSATUR without backtracking, and a predeclared signed feature menu.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+"\n")
    streams = np.random.SeedSequence(args.seed).spawn(4)
    exposure_rng, test_rng, practice_rng = [np.random.default_rng(s) for s in streams[:3]]
    exact = ExactColorings(args.vertices)
    observed_g, observed_c, exposure_attempts = exact.demonstrations(exposure_rng, args.examples, args.edge_probability)
    counts, relation = learn_relation(observed_g, observed_c)
    assert np.array_equal(relation, ~np.eye(3, dtype=bool))
    phi, allowed, actions = expose_features(observed_g, observed_c, relation)
    initial, initial_history = fit_exposure(phi, allowed, actions, args.ridge)
    test_g = graphs(test_rng, args.test_graphs, args.vertices, args.edge_probability)
    satisfiable = np.array([exact.valid(graph).any() for graph in tqdm(test_g, desc="Evaluation-only solvability")])
    training_keys = {graph.tobytes() for graph in observed_g}
    assert not any(graph.tobytes() in training_keys for graph in test_g)
    test_inputs = np.repeat(test_g, args.test_attempts, axis=0)
    test_feasible = np.repeat(satisfiable, args.test_attempts)
    evaluation_seed = int(np.random.default_rng(streams[3]).integers(2**32))

    def evaluate(weights, deterministic=False):
        reward = generate(test_inputs, weights, relation, np.random.default_rng(evaluation_seed), deterministic)[0]
        return float(reward.mean()), float(reward[test_feasible].mean())

    menu = {"uniform_legal": np.zeros(4), "dead_ends": np.array([-12, 0, 0, 0]),
            "forced_neighbors": np.array([0, -12, 0, 0]), "forced_degree": np.array([0, 0, -12, 0]),
            "reuse": np.array([0, 0, 0, 12]), "equal_signed": np.array([-12, -12, -12, 12]),
            "dead_then_forced": np.array([-120, -12, 0, 0])}
    controls = {key: dict(weights=value.tolist(), unconditional=evaluate(value)[0], conditional=evaluate(value)[1]) for key, value in menu.items()}
    lexical = evaluate(np.zeros(4), True)
    greedy = np.array([dsatur(graph) for graph in test_g])
    controls["lexical_legal"] = dict(unconditional=lexical[0], conditional=lexical[1])
    controls["dsatur"] = dict(unconditional=float(greedy.mean()), conditional=float(greedy[satisfiable].mean()))
    weights, records, parameters = initial.copy(), [], []
    recent = []
    for step in tqdm(range(args.steps+1), desc="Offline coloring practice"):
        if step % args.eval_every == 0:
            unconditional, conditional = evaluate(weights)
            train_mean = float(np.mean(recent)) if step else float("nan")
            records.append([step, unconditional, conditional, train_mean])
            parameters.append(weights.copy())
            np.savez_compressed(args.output/"trajectory.npz", records=np.array(records), weights=np.array(parameters),
                initial=initial, initial_history=initial_history, relation=relation, relation_counts=counts,
                observed_graphs=observed_g, observed_colors=observed_c, test_graphs=test_g, satisfiable=satisfiable,
                rng=json.dumps(practice_rng.bit_generator.state))
            recent = []
        if step == args.steps: break
        inputs = graphs(practice_rng, args.batch, args.vertices, args.edge_probability)
        reward, score, _ = generate(inputs, weights, relation, practice_rng)
        advantage = args.batch/(args.batch-1)*(reward-reward.mean())
        weights += args.rate*np.mean(advantage[:, None]*score, axis=0)
        assert np.isfinite(weights).all()
        recent.append(float(reward.mean()))
    summary = dict(initial=records[0], final=records[-1], initial_weights=initial.tolist(), final_weights=weights.tolist(),
        feature_names=["future_dead", "future_forced", "forced_future_degree", "reuse"],
        controls=controls, relation_counts=counts.tolist(), relation_correct=True,
        feasible_test_fraction=float(satisfiable.mean()), exposure_graph_draws=exposure_attempts,
        seed=args.seed, interpretation="Conditional validity, not a density-likelihood claim. Exact solver used only for initial examples and test solvability.")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    records, parameters = np.array(records), np.array(parameters)
    fig, axes = plt.subplots(2, 1, layout="constrained")
    axes[0].plot(records[:, 0], records[:, 2], label="Learned checker practice")
    for key in ["uniform_legal", "equal_signed", "dead_then_forced", "dsatur"]:
        axes[0].axhline(controls[key]["conditional"], linestyle="--", label=key.replace("_", " "))
    axes[0].axhline(records[0, 2], color="black", linestyle=":", label="Converged example fit")
    axes[0].set(ylabel="Single-attempt validity\non fresh colorable graphs")
    axes[0].legend()
    for column, label in enumerate(summary["feature_names"]): axes[1].plot(records[:, 0], parameters[:, column], label=label.replace("_", " "))
    axes[1].set(xlabel="Offline practice updates", ylabel="Generator weights")
    axes[1].legend()
    figure = args.output/"learning.png"
    fig.savefig(figure)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(str(figure))


if __name__ == "__main__": main()
