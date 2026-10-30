"""A single declared extension: unknown balanced nonfunctional constraint relations."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from practice_graph_constraints import policy


def context(rng, batch, vertices, probability):
    edge = rng.random((batch, vertices, vertices)) < probability
    kind = rng.integers(0, 3, size=edge.shape)
    upper = np.triu(np.where(edge, kind+1, 0), 1)
    return upper+upper.transpose(0, 2, 1)-1


def teacher_relations():
    forbidden = np.array([[0, 1, 2], [1, 2, 0], [1, 0, 2]])
    result = np.ones((3, 3, 3), dtype=bool)
    result[np.arange(3)[:, None], np.arange(3), forbidden] = False
    assert np.all(result.sum(axis=1) == 2) and np.all(result.sum(axis=2) == 2)
    return result


class Exact:
    def __init__(self, vertices, relations):
        self.edges = np.triu_indices(vertices, 1)
        self.assignments = ((np.arange(3**vertices)[:, None]//(3**np.arange(vertices-1, -1, -1))) % 3).astype(np.uint8)
        self.bad = ~relations[:, self.assignments[:, self.edges[0]], self.assignments[:, self.edges[1]]]

    def valid(self, graph):
        answer = np.ones(len(self.assignments), dtype=bool)
        for kind in range(3): answer &= ~self.bad[kind][:, graph[self.edges] == kind].any(axis=1)
        return answer

    def exposure(self, rng, number, probability):
        inputs, outputs, attempts = [], [], 0
        progress = tqdm(total=number, desc="Finite typed demonstrations", mininterval=2)
        while len(inputs) < number:
            graph = context(rng, 1, self.assignments.shape[1], probability)[0]
            valid = np.flatnonzero(self.valid(graph))
            attempts += 1
            if valid.size:
                inputs.append(graph); outputs.append(self.assignments[rng.choice(valid)])
                progress.update(1)
        progress.close()
        return np.array(inputs), np.array(outputs), attempts


def acquire(inputs, colors):
    counts = np.zeros((3, 3, 3), dtype=int)
    left, right = np.triu_indices(inputs.shape[1], 1)
    for graph, values in zip(inputs, colors):
        present = graph[left, right] >= 0
        np.add.at(counts, (graph[left[present], right[present]], values[left[present]], values[right[present]]), 1)
    return counts, counts > 0


def feature_block(inputs, allowed, previous, step, extended):
    batch, vertices, _ = inputs.shape
    edge_type = inputs[:, step, step+1:]
    compatible = extended[edge_type]
    remaining = np.einsum("bucv,buv->buc", compatible.astype(int), allowed[:, step+1:].astype(int))
    present = edge_type >= 0
    dead = ((remaining == 0) & present[:, :, None]).sum(axis=1)
    forced, pressure, eliminated = [], [], []
    future_degree = (inputs[:, step+1:, step+1:] >= 0).sum(axis=-1)
    removed = allowed[:, step+1:].sum(axis=-1)[:, :, None]-remaining
    for kind in range(3):
        selected = edge_type == kind
        one = (remaining == 1) & selected[:, :, None]
        forced.append(one.sum(axis=1))
        pressure.append((one*future_degree[:, :, None]).sum(axis=1)/vertices)
        eliminated.append((removed*selected[:, :, None]).sum(axis=1)/vertices)
    biases = np.broadcast_to(np.eye(3), (batch, 3, 3))
    reuse = (previous[:, :, None] == np.arange(3)).sum(axis=1)/vertices
    return np.concatenate([dead[:, :, None], np.stack(forced, axis=-1), np.stack(pressure, axis=-1),
                           np.stack(eliminated, axis=-1), biases, reuse[:, :, None]], axis=-1)


def generate(inputs, weights, relations, rng, target=None, deterministic=False, record_features=False):
    batch, vertices, _ = inputs.shape
    extended = np.concatenate([relations, np.ones((1, 3, 3), dtype=bool)])
    allowed = np.ones((batch, vertices, 3), dtype=bool)
    previous = np.full((batch, vertices), -1)
    alive, score = np.ones(batch, dtype=bool), np.zeros((batch, len(weights)))
    saved_phi, saved_allowed, saved_action = [], [], []
    for step in range(vertices):
        phi = feature_block(inputs, allowed, previous, step, extended)
        current = allowed[:, step].copy()
        alive &= current.any(axis=-1)
        current[~alive] = True
        probability = policy(phi, current, weights)
        if target is not None:
            action = target[:, step]
            assert np.all(alive) and np.all(current[np.arange(batch), action])
        elif deterministic: action = probability.argmax(axis=-1)
        else: action = (rng.random((batch, 1)) > probability.cumsum(axis=-1)).sum(axis=-1)
        score += alive[:, None]*(phi[np.arange(batch), action]-np.einsum("bc,bcf->bf", probability, phi))
        if record_features:
            saved_phi.append(phi); saved_allowed.append(current); saved_action.append(action)
        previous[:, step] = action
        future_relation = extended[inputs[:, step, step+1:]]
        permitted = future_relation[np.arange(batch)[:, None], np.arange(vertices-step-1), action[:, None]]
        allowed[:, step+1:] &= permitted
    left, right = np.triu_indices(vertices, 1)
    correct = extended[inputs[:, left, right], previous[:, left], previous[:, right]].all(axis=-1)
    assert np.array_equal(correct, alive)
    if record_features: return np.concatenate(saved_phi), np.concatenate(saved_allowed), np.concatenate(saved_action)
    return correct.astype(float), score


def constrained_exposure(phi, allowed, actions, radius):
    chosen = phi[np.arange(len(actions)), actions]
    history = []

    def objective(weights):
        probability = policy(phi, allowed, weights)
        value = -np.log(probability[np.arange(len(actions)), actions]).mean()
        gradient = (np.einsum("bc,bcf->bf", probability, phi)-chosen).mean(axis=0)
        return value, gradient

    def callback(weights): history.append([len(history), objective(weights)[0], np.linalg.norm(weights)])

    result = minimize(objective, np.zeros(phi.shape[-1]), jac=True, method="SLSQP",
        constraints=[dict(type="ineq", fun=lambda w: radius**2-np.dot(w, w), jac=lambda w: -2*w)],
        callback=callback, options=dict(ftol=1e-12, maxiter=2000))
    assert result.success, result.message
    value, gradient = objective(result.x)
    multiplier = max(0., -np.dot(gradient, result.x)/radius**2)
    residual = np.linalg.norm(gradient+multiplier*result.x)
    assert residual < 1e-5 and np.linalg.norm(result.x) <= radius+1e-7
    return result.x, np.array(history), dict(nll=value, kkt_residual=residual, multiplier=multiplier, steps=result.nit)


def greedy_mrv(graph, relations):
    """Generic direct MRV + least-constraining-value, without backtracking."""
    vertices = len(graph)
    assigned = np.full(vertices, -1)
    available = np.ones((vertices, 3), dtype=bool)
    for _ in range(vertices):
        unassigned = np.flatnonzero(assigned < 0)
        count = available[unassigned].sum(axis=-1)
        if np.any(count == 0): return False
        degree = (graph[unassigned][:, unassigned] >= 0).sum(axis=-1)
        node = unassigned[np.lexsort((-degree, count))[0]]
        candidates = np.flatnonzero(available[node])
        best, best_score = candidates[0], -1
        for value in candidates:
            total = 0
            for other in unassigned:
                if graph[node, other] < 0: continue
                atom = relations[graph[node, other]]
                compatible = atom[value] if node < other else atom[:, value]
                total += np.sum(available[other] & compatible)
            if total > best_score: best, best_score = value, total
        assigned[node] = best
        for other in unassigned:
            if graph[node, other] < 0: continue
            atom = relations[graph[node, other]]
            compatible = atom[best] if node < other else atom[:, best]
            available[other] &= compatible
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20103460)
    parser.add_argument("--vertices", type=int, default=10)
    parser.add_argument("--edge-probability", type=float, default=.55)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--test-graphs", type=int, default=512)
    parser.add_argument("--test-attempts", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--radius", type=float, default=12.)
    parser.add_argument("--rate", type=float, default=1.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    protocol = dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(), exploratory=True,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        design="Three unknown 3x3 allowed relations, complements of identity/cycle/transposition permutations. Six allowed pairs per relation; equal row/column degrees.",
        baseline="Supervised conditional NLL minimized over the SAME radius-12 Euclidean ball used in practice. No ridge or temperature change.",
        features="Total future dead ends; forced neighbor count, forced future degree and eliminated choices by edge type; three candidate biases; reuse.",
        practice="Unconditional typed ER contexts; complete validity from learned relations; no solver, restarts, labels or planted assignments.",
        risk="Fresh satisfiable typed ER contexts; unsatisfiable unconditional contexts have reward zero for all policies.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    streams = np.random.SeedSequence(args.seed).spawn(4)
    exposure_rng, practice_rng, test_rng, evaluation_rng = [np.random.default_rng(s) for s in streams]
    teacher = teacher_relations()
    exact = Exact(args.vertices, teacher)
    examples, solutions, attempts = exact.exposure(exposure_rng, args.examples, args.edge_probability)
    counts, learned = acquire(examples, solutions)
    assert np.array_equal(learned, teacher)
    phi, allowed, action = generate(examples, np.zeros(14), learned, exposure_rng, target=solutions, record_features=True)
    initial, exposure_history, exposure_fit = constrained_exposure(phi, allowed, action, args.radius)
    testing = context(test_rng, args.test_graphs, args.vertices, args.edge_probability)
    feasible = np.array([exact.valid(graph).any() for graph in tqdm(testing, desc="Evaluation-only satisfiability", mininterval=2)])
    assert not ({graph.tobytes() for graph in examples} & {graph.tobytes() for graph in testing})
    test_inputs = np.repeat(testing, args.test_attempts, axis=0)
    test_feasible = np.repeat(feasible, args.test_attempts)
    evaluation_seed = int(evaluation_rng.integers(2**32))

    def evaluate(weights, deterministic=False):
        reward, _ = generate(test_inputs, weights, learned, np.random.default_rng(evaluation_seed), deterministic=deterministic)
        return [float(reward.mean()), float(reward[test_feasible].mean())]

    menu = {"uniform_legal": np.zeros(14)}
    for label, indices, sign in [("dead_only", [0], -1), ("forced_only", [1, 2, 3], -1),
                               ("pressure_only", [4, 5, 6], -1), ("elimination_only", [7, 8, 9], -1),
                               ("reuse", [13], 1), ("value0", [10], 1), ("value1", [11], 1), ("value2", [12], 1)]:
        value = np.zeros(14); value[indices] = sign
        menu[label] = args.radius*value/np.linalg.norm(value)
    value = np.r_[-10., -np.ones(3), np.zeros(10)]
    menu["dead_then_forced"] = args.radius*value/np.linalg.norm(value)
    value = np.r_[-np.ones(10), np.zeros(3), 1.]
    menu["equal_signed"] = args.radius*value/np.linalg.norm(value)
    controls = {label: dict(weights=weights.tolist(), risk=evaluate(weights)) for label, weights in menu.items()}
    controls["lexical_legal"] = dict(risk=evaluate(np.zeros(14), True))
    direct = np.array([greedy_mrv(graph, learned) for graph in testing])
    controls["mrv_lcv"] = dict(risk=[float(direct.mean()), float(direct[feasible].mean())])
    weights, records, weight_history, recent = initial.copy(), [], [], []
    status = "maximum declared updates"
    for step in tqdm(range(args.steps+1), desc="Typed offline practice", mininterval=2):
        if step % args.eval_every == 0:
            risk = evaluate(weights)
            train = float(np.mean(recent)) if step else np.nan
            records.append([step, *risk, train])
            weight_history.append(weights.copy())
            np.savez_compressed(args.output/"trajectory.npz", records=np.array(records), weights=np.array(weight_history),
                initial=initial, exposure_history=exposure_history, examples=examples, solutions=solutions, learned=learned,
                relation_counts=counts, test_graphs=testing, feasible=feasible, rng=json.dumps(practice_rng.bit_generator.state))
            recent = []
            if step >= 1000:
                window = np.array(records[-10:])
                train_slope = np.polyfit(np.log(window[:, 0]), np.log(1-window[:, 3]), 1)[0]
                test_slope = np.polyfit(np.log(window[:, 0]), np.log(1-window[:, 2]), 1)[0]
                if abs(train_slope) < .003 and abs(test_slope) < .003:
                    status = "stationary training and evaluation risk"
                    break
        if step == args.steps: break
        practice_graphs = context(practice_rng, args.batch, args.vertices, args.edge_probability)
        reward, score = generate(practice_graphs, weights, learned, practice_rng)
        advantage = args.batch/(args.batch-1)*(reward-reward.mean())
        weights += args.rate*np.mean(advantage[:, None]*score, axis=0)
        weights *= min(1., args.radius/np.linalg.norm(weights))
        assert np.isfinite(weights).all() and np.linalg.norm(weights) <= args.radius+1e-10
        recent.append(float(reward.mean()))
    summary = dict(seed=args.seed, initial=records[0][1:3], final=records[-1][1:3], final_step=int(records[-1][0]), status=status,
        exposure_fit=exposure_fit, initial_weights=initial.tolist(), final_weights=weights.tolist(),
        radius=args.radius, initial_norm=float(np.linalg.norm(initial)), final_norm=float(np.linalg.norm(weights)),
        relation_counts=counts.tolist(), relation_recovered=True, feasible_fraction=float(feasible.mean()),
        exposure_attempts=attempts, controls=controls)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    records, weight_history = np.array(records), np.array(weight_history)
    fig, axes = plt.subplots(2, 1, layout="constrained")
    axes[0].plot(records[:, 0], records[:, 2], label="Offline practice")
    axes[0].axhline(records[0, 2], color="black", linestyle=":", label="Converged example fit")
    for index, key in enumerate(["uniform_legal", "dead_then_forced", "equal_signed", "mrv_lcv"]):
        axes[0].axhline(controls[key]["risk"][1], linestyle="--", color=f"C{index+1}", label=key.replace("_", " "))
    axes[0].set(ylabel="Raw validity on fresh\nsatisfiable contexts")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    axes[1].plot(records[:, 0], weight_history)
    axes[1].set(xlabel="Offline practice updates", ylabel="Fourteen generator weights")
    fig.savefig(args.output/"learning.png")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(str(args.output/"learning.png"))


if __name__ == "__main__": main()
