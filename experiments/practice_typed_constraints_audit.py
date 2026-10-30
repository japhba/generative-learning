"""Independent exact-score checks and graph-clustered paired comparisons."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from tqdm.auto import tqdm

import practice_typed_constraints as m


def exact_score(graph, weights, relation):
    exact = m.Exact(len(graph), relation)
    solutions = exact.assignments[exact.valid(graph)]
    assert len(solutions) > 0
    inputs = np.repeat(graph[None], len(solutions), axis=0)
    phi, allowed, action = m.generate(inputs, weights, relation, np.random.default_rng(0), target=solutions, record_features=True)
    probability = m.policy(phi, allowed, weights)
    chosen_p = probability[np.arange(len(action)), action].reshape(len(graph), len(solutions))
    path_p = chosen_p.prod(axis=0)
    local_score = phi[np.arange(len(action)), action]-np.einsum("bc,bcf->bf", probability, phi)
    path_score = local_score.reshape(len(graph), len(solutions), -1).sum(axis=0)
    return path_p.sum(), np.einsum("b,bf->f", path_p, path_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    rng = np.random.default_rng(20103470)
    relation = m.teacher_relations()
    errors, sampling = [], []
    for index in tqdm(range(12), desc="Exact finite-support gradient audit", mininterval=2):
        graph = m.context(rng, 1, 4, .8)[0]
        weights = rng.normal(size=14)
        weights *= 6/np.linalg.norm(weights)
        objective, gradient = exact_score(graph, weights, relation)
        numeric = np.zeros(14)
        for coordinate in range(14):
            shift = np.zeros(14); shift[coordinate] = 1e-5
            numeric[coordinate] = (exact_score(graph, weights+shift, relation)[0]-exact_score(graph, weights-shift, relation)[0])/2e-5
        error = float(np.max(abs(gradient-numeric)))
        errors.append(error)
        assert error < 1e-8
        observed, scores = m.generate(np.repeat(graph[None], 65536, axis=0), weights, relation, rng)
        assert -1e-12 <= objective <= 1+1e-12
        bounded = np.clip(objective, 0., 1.)
        standard_error = np.sqrt(bounded*(1-bounded)/len(observed))
        assert abs(observed.mean()-objective) <= 6*standard_error+1e-12
        empirical_gradient = np.mean(observed[:, None]*scores, axis=0)
        gradient_se = np.std(observed[:, None]*scores, axis=0, ddof=1)/np.sqrt(len(observed))
        assert np.all(abs(empirical_gradient-gradient) <= 6*gradient_se+1e-12)
        sampling.append(dict(world=index, exact_success=float(objective), sampled_success=float(observed.mean()),
                             success_se=float(standard_error), max_gradient_difference=float(np.max(abs(empirical_gradient-gradient)))))
    source_summary = json.loads((args.source/"summary.json").read_text())
    source_protocol = json.loads((args.source/"protocol.json").read_text())
    run = np.load(args.source/"trajectory.npz")
    attempts = source_protocol["arguments"]["test_attempts"]
    inputs = np.repeat(run["test_graphs"], attempts, axis=0)
    stream = np.random.SeedSequence(source_summary["seed"]).spawn(4)[3]
    evaluation_seed = int(np.random.default_rng(stream).integers(2**32))
    weights = {"initial": np.array(source_summary["initial_weights"]), "practice": np.array(source_summary["final_weights"])}
    weights.update({key: np.array(value["weights"]) for key, value in source_summary["controls"].items() if "weights" in value})
    samples = {}
    for label, vector in tqdm(weights.items(), desc="Matched raw generation", mininterval=2):
        reward, _ = m.generate(inputs, vector, run["learned"], np.random.default_rng(evaluation_seed))
        samples[label] = reward.reshape(len(run["test_graphs"]), attempts)
    assert abs(samples["initial"][run["feasible"]].mean()-source_summary["initial"][1]) < 1e-12
    assert abs(samples["practice"][run["feasible"]].mean()-source_summary["final"][1]) < 1e-12
    conditional = {key: value[run["feasible"]].mean(axis=-1) for key, value in samples.items()}
    n = int(run["feasible"].sum())
    bootstrap = rng.integers(n, size=(20000, n))
    comparisons = {}
    for label in weights:
        if label == "practice": continue
        difference = conditional["practice"]-conditional[label]
        means = difference[bootstrap].mean(axis=-1)
        comparisons[label] = dict(difference=float(difference.mean()), bootstrap95=np.quantile(means, [.025, .975]).tolist(),
            simultaneous_bonferroni95=np.quantile(means, [.025/(len(weights)-1), 1-.025/(len(weights)-1)]).tolist())
    np.savez_compressed(args.output/"matched_samples.npz", **samples, feasible=run["feasible"])
    result = dict(exact_gradient_max_error=max(errors), exact_gradient_checks=12, sampling=sampling,
                  conditional_graphs=n, attempts_per_graph=attempts, contrasts=comparisons,
                  note="Retrospective pilot graph-clustered bootstrap; interval does not include variation across training datasets.")
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
