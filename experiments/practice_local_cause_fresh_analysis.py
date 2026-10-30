"""Analyze the amended fresh confirmation without writing canonical plots."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import practice_drawing_library as m
import practice_drawing_sampled as s
import practice_local_cause_partial as partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.source / "protocol.json").read_text())
    inputs = json.loads((args.inputs / "summary.json").read_text())
    n, rate = protocol["arguments"]["worlds"], protocol["arguments"]["rate"]
    assert len(inputs) == n
    conditions = ["sleep", "fixed_cache"]
    runs = {c: [np.load(args.source / f"run_{i}/{c}.npz") for i in range(n)] for c in conditions}
    bootstrap = np.random.default_rng(20104299).integers(n, size=(20000, n))

    def estimate(values):
        values = np.asarray(values)
        boot = values[bootstrap].mean(axis=-1)
        return dict(mean=float(values.mean()), per_world=values.tolist(),
                    bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                    bootstrap9833333333333333=np.quantile(boot, [1/120, 119/120]).tolist())

    final = {c: np.array([r["records"][-1, 2] for r in runs[c]]) for c in conditions}
    one_step = np.array([inputs[str(i)]["one_step"]["population_kl"] for i in range(n)])
    analytic = np.array([inputs[str(i)]["analytic"]["cv_kl"] for i in range(n)])
    result = dict(worlds=n, design="Fresh seed panel with explicitly declared empty-memory amendment after initialization failure.",
                  initial=estimate([r["records"][0, 2] for r in runs["sleep"]]),
                  final={c: estimate(final[c]) for c in conditions}, one_step=estimate(one_step),
                  analytic=estimate(analytic), comparisons={}, recognition={}, certificate={}, audit={}, stopping={},
                  empty_memory={}, weight_statistics={})
    for name, values in [("fixed_cache", final["fixed_cache"]), ("one_step", one_step), ("analytic", analytic)]:
        result["comparisons"][name] = estimate(final["sleep"] - values)
    for c in conditions:
        result["recognition"][c] = dict(initial_valid=estimate([r["records"][0, 6] for r in runs[c]]),
                                        final_valid=estimate([r["records"][-1, 6] for r in runs[c]]),
                                        final_true_program_nll=estimate([r["records"][-1, 5] for r in runs[c]]))
        result["certificate"][c] = dict(final=estimate([r["records"][-1, 4] for r in runs[c]]),
                                        positive_worlds=int(sum(r["records"][-1, 4] > 0 for r in runs[c])))
        audit_rows, empty_rows = [], []
        for i, run in enumerate(runs[c]):
            initial = np.load(args.inputs / f"run_{i}/initial.npz")
            caches = list(np.split(run["caches"], np.cumsum(run["cache_sizes"])[:-1]))
            codes, weights, bound = partial.posterior(caches, initial["mass"], run["a"], run["b"])
            assert np.all(m.render(s.decode(codes)) == np.repeat(initial["targets"], run["cache_sizes"]))
            assert abs(bound + run["records"][-1, 3]) < 1e-9
            assert np.max(np.diff(run["records"][:, 3])) < 1e-9
            assert np.min(run["records"][:, 3] - run["records"][:, 1]) > -1e-9
            assert np.min(run["conditional_gains"] - rate*(1-.75*rate)*run["gradient_square"]) > -1e-10
            assert np.max(abs(run["records"][:, 4] + run["records"][:, 3] + initial["upper_ll"])) < 1e-9
            assert run["records"][0, 1] + initial["upper_ll"] >= 0
            assert np.max(np.diff(run["empty_records"][:, 1])) <= 0
            assert np.max(abs(initial["recognition"])) == 0
            if c == "fixed_cache":
                assert np.array_equal(initial["caches"], run["caches"])
                assert np.array_equal(initial["cache_sizes"], run["cache_sizes"])
            audit_rows.append(dict(world=i, updates=len(run["conditional_gains"]),
                                   minimum_gain_minus_guarantee=float(np.min(run["conditional_gains"] - rate*(1-.75*rate)*run["gradient_square"])),
                                   largest_nll_increase=float(np.diff(run["records"][:, 1]).max()),
                                   final_active_empirical_mass=float(weights.sum()),
                                   initial_upper_slack=float(run["records"][0, 1] + initial["upper_ll"])))
            empty_rows.append(dict(world=i, initial=int(run["empty_records"][0, 1]),
                                   final=int(run["empty_records"][-1, 1]), trajectory=run["empty_records"].tolist()))
        result["audit"][c] = audit_rows
        result["empty_memory"][c] = empty_rows
        result["stopping"][c] = [dict(step=int(r["records"][-1, 0]), status=str(r["status"])) for r in runs[c]]
    permutations = list(itertools.permutations(range(3)))
    entropies, distances = {"initial": [], "final": []}, {"initial": [], "final": []}
    for i, run in enumerate(runs["sleep"]):
        world = np.load(args.inputs / f"run_{i}/world.npz")
        for label, b in [("initial", world["initial_b"]), ("final", run["b"])]:
            entropies[label].append(-np.mean(np.sum(b*np.log(b), axis=-1)))
            distances[label].append(min(np.mean((world["teacher_b"] - b[list(p)])**2) for p in permutations))
    result["weight_statistics"] = dict(command_entropy={k: estimate(v) for k, v in entropies.items()},
                                       permutation_matched_mse={k: estimate(v) for k, v in distances.items()})
    (args.source / "fresh_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    display = {key: result[key] for key in ["worlds", "initial", "final", "one_step", "analytic", "comparisons", "recognition", "certificate", "weight_statistics"]}
    print(json.dumps(display, indent=2))


if __name__ == "__main__": main()
